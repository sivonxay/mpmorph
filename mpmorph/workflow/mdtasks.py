from fireworks import explicit_serialize, FireTaskBase, FWAction, Firework, Workflow
from mpmorph.runners.amorphous_maker import AmorphousMaker
from mpmorph.runners.rescale_volume import RescaleVolume
from mpmorph.analysis.md_data import parse_pressure
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.vasp.fireworks.core import MDFW
from atomate.vasp import powerups
from pymatgen.io.vasp import Poscar,Xdatcar
from atomate.common.firetasks.glue_tasks import PassCalcLocs
import shutil
import numpy as np
from pymatgen.io.vasp.sets import MITMDSet
import os

__author__ = 'Muratahan Aykol <maykol@lbl.gov>'


# TODO: 2. Add option to lead to a production run of specified length after density is found
# TODO: 3. Switch to MPRelax Parameters in MD
# TODO: 4. Database insertion?
# TODO: 5. Parser tasks

@explicit_serialize
class AmorphousMakerTask(FireTaskBase):
    """
    Create a constrained-random packed structure from composition and box dimensions.
    Required params:
        composition: (dict) a dict of target composition with integer atom numbers
                        e.g. {"V":22, "Li":10, "O":75, "B":10}
        box_scale: (float) all lattice vectors are multiplied with this scalar value.
                        e.g. edge length of a cubic simulation box.
    Optional params:
        tol (float): tolerance factor for how close the atoms can get (angstroms).
                        e.g. tol = 2.0 angstroms
        packmol_path (str): path to the packmol executable. Defaults to "packmol"
        clean (bool): whether the intermedite files generated are deleted. Defaults to True.
    """

    required_params = ["composition", "box_scale"]
    optional_params = ["packmol_path", "clean", "tol"]

    def run_task(self, fw_spec):
        glass = AmorphousMaker(self.get("composition"), self.get("box_scale"), self.get("tol", 2.0),
                               packmol_path=self.get("packmol_path", "packmol"),
                               clean=self.get("clean", True))
        structure = glass.random_packed_structure.as_dict()
        return FWAction(stored_data=structure)


@explicit_serialize
class GetPressureTask(FireTaskBase):
    required_params = ["outcar_path"]
    optional_params = ["averaging_fraction"]

    def run_task(self, fw_spec):
        p = parse_pressure(self["outcar_path"], self.get("averaging_fraction", 0.5))
        if fw_spec['avg_pres']:
            fw_spec['avg_pres'].append(p[0] * 1000)
        else:
            fw_spec['avg_pres'] = [p[0] * 1000]
        return FWAction()


@explicit_serialize
class SpawnMDFWTask(FireTaskBase):
    """
    Decides if a new MD calculation should be spawned or if density is found. If so, spawns a new calculation.
    """
    required_params = ["pressure_threshold", "max_rescales", "vasp_cmd", "wall_time",
                       "db_file", "spawn_count", "copy_calcs", "calc_home"]
    optional_params = ["averaging_fraction", "cool", "final_run", "final_run_steps",
                       "diffusion", "temperature", "priority_spec", "rsv_beta", "snaps"]

    def run_task(self, fw_spec):
        vasp_cmd = self["vasp_cmd"]
        wall_time = self["wall_time"]
        db_file = self["db_file"]
        max_rescales = self["max_rescales"]
        pressure_threshold = self["pressure_threshold"]
        spawn_count = self["spawn_count"]
        calc_home = self["calc_home"]
        copy_calcs = self["copy_calcs"]
        temperature = self.get("temperature", 2500)
        diffusion_bool = self.get("diffusion", False)
        priority_spec = self.get("priority_spec", {})
        rsv_beta = self.get("rsv_beta", 0.000002)
        snaps = self.get("snaps", 10)

        if spawn_count > max_rescales:
            # TODO: Log max rescale reached info.
            _temp_var = 1
            #return FWAction(defuse_workflow=True)

        name = ("spawnrun" + str(spawn_count))

        current_dir = os.getcwd()
        snaps = self.get("cool", False)
        averaging_fraction = self.get("averaging_fraction", 0.5)
        p = parse_pressure("./", averaging_fraction)[0]

        final_run = self.get("final_run", True)

        pressure_threshold = 5

        if np.fabs(p) > pressure_threshold:
            t = []
            # Copy the VASP outputs from previous run. Very first run get its from the initial MDWF which
            # uses PassCalcLocs. For the rest we just specify the previous dir.
            if spawn_count == 0:
                t.append(CopyVaspOutputs(calc_dir=current_dir, contcar_to_poscar=False))
            else:
                t.append(CopyVaspOutputs(calc_dir=current_dir, contcar_to_poscar=True))

            t.append(RescaleVolumeTask(initial_pressure=p * 1000.0, initial_temperature=1, beta = rsv_beta))
            t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, gamma_vasp_cmd=">>gamma_vasp_cmd<<",
                                      handler_group="md", wall_time=wall_time, gzip_output=False))
            t.append(PassCalcLocs(name=name))
            # Will implement the database insertion
            # t.append(VaspToDbTask(db_file=db_file,
            #                       additional_fields={"task_label": "density_adjustment"}))
            if copy_calcs:
                t.append(CopyCalsHome(calc_home=calc_home, run_name=name))
            t.append(SpawnMDFWTask(pressure_threshold=pressure_threshold,
                                   max_rescales=max_rescales,
                                   wall_time=wall_time,
                                   vasp_cmd=vasp_cmd,
                                   db_file=db_file,
                                   spawn_count=spawn_count + 1,
                                   copy_calcs=copy_calcs,
                                   calc_home=calc_home,
                                   averaging_fraction=averaging_fraction,
                                   cool=snaps,
                                   temperature=temperature,
                                   diffusion=diffusion_bool,
                                   final_run=final_run,
                                   priority_spec = priority_spec,
                                   rsv_beta=rsv_beta,
                                   snaps=snaps))
            new_fw = Firework(t, name=name, spec=priority_spec)
            return FWAction(stored_data={'pressure': p}, additions=[new_fw])
        else:
            fw_list = []
            if os.path.exists(os.path.join(current_dir, 'CONTCAR')):
                _poscar = Poscar.from_file(os.path.join(current_dir, 'CONTCAR'))
            else:
                _poscar = Poscar.from_file(os.path.join(current_dir, 'POSCAR'))

            name = str(_poscar.structure.composition.reduced_formula)
            if final_run or snaps:
                if diffusion_bool:
                    _steps = 40000
                    _name = "diffusion_run"
                else:
                    _steps = 10000
                    _name = "longrun"
                fw_list = self.get_final_run_fws(_poscar.structure, name=_name, copy_calcs=copy_calcs,
                                                 calc_home=calc_home, target_steps=_steps, temperature=temperature,
                                                 priority_spec=priority_spec)
            if snaps:
                t = []
                t.append(CopyVaspOutputs(calc_loc=True, contcar_to_poscar=True, additional_files=["XDATCAR"]))
                t.append(StructureSamplerTask(copy_calcs=copy_calcs, calc_home=calc_home, n_snapshots=snaps, priority_spec=priority_spec))
                if len(fw_list) > 0:
                    new_fw = Firework(t, name=name + "structure_sampler", parents=fw_list[len(fw_list)-1], spec=priority_spec)
                else:
                    new_fw = Firework(t, name=name + "structure_sampler", spec=priority_spec)
                fw_list.append(new_fw)
            if snaps or final_run:
                wf = Workflow(fw_list, name=name + "_" + str(temperature) + "_longruns")
                wf = powerups.add_modify_incar_envchk(wf)
                return FWAction(stored_data={'pressure':p, 'density_calculated': True}, additions=wf)
            return FWAction(stored_data={'pressure':p, 'density_calculated': True})

    def get_final_run_fws(self, structure, target_steps=40000, copy_calcs=False, calc_home=None,
                          run_steps=5000, run_time = 86400, temperature=2500, vasp_cmd=">>vasp_cmd<<", db_file=None, name="longrun",
                   optional_MDWF_params=None, override_default_vasp_params=None, vasp_input_set=None, priority_spec={}):
        fw_list = []
        _steps = 0
        spawn_count = 0

        optional_MDWF_params = optional_MDWF_params or {}
        optional_MDWF_params['spec'] = priority_spec
        override_default_vasp_params = override_default_vasp_params or {}
        override_default_vasp_params['user_incar_settings'] = override_default_vasp_params.get(
            'user_incar_settings') or {}
        override_default_vasp_params['user_incar_settings'].update({"ISIF": 1, "LWAVE": False})

        fw1 = MDFW(structure=structure, start_temp=temperature, end_temp=temperature, nsteps=run_steps,
                   name=name + "_" + str(spawn_count), vasp_input_set=vasp_input_set, db_file=db_file,
                   vasp_cmd=vasp_cmd, wall_time=run_time, copy_vasp_outputs=False, override_default_vasp_params=override_default_vasp_params,
                   **optional_MDWF_params)
        _steps += run_steps
        spawn_count += 1
        fw_list.append(fw1)

        while _steps < target_steps:
            _name = (name + "_" + str(spawn_count))
            t = []
            t.append(CopyVaspOutputs(calc_loc=True, contcar_to_poscar=True, additional_files=["XDATCAR", "OSZICAR", "DOSCAR"]))
            if spawn_count == 1:
                if copy_calcs:
                    t.append(CopyCalsHome(calc_home=calc_home, run_name=name + "_0"))
            t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, gamma_vasp_cmd=">>gamma_vasp_cmd<<",
                                      handler_group="md", wall_time=run_time, gzip_output=False))
            if copy_calcs:
                t.append(CopyCalsHome(calc_home=calc_home, run_name=_name))
            t.append(PassCalcLocs(name=_name))
            new_fw = Firework(tasks=t, name=_name, parents=[fw_list[spawn_count-1]], spec=priority_spec)
            _steps += run_steps
            spawn_count += 1
            fw_list.append(new_fw)
        return fw_list

@explicit_serialize
class RescaleVolumeTask(FireTaskBase):
    """
    Volume rescaling
    """
    required_params = ["initial_temperature", "initial_pressure"]
    optional_params = ["target_pressure", "target_temperature", "target_pressure", "alpha", "beta"]

    def run_task(self, fw_spec):
        # Initialize volume correction object with last structure from last_run
        initial_temperature = self["initial_temperature"]
        initial_pressure = self["initial_pressure"]
        target_temperature = self.get("target_temperature", initial_temperature)
        target_pressure = self.get("target_pressure", 0.0)
        alpha = self.get("alpha", 10e-6)
        beta = self.get("beta", 10e-7)
        corr_vol = RescaleVolume.of_poscar(poscar_path="./POSCAR", initial_temperature=initial_temperature,
                                           initial_pressure=initial_pressure,
                                           target_pressure=target_pressure,
                                           target_temperature=target_temperature, alpha=alpha, beta=beta)
        # Rescale volume based on temperature difference first. Const T will return no volume change:
        corr_vol.by_thermo(scale='temperature')
        # TO DB ("Rescaled volume due to delta T: ", corr_vol.structure.volume)
        # Rescale volume based on pressure difference:
        corr_vol.by_thermo(scale='pressure')
        # TO DB ("Rescaled volume due to delta P: ", corr_vol.structure.volume)
        corr_vol.poscar.write_file("./POSCAR")
        # Pass the rescaled volume to Poscar
        return FWAction(stored_data=corr_vol.structure.as_dict())

@explicit_serialize
class CopyCalsHome(FireTaskBase):
    required_params = ["calc_home", "run_name"]
    optional_params = ["files"]

    def run_task(self, fw_spec):
        default_list = ["INCAR", "POSCAR", "CONTCAR", "OUTCAR", "POTCAR", "vasprun.xml", "XDATCAR", "OSZICAR", "DOSCAR"]
        files = self.get("files", default_list)
        calc_home = self["calc_home"]
        run_name = self["run_name"]
        target_dir = os.path.join(calc_home, run_name)
        if not os.path.exists(calc_home):
            os.mkdir(calc_home)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for f in files:
            try:
                shutil.copy2(f, target_dir)
            except:
                pass
        return FWAction()

@explicit_serialize
class StructureSamplerTask(FireTaskBase):
    """

    """
    required_params = ["copy_calcs", "calc_home"]
    optional_params = ["n_snapshots", "priority_spec"]

    def run_task(self, fw_spec):
        copy_calcs = self["copy_calcs"]
        calc_home = self["calc_home"]
        n = self.get("n_snapshots", 10)
        priority_spec = self.get("priority_spec", {})

        current_dir = os.getcwd()
        xdatcar_file = os.path.join(current_dir, 'XDATCAR')
        wfs = get_wf_structure_sampler(xdatcar_file=xdatcar_file, sim_anneal=True, copy_calcs=copy_calcs,
                                       calc_home=calc_home, n=n, db_file=None, priority_spec=priority_spec)
        return FWAction(additions=wfs)

@explicit_serialize
class RelaxStaticTask(FireTaskBase):

    required_params = ["copy_calcs", "calc_home"]
    optional_params = ["name", "db_file", "snap_num", "priority_spec", "temps"]
    def run_task(self, fw_spec):
        copy_calcs = self["copy_calcs"]
        calc_home = self["calc_home"]
        snap_num = self.get("snap_num", 0)
        db_file = self.get("db_file", None)
        priority_spec = self.get("priority_spec", {})

        if os.path.exists(os.path.join(os.getcwd(),'XDATCAR.gz')):
            xdat = Xdatcar(os.path.join(os.getcwd(),'XDATCAR.gz'))
        else:
            xdat = Xdatcar(os.path.join(os.getcwd(), 'XDATCAR'))
        structure = xdat.structures[len(xdat.structures)-1]

        wfs = get_relax_static_wf([structure], name = "relax_static", copy_calcs = copy_calcs,
                                  calc_home=calc_home, snap=snap_num, priority_spec=priority_spec)

        if snap_num == 0:
            name = str(structure.composition.reduced_formula)
            temps = self.get("temps", [500, 1000, 1500])
            for temp in temps:
                _wf = get_wf_density(structure=structure, temperature=temp, pressure_threshold=5,
                                     name=name + "_snap_" + str(snap_num) + '_diffusion_' + str(temp), db_file=db_file,
                                     copy_calcs=copy_calcs, calc_home=calc_home, cool=False, diffusion=True,
                                     priority_spec=priority_spec)
                wfs.append(_wf)

        return FWAction(additions=wfs)

@explicit_serialize
class DiffusionTask(FireTaskBase):

    required_params = ["copy_calcs", "calc_home", "snap_num"]
    optional_params = ["temps", "name", "db_file", "priority_spec", "structure"]
    def run_task(self, fw_spec):
        copy_calcs = self["copy_calcs"]
        calc_home = self["calc_home"]
        snap_num = self["snap_num"]
        db_file = self.get("db_file", None)
        priority_spec = self.get("priority_spec", {})
        structure = self.get("structure", None)
        if structure == None:
            if os.path.exists(os.path.join(os.getcwd(),'XDATCAR.gz')):
                xdat = Xdatcar(os.path.join(os.getcwd(),'XDATCAR.gz'))
            else:
                xdat = Xdatcar(os.path.join(os.getcwd(), 'XDATCAR'))
            structure = xdat.structures[len(xdat.structures)-1]

        name = str(structure.composition.reduced_formula)
        temps = self.get("temps", [500, 1000, 1500])
        wfs = []
        for temp in temps:
            _wf = get_wf_density(structure=structure, temperature=temp, pressure_threshold=5,
                                name = name+"_snap_"+str(snap_num)+'_diffusion_'+str(temp), db_file=db_file,
                                copy_calcs=copy_calcs, calc_home=calc_home, cool=False, diffusion=True, priority_spec=priority_spec)
            _wf = powerups.add_modify_incar_envchk(_wf)
            wfs.append(_wf)
        return FWAction(additions=wfs)

@explicit_serialize
class WriteSetTask(FireTaskBase):
    required_params = ["start_temp", "end_temp", "nsteps"]
    optional_params = ["override_default_vasp_params"]
    def run_task(self, fw_spec):
        start_temp = self["start_temp"]
        end_temp = self["end_temp"]
        nsteps = self["nsteps"]
        pos = Poscar.from_file(os.path.join(os.getcwd(), 'POSCAR'))
        structure = pos.structure
        override_default_vasp_params = self.get("override_default_vasp_params", {})
        vasp_input_set = MITMDSet(structure, start_temp=start_temp, end_temp=end_temp,
                                                    nsteps=nsteps, **override_default_vasp_params)
        vasp_input_set.write_input(".")
        return FWAction()

@explicit_serialize
class DoNothingTask(FireTaskBase):

    def run_task(self, fw_spec):
        return FWAction()

@explicit_serialize
class VaspMdToDbTask(FireTaskBase):
    pass


@explicit_serialize
class VaspMdToDiffusion(FireTaskBase):
    pass


@explicit_serialize
class VaspMdToStructuralAnalysis(FireTaskBase):
    pass

from mpmorph.workflow.workflows import get_wf_structure_sampler, get_relax_static_wf, get_wf_density