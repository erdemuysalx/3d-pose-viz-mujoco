import mujoco_py
from mujoco_py import MjSim, MjViewer, functions


class Environment:
    def __init__(self, xml_path):
        self.mj_path = mujoco_py.utils.discover_mujoco()
        self.xml_path = xml_path
        self.model = mujoco_py.load_model_from_path(self.xml_path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.body_names = self.model.geom_names
        self.joint_names = self.model.joint_names
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.step = 0

        mujoco_py.ignore_mujoco_warnings()
        self.sim.reset()

    def print_model_info(self):
        print(f"Sim state {self.sim.get_state()}\n")
        print(f"Model geom names: {self.model.geom_names} in total {len(self.model.geom_names)}\n")
        print(f"Model joint names: {self.model.joint_names} in total {len(self.model.joint_names)}\n")
        print(f"Number of data ctrl: {len(self.sim.data.ctrl)}\n")
        print(f"Sim data ctrl: {self.sim.data.ctrl}\n")
        print(f"Number of data qpos: {len(self.sim.data.qpos)}\n")
        print(f"Sim data qpos: {self.sim.data.qpos}\n")
    
    def _get_state(self):
        return self.sim.get_state()

    def _set_state(self, state):
        self.sim.set_state(state)

    def _set_joint_qpos(self, joint, state):
        self.sim.set_joint_qpos(joint, state)

    def _get_joint_qpos(self, joint):
        print(f"{joint} qpos:", self.sim.data.get_joint_qpos(joint))

    def _get_body_xpos(self, body):
        print(f"{body} xpos:", self.sim.data.get_body_xpos(body))

    def get_target_pos(self):
        NotImplementedError()

    def set_target_pos(self, target):
        for ix, data in enumerate(target):
            self.model.body_pos[ix] = data
            if ix == 59:
                break
        self.sim.set_constants()

    def pose_generator(self):
        NotImplementedError()

    def set_pose(self, pose):
        self.sim.data.qpos[:self.model.nq] = pose
        #self.sim.data.qpos[self.model.nq] += 1.0

    def sim_save(self):
        functions.mj_saveLastXML(
            "assets/mujoco_models/saved_humanoid.xml",
            self.model,
            "", 
            512
        )