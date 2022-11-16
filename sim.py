import os
import numpy as np
import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer


mujoco_py.ignore_mujoco_warnings()
mj_path = mujoco_py.utils.discover_mujoco()

defaul_xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
xml_path = 'assets/mujoco_models/humanoid_1205_vis_single_v1.xml'

model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
viewer = MjViewer(sim)

step = 0
body_names = model.geom_names
joint_names = model.joint_names


def print_model_info(sim):
    print(f"Sim state {sim.get_state()}\n")
    print(f"Model geom names: {model.geom_names} in total {len(model.geom_names)}\n")
    print(f"Model joint names: {model.joint_names} in total {len(model.joint_names)}\n")
    print(f"Number of data ctrl: {len(sim.data.ctrl)}\n")
    print(f"Sim data ctrl: {sim.data.ctrl}\n")
    print(f"Number of data qpos: {len(sim.data.qpos)}\n")
    print(f"Sim data qpos: {sim.data.qpos}\n")


if __name__ == '__main__':
    sim.reset()
    init_state = sim.get_state()
    print_model_info(sim)

    while True:
        sim.reset()

        for i in range(1000):
            sim.forward()
            viewer.render()

        step += 1
        if step > 1000 and os.getenv('TESTING') is not None:
            break