import numpy as np
from environment import Environment


TEST_XML = 'assets/mujoco_models/mocap_v2.xml'
MODEL_XML = "assets/mujoco_models/humanoid.xml"


def main():
    env = Environment(MODEL_XML)
    env.sim.reset()
    # print("qpos state shape: ", env.sim.data.qpos.shape)
    # print("number of nqs: ", env.model.nq)
    while True:
        #env.sim.reset()
        #env.set_pose(pose)
        env.sim.forward()
        env.viewer.render()
        env.step += 1
        # if env.step == 100:
        #     env.sim_save()
        #     break
        
if __name__ == '__main__':
    main()