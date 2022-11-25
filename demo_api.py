import numpy as np
from environment import Environment



MODEL_XML = "assets/mujoco_models/humanoid.xml"




def main():
    env = Environment(MODEL_XML)
    env.sim.reset()
    while True:
        #env.sim.reset()

        env.sim.forward()
        env.viewer.render()

        env.step += 1
        # if env.step == 100:
        #     env.sim_save()
        #     break
        
if __name__ == '__main__':
    main()