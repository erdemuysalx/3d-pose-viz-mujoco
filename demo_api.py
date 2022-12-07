import numpy as np
from environment import Environment


TEST_XML = 'assets/mujoco_models/mocap_v2.xml'
MODEL_XML = "assets/mujoco_models/humanoid.xml"


def main():
    env = Environment(MODEL_XML)
    env.sim.reset()
    print("Number of bodies: ", (env.model.body_names))
    #print("Number of joints: ", len(env.model.joint_names))
    #print("Name of joints: ", env.model.joint_names)
    # print("Number of qpos: ", env.model.nq)
    while True:
        #env.sim.reset()
        #env.set_pose(pose)
        #env.sim.forward()
        #env.viewer.render()
        env.step += 1
        # if env.step == 100:
        #     env.sim_save()
        #     break

if __name__ == '__main__':
    main()













    # print("Spine_x addres: ",env.model.get_joint_qpos_addr('Spine_x'))
    # print("Spine_y addres: ",env.model.get_joint_qpos_addr('Spine_y'))
    # print("Spine_z addres: ",env.model.get_joint_qpos_addr('Spine_z'))
    # print("Spine1_x addres: ",env.model.get_joint_qpos_addr('Spine1_x'))
    # print("Spine1_y addres: ",env.model.get_joint_qpos_addr('Spine1_y'))
    # print("Spine1_z addres: ",env.model.get_joint_qpos_addr('Spine1_z'))
    # print("Spine2_x addres: ",env.model.get_joint_qpos_addr('Spine2_x'))
    # print("Spine2_y addres: ",env.model.get_joint_qpos_addr('Spine2_y'))
    # print("Spine2_z addres: ",env.model.get_joint_qpos_addr('Spine2_z'))
    # print("Spine3_x addres: ",env.model.get_joint_qpos_addr('Spine3_x'))
    # print("Spine3_y addres: ",env.model.get_joint_qpos_addr('Spine3_y'))
    # print("Spine3_z addres: ",env.model.get_joint_qpos_addr('Spine3_z'))
    # print("Neck_x addres: ",env.model.get_joint_qpos_addr('Neck_x'))
    # print("Neck_y  addres: ",env.model.get_joint_qpos_addr('Neck_y'))
    # print("Neck_z  addres: ",env.model.get_joint_qpos_addr('Neck_z'))
    # print("Head_x addres: ",env.model.get_joint_qpos_addr('Head_x'))
    # print("Head_y addres: ",env.model.get_joint_qpos_addr('Head_y'))
    # print("Head_z addres: ",env.model.get_joint_qpos_addr('Head_z'))
    # print("RightShoulder_x addres: ",env.model.get_joint_qpos_addr('RightShoulder_x'))
    # print("RightShoulder_y addres: ",env.model.get_joint_qpos_addr('RightShoulder_y'))
    # print("RightShoulder_z addres: ",env.model.get_joint_qpos_addr('RightShoulder_z'))
    # print("RightArm_x addres: ",env.model.get_joint_qpos_addr('RightArm_x'))
    # print("RightArm_y addres: ",env.model.get_joint_qpos_addr('RightArm_y'))
    # print("RightArm_z addres: ",env.model.get_joint_qpos_addr('RightArm_z'))
    # print("RightForeArm_z addres: ",env.model.get_joint_qpos_addr('RightForeArm_z'))
    # print("RightHand_x addres: ",env.model.get_joint_qpos_addr('RightHand_x'))
    # print("RightHand_y addres: ",env.model.get_joint_qpos_addr('RightHand_y'))
    # print("RightHand_z addres: ",env.model.get_joint_qpos_addr('RightHand_z'))
    # print("LeftShoulder_x addres: ",env.model.get_joint_qpos_addr('LeftShoulder_x'))
    # print("LeftShoulder_y addres: ",env.model.get_joint_qpos_addr('LeftShoulder_y'))
    # print("LeftShoulder_z addres: ",env.model.get_joint_qpos_addr('LeftShoulder_z'))
    # print("LeftArm_x addres: ",env.model.get_joint_qpos_addr('LeftArm_x'))
    # print("LeftArm_y addres: ",env.model.get_joint_qpos_addr('LeftArm_y'))
    # print("LeftArm_z addres: ",env.model.get_joint_qpos_addr('LeftArm_z'))
    # print("LeftForeArm_z addres: ",env.model.get_joint_qpos_addr('LeftForeArm_z'))
    # print("LeftHand_x addres: ",env.model.get_joint_qpos_addr('LeftHand_x'))
    # print("LeftHand_y addres: ",env.model.get_joint_qpos_addr('LeftHand_y'))
    # print("LeftHand_z addres: ",env.model.get_joint_qpos_addr('LeftHand_z'))
    # print("RightUpLeg_x addres: ",env.model.get_joint_qpos_addr('RightUpLeg_x'))
    # print("RightUpLeg_y addres: ",env.model.get_joint_qpos_addr('RightUpLeg_y'))
    # print("RightUpLeg_z addres: ",env.model.get_joint_qpos_addr('RightUpLeg_z'))
    # print("RightLeg_x addres: ",env.model.get_joint_qpos_addr('RightLeg_x'))
    # print("RightFoot_x addres: ",env.model.get_joint_qpos_addr('RightFoot_x'))
    # print("RightFoot_y addres: ",env.model.get_joint_qpos_addr('RightFoot_y'))
    # print("RightFoot_z addres: ",env.model.get_joint_qpos_addr('RightFoot_z'))
    # print("LeftUpLeg_x addres: ",env.model.get_joint_qpos_addr('LeftUpLeg_x'))
    # print("LeftUpLeg_z addres: ",env.model.get_joint_qpos_addr('LeftUpLeg_z'))
    # print("LeftUpLeg_z addres: ",env.model.get_joint_qpos_addr('LeftUpLeg_z'))
    # print("LeftLeg_x addres: ",env.model.get_joint_qpos_addr('LeftLeg_x'))
    # print("LeftFoot_x addres: ",env.model.get_joint_qpos_addr('LeftFoot_x'))
    # print("LeftFoot_y addres: ",env.model.get_joint_qpos_addr('LeftFoot_y'))
    # print("LeftFoot_z addres: ",env.model.get_joint_qpos_addr('LeftFoot_z'))