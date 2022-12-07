import numpy as np
import torch
from torch.utils.data import DataLoader
from utils   import h36motion as datasets
from utils.data_utils import *
from utils.parser import args
from environment import Environment


TEST_XML = 'assets/mujoco_models/mocap_v2.xml'
MODEL_XML = "assets/mujoco_models/humanoid.xml"

EXC_JOINT_LIST = ['Hips', 'Spine2_x', 'Spine2_y', 'Spine2_z',
                  'Spine3_x', 'Spine3_y', 'Spine3_z', 'RightForeArm_z',
                  'LeftForeArm_z', 'RightLeg_x', 'LeftLeg_x']

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device: %s'%device)

def pose_generator():
    print()
    print()

    env = Environment(MODEL_XML)
    env.sim.reset()

    actions = define_actions(args.actions_to_consider)
    
    euler_order = [1, 2, 0]

    joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
                "LeftFoot",
                "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                "LeftForeArm",
                "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
                "RightForeArm",
                "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]
    
    dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                            43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                            86])


    for action in actions:
        dataset = datasets.Datasets(
            args.data_dir,
            args.input_n, 
            args.output_n, 
            args.skip_rate, 
            split=2, 
            actions=[action]
        )

        print('>>> test action for sequences: {:d}\n'.format(dataset.__len__()))

        data_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size_test, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )

        print()
        print()

        for ix, batch in enumerate(data_loader):
            #batch=batch.to(device)           
            #sequences_gt=batch[:, 0:args.input_n, :]
            for samples in batch:
                for sample in samples:
                    #sample = np.resize(sample, 59)
                    #sample[48:] = 0

                    # exc_joint_indices = get_redundant_joint_indices(env, EXC_JOINT_LIST)
                    # for ix in exc_joint_indices:
                    #     sample = _insert(sample, ix)

                    new_qpos = np.zeros(59)

                    ## Mapping from pose to model
                    # For hip joints 2 4 5 6
                    
                    # new_qpos[4:7] = sample[3:6]         # Hip
                    new_qpos[45:48] = sample[3:6]       # RHip  RightUpLeg
                    new_qpos[48] = sample[6]            # RKnee  RightLeg
                    new_qpos[49:52] = sample[9:12]      # RFoot  RightFoot

                    new_qpos[52:55] = sample[19:22]     # LHip  LeftUpLeg
                    new_qpos[55] = sample[22]           # LKnee  LeftLeg
                    new_qpos[56:59] = sample[25:28]     # LFoot  LeftFoot
 
                    new_qpos[7:10] = sample[35:38]      # Spine  Spine
                    new_qpos[10:13] = sample[38:41]     # Spine1   Spine1
                    new_qpos[19:22] = sample[41:44]     # Neck  Neck
                    new_qpos[22:25] = sample[44:47]     # Head  Head
   
                    new_qpos[35:38] = sample[50:53]     # LShoulder  LeftShoulder
                    #new_qpos[38:41] = sample[53:56]     # LElbow  LeftArm
                    new_qpos[41] = sample[58]           # LElbow  LeftArm 56 57 58
                    new_qpos[42:45] = sample[59:62]     # LWrist  LeftHand
   
                    new_qpos[25:28] = sample[74:77]     # RShoulder  RightShoulder
                    #new_qpos[28:31] = sample[77:80]     # RElbow  RightArm
                    new_qpos[31] = sample[82]           # RElbow  RightArm 79 80 81
                    new_qpos[32:35] = sample[83:86]     # RWrist  RightHand

                    
                    # Not Used new_qpos[13:16] = sample[21:24]  # Spine2
                    # Not Used new_qpos[16:19] = sample[21:24]  # Spine3

                    env.set_pose(new_qpos)
                    env.sim.forward()
                    env.viewer.render()
                    env.step += 1

def get_redundant_joint_indices(env, joint_list):
    """
    Returns redundant joint's indices i.e. static joints.
    """
    joint_addrs = []
    for joint in joint_list:
        addr = env._get_joint_qpos_addr(joint)
        if isinstance(addr, tuple):
            for each in addr:
                if each == 7:
                    continue
                joint_addrs.append(each)
        else:
            joint_addrs.append(addr)
    return joint_addrs

def _insert(arr, index):
    """
    Inserts element into given index.
    """
    arr = np.insert(arr, index, 0)
    return arr

if __name__ == '__main__':
    pose_generator()


















# class Visualizer(Environment):
#     def __init__(self, xml_path):
#         super().__init__(xml_path)
#         self.pose_gen = self.pose_generator()
#         self.pose = next(self.pose_gen)

#     def pose_generator(self):
#         actions = define_actions(args.actions_to_consider)
#         for action in actions:
#             dataset_test = datasets.Datasets(
#                 args.data_dir,
#                 args.input_n, 
#                 args.output_n, 
#                 args.skip_rate, 
#                 split=2, 
#                 actions=[action]
#             )

#             print('>>> test action for sequences: {:d}'.format(dataset_test.__len__()))

#             data_loader = DataLoader(
#                 dataset_test, 
#                 batch_size=args.batch_size_test, 
#                 shuffle=False, 
#                 num_workers=0, 
#                 pin_memory=True
#             )

#             with torch.no_grad():
#                 for ix, batch in enumerate(data_loader):
#                         for samples in batch:
#                             for sample in samples:
#                                 yield sample


# def main():
#     env = Visualizer(MODEL_XML)
#     env.sim.reset()
#     while True:
#         #env.sim.reset()
#         env.set_pose(env.pose)
#         env.sim.forward()
#         env.viewer.render()
#         env.step += 1
#         # if env.step == 100:
#         #     env.sim_save()
#         #     break
