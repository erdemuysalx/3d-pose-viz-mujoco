import os
import sys
import time
import math
import pickle
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from STSGCN.utils import h36motion as datasets
from STSGCN.utils.data_utils import *
from STSGCN.utils.parser import args as STSGCN_args
from environment import Environment

## For MOCAP data generation
sys.path.append(os.getcwd())
from ego_pose.utils.metrics import *
from ego_pose.utils.tools import *
from ego_pose.utils.egomimic_config import Config
from ego_pose.envs.humanoid_v1 import HumanoidEnv
from envs.visual.humanoid_vis import HumanoidVisEnv


parser = argparse.ArgumentParser()
parser.add_argument('--vis-model', default='humanoid_h36m_v4')
parser.add_argument('--data-generator', default='h36m')

parser.add_argument('--egomimic-cfg', default='subject_03')
parser.add_argument('--egomimic-iter', type=int, default=3000)
parser.add_argument('--egomimic-tag', default='')
parser.add_argument('--data', default='test')
parser.add_argument('--mode', default='vis')

args = parser.parse_args()


TEST_XML = 'assets/mujoco_models/mocap_v2.xml'
MODEL_XML = 'assets/mujoco_models/%s.xml' % (args.vis_model)
H36M_MODEL_XML = 'assets/mujoco_models/humanoid_h36m_v4.xml'


def mocap_pose_generator():
    #pass
    cfg = Config(args.egomimic_cfg)
    dt = 1 / 30.0


    res_base_dir = 'results'
    em_res_path = '%s/egomimic/%s/results/iter_%04d_%s%s.p' % (res_base_dir, args.egomimic_cfg, args.egomimic_iter, args.data, args.egomimic_tag)
    em_res, em_meta = pickle.load(open(em_res_path, 'rb')) if args.egomimic_cfg is not None else (None, None)
    remove_noisy_hands(em_res)


    def update_pose():
        print('take_ind: %d, fr: %d, mfr int: %d' % (take_ind, fr, mfr_int))
        new_qpos = traj_orig[fr, :]        
        env.set_pose(new_qpos)
        env.sim.forward()
        env.step += 1

    def load_take():
        global traj_orig
        algo_res, algo = algos[algo_ind]
        if algo_res is None:
            return
        take = takes[take_ind]
        print('%s ---------- %s' % (algo, take))
        traj_orig = algo_res['traj_orig'][take]

    traj_orig = None
    takes = cfg.takes[args.data]
    algos = [(em_res, 'ego mimic')]
    algo_ind = 0
    take_ind = 0
    ss_ind = 0
    mfr_int = 10
    show_gt = False
    load_take()

    """render or select part of the clip"""
    T = 10
    fr = 0
    paused = False
    stop = False
    reverse = False

    update_pose()
    t = 0
    env = Environment(MODEL_XML)
    env.sim.reset()
    while not stop:
        if t >= math.floor(T):
            if not reverse and fr < traj_orig.shape[0] - 1:
                fr += 1
                update_pose()
            elif reverse and fr > 0:
                fr -= 1
                update_pose()
            t = 0

        env.viewer.render()
        if not paused:
            t += 1

def h36m_pose_generator():
    # Create an environment
    # env = Environment(MODEL_XML)
    # env.sim.reset()
    
    actions = define_actions(STSGCN_args.actions_to_consider)
    dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                            43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                            86])

    for action in actions:
        dataset = datasets.Datasets(
            STSGCN_args.data_dir,
            STSGCN_args.input_n, 
            STSGCN_args.output_n, 
            STSGCN_args.skip_rate, 
            split=2, 
            actions=[action]
        )

        print('>>> test action for sequences: {:d}\n'.format(dataset.__len__()))

        data_loader = DataLoader(
            dataset, 
            batch_size=STSGCN_args.batch_size_test, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
        for ix, batch in enumerate(data_loader):
            #batch=batch.to(device)           
            #sequences_gt=batch[:, 0:args.input_n, :]
            sequences_gt=batch[:, STSGCN_args.input_n:STSGCN_args.input_n+STSGCN_args.output_n, dim_used]
            for samples in batch:
                print("batch shape: ", samples.shape)
                for sample in samples:
                    yield sample

                    #sample = np.resize(sample, 59)
                    #sample[48:] = 0
                    # exc_joint_indices = get_redundant_joint_indices(env, EXC_JOINT_LIST)
                    # for ix in exc_joint_indices:
                    #     sample = insert(sample, ix)
                    # print("Sample size", sample.shape)

                    ## Mapping from pose to model
                    # new_qpos = np.zeros(59)

                    # new_qpos[0:7] = sample[0:7]         # Hip
                    # new_qpos[7:10] = 0.00000000e+00, 0.00000000e+00, 0.00000000e+00       # Spine  Spine
                    # new_qpos[10:13] = sample[21:24]     # Spine1   Spine1
                    # new_qpos[13:16] = 0.00000000e+00, 0.00000000e+00, 0.00000000e+00     # Spine2
                    # new_qpos[16:19] = 0.00000000e+00, 0.00000000e+00, 0.00000000e+00     # Spine3

                    # new_qpos[19:22] = sample[24:27]     # Neck  Neck
                    # new_qpos[22:25] = sample[27:30]     # Head  Head
                    # # 30-33 
                    # new_qpos[25:28] = sample[40:43]     # RShoulder  RightShoulder
                    # new_qpos[28:31] = sample[43:46]     # RElbow  RightArm
                    # new_qpos[31:32] = sample[46:47]     # RElbow  RightArm 79 80 81
                    # new_qpos[32:35] = 0.00000000e+00, 0.00000000e+00, 0.00000000e+00     # RWrist  RightHand

                    # new_qpos[35:38] = sample[33:36]     # LShoulder  LeftShoulder
                    # new_qpos[38:41] = sample[36:39]     # LElbow  LeftArm
                    # new_qpos[41:42] = sample[39:40]     # LElbow  LeftArm 56 57 58
                    # new_qpos[42:45] = 0.00000000e+00, 0.00000000e+00, 0.00000000e+00    # LWrist  LeftHand

                    # new_qpos[45:48] = sample[7:10]      # RHip  RightUpLeg
                    # new_qpos[48:49] = sample[10:11]     # RKnee  RightLeg
                    # new_qpos[49:52] = sample[11:14]     # RFoot  RightFoot

                    # new_qpos[52:55] = sample[14:17]     # LHip  LeftUpLeg
                    # new_qpos[55:56] = sample[17:18]     # LKnee  LeftLeg
                    # new_qpos[56:59] = sample[18:21]     # LFoot  LeftFoot
 
                    # env.set_pose(new_qpos)
                    # env.sim.forward()
                    # env.viewer.render()
                    # env.step += 1

# def get_redundant_joint_indices(env, joint_list):
#     """
#     Returns redundant joint's indices i.e. static joints.
#     """
#     joint_addrs = []
#     for joint in joint_list:
#         addr = env._get_joint_qpos_addr(joint)
#         if isinstance(addr, tuple):
#             for each in addr:
#                 if each == 7:
#                     continue
#                 joint_addrs.append(each)
#         else:
#             joint_addrs.append(addr)
#     return joint_addrs

def main():
    env = Environment(H36M_MODEL_XML)
    env.sim.reset()

    if args.data == 'mocap':
        mocap_pose_generator()
    else: 
        #h36m_pose_generator()
        pose_gen = h36m_pose_generator()
        while True:
            sample = next(pose_gen)

            ## Mapping from pose to model
            new_qpos = np.zeros(59)

            new_qpos[0:7] = sample[0:7]         # Hip
            new_qpos[7:10] = 0.00000000e+00, 0.00000000e+00, 0.00000000e+00       # Spine  Spine
            new_qpos[10:13] = sample[21:24]     # Spine1   Spine1
            new_qpos[13:16] = 0.00000000e+00, 0.00000000e+00, 0.00000000e+00     # Spine2
            new_qpos[16:19] = 0.00000000e+00, 0.00000000e+00, 0.00000000e+00     # Spine3

            new_qpos[19:22] = sample[24:27]     # Neck  Neck
            new_qpos[22:25] = sample[27:30]     # Head  Head
            # 30-33 
            new_qpos[25:28] = sample[40:43]     # RShoulder  RightShoulder
            new_qpos[28:31] = sample[43:46]     # RElbow  RightArm
            new_qpos[31:32] = sample[46:47]     # RElbow  RightArm 79 80 81
            new_qpos[32:35] = 0.00000000e+00, 0.00000000e+00, 0.00000000e+00     # RWrist  RightHand

            new_qpos[35:38] = sample[33:36]     # LShoulder  LeftShoulder
            new_qpos[38:41] = sample[36:39]     # LElbow  LeftArm
            new_qpos[41:42] = sample[39:40]     # LElbow  LeftArm 56 57 58
            new_qpos[42:45] = 0.00000000e+00, 0.00000000e+00, 0.00000000e+00    # LWrist  LeftHand

            new_qpos[45:48] = sample[7:10]      # RHip  RightUpLeg
            new_qpos[48:49] = sample[10:11]     # RKnee  RightLeg
            new_qpos[49:52] = sample[11:14]     # RFoot  RightFoot

            new_qpos[52:55] = sample[14:17]     # LHip  LeftUpLeg
            new_qpos[55:56] = sample[17:18]     # LKnee  LeftLeg
            new_qpos[56:59] = sample[18:21]     # LFoot  LeftFoot

            env.set_pose(new_qpos)
            env.sim.forward()
            env.viewer.render()
            env.step += 1

if __name__ == '__main__':
    main()


















                    # new_qpos[4:7] = sample[3:6]         # Hip
                    # new_qpos[45:48] = sample[3:6]       # RHip  RightUpLeg
                    # new_qpos[48] = sample[6]            # RKnee  RightLeg
                    # new_qpos[49:52] = sample[9:12]      # RFoot  RightFoot

                    # new_qpos[52:55] = sample[19:22]     # LHip  LeftUpLeg
                    # new_qpos[55] = sample[22]           # LKnee  LeftLeg
                    # new_qpos[56:59] = sample[25:28]     # LFoot  LeftFoot
 
                    # new_qpos[7:10] = sample[35:38]      # Spine  Spine
                    # new_qpos[10:13] = sample[38:41]     # Spine1   Spine1
                    # # Not Used new_qpos[13:16] = sample[21:24]  # Spine2
                    # # Not Used new_qpos[16:19] = sample[21:24]  # Spine3
                    # new_qpos[19:22] = sample[41:44]     # Neck  Neck
                    # new_qpos[22:25] = sample[44:47]     # Head  Head
   
                    # new_qpos[35:38] = sample[50:53]     # LShoulder  LeftShoulder
                    # #new_qpos[38:41] = sample[53:56]     # LElbow  LeftArm
                    # new_qpos[41] = sample[58]           # LElbow  LeftArm 56 57 58
                    # new_qpos[42:45] = sample[59:62]     # LWrist  LeftHand
   
                    # new_qpos[25:28] = sample[74:77]     # RShoulder  RightShoulder
                    # #new_qpos[28:31] = sample[77:80]     # RElbow  RightArm
                    # new_qpos[31] = sample[82]           # RElbow  RightArm 79 80 81
                    # new_qpos[32:35] = sample[83:86]     # RWrist  RightHand









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