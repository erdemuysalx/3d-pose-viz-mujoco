import numpy as np
import torch
from torch.utils.data import DataLoader
from utils   import h36motion as datasets
from utils.data_utils import *
from utils.parser import args
from environment import Environment


TEST_XML = 'assets/mujoco_models/mocap_v2.xml'
MODEL_XML = "assets/mujoco_models/humanoid.xml"


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device: %s'%device)


def pose_generator():
    env = Environment(MODEL_XML)
    env.sim.reset()
    
    actions = define_actions(args.actions_to_consider)
    dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                            43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                            86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97])
    print(len(dim_used))
    for action in actions:
        dataset = datasets.Datasets(
            args.data_dir,
            args.input_n, 
            args.output_n, 
            args.skip_rate, 
            split=2, 
            actions=[action]
        )

        print('>>> test action for sequences: {:d}'.format(dataset.__len__()))

        data_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size_test, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )

        for ix, batch in enumerate(data_loader):
            #batch=batch.to(device)
                            
            print(batch.shape)
            sequences_gt=batch[:, 0:args.input_n, dim_used]
            for samples in sequences_gt:
                for sample in samples:
                    #data = sample.cpu()
                    env.set_pose(sample)
                    env.sim.forward()
                    env.viewer.render()
                    env.step += 1

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
