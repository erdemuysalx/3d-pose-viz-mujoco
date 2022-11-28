import numpy as np


# Joints in H3.6M
# Data has 32 joints, but only 16 of them move
# Indices as follows
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hips'
H36M_NAMES[1]  = 'RightUpLeg'
H36M_NAMES[2]  = 'RightLeg'
H36M_NAMES[3]  = 'RightFoot'
H36M_NAMES[6]  = 'LeftUpLeg'
H36M_NAMES[7]  = 'LeftLeg'
H36M_NAMES[8]  = 'LeftFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Spine1'
H36M_NAMES[14] = 'Neck'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LeftShoulder'
H36M_NAMES[18] = 'LeftArm'
H36M_NAMES[19] = 'LeftHand'
H36M_NAMES[25] = 'RightShoulder'
H36M_NAMES[26] = 'RightArm'
H36M_NAMES[27] = 'RightHand'


def to_str(value):
    """
    Converts list to a string
    """
    return ' '.join(str(el) for el in value)

def list_to_dict(joint_list, pose_list):
    """
    Merges two list into a dictionary
    """
    return dict(zip(joint_list, pose_list))

def list_to_3el_list(li):
    return [list(li[i:i+3]) for i in range(0, len(li), 3)]

def load_npz(filename, data_path='dataset/data_3d_h36m.npz'):
    keys = []
    vals = [] 
    with np.load(data_path, allow_pickle=True) as data:
        print(data.files)
        for value in data['p3d0'].item():
            key = value
            values = data['p3d0'].item()[value]
            keys.append(key)
            vals.append(values)
        d = dict(zip(keys, vals))
    return d[filename]