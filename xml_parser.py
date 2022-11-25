import xml.etree.ElementTree as ET
from utils import to_str
 

class XMLParser:
    """
    XMLParser class for manipulating MuJoCo models
    """

    def __init__(self, path:str) -> None:
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()

    def get_childs(self, tag:str) -> list:
        """
        Return a list full of childs with given tag

        Args: 
            tag -- Target tag name
        Returns:
            childs -- List of child names
        """
        childs = []
        for t in self.root.iter(tag):
            if t.attrib.get('name'):
                childs.append(t.attrib.get('name'))
        return childs

    def get_joint_pos(self, target:str) -> list:
        """
        Returns position of given joint

        Args:
            target -- Target joint's name

        Returns:
            list -- Target joint's position values for x, y, and z axises
        """
        for t in self.root.iter('joint'):
            if t.attrib.get('name') == target:
                pos = t.attrib.get('pos')
                # print(type(pos), pos)
                return [float(el) for el in pos.split(' ')]

    def set_joint_pos(self, target:str, value:list):
        """
        Modifies position of given joint

        Args: 
            target -- Target joint's name
            value -- Target joint's new position value

        Returns:
            None
        """
        new_pos = to_str(value)
        for t in self.root.iter('joint'):
            if t.attrib.get('name') == target:
                t.attrib['pos'] = new_pos
                self.tree.write('assets/mujoco_models/modified.xml')


# if __name__ == '__main__':
#     p = XMLParser('test.xml')
#     x, y, z = p.get_joint_pos(target='RightShoulder_x')
#     p.set_joint_pos(target='RightShoulder_x', value=[0.50, 0.50, 0.50])
