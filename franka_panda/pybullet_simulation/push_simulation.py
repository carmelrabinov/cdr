import math
import os
import numpy as np
import pybullet as p
import yaml

from scipy.ndimage.morphology import binary_dilation, generate_binary_structure

from franka_panda.pybullet_simulation.base_simulation import PandaSimulation
from franka_panda.pybullet_simulation.utils import AttrDict


class PushPandaSimulation(PandaSimulation):
    def __init__(self, use_ui: bool = True, config_path: str = "configs/default_push.yml"):
        config = self.load_config(config_path)
        self._config = config
        super().__init__(use_ui=use_ui,
                         image_width=config.image_width,
                         image_hight=config.image_hight,
                         camera_params=config.camera_params)

    def load_config(self, config_path):
        if not os.path.exists(config_path):
            config_path = os.path.join("franka_panda", "pybullet_simulation", config_path)
        assert os.path.exists(config_path), f"Config path {config_path} does not exists"
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return self.parse_config(config)

    def add_config_attribute(self, config, name, value):
        if name not in config:
            config[name] = value

    def parse_config(self, config_dict):
        self.add_config_attribute(config_dict, "random_camera", False)
        return AttrDict(config_dict)

    def reset(self):
        super().reset()

        self._state_space_x_limits = np.array([0.3, -0.3])
        self._state_space_z_limits = np.array([-0.3, -0.7])
        self.y = 0.02

        # Note: changing set camera will require to change im2pos x and z limits
        self.set_camera(distance=0.5, yaw=0., pitch=-89., fov=90)
        self._im2pos_x_lim = [-0.365, 0.365]  # do not touch
        self._im2pos_z_lim = [-0.95, -0.24]  # do not touch
        self.robot_video = []
        # self.validate_im2pos_cordinates()

        self._im2pos_lim_min = np.array([-0.365, -0.95])  # do not touch
        self._im2pos_lim_max = np.array([0.365, -0.24])  # do not touch

    def load_state_space_frame(self, hight: float = 0.0005, half_extent: float = 0.009, color: np.ndarray = [1, 0, 0, 1]) -> int:
        meshScale = np.array([1., 1., 1., 1.])
        shapeTypes = [p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX]
        meshScales = [meshScale, meshScale, meshScale, meshScale]
        length = np.abs(np.diff(self._state_space_x_limits)[0])  #x axis
        width = np.abs(np.diff(self._state_space_z_limits)[0])  #z axis
        fileNames = ["", ""]
        shift1 = [0, 0., 0]
        shift2 = [length/2, 0., -width/2]
        shift3 = [0, 0., -width]
        shift4 = [-length/2, 0., -width/2]
        position = [self._state_space_x_limits[1] + length / 2, hight, self._state_space_z_limits[0]]

        shifts = np.array([shift1, shift2, shift3, shift4])
        halfExtents = np.array([[length/2, hight/2, half_extent/2],
                                [half_extent/2, hight/2, width/2],
                                [length/2, hight/2, half_extent/2],
                                [half_extent/2, hight/2, width/2]])

        visualShapeId = p.createVisualShapeArray(shapeTypes=shapeTypes,
                                                 halfExtents=halfExtents,
                                                 fileNames=fileNames,
                                                 visualFramePositions=shifts,
                                                 meshScales=meshScales)

        collisionShapeId = p.createCollisionShapeArray(shapeTypes=shapeTypes,
                                                       halfExtents=halfExtents,
                                                       fileNames=fileNames,
                                                       collisionFramePositions=shifts,
                                                       meshScales=meshScales)

        mb_id = p.createMultiBody(baseMass=0,
                                  baseInertialFramePosition=[0, 0, 0],
                                  baseCollisionShapeIndex=collisionShapeId,
                                  baseVisualShapeIndex=visualShapeId,
                                  basePosition=position,
                                  useMaximalCoordinates=False)

        p.changeVisualShape(mb_id, -1, rgbaColor=color, physicsClientId=self.my_id)
        self.state_space_frame = mb_id
        return mb_id

    def im2pos_coordinates(self, image_x, image_z):
        """
        move from image pixels coordinates to world coordinates
        Note: this is true only for overhead view only
        """
        x = self._im2pos_x_lim[1] - (self._im2pos_x_lim[1] - self._im2pos_x_lim[0]) * image_x/(self._image_width-1)
        z = self._im2pos_z_lim[1] - (self._im2pos_z_lim[1] - self._im2pos_z_lim[0]) * image_z/(self._image_hight-1)
        return x, z

    def pos2im_coordinates(self, x, z):
        """
        move from world coordinates to image pixels coordinates
        Note: this is true only for overhead view only
        """
        pix_x = int((self._image_hight-1) * (self._im2pos_x_lim[1] - x) / (self._im2pos_x_lim[1] - self._im2pos_x_lim[0]))
        pix_z = int((self._image_hight-1) * (self._im2pos_z_lim[1] - z) / (self._im2pos_z_lim[1] - self._im2pos_z_lim[0]))
        return pix_x, pix_z

    def im2pos_coordinates_array(self, image_coordinates: np.ndarray) -> np.ndarray:
        """
        move from image pixels coordinates to world coordinates
        Note: this is true only for overhead view only
        param: image_coordinates: n_coordinates X 2 array
        """
        assert len(image_coordinates.shape) == 2 and image_coordinates.shape[1] == 2, "Error in converting im2pos coordinates"
        world_coordinates = self._im2pos_lim_max - (self._im2pos_lim_max - self._im2pos_lim_min) * image_coordinates/(self._image_width-1)
        return world_coordinates

    def pos2im_coordinates_array(self, world_coordinates: np.ndarray) -> np.ndarray:
        """
        move from world coordinates to image pixels coordinates
        Note: this is true only for overhead view only
        param: world_coordinates: n_coordinates X 2 array
        """
        assert len(world_coordinates.shape) == 2 and world_coordinates.shape[1] == 2, "Error in converting pos2im coordinates"
        image_coordinates = ((self._image_hight-1) * (self._im2pos_lim_max - world_coordinates) / (self._im2pos_lim_max - self._im2pos_lim_min)).astype(int)
        assert np.all(image_coordinates < self._image_hight) and np.all(image_coordinates >= 0), "Error in converting pos2im coordinates"
        return image_coordinates

    def get_overhead_object_segmentation(self):
        """
        returns segmentation mask of the object
        """
        rendered_img, seg = self.render_overhead(save_render=False, return_seg_mask=True)
        seg_mask = self.get_segmantation_mask_from_segmentation(seg)
        return seg_mask, rendered_img

    @staticmethod
    def dilate_mask(mask, size=5):
        """run morphology dilate operation on binary segmentation mask"""
        struct = generate_binary_structure(2, 1)
        if len(np.shape(mask)) == 3:
            return np.expand_dims(binary_dilation(mask[:, :, 0], structure=struct, iterations=size), axis=2)
        else:
            binary_dilation(mask, structure=struct, iterations=size)

    @staticmethod
    def sample_from_mask(mask):
        assert np.any(mask), "Error, mask must contain samples"
        object_z, object_x, _ = np.where(mask)
        n_points = len(object_x)
        rand_index = np.random.choice(n_points, 1)[0]
        image_x, image_z = object_x[rand_index], object_z[rand_index]
        assert mask[image_z, image_x], "Error, sampled point must be on mask"
        return image_x, image_z

    def sample_point_on_object(self, object_id: int) -> np.ndarray:
        """
        sample a random cartesian coordinate from the objects segmentation mask
        :param object_id: object to sample from
        :return: cartesian point which is on the objects
        """
        # get object position map
        _, object_mask = self.render_overhead(save_render=False, return_seg_mask=True)
        object_mask = object_mask == object_id

        if not object_mask.any():
            print(f"Warning, try to sample point from object: {object_id}, which has an empty segmentation mask!")
            return None

        # return position in real world coordinates
        image_x, image_z = self.sample_from_mask(object_mask)
        x, z = self.im2pos_coordinates(image_x, image_z)
        position = np.array([x, z])
        return position

    def validate_im2pos_cordinates(self):
        for x in self._im2pos_x_lim:
            for z in self._im2pos_z_lim:
                self.load_cube(size=0.01, position=[x, 0, z])
        self.render(save_render=False)
        self.step_to_image_point(0, 0)
        self.step_to_image_point(0, 127)
        self.step_to_image_point(127, 127)
        self.step_to_image_point(127, 0)


class Sim2RealPushPandaSimulation(PushPandaSimulation):
    def reset(self):
        super().reset()

        self._x_limits = np.array([0.23, -0.23])
        self._z_limits = np.array([-0.32, -0.68])
        self.y = 0.02

        # Note: changing set camera will require to change im2pos x and z limits
        if self._camera_params is not None:
            position = self._camera_params["position"] if "position" in self._camera_params else [0., 0.63, -0.87]
            target = self._camera_params["target"] if "target" in self._camera_params else [0., 0., -0.5]
            fov = self._camera_params["fov"] if "fov" in self._camera_params else 41
            far_val = self._camera_params["far_val"] if "far_val" in self._camera_params else 4.
            self.set_camera(position=position, target=target, fov=fov, far_val=far_val)
        else:
            self.set_camera(position=[0., 0.6, -0.85], target=[0., 0., -0.51], fov=41, far_val=4.)

        self._im2pos_x_lim = [-0.365, 0.365]  # do not touch
        self._im2pos_z_lim = [-0.95, -0.24]  # do not touch
        self.robot_video = []
        # self.validate_im2pos_cordinates()

