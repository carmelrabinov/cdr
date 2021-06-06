import os
import numpy as np

from franka_panda.environment import PushPandaEnv
from franka_panda.pybullet_simulation.push_simulation import Sim2RealPushPandaSimulation


class PybulletPushPandaEnv(PushPandaEnv):
    def __init__(self, config_path: str = "configs/default_push.yml",
                 load_frame: bool = False,
                 random_texture: bool = True,
                 random_light: bool = True,
                 random_size: bool = False,
                 random_camera: bool = False,
                 use_ui: bool = False,
                 textures_path: str = '../textures',
                 alternative_textures_path: str = '../textures'):
        super().__init__()
        # transform to sim_coordinates
        self.convert_real_2_sim_spaces()

        self._simulation = Sim2RealPushPandaSimulation(use_ui=use_ui, config_path=config_path)
        self._default_orientation = self._simulation.default_orientation

        # objects parameters
        self._object_id = None
        self._random_sized_object = random_size

        # env parameters
        self._random_light = random_light
        self._random_camera = random_camera
        if load_frame:
            self._simulation.load_state_space_frame()
        self._random_texture = random_texture
        self._simulation.texture_dir = textures_path
        self._simulation.alternative_texture_dir = alternative_textures_path
        if not os.path.exists(textures_path):
            print(f"Warning, texture directory: {textures_path} does not exists, will not use textures")
            self._simulation.texture_dir = None
            self._random_texture = False
        if not os.path.exists(alternative_textures_path):
            print(f"Warning, alternative texture directory: {alternative_textures_path} does not exists, will not use textures")
            self._simulation.alternative_texture_dir = None
            self._random_texture = False

        self.reset()
        self.reset_environment_appearance()

    @property
    def state(self) -> np.ndarray:
        raise NotImplementedError

    @state.setter
    def state(self, state: np.ndarray):
        raise NotImplementedError

    def render(self, mode: str = "regular") -> np.ndarray:
        if mode not in self.metadata["render.modes"]:
            print(f"Warning! no render mode: {mode}")
        if mode == "alternative":
            return self._simulation.render(alternative=True)

        image, segmantaion_mask = self._simulation.render(return_seg_mask=True)
        if mode == "segmentation":
            return segmantaion_mask

        return image

    def close(self):
        self._simulation.disconnect()

    def step(self, action: np.ndarray) -> [np. ndarray, float, bool, dict]:
        """
        :param action:
        :return: observation, reward, done, info
        """
        assert len(action) == 4, "Action space must be a 4 length vector [x_init, y_init, x_goal, y_goal]"

        # validate action space
        cliped_action = self._clip_to_action_space(action)
        if np.linalg.norm(action - cliped_action) > 1.e-4:
            print(f"Warning, action provided was out of action space and was cliped from {action} to {cliped_action}")

        # move to source position
        source = [cliped_action[0], self._ee_push_hight, cliped_action[1]] + self._default_orientation
        self._simulation.step_to_state(source, eps=0.005)

        # record the actual source coordinates
        actual_source = np.array(self._simulation.ee_position[[0, 2]])

        # move to target position
        target = [cliped_action[2], self._ee_push_hight, cliped_action[3]] + self._default_orientation
        self._simulation.step_to_state(target, eps=0.005)

        # record the actual target coordinates
        actual_target = np.array(self._simulation.ee_position[[0, 2]])

        # lift panda arm and render
        self._simulation.panda_robot.reset()
        observation = self.render()

        done = self.out_of_state_space(self.state)
        actual_action = np.concatenate((actual_source, actual_target))
        info = {"actual_action": actual_action}

        return observation, 0., done, info

    def reset_environment_appearance(self):
        # random light
        if self._random_light:
            self._simulation.init_random_light_directions()

        # random texture
        if self._random_texture:
            self._simulation.init_random_textures()

        # random camera
        if self._random_camera:
            self._simulation.init_random_cameras()

    def reset(self) -> None:
        raise NotImplementedError

    def image2world(self, image_coordinates: np.ndarray):
        """ move from image pixels to world coordinates """
        return self._simulation.image2world(image_coordinates)

    def world2image(self, world_coordinates: np.ndarray):
        """ move from world coordinates to image pixels """
        return self._simulation.world2image(world_coordinates)


class PybulletPushCubePandaEnv(PybulletPushPandaEnv):

    @property
    def state(self) -> np.ndarray:
        id = list(self._simulation.objects)[-1]
        return self._simulation.get_object_xz_position(id)

    @state.setter
    def state(self, state: np.ndarray):
        if self._object_id is not None:
            self._simulation.remove_object(self._object_id)
        cube_size = np.random.uniform(low=0.018, high=0.022) if self._random_sized_object else 0.018
        position = [state[0], 0.1, state[1]]
        self._object_id = self._simulation.load_cube(size=cube_size, position=position, mass=0.1, friction=0.3)
        self._simulation.let_objects_settle()
        self._simulation.panda_robot.reset()

    def reset(self) -> None:

        self._simulation.panda_robot.reset()

        # remove cube if exists, and load new one
        cube_potision = np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(-0.6, -0.4)])
        self.state = cube_potision


class PybulletPushRopePandaEnv(PybulletPushPandaEnv):

    @property
    def state(self) -> np.ndarray:
        dummy_state = np.array([0., -0.5])
        return dummy_state

    @state.setter
    def state(self, state: np.ndarray):
        self._simulation.panda_robot.reset()

    def reset(self) -> None:

        self._simulation.panda_robot.reset()
        self._simulation.y = 0.015
        # remove rope if exists, and load new one
        if self._object_id is not None:
            self._simulation.remove_object(self._object_id)
        self._object_id = self._simulation.load_rope(random_positions=True,
                                                     random_rope=self._random_sized_object,
                                                     random_texture=self._random_texture)
        self._simulation.let_objects_settle()

    def sample_point_on_rope(self):
        return self._simulation.sample_point_on_object(object_id=self._object_id)

