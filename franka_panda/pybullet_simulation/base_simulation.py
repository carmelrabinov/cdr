import os
import time
import cv2
import glob
import numpy as np
import math
from typing import List
import pybullet as p
import pybullet_data as pd

from franka_panda.pybullet_simulation.robot import PandaRobotGripper as panda_robot
from franka_panda.pybullet_simulation.rope import TextureRope


class PandaSimulation:

    def __init__(self, use_ui: bool = True,
                 image_width=128, image_hight=128,
                 camera_params: dict = None):

        # initiate pybullet
        self.my_id = p.connect(p.GUI if use_ui else p.DIRECT)
        print(f"Pybullet simulation connected to id {self.my_id}")
        p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000, physicsClientId=self.my_id)
        p.setAdditionalSearchPath(pd.getDataPath(), physicsClientId=self.my_id)
        p.setGravity(0, -9.8, 0, physicsClientId=self.my_id)

        self.panda_robot = panda_robot(my_id=self.my_id)
        self.objects = set()
        self.enable_rendering = False
        self.use_ui = use_ui

        # time
        fps = 240  # 240.
        self._control_dt = 1. / fps
        p.setTimeStep(self._control_dt, physicsClientId=self.my_id)
        self.t = 0.
        self._state_t = 0.
        self._cur_state_index = 0
        self._render_t = 0
        self.real_time = False  # run simulation in real time
        self.recoder_robot = False

        self._state_durations = [10]
        self.plane_id = self.load_plane(checker_board=True)
        self.state_space_frame = None
        self.velocity_sensitivity = 0.01

        self.print_msg = False
        self._max_steps_per_state = 600

        # image and video
        render_fps = 3
        self._render_dt = 1. / render_fps
        self._image_width = image_width
        self._image_hight = image_hight
        self.tray_textures_path = os.path.join(os.getcwd(),"../textures")

        # set domain randomization parameters
        self._texture_dir = None
        self._alternative_texture_dir = None
        self._textures = {self.plane_id: (self.default_texture, self.default_color, True)}
        self._alternative_textures = self._textures

        self._light_direction = [0, 1, 1]
        self._alternative_light_direction = self._light_direction

        self._camera_params = camera_params
        if camera_params is not None:
            self._render_matrixs = self.set_camera_from_params(camera_params)
        else:
            self._render_matrixs = self.set_camera(distance=0.5, yaw=0., pitch=-89., fov=90)
        self._alternative_render_matrixs = self._render_matrixs

        # save view of the robot
        self._projection_matrix_robot, self._view_matrix_robot = self.calc_camera_params(distance=1.24, yaw=408.4,
                                                                                         pitch=-34.8, fov=60,
                                                                                         near_val=0.01, far_val=10.,
                                                                                         target=[-0.1, 0.28, -0.225])

        # save overhead view
        self._projection_matrix_overhead, self._view_matrix_overhead = self.calc_camera_params(distance=0.5, yaw=0.,
                                                                                               pitch=-89., fov=90)
        # self.default_orientation = list(p.getQuaternionFromEuler([math.pi / 2., 0., 0.]))
        self.default_orientation = list(p.getQuaternionFromEuler([math.pi / 2., math.pi / 2., 0.]))
        self.reset()
        self.panda_robot.reset()

    def disconnect(self):
        p.disconnect(self.my_id)

    @property
    def texture_dir(self) -> str:
        return self._texture_dir

    @texture_dir.setter
    def texture_dir(self, texture_dir: str):
        assert os.path.exists(texture_dir), "Error texture directory path does not exists"
        print(f"Set {texture_dir} as texture directory")
        self._texture_dir = texture_dir

    @property
    def alternative_texture_dir(self) -> str:
        return self._alternative_texture_dir

    @alternative_texture_dir.setter
    def alternative_texture_dir(self, texture_dir: str):
        assert os.path.exists(texture_dir), "Error texture directory path does not exists"
        print(f"Set {texture_dir} as alternative texture directory")
        self._alternative_texture_dir = texture_dir

    @property
    def ee_state(self) -> np.ndarray:
        return self.panda_robot.get_ee_state()

    @property
    def joints_state(self) -> np.ndarray:
        return np.concatenate((self.panda_robot.get_robot_state()[0], self.panda_robot.get_robot_state()[1]))

    @property
    def joints_positions(self) -> np.ndarray:
        return np.array(self.panda_robot.get_robot_state()[0], dtype=float)

    @property
    def joints_velocities(self) -> np.ndarray:
        return np.array(self.panda_robot.get_robot_state()[1], dtype=float)

    @property
    def ee_position(self) -> np.ndarray:
        return self.panda_robot.get_ee_state()[:3]

    @property
    def ee_orientation(self) -> np.ndarray:
        return self.panda_robot.get_ee_state()[3:7]

    @property
    def video(self) -> np.ndarray:
        return self._video

    @property
    def video_2(self) -> np.ndarray:
        return self._video_2

    @property
    def segmentation_mask(self) -> np.ndarray:
        return self._video_seg_mask

    @property
    def states_and_actions(self) -> np.ndarray:
        return self._action_state_pairs

    @staticmethod
    def _get_random_color():
        # return [0.2, 0.2, 0.2, 0.8] # black
        # return [0.95, 0.95, 0.95, 0.8] # white
        return [np.random.rand(), np.random.rand(), np.random.rand(), np.random.uniform(0.85, 0.95)]

    @staticmethod
    def _get_random_light_direction():
        return [np.random.uniform(-5, 5), np.random.uniform(1., 6), np.random.uniform(-5, 5)]

    def _get_random_camera_matrix(self):
        mean_fov = self._camera_params["fov"]
        mean_position = np.array(self._camera_params["position"])
        mean_target = np.array(self._camera_params["target"])
        fov = np.random.uniform(mean_fov - 4, mean_fov + 2)
        position = np.random.uniform(mean_position - 0.015, mean_position + 0.015)
        target = np.random.uniform(mean_target - 0.015, mean_target + 0.015)
        projection_matrix, view_matrix = self.calc_camera_params(fov=fov, target=target, position=position)
        return {"projection_matrix": projection_matrix, "view_matrix": view_matrix}

    def _get_random_textures(self, texture_dir: str = None, texture_plane: bool = True, texture_frame: bool = True):
        """ saves 2 textures"""
        assert self._texture_dir is not None or texture_dir is not None, "Error, need to define path textures directory"
        texture_dir = texture_dir if texture_dir is not None else self._texture_dir
        assert os.path.exists(texture_dir), f"Texture directory does not exists: {texture_dir}"
        textures = glob.glob(os.path.join(texture_dir, '**', '*.jpg'), recursive=True)
        textures_dict = {}

        objects = list(self.objects)

        # if texture_plane will also init for plane
        if texture_plane:
            objects += [self.plane_id]

        # if texture_frame will also init for action space frame
        if texture_frame:
            objects += [self.state_space_frame] if self.state_space_frame is not None else []

        for object in objects:
            random_texture_path = textures[np.random.randint(0, len(textures))]
            texture_id = p.loadTexture(random_texture_path, physicsClientId=self.my_id)
            color = [np.random.rand(), np.random.rand(), np.random.rand(), 1]
            # apply texture to plain with probability 95% (otherwise apply only color)
            if object == self.plane_id:
                apply_texture = np.random.uniform(0, 1) < 0.95
            # apply texture to objects or frame with probability 15% (otherwise apply only color)
            else:
                apply_texture = np.random.uniform(0, 1) < .15
            textures_dict[object] = (texture_id, color, apply_texture)
        return textures_dict

    def set_random_light(self, light_direction: list = None):
        """ set light direction, if not provided will use random direction"""
        self._light_direction = self._get_random_light_direction() if light_direction is None else light_direction
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.my_id, lightPosition=self._light_direction)

    def apply_textures(self, textures_dict: dict):
        """ apply textures, if not provided will use random textures"""
        for object, (texture, color, apply_texture) in textures_dict.items():
            if apply_texture:
                p.changeVisualShape(object, -1, textureUniqueId=texture, rgbaColor=self.default_color, physicsClientId=self.my_id)
            else:
                p.changeVisualShape(object, -1, textureUniqueId=self.default_texture, rgbaColor=color, physicsClientId=self.my_id)

    def init_random_light_directions(self):
        self._light_direction = self._get_random_light_direction()
        self._alternative_light_direction = self._get_random_light_direction()

    def init_random_cameras(self):
        self._render_matrixs = self._get_random_camera_matrix()
        self._alternative_render_matrixs = self._get_random_camera_matrix()

    def init_random_textures(self):
        self._alternative_textures = self._get_random_textures(texture_dir=self.alternative_texture_dir)
        self._textures = self._get_random_textures(texture_dir=self.texture_dir)

    def reset(self) -> None:
        self._state_t = 0.
        self._cur_state_index = 0
        self._render_t = 0
        self.set_camera()
        self.t = 0.
        self._video = []
        self._video_seg_mask = []
        self._video_2 = []
        self._action_state_pairs = {}
        for key in ["actions", "ee_positions", "ee_orientations", "joints_velocities", "joints_positions", "collisions"]:
            self._action_state_pairs[key] = []

    def step_to_state(self, state: np.ndarray, max_iter: int = None, eps: float = 0.01,
                      stop_at_collision: bool = False, closed_gripper: bool = True) -> bool:
        """
        perform motion planning and execute, reaching from current state to defined state
        :param state: a goal state to reach
        :param max_iter: max number of iteration to perform
        :param eps: Euclidean distance tolerance for reaching to goal state
        :param stop_at_collision: whether to stop when colliding with some object
        :param closed_gripper: whether to ensure close gripper while moving
        :return bool: whether a collision has happened
        """
        target_pos = np.array(state[:3], dtype=float)
        # target_orn = state[3:7] if len(state) > 3 else None
        max_iter = self._max_steps_per_state if max_iter is None else max_iter
        curr_pos = self.ee_position
        steps_counter = 1
        is_collision = False
        while np.linalg.norm(curr_pos - target_pos, 2) > eps:
            self.panda_robot.set_xyz(state, closed_gripper=closed_gripper)
            curr_state, _is_collision = self.step()
            is_collision = True if _is_collision else is_collision
            curr_pos = curr_state[:3]
            if stop_at_collision and is_collision:
                # print_msg()
                # return False
                break
            if steps_counter >= max_iter:
                # print_msg()
                # return False
                break
            steps_counter += 1
        print(f"iterations: {steps_counter}, target: {state[:3]}, reached: {self.ee_position}, error: {100 * np.linalg.norm(self.ee_position - state[:3])}") if self.print_msg else None
        return is_collision

    def step(self) -> tuple:
        """
        perform one pybullet simulation step
        return (end-effector state, collision detector flag)
        """
        self._render_t += self._control_dt
        self.t += self._control_dt
        if self.enable_rendering and self._render_t >= self._render_dt:
            self.render_robot() if self.recoder_robot else self.render()
            self._render_t = 0
        p.stepSimulation(physicsClientId=self.my_id)
        if self.real_time:
            time.sleep(self._control_dt)

        return self.ee_state, self.panda_robot.is_collision()

    def calc_camera_params(self, distance: float = 0.8, yaw: float = 5., pitch: float = -45., fov: float = 80,
                   near_val: float = 0.01, far_val: float = 4., target: list = [0., -0.13, -0.6], position: list = None):
        if self.use_ui:
            p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1, physicsClientId=self.my_id, lightPosition=self._light_direction)
            p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=[0., -0.13, -0.6], physicsClientId=self.my_id)
        if position is not None:
            view_matrix = p.computeViewMatrix(cameraEyePosition=position, cameraTargetPosition=target,
                                              cameraUpVector=[0, 1, 0], physicsClientId=self.my_id)
        else:
            view_matrix = p.computeViewMatrixFromYawPitchRoll(distance=distance, yaw=yaw, pitch=pitch, upAxisIndex=1,
                                                              cameraTargetPosition=target, roll=0, physicsClientId=self.my_id)
        projection_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=1., nearVal=near_val, farVal=far_val,
                                                         physicsClientId=self.my_id)
        return projection_matrix, view_matrix

    def set_camera(self, distance: float = 0.8, yaw: float = 5., pitch: float = -45., fov: float = 80,
                   near_val: float = 0.01, far_val: float = 1., target: list = [0., -0.13, -0.6], position: list = None) -> dict:
        self._projection_matrix, self._view_matrix = self.calc_camera_params(distance=distance, yaw=yaw, pitch=pitch,
                                                                             fov=fov, near_val=near_val,
                                                                             far_val=far_val, target=target, position=position)
        return {"view_matrix": self._view_matrix, "projection_matrix": self._projection_matrix}

    def set_camera_from_params(self, camera_params: dict) -> dict:
        if "position" in camera_params:
            position = camera_params["position"]
        else:
            position = [0., 0.63, -0.87]
            print(f"Warning, camera params does not specify position, uses default value: {position}")
        if "target" in camera_params:
            target = camera_params["target"]
        else:
            target = [0., 0., -0.5]
            print(f"Warning, camera params does not specify target, uses default value: {target}")
        if "fov" in camera_params:
            fov = camera_params["fov"]
        else:
            fov = 41
            print(f"Warning, camera params does not specify fov, uses default value: {fov}")
        if "far_val" in camera_params:
            far_val = camera_params["far_val"]
        else:
            far_val = 4.
            print(f"Warning, camera params does not specify far_val, uses default value: {far_val}")
        return self.set_camera(position=position, target=target, fov=fov, far_val=far_val)

    def render_robot(self):
        """ render an image from a perspective such that the robot is seen (useful for view the robot action) """
        robot_img = self.render(save_render=False,
                                view_matrix=self._view_matrix_robot,
                                projection_matrix=self._projection_matrix_robot,
                                high_resolution=True, return_seg_mask=False)
        self.robot_video.append(robot_img)
        return robot_img

    def render_overhead(self, save_render=False, return_seg_mask=False):
        """ render with overhead camera"""
        return self.render(save_render=save_render,
                           return_seg_mask=return_seg_mask,
                           view_matrix=self._view_matrix_overhead,
                           projection_matrix=self._projection_matrix_overhead)

    def render(self, save_render=True, return_seg_mask=False, view_matrix=None, projection_matrix=None,
               alternative=False, high_resolution: bool = False) -> np.ndarray:
        """
        render pybullet scene
        :return: a rendered scene image
        """
        if view_matrix is None or projection_matrix is None:
            render_metrixs = self._alternative_render_matrixs if alternative else self._render_matrixs
            view_matrix = render_metrixs["view_matrix"]
            projection_matrix = render_metrixs["projection_matrix"]
        light_direction = self._alternative_light_direction if alternative else self._light_direction

        self.apply_textures(self._alternative_textures) if alternative else self.apply_textures(self._textures)
        # (width, hight) = (1024, 1024) if high_resolution else (self._image_width, self._image_hight)
        img_data = p.getCameraImage(self._image_width,
                                    self._image_hight,
                                    view_matrix,
                                    projection_matrix,
                                    shadow=True,
                                    lightDirection=light_direction,
                                    # renderer=p.ER_TINY_RENDERER,
                                    flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                    physicsClientId=self.my_id)

        width, height, rgb_img, depth_img, seg_img = img_data

        rendered_img = np.reshape(rgb_img, (height, width, 4))[:, :, :3]

        img = np.expand_dims(np.array(rendered_img), axis=0)
        if save_render:
            self._video = img if len(self._video) < 1 else np.append(self._video, img, axis=0)

        if return_seg_mask:
            seg_img = np.reshape(seg_img, (height, width, 1))
            return rendered_img, seg_img

        return rendered_img

    def get_segmantation_mask_from_segmentation(self, seg):
        seg_mask = np.zeros_like(seg).astype(bool)
        for object_id in self.objects:
            seg_mask = np.bitwise_or(seg_mask, seg == object_id)
        return seg_mask

    def save_as_mp4(self, video: List[np.ndarray], save_path='simulation', fps: int = 10) -> None:
        """
        save a stream of images as videos file
        :param video: a list of np.ndarray frames
        :param save_path: path to save video
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 0x00000021
        vid = cv2.VideoWriter(save_path + ".mp4", fourcc, fps, (self._image_width, self._image_hight))
        for frame in video:
            vid.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        vid.release()

    def save_as_npy(self, video: List[np.ndarray], save_path='simulation') -> None:
        """
        save a stream of images as videos file
        :param video: a list of np.ndarray frames
        :param save_path: path to save video
        """
        np.save(save_path, np.array(video))

    @staticmethod
    def change_object_mass(object_id: int, mass: float, change_interia: bool = True) -> None:
        if change_interia:
            dimensions = p.getVisualShapeData(object_id)[0][3]
            inertia = [(1 / 12) * mass * (dimensions[1] ** 2 + dimensions[2] ** 2),
                       (1 / 12) * mass * (dimensions[0] ** 2 + dimensions[2] ** 2),
                       (1 / 12) * mass * (dimensions[0] ** 2 + dimensions[1] ** 2)]
            p.changeDynamics(object_id, -1, mass=mass, localInertiaDiagonal=inertia)
        else:
            p.changeDynamics(object_id, -1, mass=mass)

    @staticmethod
    def change_object_friction(object_id: int, friction: float) -> None:
        p.changeDynamics(object_id, -1, lateralFriction=friction)

    @staticmethod
    def let_objects_settle(steps: int = 100) -> None:
        # let object fall and settle
        for i in range(steps):
            p.stepSimulation()

    def is_object_moving(self, object_id: int):
        return np.linalg.norm(self.get_object_velocity(object_id)) > self.velocity_sensitivity

    def get_object_velocity(self, object_id: int) -> np.ndarray:
        return np.array(p.getBaseVelocity(object_id, physicsClientId=self.my_id), dtype=float)

    def get_object_state(self, object_id: int) -> np.ndarray:
        state = p.getBasePositionAndOrientation(object_id, physicsClientId=self.my_id)
        return np.concatenate((state[0], state[1]))

    def get_object_orientation(self, object_id: int) -> np.ndarray:
        return self.get_object_state(object_id)[3:7]

    def get_object_position(self, object_id: int) -> np.ndarray:
        return self.get_object_state(object_id)[:3]

    def get_object_xz_position(self, object_id: int) -> np.ndarray:
        return self.get_object_state(object_id)[[0, 2]]

    def load_lego(self, position: np.ndarray = [0.1, 0.3, -0.5], mass: float = 5., color: np.ndarray = [1, 0, 0, 1], friction: float = 1.) -> int:
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        id = p.loadURDF("lego/lego.urdf", position, flags=flags, globalScaling=2., physicsClientId=self.my_id)
        if mass:
            self.change_object_mass(id, mass=mass)
        if friction:
            self.change_object_friction(id, friction=friction)
        p.changeVisualShape(id, -1, rgbaColor=color, physicsClientId=self.my_id)
        self.objects.add(id)
        return id

    def load_cube(self, size: float = 0.03, position: np.ndarray = [0.1, 0.3, -0.5], mass: float = 4.,
                  color: np.ndarray = [1, 0, 0, 1], friction: float = 1.) -> int:
        collision_box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size], physicsClientId=self.my_id)
        visual_box = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=color,
                                         physicsClientId=self.my_id)
        id = p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=collision_box, baseVisualShapeIndex=visual_box,
            basePosition=position, physicsClientId=self.my_id)
        if mass:
            self.change_object_mass(id, mass=mass)
        if friction:
            self.change_object_friction(id, friction=friction)
        self.objects.add(id)
        return id

    def load_plane(self, random_texture: bool = False, checker_board: bool = False) -> int:
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        baseOrientation = p.getQuaternionFromEuler([-np.pi / 2, 0, 0])
        plane_path = os.path.join('./plane/plane_mesh.urdf')
        if not os.path.exists(plane_path):
            plane_path = os.path.join('./franka_panda_pybullet/plane/plane_mesh.urdf')
        if checker_board:
            plane_path = "plane.urdf"
        id = p.loadURDF(plane_path, basePosition=[0, 0, 0], baseOrientation=baseOrientation, flags=flags, physicsClientId=self.my_id)
        self.default_color, self.default_texture = p.getVisualShapeData(id, -1)[0][-2:]
        if random_texture:
            self.load_texture(object_id=id, random=True, path=self.tray_textures_path)
        return id

    def load_rope(self, position: np.ndarray = [-0.2, 0.2, -0.6],
                  orientation: np.ndarray = list(p.getQuaternionFromEuler([math.pi / 2., math.pi / 2, 0.])),
                  color: np.ndarray = [1, 0, 0, 1],
                  random_texture: bool = False,
                  length: int = 42,
                  thickness: float = 0.021,
                  random_positions: bool = False,
                  random_rope: bool = False) -> int:

        # random position and orientation
        if random_positions:
            x = np.random.rand()
            # horizontal
            if x <= 0.25:
                orientation = list(p.getQuaternionFromEuler([math.pi / 2., math.pi / 2, 0.]))
                position = [np.random.uniform(-0.3, -0.1), 0.2, np.random.uniform(-0.7, -0.5)]
            # /
            elif x <= 0.5:
                orientation = list(p.getQuaternionFromEuler([math.pi / 2., math.pi / 1.5, 0.]))
                position = [-0.2, 0.2, np.random.uniform(-0.55, -0.35)]
            # \
            elif x <= 0.75:
                orientation = list(p.getQuaternionFromEuler([math.pi / 2., math.pi / 4., 0.]))
                position = [np.random.uniform(-0.2, -0.1), 0.2, np.random.uniform(-0.8, -0.6)]
            # vertical
            else:
                orientation = list(p.getQuaternionFromEuler([math.pi / 2., 0., 0.]))
                position = [np.random.uniform(-0.15, 0.15), 0.2, -0.8]

        # random length and thickness
        if random_rope:
            length = np.random.randint(35, 45)
            thickness = np.random.uniform(0.02, 0.025)

        # generate object
        id = TextureRope.load(basePosition=position, baseOrientation=orientation,
                              length=length, thickness=thickness,
                              physicsClientId=self.my_id)
        p.changeVisualShape(id, -1, rgbaColor=color, physicsClientId=self.my_id)

        # apply random texture
        if random_texture:
            self.load_texture(object_id=id, random=True, path=self.tray_textures_path)

        # save to objects
        self.objects.add(id)
        self.rope_id = id

        return id

    def load_texture(self, object_id: int, path: str, random: bool = True) -> None:
        """
        load and apply a texture from image file
        :param object_id: object id to apply texture on
        :param path: path to images directory if random is True else path to texture image.
        :param random: load random image from directory if True
        :return: None
        """
        if random:
            textures = os.listdir(path)
            random_texture_path = os.path.join(path, textures[np.random.randint(0, len(textures))])
        else:
            random_texture_path = path
        texture_id = p.loadTexture(random_texture_path, physicsClientId=self.my_id)
        p.changeVisualShape(object_id, -1, textureUniqueId=texture_id, physicsClientId=self.my_id)

    def remove_object(self, obj_id):
        p.removeBody(obj_id, physicsClientId=self.my_id)
        if obj_id in self.objects:
            self.objects.remove(obj_id)

    def world2image(self, world_coordinates: np.ndarray):
        """
        move from world coordinates to image pixels
        """
        x = world_coordinates[0]
        y = world_coordinates[1]
        projectionMatrix = np.asarray(self._render_matrixs["projection_matrix"]).reshape([4,4],order='F')
        viewMatrix = np.asarray(self._render_matrixs["view_matrix"]).reshape([4,4],order='F')
        tran_pix_world = np.matmul(projectionMatrix, viewMatrix)
        z = 0.  #z = 2 * depth_np_arr[h, w] - 1
        worldPos = np.asarray([x, z, y, 1])   # y is set to be up axis in simulation
        pixels = np.matmul(tran_pix_world, worldPos)
        pixels /= pixels[-1]   # normalize w=1
        # print(pixels)
        w = (pixels[0] * self._image_width + self._image_width)/2. / pixels[2]  # normalize so z=1 in pixel space
        h = (pixels[1] * self._image_hight + self._image_hight)/2. / pixels[2]  # normalize so z=1 in pixel space
        h = self._image_hight - h   # hight is negative in simulation
        return np.array([int(w), int(h)])

    def image2world(self, image_coordinates: np.ndarray):
        """
        move from image pixels to world coordinates
        """
        # TODO: this is wrong
        w = image_coordinates[0]
        h = image_coordinates[1]
        projectionMatrix = np.asarray(self._render_matrixs["projection_matrix"]).reshape([4,4],order='F')
        viewMatrix = np.asarray(self._render_matrixs["view_matrix"]).reshape([4,4],order='F')
        tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
        x = (2 * w - self._image_width) / self._image_width
        h = self._image_hight - h
        y = -(2 * h - self._image_hight) / self._image_hight  # be careful！ deepth and its corresponding position
        z = 1  #z = 2 * depth_np_arr[h, w] - 1
        pixPos = np.asarray([x, y, z, 1])
        # print(pixPos)
        position = np.matmul(tran_pix_world, pixPos)
        position /= position[-1]
        # position = position / position[1]
        return np.array([position[0], position[2]])

        # this is probably the correct one
        projectionMatrix = np.asarray(projection_matrix).reshape([4,4],order='F')
        viewMatrix = np.asarray(view_matrix).reshape([4,4],order='F')
        tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
        x = (2 * w - self._image_width) / self._image_width
        y = -(2 * h - self._image_hights) / self._image_hights  # be careful！ deepth and its corresponding position
        z = 2 * depth_np_arr[h, w] - 1
        pixPos = np.asarray([x, y, z, 1])
        position = np.matmul(tran_pix_world, pixPos)
