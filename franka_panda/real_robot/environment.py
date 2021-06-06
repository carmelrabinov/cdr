import numpy as np


from franka_panda.environment import PushPandaEnv
from franka_panda.real_robot.image_utils import generate_calibration_coordinates, \
    generate_oposite_calibration_coordinates, get_cube_mask, get_cube_pixel_coordinates
from franka_panda.real_robot.redis_pubsub import RedisPubSub


class RealPushCubePandaEnv(PushPandaEnv):
    def __init__(self):
        super(RealPushCubePandaEnv, self).__init__()
        self._redis_pub_sub = RedisPubSub(pickle_encoding='latin1')

        # panda parameters
        self._joint_pos_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self._joint_pos_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        # TODO: find out the default orientation
        self._default_orientation = []

        # panda gripper parameters
        self._gripper_input_open = -1.
        self._gripper_input_close = 1.
        self._gripper_input_min = np.array([min(self._gripper_input_open, self._gripper_input_close)])
        self._gripper_input_max = np.array([max(self._gripper_input_open, self._gripper_input_close)])
        width_range = self._gripper_max_width - self._gripper_min_width
        action_range = self._gripper_input_open - self._gripper_input_close
        self._gripper_action_scale = width_range / action_range
        self._gripper_action_offset = self._gripper_input_open * self._gripper_action_scale - self._gripper_max_width

        # camera params
        self._perspective_matrix = self._load_perspective_matrix()
        self._oposite_perspective_matrix = self._load_oposite_perspective_matrix()

    def _load_perspective_matrix(self) -> np.ndarray:
        return np.load("calibration/perspective_matrix.npy")

    def _load_oposite_perspective_matrix(self) -> np.ndarray:
        return np.load("calibration/oposite_perspective_matrix.npy")

    def _calibrate_perspective_matrix(self) -> None:
        perspective_matrix = generate_calibration_coordinates(image=self.render(), ordered_pixel_coord=None)
        np.save("calibration/perspective_matrix.npy", perspective_matrix)

    def _calibrate_oposite_perspective_matrix(self)-> None:
        perspective_matrix = generate_oposite_calibration_coordinates(image=self.render(), ordered_pixel_coord=None)
        np.save("calibration/oposite_perspective_matrix.npy", perspective_matrix)

    @property
    def _gripper_min_width(self):
        return 0.0001

    @property
    def _gripper_max_width(self):
        return 0.08

    @property
    def state(self) -> np.ndarray:
        observation = self.render()
        cube_image_coordinates = get_cube_pixel_coordinates(observation)
        return self._im2pos_coordinates(cube_image_coordinates)

    def image2world(self, image_coordinates: np.ndarray) -> np.ndarray:
        """ move from image pixels coordinates to world coordinates """
        image_coordinates = np.flip(image_coordinates) # TODO: need to verify
        res = self._perspective_matrix.dot(np.array([image_coordinates[0], image_coordinates[1], 1]))
        world_coordinates = res / res[2]
        return world_coordinates[:2]

    def world2image(self, world_coordinates: np.ndarray):
        """
        move from world coordinates to image pixels
        """
        res = self._oposite_perspective_matrix.dot(np.array([world_coordinates[0], world_coordinates[1], 1]))
        image_coordinates = res / res[-1]
        image_coordinates = np.flip(image_coordinates[:2]).astype(np.int)  # TODO: need to verify
        return image_coordinates

    def render(self, mode: str = "regular") -> np.ndarray:
        # TODO: make adjustable image size
        image =  self._redis_pub_sub.run_method('get_camera_obs',
                                                camera_obs_dim=128,
                                                crop=None,
                                                flip_image=True,
                                                resize=True,
                                                resize_to_square=True)

        if mode == "segmentation":
            segmentation_mask = get_cube_mask(image, use_BGR=False)
            return segmentation_mask

        return image

    def close(self):
        self._redis_pub_sub.run_method('close')

    def _move_panda_to_coordinate(self, coordinates) -> None:
        succ, target_joints_from_ik = self.redis_pub_sub.run_method('inverse_kinematics',
                                                                    target_pos=coordinates,
                                                                    target_ori=self._default_orientation)
        assert succ, 'Inverse Kinematics Failed!'

        target_joints = target_joints_from_ik.clip(self._joint_pos_min, self._joint_pos_max)
        if not np.equal(target_joints_from_ik, target_joints).all():
            print('JOINTS CLIPPED')
            print('  IK Values: {}'.format(target_joints_from_ik))
            print('  Clipped  : {}'.format(target_joints))

        # Move arm sync with impedance control or without (set motion_planning=True)
        self.redis_pub_sub.run_method('move_to_joint_position', target_joints=target_joints,
                                      motion_planning=True, threshold=5e-2)

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
        source_pos = np.concatenate([cliped_action[:2], [self._ee_push_hight]])
        self._move_panda_to_coordinate(source_pos)

        # move to target position
        target_pos = np.concatenate([cliped_action[2:4], [self._ee_push_hight]])
        self._move_panda_to_coordinate(target_pos)

        # lift panda arm and render
        self.reset()
        observation = self.render()

        return observation, 0., 0., {}

    def reset(self) -> None:
        # TODO: add safety assertion for initial cartestian position to be inside the bounding box?
        print('In PandaEnv.reset()')
        self.redis_pub_sub.run_method('untuck')
        self.redis_pub_sub.run_method('move_gripper', target_width=-self._gripper_action_offset, wait_for_result=False)



