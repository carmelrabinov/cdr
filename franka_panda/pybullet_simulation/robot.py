import os
import numpy as np
import math
import pybullet as p


class PandaRobot(object):
    def __init__(self, my_id: int, offset: np.ndarray = [0, 0, 0]):

        # init bullet client
        self.my_id = my_id
        p.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.position_sensitivity = 0.005
        self.velocity_sensitivity = 0.0001

        # init panda robot
        self._offset = np.array(offset)
        orn = [-0.707107, 0.0, 0.0, 0.707107]  # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        self._robot_id = p.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]) + self._offset, orn,
                                                 useFixedBase=True,
                                    flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                                    physicsClientId=self.my_id)
        p.setCollisionFilterPair(self._robot_id, self._robot_id, 6, 8, 0, physicsClientId=self.my_id)
        p.setCollisionFilterPair(self._robot_id, self._robot_id, 9, 10, 0, physicsClientId=self.my_id)

        self._end_effector_index = 11
        self._num_dofs = 7
        self._force = 500
        # self.reset_joint_positions = [0.927, -0.0574247, 0.3015, -0.9111, 0.0289, 0.91, 2.031, 0.0, 0.0]
        self.reset_joint_positions = [0.927, -0.0574247, 0.3015, -0.9111, 0.0289, 0.91, math.pi/4, 0.0, 0.0]
        self.reset()

        # init all joints info
        self._number_of_all_joints = p.getNumJoints(self._robot_id, physicsClientId=self.my_id)
        joints_info = self._get_joints_properties()
        self._joint_names, self._joint_types, self._joints_lower_bounds, self._joints_upper_bounds = zip(*joints_info)
        # externalize only controlable joints
        enumerated_controlled_joint_info = [
            (i, joint_name, joint_type, lower_bound, upper_bound)
            for i, (joint_name, joint_type, lower_bound, upper_bound) in enumerate(joints_info)
            if joint_type != p.JOINT_FIXED
        ]
        res = list(zip(*enumerated_controlled_joint_info))
        self._external_to_internal_joint_index = res[0]
        self.joint_names = res[1]
        self.joint_types = res[2]
        self.joints_lower_bounds = res[3]
        self.joints_upper_bounds = res[4]
        self.joints_ranges = [5.8, 3.5, 5.8, 3.1, 5.8, 3.8, 5.8, 0.04, 0.04],
        self.number_of_joints = len(self._external_to_internal_joint_index)

    def is_close(self, target_joints, source_joints=None):
        distance = self.get_distance(target_joints, source_joints)
        return distance < self.position_sensitivity

    def get_distance(self, target_joints, source_joints=None):
        assert len(target_joints) == self.number_of_joints
        if source_joints is None:
            source_joints = self.get_robot_state()[0]
        assert len(source_joints) == self.number_of_joints
        return np.linalg.norm(np.array(source_joints) - np.array(target_joints))

    def is_moving(self):
        current_speed = self.get_current_speed()
        return current_speed > self.velocity_sensitivity

    def set_joints(self, joint_positions: np.ndarray) -> None:
        self._joints_positions = joint_positions
        control_mode = p.POSITION_CONTROL
        max_force = self._force
        base_list = np.ones_like(joint_positions)
        p.setJointMotorControlArray(self._robot_id,
                                      jointIndices=self._external_to_internal_joint_index,
                                      controlMode=control_mode,
                                      targetPositions=joint_positions,
                                      targetVelocities=list(base_list[:-2] * 0.01) + [0.0, 0.0],
                                      # forces=np.array([1, 1, 0.3, 0.3, 0.3, 0.3, 1., 0.1, 0.1])*max_force * 10000000000,
                                      positionGains=list(base_list[:-2] * 0.01) + [1., 1.],
                                      velocityGains=list(base_list[:-2] * 0.8) + [1., 1.],
                                      physicsClientId=self.my_id)

    def get_robot_state(self):
        joint_position_velocity_pairs = [
            (t[0], t[1])
            for t in p.getJointStates(self._robot_id, self._external_to_internal_joint_index, physicsClientId=self.my_id)
        ]
        return list(zip(*joint_position_velocity_pairs))

    def get_current_speed(self):
        velocities = self.get_robot_state()[1]
        return np.linalg.norm(velocities)

    def is_moving(self):
        current_speed = self.get_current_speed()
        return current_speed > self.velocity_sensitivity

    def set_xyz(self, state: np.ndarray, closed_gripper=True) -> None:
        position = state[:3]
        orientation = state[3:7] if len(state) > 3 else None
        joint_positions = self.inverse_kinematics(position=position, orientation=orientation)
        if closed_gripper:
            # self.close_gripper()
            joint_positions = tuple(list(joint_positions[:-2]) + [0., 0.])

        self.set_joints(joint_positions)

    def inverse_kinematics(self, position: np.ndarray, orientation: np.ndarray = None) -> np.ndarray:
        if orientation is None:
            # # default orientation is exactly above object
            # orientation = p.getQuaternionFromEuler([math.pi / 2., 0., 0.])
            joint_positions = p.calculateInverseKinematics(self._robot_id,
                                                                 endEffectorLinkIndex=self._end_effector_index,
                                                                 targetPosition=position,
                                                                 lowerLimits=self.joints_lower_bounds,
                                                                 upperLimits=self.joints_upper_bounds,
                                                                 jointRanges=self.joints_ranges,
                                                                 restPoses=self.reset_joint_positions,
                                                                 maxNumIterations=20,
                                                                 physicsClientId=self.my_id)
        else:
            joint_positions = p.calculateInverseKinematics(self._robot_id,
                                                                 endEffectorLinkIndex=self._end_effector_index,
                                                                 targetPosition=position,
                                                                 targetOrientation=orientation,
                                                                 lowerLimits=self.joints_lower_bounds,
                                                                 upperLimits=self.joints_upper_bounds,
                                                                 jointRanges=self.joints_ranges,
                                                                 restPoses=self.reset_joint_positions,
                                                                 maxNumIterations=20,
                                                                 physicsClientId=self.my_id)
        return joint_positions

    def accurate_inverse_kinematics(self, position: np.ndarray, orientation: np.ndarray = None,
                                    max_iter: int = 20, th: float = 1.e-2):
        # TODO: currently not supported, problems with collisions
        close_enough = False
        iter = 0
        dist2 = 1e30
        while (not close_enough and iter < max_iter):
            joint_positions = self.inverse_kinematics(position, orientation)
            for i in range(self._num_dofs):
                p.resetJointState(self._robot_id, i, joint_positions[i], physicsClientId=self.my_id)
            ls = p.getLinkState(self._robot_id, self._end_effector_index, physicsClientId=self.my_id)
            newPos = ls[4]
            diff = [position[0] - newPos[0], position[1] - newPos[1], position[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            close_enough = (dist2 < th)
            iter = iter + 1
        # print ("Num iter: "+str(iter) + "threshold: "+str(dist2))
        return joint_positions

    def save_joints_positions(self) -> None:
        self._saved_joints_positions = [pos[0] for pos in p.getJointStates(self._robot_id, self._external_to_internal_joint_index, physicsClientId=self.my_id)]

    def reset_saved_joints_positions(self) -> None:
        self.reset(self._saved_joints_positions)

    def reset(self, joints_positions=None) -> None:
        if joints_positions is None:
            joints_positions = self.reset_joint_positions
        self._joints_positions = joints_positions
        index = 0
        for j in range(p.getNumJoints(self._robot_id, physicsClientId=self.my_id)):
            # p.changeDynamics(self._robot_id, j, linearDamping=0, angularDamping=0, physicsClientId=self.my_id)
            info = p.getJointInfo(self._robot_id, j, physicsClientId=self.my_id)

            # print("info=",info)
            jointName, jointType = info[1: 3]
            if (jointType == p.JOINT_PRISMATIC):
                p.resetJointState(self._robot_id, j, joints_positions[index], physicsClientId=self.my_id)
                index = index + 1
            if (jointType == p.JOINT_REVOLUTE):
                p.resetJointState(self._robot_id, j, joints_positions[index], physicsClientId=self.my_id)
                index = index + 1

    def _get_joints_properties(self):
        joints_info = [
            p.getJointInfo(self._robot_id, i, physicsClientId=self.my_id) for i in range(self._number_of_all_joints)
        ]
        joints_info = [(t[1], t[2], t[8], t[9]) for t in joints_info]
        return joints_info

    def get_ee_state(self) -> np.ndarray:
        state = p.getLinkState(self._robot_id, self._end_effector_index,
                               computeLinkVelocity=1,
                               computeForwardKinematics=1,
                               physicsClientId=self.my_id)

        # Cartesian 6D pose:
        pos = state[4]
        orn = state[5]
        velocity = state[6]

        return np.concatenate((np.array(pos), np.array(orn)))

    def get_link_poses(self):
        link_poses = []
        for i in self._external_to_internal_joint_index:
            state = p.getLinkState(self._robot_id, i, physicsClientId=self.my_id)
            position = state[4]
            orientation = state[5]
            link_poses.append((position, orientation))
        return link_poses

    def get_collisions(self):
        a = [
            contact
            for contact in p.getContactPoints(self._robot_id, physicsClientId=self.my_id) if contact[8] < -0.0001
        ]
        return a

    def is_collision(self):
        collisions = self.get_collisions()
        return len(collisions) > 0


class PandaRobotGripper(PandaRobot):
    def __init__(self, my_id: int, offset: np.ndarray = [0, 0, 0]):
        PandaRobot.__init__(self, my_id, offset)

        self.open_finger_positions = [0.3, 0.3]
        self.closed_finger_positions = [0.0, 0.]
        self.gripper_force = 50000

    def open_gripper(self) -> None:
        p.setJointMotorControlArray(self._robot_id,
                                    jointIndices=[9,10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.open_finger_positions,
                                    physicsClientId=self.my_id)

    def close_gripper(self) -> None:
        p.setJointMotorControlArray(self._robot_id,
                                    jointIndices=[9,10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.closed_finger_positions,
                                    physicsClientId=self.my_id)
