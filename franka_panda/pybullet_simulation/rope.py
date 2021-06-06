import pybullet_data
import pybullet as p
import numpy as np


class TextureRope:

    @classmethod
    def load(cls, basePosition: np.ndarray = [0, 0, 1], baseOrientation: np.ndarray = [0, 0, 0, 1],
             length: int = 80, thickness: float = 0.015, physicsClientId: int = 0) -> int:

        sphereRadius = thickness / 2.
        dist_coef = 1.01
        colBoxId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius, physicsClientId=physicsClientId)
        visualShapeId = -1
        mass = 0.1
        link_Masses = []
        linkCollisionShapeIndices = []
        linkVisualShapeIndices = []
        linkPositions = []
        linkOrientations = []
        linkInertialFramePositions = []
        linkInertialFrameOrientations = []
        indices = []
        jointTypes = []
        axis = []

        for i in range(length):

            link_Masses.append(mass)
            linkCollisionShapeIndices.append(colBoxId)
            linkVisualShapeIndices.append(visualShapeId)
            linkPositions.append([0, sphereRadius * dist_coef, 0])
            linkOrientations.append([0, 0, 0, 1])
            linkInertialFramePositions.append([0, 0, 0])
            linkInertialFrameOrientations.append([np.random.rand(), np.random.rand(), np.random.rand(), 1])
            indices.append(i)
            jointTypes.append(p.JOINT_REVOLUTE)

            axis.append([0, 0, 1])

        id = p.createMultiBody(mass,
                                  colBoxId,
                                  visualShapeId,
                                  basePosition,
                                  baseOrientation,
                                  linkMasses=link_Masses,
                                  linkCollisionShapeIndices=linkCollisionShapeIndices,
                                  linkVisualShapeIndices=linkVisualShapeIndices,
                                  linkPositions=linkPositions,
                                  linkOrientations=linkOrientations,
                                  linkInertialFramePositions=linkInertialFramePositions,
                                  linkInertialFrameOrientations=linkInertialFrameOrientations,
                                  linkParentIndices=indices,
                                  linkJointTypes=jointTypes,
                                  linkJointAxis=axis,
                                  useMaximalCoordinates=True,
                                  physicsClientId=physicsClientId)

        anistropicFriction = [1, 0.1, 0.1]
        p.changeDynamics(id, -1, anisotropicFriction=anistropicFriction, lateralFriction=0.2, #rollingFriction=0.0001, spinningFriction=0.0001, restitution=0.2,
                         physicsClientId=physicsClientId)
        for i in range(p.getNumJoints(id, physicsClientId=physicsClientId)):
            p.getJointInfo(id, i, physicsClientId=physicsClientId)
            p.changeDynamics(id, i, anisotropicFriction=anistropicFriction, lateralFriction=0.2, #rollingFriction=0.0001, spinningFriction=0.0001, restitution=0.2,
                             physicsClientId=physicsClientId)
        return id

    @classmethod
    def apply_texture(cls, texture_id, object_id, physicsClientId):
        for i in range(p.getNumJoints(object_id, physicsClientId=physicsClientId)):
            p.changeVisualShape(object_id, linkIndex=i, shapeIndex=1, textureUniqueId=texture_id, physicsClientId=physicsClientId)


if __name__ == '__main__':

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.createCollisionShape(p.GEOM_PLANE)
    p.changeDynamics(plane, -1, lateralFriction=1)
    p.createMultiBody(0, plane)
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)

    sphereUid = TextureRope.load(length=40)

    while True:
        p.stepSimulation()





