import time
from hitsz_qy_hummingbird.base_FWMAV.MAV.MAV_params import ParamsForBaseMAV
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
import numpy as np
import pybullet as p
import pybullet_data
from pprint import pprint

class BaseMAV:
    """
    This class should have as few functions as possible,
    concentrating all PyBullet environment functionalities internally.
    """

    def __init__(self,
                 mav_params: ParamsForBaseMAV,
                 if_gui: bool,
                 if_fixed: bool):
        """
        the urdf should be loaded and the variables should be set up
        the joints and rods are defined accordingly,
        and the configuration should be set up.
        """

        self.params = mav_params
        self.params.initial_orientation = p.getQuaternionFromEuler(self.params.initial_rpy)
        self.data = {}
        self.logger = GLOBAL_CONFIGURATION.logger

        self.right_stroke_joint = self.params.right_stroke_joint
        self.right_rotate_joint = self.params.right_rotate_joint
        self.left_stroke_joint = self.params.left_stroke_joint
        self.left_rotate_joint = self.params.left_rotate_joint
        self.right_rod_link = self.params.right_rod_link
        self.left_rod_link = self.params.left_rod_link
        self.right_wing_link = self.params.right_wing_link
        self.left_wing_link = self.params.left_wing_link

        self.right_stroke_amp = 0
        self.right_stroke_vel = 0
        self.right_stroke_acc = 0
        self.target_right_stroke_amp = 0
        self.target_right_stroke_vel = 0
        self.target_right_stroke_acc = 0

        self.left_stroke_amp = 0
        self.left_stroke_vel = 0
        self.left_stroke_acc = 0
        self.target_left_stroke_amp = 0
        self.target_left_stroke_vel = 0
        self.target_left_stroke_acc = 0

        self.right_rotate_amp = 0
        self.right_rotate_vel = 0

        self.left_rotate_amp = 0
        self.left_rotate_vel = 0

        if if_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        #COV_ENABLE_RGB_BUFFER_PREVIEW：启用或禁用RGB缓冲区预览。
        #COV_ENABLE_DEPTH_BUFFER_PREVIEW：启用或禁用深度缓冲区预览。
        #COV_ENABLE_SEGMENTATION_MARK_PREVIEW：启用或禁用分割标记预览。0，禁用
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        # 添加资源路径，p.setAdditionalSearchPath(pybullet_data.getDataPath())用于设置模型加载路径。这个函数接受一个参数，表示要添加的搜索路径。pybullet_data.getDataPath()返回PyBullet内置数据的路径，其中包含一些预定义的模型和环境，例如平面、机器人等。
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print(pybullet_data.getDataPath())
        p.setGravity(self.params.gravity[0],
                     self.params.gravity[1],
                     self.params.gravity[2])

        p.loadURDF("plane.urdf")
        # # # 加载坐标轴
        # axis_length = 0.1
        # axis_visual = p.addUserDebugLine([0, 0, 0], [axis_length, 0, 0], [0.1, 0, 0], parentObjectUniqueId=-1, parentLinkIndex=-1)
        # p.addUserDebugLine([0, 0, 0], [0, axis_length, 0], [0, 0.1, 0], parentObjectUniqueId=-1, parentLinkIndex=-1)
        # p.addUserDebugLine([0, 0, 0], [0, 0, axis_length], [0, 0, 1], parentObjectUniqueId=-1, parentLinkIndex=-1)

        # # 添加XYZ标签
        # p.addUserDebugText('X', [axis_length, 0, 0], textColorRGB=[1, 0, 0], textSize=2, parentObjectUniqueId=axis_visual)
        # p.addUserDebugText('Y', [0, axis_length, 0], textColorRGB=[0, 1, 0], textSize=2, parentObjectUniqueId=axis_visual)
        # p.addUserDebugText('Z', [0, 0, axis_length], textColorRGB=[0, 0, 1], textSize=2, parentObjectUniqueId=axis_visual)

        # startPos = [-3, 2, 0.2]
        # startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        # boxId = p.loadURDF("D://graduate//fwmav//fsolidwork//240118//build7//urdf//build7.urdf", startPos, startOrientation)
        p.loadURDF("samurai.urdf",
            physicsClientId=self.physics_client
            )
        
        # p.loadURDF("build.urdf",
        #             physicsClientId=self.physics_client
        #            )
        
        # p.loadURDF("duck_vhacd.urdf",
        #                [-1, 0, 1],
        #                p.getQuaternionFromEuler([0.5*np.pi, 0, 0.5*np.pi]),
        #                physicsClientId=self.physics_client
        #                )    
        # p.loadURDF("teddy_vhacd.urdf",
        #                [0, -1, .1],
        #                p.getQuaternionFromEuler([0, 0, 0]),
        #                physicsClientId=self.physics_client
        #                )
       
        #URDF_USE_INERTIA_FROM_FILE：默认情况下，Bullet 根据碰撞形状的质量和体积重新计算惯性张量。 如果您可以提供更准确的惯性张量，请使用此标志。
        self.body_unique_id = p.loadURDF(fileName="D://graduate//fwmav//simul2024//240201//QY-hummingbird-main//URDFdir//newbird24.urdf",
                                         basePosition=self.params.initial_xyz,
                                         baseOrientation=self.params.initial_orientation,
                                         flags=p.URDF_USE_INERTIA_FROM_FILE)
        
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]]   #红 绿  蓝 白
        p.changeVisualShape(self.body_unique_id, -1, rgbaColor=colors[3]) #torso 白
        p.changeVisualShape(self.body_unique_id, self.right_wing_link, rgbaColor=colors[0]) #right wing 红
        p.changeVisualShape(self.body_unique_id, self.left_wing_link, rgbaColor=colors[2]) #left wing 蓝
        
        print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        print("flapper的信息：")
        # 可以使用的关节
        available_joints_indexes = [i for i in range(p.getNumJoints(self.body_unique_id)) if p.getJointInfo(self.body_unique_id, i)[2] != p.JOINT_FIXED]
        print('可以使用的关节')
        pprint([p.getJointInfo(self.body_unique_id, i)[1] for i in available_joints_indexes])
        joint_num = p.getNumJoints(self.body_unique_id)
        print("flapper的关节数量为：", joint_num)

        #在下限》上限的情况下忽略值。                    slider和revolute(hinge)类型的位移最小值：0.0
        #                                               #slider和revolute(hinge)类型的位移最大值：-1.0
        for joint_index in range(joint_num):
            info_tuple = p.getJointInfo(self.body_unique_id, joint_index)
            print(f"关节序号：{info_tuple[0]}\n\
                    关节名称：{info_tuple[1]}\n\
                    关节类型：{info_tuple[2]}\n\
                    机器人第一个位置的变量索引：{info_tuple[3]}\n\
                    机器人第一个速度的变量索引：{info_tuple[4]}\n\
                    保留参数：{info_tuple[5]}\n\
                    关节的阻尼大小：{info_tuple[6]}\n\
                    关节的摩擦系数：{info_tuple[7]}\n\
                    slider和revolute(hinge)类型的位移最小值：{info_tuple[8]}\n\
                    slider和revolute(hinge)类型的位移最大值：{info_tuple[9]}\n\
                    关节驱动的最大值：{info_tuple[10]}\n\
                    关节的最大速度：{info_tuple[11]}\n\
                    节点名称：{info_tuple[12]}\n\
                    局部框架中的关节轴系：{info_tuple[13]}\n\
                    父节点frame的关节位置：{info_tuple[14]}\n\
                    父节点frame的关节方向：{info_tuple[15]}\n\
                    父节点的索引，若是基座返回-1：{info_tuple[16]}\n\n")
            
        for link_id in range(0,4):
            link_tuple = p.getLinkState(self.body_unique_id, link_id)
            print(link_id)
            print(f"质心世界位置：{link_tuple[0]}\n\
                    质心的世界方向: {link_tuple[1]}\n\
                    以 URDF 链接框架表示的惯性框架（质心）的局部位置偏移：{link_tuple[2]}\n\
                    以 URDF 链接框架表示的惯性框架的局部方向：{link_tuple[3]}\n\
                    URDF 链接框架的世界位置：{link_tuple[4]}\n\
                    URDF 链接框架的世界方向：{link_tuple[5]}\n\
                  ")
        # 仅当 computeLinkVelocity 非零时才返回:
        # worldLinkLinearVelocity: {link_tuple[6]}\n\
        # worldLinkAngularVelocity: {link_tuple[7]}\n\
            
        #计算局部惯性对角线是否为零，local inertia diagonal（局部惯性对角线）
        #getDynamicsInfovec3[2]：list of 3 floats
        #局部惯性对角线。 请注意，连杆和底座以质心为中心，并与惯性主轴对齐。惯性张量描述了物体绕其质心旋转时的惯性特性。它是一个对称矩阵，可以对角化，即可以找到一个坐标系，使得惯性张量在该坐标系下的表示为对角矩阵。这个对角矩阵的对角线元素就是局部惯性对角线。
        self.if_valid_urdf = True
        for linkid in range(p.getNumJoints(self.body_unique_id)):
            inertia = p.getDynamicsInfo(self.body_unique_id, linkid)[2]
            if sum(inertia) == 0:
                self.if_valid_urdf = False

        #stepSimulation 将在单个正向动力学仿真步骤中执行所有操作，例如碰撞检测、约束求解和集成。 默认时间步长为 1/240 秒，可以使用 setTimeStep 或 setPhysicsEngineParameter API 更改
        #求解器迭代次数和接触、摩擦和非接触关节的误差减少参数 (erp) 与时间步长有关。 如果您更改时间步长，您可能需要相应地重新调整这些值，尤其是 erp 值。
        p.setTimeStep(1 / GLOBAL_CONFIGURATION.TIMESTEP)

        #URDF、SDF 和 MJCF 将铰接体指定为无环的树结构。 “createConstraint”允许连接主体的特定链接以形成闭环。 还可以在对象之间以及对象与特定世界框架之间创建任意约束。
        #createConstraint 将返回一个整数唯一 id，可用于更改或删除约束。
        self.if_fixed = if_fixed
        if self.if_fixed:
            self.constraint_id = p.createConstraint(self.body_unique_id,
                                                    parentLinkIndex=-1,
                                                    childBodyUniqueId=-1,
                                                    childLinkIndex=-1,
                                                    jointType=p.JOINT_FIXED,
                                                    jointAxis=[0, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=self.params.initial_xyz,
                                                    childFrameOrientation=self.params.initial_orientation)
        
        self.camera_follow()

        #使用 changeDynamics 更改质量、摩擦和恢复系数等属性
        #change the lower limit of a joint, also requires
        #jointUpperLimit otherwise it is ignored. note that at
        #the moment, the joint limits are not updated in
        #'getJointInfo'!
        p.changeDynamics(self.body_unique_id,
                         self.right_wing_link,
                         jointLowerLimit=-1 * self.params.max_angle_of_rotate,
                         jointUpperLimit=self.params.max_angle_of_rotate)

        #给定关节的最大关节速度，如果在约束求解期间超过它，则将其固定。 默认最大关节速度为 100 个单位。
        p.changeDynamics(self.body_unique_id,
                         self.right_wing_link,
                         lateralFriction=0,
                         linearDamping=0,
                         angularDamping=0,
                         maxJointVelocity=self.params.max_joint_velocity)

        p.changeDynamics(self.body_unique_id,
                         self.left_wing_link,
                         jointLowerLimit=-1 * self.params.max_angle_of_rotate,
                         jointUpperLimit=self.params.max_angle_of_rotate)

        p.changeDynamics(self.body_unique_id,
                         self.left_wing_link,
                         lateralFriction=0,
                         linearDamping=0,
                         angularDamping=0,
                         maxJointVelocity=self.params.max_joint_velocity)

        # p.changeDynamics(self.body_unique_id,
        #                  self.right_rod_link,
        #                  jointLowerLimit=-1 * self.params.max_angle_of_stroke,
        #                  jointUpperLimit=self.params.max_angle_of_stroke)

        p.changeDynamics(self.body_unique_id,
                         self.right_rod_link,
                         lateralFriction=0,
                         linearDamping=0,
                         angularDamping=0,
                         maxJointVelocity=self.params.max_joint_velocity)

        # p.changeDynamics(self.body_unique_id,
        #                  self.right_rod_link,
        #                  jointLowerLimit=-1 * self.params.max_angle_of_stroke,
        #                  jointUpperLimit=self.params.max_angle_of_stroke,)

        p.changeDynamics(self.body_unique_id,
                         self.left_rod_link,
                         lateralFriction=0,
                         linearDamping=0,
                         angularDamping=0,
                         maxJointVelocity=self.params.max_joint_velocity)
        
        #通过为一个或多个关节电机设置所需的控制模式来控制机器人
        #关节电机控制器的实际实现对于 POSITION_CONTROL 和 VELOCITY_CONTROL 是作为约束，对于 TORQUE_CONTROL 是作为外力
        #参数force: 在 POSITION_CONTROL 和 VELOCITY_CONTROL 中，这是用于达到目标值的最大电机力。 在 TORQUE_CONTROL 中，这是每个模拟步骤要应用的力/扭矩。
        p.setJointMotorControl2(self.body_unique_id,
                                self.right_stroke_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                force=0)

        p.setJointMotorControl2(self.body_unique_id,
                                self.left_stroke_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                force=0)

        p.setJointMotorControl2(self.body_unique_id,
                                self.right_rotate_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                force=0)

        p.setJointMotorControl2(self.body_unique_id,
                                self.left_rotate_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                force=0)

    def camera_follow(self):
        """
        the camera should be always targeted at the MAV
        """
        position, orientation = p.getBasePositionAndOrientation(self.body_unique_id)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=120,
                                     cameraPitch=-20,
                                     cameraTargetPosition=position)

    def step(self):
        """
        The torque and the force are applied, and the influence is calculated
        """
        GLOBAL_CONFIGURATION.TICKTOCK = GLOBAL_CONFIGURATION.TICKTOCK + 1
        p.stepSimulation()
        self.camera_follow()
        time.sleep(self.params.sleep_time)
        self.joint_state_update()

    def joint_state_update(self):
        """
        The joint state should be renewed
        """
        pos, vel, _, _ = p.getJointState(self.body_unique_id,
                                         self.right_stroke_joint)
        self.right_stroke_amp = pos
        self.right_stroke_acc = (vel - self.right_stroke_vel) / GLOBAL_CONFIGURATION.TIMESTEP
        self.right_stroke_vel = vel

        pos, vel, _, _ = p.getJointState(self.body_unique_id,
                                         self.left_stroke_joint)
        self.left_stroke_amp = pos
        self.left_stroke_acc = (vel - self.left_stroke_vel) / GLOBAL_CONFIGURATION.TIMESTEP
        self.left_stroke_vel = vel

        pos, vel, _, _ = p.getJointState(self.body_unique_id,
                                         self.right_rotate_joint)
        self.right_rotate_amp = pos
        self.right_rotate_vel = vel

        pos, vel, _, _ = p.getJointState(self.body_unique_id,
                                         self.left_rotate_joint)
        self.left_rotate_amp = pos
        self.left_rotate_vel = vel

    def joint_control(self,
                      target_right_stroke_amp=None,
                      target_right_stroke_vel=None,
                      target_left_stroke_amp=None,
                      target_left_stroke_vel=None,
                      right_input_torque=None,
                      left_input_torque=None):
        """
        the motor is controlled according to the control mode
        """
        #电机控制器的实际实现对于POSITION_CONTROL 和 VELOCITY_CONTROL 是作为约束
        #参数positionGain和velocityGain是用于约束误差最小化的参数，error =position_gain*(desired_position-actual_position)+velocity_gain*(desired_velocity-actual_velocity)
        if target_right_stroke_amp is not None and target_right_stroke_vel is not None and target_left_stroke_amp is not None and target_left_stroke_vel is not None:

            p.setJointMotorControl2(self.body_unique_id,
                                    self.right_stroke_joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_right_stroke_amp,
                                    targetVelocity=target_right_stroke_vel,
                                    force=self.params.max_force,
                                    positionGain=self.params.position_gain,
                                    velocityGain=self.params.velocity_gain,
                                    maxVelocity=self.params.max_joint_velocity)

            p.setJointMotorControl2(self.body_unique_id,
                                    self.left_stroke_joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_left_stroke_amp,
                                    targetVelocity=target_left_stroke_vel,
                                    force=self.params.max_force,
                                    positionGain=self.params.position_gain,
                                    velocityGain=self.params.velocity_gain,
                                    maxVelocity=self.params.max_joint_velocity)

        elif target_right_stroke_vel is not None and target_left_stroke_vel is not None:
            p.setJointMotorControl2(self.body_unique_id,
                                    self.right_stroke_joint,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=target_right_stroke_vel,
                                    force=self.params.max_force,
                                    maxVelocity=self.params.max_joint_velocity)
            p.setJointMotorControl2(self.body_unique_id,
                                    self.left_stroke_joint,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=target_left_stroke_vel,
                                    force=self.params.max_force,
                                    maxVelocity=self.params.max_joint_velocity)

        elif right_input_torque is not None and left_input_torque is not None:
            p.setJointMotorControl2(self.body_unique_id,
                                    self.right_stroke_joint,
                                    controlMode=p.TORQUE_CONTROL,
                                    force=right_input_torque,
                                    maxVelocity=self.params.max_joint_velocity)
            p.setJointMotorControl2(self.body_unique_id,
                                    self.left_stroke_joint,
                                    controlMode=p.TORQUE_CONTROL,
                                    force=left_input_torque,
                                    maxVelocity=self.params.max_joint_velocity)
        elif target_right_stroke_amp is not None and target_left_stroke_amp is not None:
            p.setJointMotorControl2(self.body_unique_id,
                                    self.right_stroke_joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_right_stroke_amp,
                                    force=self.params.max_force,
                                    maxVelocity=self.params.max_joint_velocity)

            p.setJointMotorControl2(self.body_unique_id,
                                    self.left_stroke_joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_left_stroke_amp,
                                    force=self.params.max_force,
                                    maxVelocity=self.params.max_joint_velocity)
            
        elif target_right_stroke_amp is not None:
            p.setJointMotorControl2(self.body_unique_id,
                                    self.right_stroke_joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_right_stroke_amp,
                                    force=self.params.max_force,
                                    maxVelocity=self.params.max_joint_velocity)

    def get_state_for_motor_torque(self):
        """
        return the  right_stroke_amp    right_stroke_vel    right_stroke_acc    right_torque
                    left_stroke_amp     left_stroke_vel     left_stroke_acc     left_torque
        """
        #getJointState[3]返回: appliedJointMotorTorque,这是在最后一步模拟期间应用的电机扭矩。 请注意，这只适用于 VELOCITY_CONTROL 和 POSITION_CONTROL
        right_torque = p.getJointState(self.body_unique_id,
                                       self.right_stroke_joint)[3]
        left_torque = p.getJointState(self.body_unique_id,
                                      self.left_stroke_joint)[3]
        return (self.right_stroke_amp,
                self.right_stroke_vel,
                self.right_stroke_acc,
                right_torque,
                self.left_stroke_amp,
                self.left_stroke_vel,
                self.left_stroke_acc,
                left_torque)

    def get_state_for_wing(self):
        """
        return the  right_stroke_angular_velocity   right_rotate_angular_velocity
                    right_c_axis        right_r_axis        right_z_axis
                    left_stroke_angular_velocity    left_rotate_angular_velocity
                    left_c_axis         left_r_axis         left_z_axis
        """
        #getLinkState[7]返回：笛卡尔世界角速度，worldLinkAngularVelocity，vec3, list of 3 floats
        right_sum_angular_velocity = np.array(
            p.getLinkState(self.body_unique_id,
                           self.right_wing_link,
                           computeLinkVelocity=1)[7])
        right_stroke_angular_velocity = np.array(
            p.getLinkState(self.body_unique_id,
                           self.right_rod_link,
                           computeLinkVelocity=1)[7])
        right_rotate_angular_velocity = right_sum_angular_velocity - right_stroke_angular_velocity

        left_sum_angular_velocity = np.array(
            p.getLinkState(self.body_unique_id,
                           self.left_wing_link,
                           computeLinkVelocity=1)[7])
        left_stroke_angular_velocity = np.array(
            p.getLinkState(self.body_unique_id,
                           self.left_rod_link,
                           computeLinkVelocity=1)[7])
        left_rotate_angular_velocity = left_sum_angular_velocity - left_stroke_angular_velocity

        #getLinkState[5]返回：URDF 链接框架的世界方向，worldLinkFrameOrientation，vec4, list of 4 floats
        right_orientation = np.array(
            p.getMatrixFromQuaternion(
                p.getLinkState(self.body_unique_id,
                               self.right_wing_link)[5])).reshape(3, 3)

        left_orientation = np.array(
            p.getMatrixFromQuaternion(
                p.getLinkState(self.body_unique_id,
                               self.left_wing_link)[5])).reshape(3, 3)

        #rcz axis
        right_r_axis = right_orientation[:, self.params.rotate_axis]
        right_c_axis = -1 * right_orientation[:, self.params.stroke_axis]
        if right_stroke_angular_velocity.dot(right_c_axis) > 0:
            right_z_axis = -1 * right_orientation[:, self.params.the_left_axis]
        else:
            right_z_axis = right_orientation[:, self.params.the_left_axis]

        left_r_axis = -1 * left_orientation[:, self.params.rotate_axis]
        left_c_axis = -1 * left_orientation[:, self.params.stroke_axis]
        if left_stroke_angular_velocity.dot(left_c_axis) < 0:
            left_z_axis = -1 * left_orientation[:, self.params.the_left_axis]
        else:
            left_z_axis = left_orientation[:, self.params.the_left_axis]

        return [right_stroke_angular_velocity,
                right_rotate_angular_velocity,
                right_c_axis, right_r_axis, right_z_axis,
                left_stroke_angular_velocity,
                left_rotate_angular_velocity,
                left_c_axis, left_r_axis, left_z_axis]

    def set_link_force_world_frame(self,
                                   link_id,
                                   position_bias: np.ndarray,
                                   force: np.ndarray):
        """
        """
        force = np.array(force)

        if force.dot(force) == 0:
            return
        #getLinkState[4]返回：worldLinkFramePosition，vec3, list of 3 floats，URDF 链接框架的世界位置
        pos = p.getLinkState(self.body_unique_id,
                             link_id)[4]
        pos = np.array(pos)
        position_bias = np.array(position_bias)
        #您可以使用 applyExternalForce 和 applyExternalTorque 对实体施加力或扭矩。 请注意，此方法仅在使用 stepSimulation 显式步进模拟时才有效，
        p.applyExternalForce(self.body_unique_id,
                             link_id,
                             forceObj=force,
                             posObj=pos + position_bias,
                             flags=p.WORLD_FRAME)
        self.draw_a_line(pos + position_bias,
                         pos + position_bias + force,
                         [1, 0, 0],
                         f'{link_id}')

    def set_link_torque_world_frame(self,
                                    linkid,
                                    torque: np.ndarray):
        """
        """
        torque = np.array(torque)
        if torque.dot(torque) == 0:
            return
        p.applyExternalTorque(self.body_unique_id,
                              linkid,
                              torqueObj=torque,
                              flags=p.WORLD_FRAME)

    def draw_a_line(self,
                    start,
                    end,
                    line_color,
                    name):
        if "debugline" not in self.data:
            self.data["debugline"] = {}
        if name not in self.data["debugline"]:
            self.data["debugline"][name] = -100
        #addUserDebugLine 将返回一个非负的唯一 id，使用“replaceItemUniqueId”时，它将返回 replaceItemUniqueId
        self.data["debugline"][name] = p.addUserDebugLine(start,
                                                          end,
                                                          lineColorRGB=line_color,
                                                          replaceItemUniqueId=self.data["debugline"][name])

    def get_constraint_state(self):
        if self.if_fixed is False:
            #raise AttributeError("the force&torque sensor is not enabled")
            return [0,0,0,0,0,0]
        #getConstraintState（获取约束状态）：给定一个唯一的约束 id，您可以在最近的模拟步骤中查询应用的约束力。 输入是一个约束唯一 id，输出是一个约束力向量，其维度是受约束影响的自由度（例如，固定约束影响 6 个自由度）
        return p.getConstraintState(self.constraint_id)

    def reset(self):
        """
        reset the base position orientation and velocity
        """
        self.logger.debug("The MAV is Reset")
        p.resetBasePositionAndOrientation(self.body_unique_id,
                                          posObj=self.params.initial_xyz,
                                          ornObj=self.params.initial_orientation)

    def close(self):
        """
        close this environment
        """
        self.logger.debug("The MAV is closed")
        p.removeBody(self.body_unique_id)
        p.disconnect(self.physics_client)

    def housekeeping(self):
        """
        all the cache will be removed
        """
        self.data.clear()
