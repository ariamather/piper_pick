#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf
from sensor_msgs.msg import JointState
from piper_msgs.msg import PosCmd
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped


# float64 x
# float64 y
# float64 z
# float64 roll
# float64 pitch
# float64 yaw
# float64 gripper    # 单位：米    范围：0 ~ 0.08米
# int32 mode1
# int32 mode2

class PIPER:
    def __init__(self, broadcast_tf=False):
        
        self.broadcast_tf = broadcast_tf
        
        # 发布控制piper机械臂话题
        self.pub_descartes = rospy.Publisher('pos_cmd', PosCmd, queue_size=10)
        self.pub_joint = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.descartes_msgs = PosCmd()
        
        # 订阅实际回传的关节状态
        self.joint_states_sub = rospy.Subscriber('/joint_states_single', JointState, self.joint_states_callback)
        self.current_joint_positions = [0.0] * 7
        
        # 订阅末端执行器位置
        self.end_pose_sub = rospy.Subscriber('/end_pose', PoseStamped, self.end_pose_callback)
        self.current_end_pose = PoseStamped()  # 存储当前末端位姿
        if tf:
            self.tf_broadcaster = tf.TransformBroadcaster()


    def joint_states_callback(self, msg):
        """ 接收并更新当前关节角度 """
        if len(msg.name) == 7 and len(msg.position) == 7:
            self.current_joint_positions = msg.position
            # 打印可以注释掉
            # print("Current joint positions:", self.current_joint_positions)
            
    def end_pose_callback(self, msg):
        """ 接收并保存末端执行器位置 """
        self.current_end_pose = msg
        # 打印可选
        # print("Current end effector pose:", self.current_end_pose)
        if self.broadcast_tf:
            # 提取平移和四元数旋转
            position = msg.pose.position
            orientation = msg.pose.orientation

            # 发布 tf：从 base_link 到 end_effector
            self.tf_broadcaster.sendTransform(
                (position.x, position.y, position.z),
                (orientation.x, orientation.y, orientation.z, orientation.w),
                msg.header.stamp,                # 使用原始时间戳
                "end_effector",                  # 子坐标系名称
                "base_link"                      # 父坐标系名称
            )

    def get_joint_positions(self):
        """ 获取当前的关节角度列表 """
        return self.current_joint_positions
    
    def get_end_pose(self):
        """ 获取当前末端执行器位置 PoseStamped """
        return self.current_end_pose
    
    def init_pose(self):
        self.joint_control_piper([0.0,0.0,0.0,0.0,0.0,0.0,0.07])
        
        
        
    def descartes_control_piper(self,x,y,z,roll,pitch,yaw,gripper):
        self.descartes_msgs.x = x
        self.descartes_msgs.y = y
        self.descartes_msgs.z = z
        self.descartes_msgs.roll = roll
        self.descartes_msgs.pitch = pitch
        self.descartes_msgs.yaw = yaw
        self.descartes_msgs.gripper = gripper
        self.pub_descartes.publish(self.descartes_msgs)
        print(f"send descartes control piper command{x,y,z,roll,pitch,yaw,gripper}")
    
    def joint_control_piper(self,pose):
        if len(pose) != 7:
            print("joint control piper command error")
            return
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = rospy.Time.now()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position = pose
        self.pub_joint.publish(joint_states_msgs)
        # self.rate.sleep()
        print(f"send joint control piper command{pose}")
    
     
# test code
if __name__ == '__main__':
    rospy.init_node('control_piper_node', anonymous=True)
    piper = PIPER(broadcast_tf=False) 
    rospy.on_shutdown(piper.init_pose)
    piper.init_pose()
    rospy.sleep(1)
    piper.joint_control_piper([0.2,0.0,0.0,0.0,0.0,0.0,0.07])
    rospy.sleep(1)
    # piper.descartes_control_piper(0.5, 0.0, 0.4, 0.0, 1.57, 0.0, 0.07)
    # 保持节点运行并监听外部程序的调用
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        joint_positions = piper.get_joint_positions()
        print("Current joint positions:", joint_positions)
        end_pose = piper.get_end_pose()
        pos = end_pose.pose.position
        ori = end_pose.pose.orientation
        print(f"End effector position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}")
        print(f"Orientation (quaternion): x={ori.x:.3f}, y={ori.y:.3f}, z={ori.z:.3f}, w={ori.w:.3f}")

        rate.sleep()
