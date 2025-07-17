#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from piper_control import PIPER

if __name__ == '__main__':
    rospy.init_node('piper_tf_broadcaster')
    piper = PIPER(broadcast_tf=True)  # 创建PIPER实例并启用TF广播
    # piper.init_pose()  # 初始化PIPER位置
    rospy.spin()  # 持续等待消息并广播TF
