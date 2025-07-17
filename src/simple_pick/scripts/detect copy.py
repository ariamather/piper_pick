#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import apriltag
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from piper_control import PIPER
from math import pi
class AprilTagDetector:
    def __init__(self):
        self.bridge = CvBridge()
        options = apriltag.DetectorOptions(families="tag25h9")
        self.detector = apriltag.Detector(options)
        self.piper = PIPER(broadcast_tf=False)  # 创建PIPER实例
        self.piper.init_pose()
        self.piper.init_pose()
        rospy.sleep(2)
        self.piper.joint_control_piper([0.0, 1.57, -1.57, 0.0, 0.9, 0.0, 0.07])  # 初始化PIPER位置
        rospy.sleep(4)
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.pc_sub = rospy.Subscriber("/camera/depth/points", PointCloud2, self.pointcloud_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.last_depth = None
        self.last_pointcloud = None

    def image_callback(self, msg):
        # print("Image callback triggered")
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray)

        if not results:
            return

        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners.astype(int)
            cx, cy = int(r.center[0]), int(r.center[1])
            cv2.rectangle(image, tuple(ptA), tuple(ptC), (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

            if self.last_pointcloud:
                try:
                    # 从点云中读取该像素点的 (x, y, z)
                    gen = pc2.read_points(self.last_pointcloud, field_names=("x", "y", "z"), skip_nans=False,
                                          uvs=[[cx, cy]])
                    point = next(gen)
                    x, y, z = point
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        # 创建点在 camera_link 坐标系中的表示
                        point_camera = PointStamped()
                        point_camera.header = self.last_pointcloud.header
                        point_camera.point.x = x
                        point_camera.point.y = y
                        point_camera.point.z = z

                        # 查找并应用 TF 变换到 base_link 坐标系
                        transform = self.tf_buffer.lookup_transform("base_link",
                                                                    point_camera.header.frame_id,
                                                                    rospy.Time(0),
                                                                    rospy.Duration(1.0))  # 等待最多1秒

                        point_base = tf2_geometry_msgs.do_transform_point(point_camera, transform)
                        print(f"AprilTag ID: {r.tag_id}")
                        print(f"Camera frame position: ({x:.3f}, {y:.3f}, {z:.3f})")
                        print(f"base_link frame position: ({point_base.point.x:.3f}, {point_base.point.y:.3f}, {point_base.point.z:.3f})")
                        piper_position = [point_base.point.x-0.2, point_base.point.y, point_base.point.z+0.15]
                        piper_orientation = [0.0, pi/2 *1.3, 0.0]
                        self.piper.descartes_control_piper(piper_position[0], piper_position[1], piper_position[2],
                                                           piper_orientation[0], piper_orientation[1], piper_orientation[2],
                                                           0.07)
                        rospy.sleep(5)
                        piper_position = [point_base.point.x-0.16, point_base.point.y, point_base.point.z+0.1]
                        piper_orientation = [0.0, pi/2 *1.25, 0.0]
                        self.piper.descartes_control_piper(piper_position[0], piper_position[1], piper_position[2],
                                                           piper_orientation[0], piper_orientation[1], piper_orientation[2],
                                                           0.07)
                        rospy.sleep(1.5)
                        
                        piper_position = [point_base.point.x-0.1, point_base.point.y, point_base.point.z+0.08]
                        piper_orientation = [0.0, pi/2 *1.25, 0.0]
                        self.piper.descartes_control_piper(piper_position[0], piper_position[1], piper_position[2],
                                                           piper_orientation[0], piper_orientation[1], piper_orientation[2],
                                                           0.07)
                        rospy.sleep(1.5)
                        self.piper.descartes_control_piper(piper_position[0], piper_position[1], piper_position[2],
                                                           piper_orientation[0], piper_orientation[1], piper_orientation[2],
                                                           0.02)
                        rospy.sleep(1)
                        self.piper.joint_control_piper([0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.02])
                        rospy.sleep(1)
                        self.piper.joint_control_piper([0.0, 0, 0, 0.0, 0.0, 0.0, 0.02])
                        rospy.sleep(100)
                        exit()
                        
                except (StopIteration, tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn(f"Point transform failed: {e}")

        cv2.imshow("AprilTag Detection", image)
        cv2.waitKey(1)

    def depth_callback(self, msg):
        self.last_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def pointcloud_callback(self, msg):
        self.last_pointcloud = msg

if __name__ == '__main__':
    rospy.init_node("apriltag_detector_node")
    detector = AprilTagDetector()
    rospy.spin()
