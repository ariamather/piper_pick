#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from piper_control import PIPER
from math import pi
from ultralytics import YOLO  # 使用YOLOv8

class YOLODetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO("model/best.pt")  # 替换为你的训练模型路径或官方模型
        self.piper = PIPER(broadcast_tf=False)
        self.piper.init_pose()
        rospy.sleep(2)
        self.piper.joint_control_piper([0.0, 1.57, -1.57, 0.0, 0.9, 0.0, 0.07])
        rospy.sleep(4)

        self.task_active = False
        rospy.Subscriber("/task", String, self.task_callback)
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.pc_sub = rospy.Subscriber("/camera/depth/points", PointCloud2, self.pointcloud_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.last_depth = None
        self.last_pointcloud = None

    def task_callback(self, msg):
        if msg.data == "pick1":
            rospy.loginfo("Received task: pick1")
            self.task_active = True
        else:
            self.task_active = False

    def image_callback(self, msg):
        if not self.task_active:
            return
        
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model(image)[0]  # 进行检测

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if self.last_pointcloud:
                try:
                    gen = pc2.read_points(self.last_pointcloud, field_names=("x", "y", "z"), skip_nans=False, uvs=[[cx, cy]])
                    point = next(gen)
                    x, y, z = point
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        point_camera = PointStamped()
                        point_camera.header = self.last_pointcloud.header
                        point_camera.point.x = x
                        point_camera.point.y = y
                        point_camera.point.z = z

                        transform = self.tf_buffer.lookup_transform("base_link",
                                                                    point_camera.header.frame_id,
                                                                    rospy.Time(0),
                                                                    rospy.Duration(1.0))
                        point_base = tf2_geometry_msgs.do_transform_point(point_camera, transform)
                        rospy.loginfo(f"YOLO target at base_link: ({point_base.point.x:.3f}, {point_base.point.y:.3f}, {point_base.point.z:.3f})")

                        # 执行抓取动作
                        self.perform_pick(point_base.point.x, point_base.point.y, point_base.point.z)
                        self.task_active = False  # 完成一次抓取后重置任务
                        break

                except (StopIteration, tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn(f"Transform failed: {e}")

        cv2.imshow("YOLO Detection", image)
        cv2.waitKey(1)

    def perform_pick(self, x, y, z):
        target_positions = [
            (x - 0.2, y, z + 0.15, pi / 2 * 1.3),
            (x - 0.16, y, z + 0.1, pi / 2 * 1.25),
            (x - 0.1, y, z + 0.08, pi / 2 * 1.25)
        ]
        for pos in target_positions:
            self.piper.descartes_control_piper(pos[0], pos[1], pos[2], 0.0, pos[3], 0.0, 0.07)
            rospy.sleep(1.5)
        self.piper.descartes_control_piper(pos[0], pos[1], pos[2], 0.0, pos[3], 0.0, 0.02)
        rospy.sleep(1)
        self.piper.joint_control_piper([0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.02])
        rospy.sleep(1)
        self.piper.joint_control_piper([0.0, 0, 0, 0.0, 0.0, 0.0, 0.02])
        rospy.sleep(100)
        exit()

    def depth_callback(self, msg):
        self.last_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def pointcloud_callback(self, msg):
        self.last_pointcloud = msg

if __name__ == '__main__':
    rospy.init_node("yolo_detector_node")
    detector = YOLODetector()
    rospy.spin()
