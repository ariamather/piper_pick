#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String, Bool # 引入Bool消息类型
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from piper_control import PIPER
from math import pi
from ultralytics import YOLO  # 使用YOLOv8

class YOLODetector:
    def __init__(self):
        # 初始化
        self.bridge = CvBridge()
        self.model = YOLO("model/best.pt")  # 替换为你的训练模型路径或官方模型
        self.piper = PIPER(broadcast_tf=False)

        # 机械臂移动到初始观察姿态
        rospy.loginfo("Initializing robot pose...")
        self.piper.init_pose()
        rospy.sleep(2)
        rospy.loginfo("Robot is ready.")

        # 任务状态变量
        self.current_task = None       # e.g., "pick", "place"
        self.target_class_id = -1      # 目标物体的类别ID
        self.place_y_spacing = 0.11     # 放置时y轴的间距

        # ROS Subscribers
        rospy.Subscriber("/task", String, self.task_callback)
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.pc_sub = rospy.Subscriber("/camera/depth/points", PointCloud2, self.pointcloud_callback)

        # ROS Publisher
        self.finished_pub = rospy.Publisher("/finished", Bool, queue_size=1)

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 缓存最新的深度图和点云f
        self.last_depth = None
        self.last_pointcloud = None

    def task_callback(self, msg):
        task_name = msg.data
        rospy.loginfo(f"Received task: {task_name}")

        if task_name.startswith("pick"):
            try:
                # 观察姿态
                self.piper.joint_control_piper([0.0, 1.57, -1.57, 0.0, 0.9, 0.0, 0.07])
                rospy.sleep(6)
                # e.g. pick1 -> class_id 0
                class_num = int(task_name.replace("pick", ""))
                if 1 <= class_num <= 4:
                    self.target_class_id = class_num - 1
                    self.current_task = "pick"
                    rospy.loginfo(f"Set to pick object with class ID: {self.target_class_id}")
                else:
                    rospy.logwarn("Invalid pick number. Must be between 1 and 4.")
            except ValueError:
                rospy.logwarn(f"Invalid task format: {task_name}")

        elif task_name.startswith("place"):
            try:
                place_num = int(task_name.replace("place", ""))
                if 1 <= place_num <= 4:
                    self.current_task = "place" # 阻止YOLO检测
                    # 计算放置位置
                    # x=0.5，y=-0.1+k*i，z=0.2 (k=0.1, i从0开始)
                    target_x = 0.5
                    target_y = -0.13 + self.place_y_spacing * (place_num - 1)
                    target_z = 0.2
                    rospy.loginfo(f"Performing place action at ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
                    self.perform_place(target_x, target_y, target_z)
                    self.publish_finished()
                    self.reset_state()
                else:
                    rospy.logwarn("Invalid place number. Must be between 1 and 4.")
            except ValueError:
                rospy.logwarn(f"Invalid task format: {task_name}")
        else:
            rospy.logwarn(f"Unknown task: {task_name}")
            self.reset_state()

    def image_callback(self, msg):
        # 只在 "pick" 任务激活时运行
        if self.current_task != "pick":
            return
        
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            results = self.model(image)[0]  # 进行检测

            for box in results.boxes:
                detected_class_id = int(box.cls[0])
                
                # 检查是否是我们正在寻找的目标
                if detected_class_id == self.target_class_id:
                    # print("See the goal")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(image, f"ID: {detected_class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


                    if self.last_pointcloud:
                        try:
                            # 从点云中获取目标点的3D坐标
                            gen = pc2.read_points(self.last_pointcloud, field_names=("x", "y", "z"), skip_nans=False, uvs=[[cx, cy]])
                            point = next(gen)
                            x, y, z = point

                            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                point_camera = PointStamped()
                                point_camera.header = self.last_pointcloud.header
                                point_camera.point.x = x
                                point_camera.point.y = y
                                point_camera.point.z = z

                                # 坐标转换到base_link
                                transform = self.tf_buffer.lookup_transform("base_link",
                                                                            point_camera.header.frame_id,
                                                                            rospy.Time(0),
                                                                            rospy.Duration(1.0))
                                point_base = tf2_geometry_msgs.do_transform_point(point_camera, transform)
                                rospy.loginfo(f"Target ID {self.target_class_id} at base_link: ({point_base.point.x:.3f}, {point_base.point.y:.3f}, {point_base.point.z:.3f})")

                                # 执行抓取动作
                                self.perform_pick(point_base.point.x, point_base.point.y, point_base.point.z)
                                
                                # 任务完成
                                self.publish_finished()
                                self.reset_state() # 重置任务状态，等待新任务
                                
                                #cv2.imshow("YOLO Detection", image)
                                cv2.waitKey(1)
                                return # 找到并处理完目标后，退出回调

                        except (StopIteration, tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                            rospy.logwarn(f"Could not get 3D point or transform: {e}")
            
            #cv2.imshow("YOLO Detection", image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

    def perform_pick(self, x, y, z):
        rospy.loginfo("Starting pick sequence...")
        # 抓取前的准备位置（物体上方）
        self.piper.descartes_control_piper(x-0.20, 0.99 * y -0.01, z + 0.15, 0.0, pi / 2, 0.0, 0.07)
        rospy.sleep(2)

        # 接近物体
        self.piper.descartes_control_piper(x-0.2, 0.99 * y -0.01, z + 0.08, 0.0, pi / 2+0.1, 0.0, 0.07)
        rospy.sleep(2)

        # 接近物体
        self.piper.descartes_control_piper(x-0.17, 0.99 * y-0.01, z + 0.035, 0.0, pi / 2+0.1, 0.0, 0.07)
        rospy.sleep(2)
        
        # 接近物体
        self.piper.descartes_control_piper(x-0.13, 0.99 * y-0.01, z + 0.035, 0.0, pi / 2+0.1, 0.0, 0.07)
        rospy.sleep(2)
        
        # 抓取（闭合夹爪）
        self.piper.descartes_control_piper(x-0.1, 0.99 * y-0.01, z + 0.02, 0.0, pi / 2+0.1, 0.0, 0.02)
        rospy.sleep(1.5)

        # 抬起物体
        self.piper.descartes_control_piper(x-0.1, 0.99 * y-0.01, z + 0.15, 0.0, pi / 2, 0.0, 0.02)
        rospy.sleep(2)

        # 回到初始姿态
        self.piper.joint_control_piper([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02])
        rospy.sleep(2)
        rospy.loginfo("Pick sequence finished.")

    def perform_place(self, x, y, z):
        rospy.loginfo("Starting place sequence...")
        # 移动到放置点上方
        self.piper.descartes_control_piper(x, y, z + 0.1, 0.0, pi / 2, 0.0, 0.02)
        rospy.sleep(2)
        
        # 移动到放置点
        self.piper.descartes_control_piper(x, y, z, 0.0, pi / 2, 0.0, 0.02)
        rospy.sleep(1.5)

        # 释放物体（打开夹爪）
        self.piper.descartes_control_piper(x, y, z, 0.0, pi / 2, 0.0, 0.07)
        rospy.sleep(1)

        # 抬起机械臂
        self.piper.descartes_control_piper(x, y, z + 0.1, 0.0, pi / 2, 0.0, 0.07)
        rospy.sleep(2)

        # 回到初始姿态
        self.piper.init_pose()
        rospy.sleep(2)

        rospy.loginfo(f"Placed object. Waiting for 5 seconds...")
        rospy.sleep(5)
        rospy.loginfo("Place sequence finished.")

    def publish_finished(self):
        """发布任务完成信号"""
        finished_msg = Bool()
        finished_msg.data = True
        self.finished_pub.publish(finished_msg)
        rospy.loginfo("Task finished. Published 'True' to /finished topic.")
    
    def reset_state(self):
        """重置任务状态"""
        self.current_task = None
        self.target_class_id = -1

    def depth_callback(self, msg):
        self.last_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def pointcloud_callback(self, msg):
        self.last_pointcloud = msg

if __name__ == '__main__':
    rospy.init_node("yolo_detector_node")
    try:
        detector = YOLODetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()