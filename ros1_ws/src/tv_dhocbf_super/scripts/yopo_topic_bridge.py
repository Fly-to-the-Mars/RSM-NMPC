#!/usr/bin/env python3
"""Bridge YOPO simulator topics to the RSM-NMPC/SUPER topic contract."""

import math
from typing import Optional

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2


def quat_to_rot(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


class YopoTopicBridge:
    def __init__(self) -> None:
        self.odom: Optional[Odometry] = None
        self.max_lidar_points = int(rospy.get_param("~max_lidar_points", 8000))
        self.odom_pub = rospy.Publisher("/lidar_slam/odom", Odometry, queue_size=10)
        self.local_pub = rospy.Publisher("/cloud_registered", PointCloud2, queue_size=2)
        self.global_pub = rospy.Publisher("/global_pc", PointCloud2, queue_size=1, latch=True)
        self.cmd_pub = rospy.Publisher("/so3_control/pos_cmd", PositionCommand, queue_size=10)
        rospy.Subscriber("/sim/odom", Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber("/lidar_points", PointCloud2, self.lidar_cb, queue_size=1)
        rospy.Subscriber("/mock_map", PointCloud2, self.map_cb, queue_size=1)
        rospy.Subscriber("/planning/pos_cmd", PositionCommand, self.cmd_cb, queue_size=1)

    def odom_cb(self, msg: Odometry) -> None:
        self.odom = msg
        out = Odometry()
        out.header = msg.header
        out.header.frame_id = "world"
        out.child_frame_id = "base_link"
        out.pose = msg.pose
        out.twist = msg.twist
        self.odom_pub.publish(out)

    def lidar_cb(self, msg: PointCloud2) -> None:
        if self.odom is None:
            return
        p = self.odom.pose.pose.position
        origin = np.array([p.x, p.y, p.z], dtype=float)
        rot = quat_to_rot(self.odom.pose.pose.orientation)
        pts = []
        for i, q in enumerate(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if i >= self.max_lidar_points:
                break
            body = np.array([q[0], q[1], q[2]], dtype=float)
            world = origin + rot.dot(body)
            if all(math.isfinite(v) for v in world):
                pts.append((float(world[0]), float(world[1]), float(world[2])))
        header = msg.header
        header.frame_id = "world"
        header.stamp = rospy.Time.now()
        self.local_pub.publish(point_cloud2.create_cloud_xyz32(header, pts))

    def map_cb(self, msg: PointCloud2) -> None:
        msg.header.frame_id = "world"
        if msg.header.stamp.to_sec() == 0.0:
            msg.header.stamp = rospy.Time.now()
        self.global_pub.publish(msg)

    def cmd_cb(self, msg: PositionCommand) -> None:
        self.cmd_pub.publish(msg)


def main() -> None:
    rospy.init_node("yopo_topic_bridge")
    YopoTopicBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
