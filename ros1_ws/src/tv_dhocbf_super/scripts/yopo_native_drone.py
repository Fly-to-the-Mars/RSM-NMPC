#!/usr/bin/env python3
"""Lightweight low-level executor for the YOPO native sensor simulator."""

import math
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from visualization_msgs.msg import Marker


def quat_from_yaw(yaw: float):
    qz = math.sin(0.5 * yaw)
    qw = math.cos(0.5 * yaw)
    return 0.0, 0.0, qz, qw


class YopoNativeDrone:
    def __init__(self) -> None:
        self.rate = float(rospy.get_param("~rate", 100.0))
        self.kp = np.array(rospy.get_param("~kp", [5.7, 5.7, 4.2]), dtype=float)
        self.kv = np.array(rospy.get_param("~kv", [3.4, 3.4, 4.0]), dtype=float)
        self.acc_limit = float(rospy.get_param("~acc_limit", 9.0))
        self.z_acc_limit = float(rospy.get_param("~z_acc_limit", 4.0))
        self.mesh = str(rospy.get_param("~mesh_resource", "package://perfect_drone_sim/meshes/yunque-M.dae"))
        self.pos = np.array(
            [
                float(rospy.get_param("~init_x", 0.0)),
                float(rospy.get_param("~init_y", -50.0)),
                float(rospy.get_param("~init_z", 1.5)),
            ],
            dtype=float,
        )
        self.vel = np.zeros(3, dtype=float)
        self.yaw = float(rospy.get_param("~init_yaw", 1.5708))
        self.yaw_rate = 0.0
        self.cmd: Optional[PositionCommand] = None
        self.odom_pub = rospy.Publisher("/sim/odom", Odometry, queue_size=20)
        self.mesh_pub = rospy.Publisher("/quadrotor_simulator_so3/uav", Marker, queue_size=1)
        rospy.Subscriber("/planning/pos_cmd", PositionCommand, self.cmd_cb, queue_size=1)

    def cmd_cb(self, msg: PositionCommand) -> None:
        self.cmd = msg

    def step(self, dt: float) -> None:
        if self.cmd is None:
            acc = np.array([0.0, 0.0, 2.0 * (1.5 - self.pos[2]) - 1.2 * self.vel[2]])
            yaw_des = self.yaw
        else:
            cmd = self.cmd
            p_des = np.array([cmd.position.x, cmd.position.y, cmd.position.z], dtype=float)
            v_des = np.array([cmd.velocity.x, cmd.velocity.y, cmd.velocity.z], dtype=float)
            a_ff = np.array([cmd.acceleration.x, cmd.acceleration.y, cmd.acceleration.z], dtype=float)
            finite_p = np.isfinite(p_des)
            finite_v = np.isfinite(v_des)
            p_err = np.where(finite_p, p_des - self.pos, 0.0)
            v_err = np.where(finite_v, v_des - self.vel, 0.0)
            acc = a_ff + self.kp * p_err + self.kv * v_err
            yaw_des = cmd.yaw if math.isfinite(cmd.yaw) else self.yaw
        xy_norm = float(np.linalg.norm(acc[:2]))
        if xy_norm > self.acc_limit:
            acc[:2] *= self.acc_limit / xy_norm
        acc[2] = float(np.clip(acc[2], -self.z_acc_limit, self.z_acc_limit))
        self.vel += acc * dt
        self.vel[:2] = np.clip(self.vel[:2], -6.5, 6.5)
        self.vel[2] = float(np.clip(self.vel[2], -2.5, 2.5))
        self.pos += self.vel * dt
        yaw_err = math.atan2(math.sin(yaw_des - self.yaw), math.cos(yaw_des - self.yaw))
        self.yaw_rate = float(np.clip(3.0 * yaw_err, -2.2, 2.2))
        self.yaw += self.yaw_rate * dt

    def publish(self) -> None:
        now = rospy.Time.now()
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position = Point(float(self.pos[0]), float(self.pos[1]), float(self.pos[2]))
        qx, qy, qz, qw = quat_from_yaw(self.yaw)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear = Vector3(float(self.vel[0]), float(self.vel[1]), float(self.vel[2]))
        odom.twist.twist.angular = Vector3(0.0, 0.0, float(self.yaw_rate))
        self.odom_pub.publish(odom)

        marker = Marker()
        marker.header = odom.header
        marker.ns = "mesh"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.mesh_resource = self.mesh
        marker.pose = odom.pose.pose
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color.r = 0.95
        marker.color.g = 0.95
        marker.color.b = 0.95
        marker.color.a = 1.0
        self.mesh_pub.publish(marker)

    def run(self) -> None:
        rate = rospy.Rate(self.rate)
        last = rospy.Time.now()
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = max(1e-3, min(0.05, (now - last).to_sec()))
            last = now
            self.step(dt)
            self.publish()
            rate.sleep()


def main() -> None:
    rospy.init_node("yopo_native_drone")
    YopoNativeDrone().run()


if __name__ == "__main__":
    main()
