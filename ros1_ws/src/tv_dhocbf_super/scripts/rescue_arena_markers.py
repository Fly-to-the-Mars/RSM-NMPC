#!/usr/bin/env python3
"""Semantic RViz markers for the TV-DHOCBF rescue arena."""

import math

import rospy
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


FRAME_ID = "world"


def color(r, g, b, a=1.0):
    return ColorRGBA(float(r), float(g), float(b), float(a))


def point(x, y, z):
    return Point(float(x), float(y), float(z))


def quat_from_yaw(yaw):
    qz = math.sin(0.5 * yaw)
    qw = math.cos(0.5 * yaw)
    return 0.0, 0.0, qz, qw


def base_marker(ns, mid, marker_type):
    m = Marker()
    m.header.frame_id = FRAME_ID
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = mid
    m.type = marker_type
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def cube(ns, mid, xyz, scale, yaw, rgba):
    m = base_marker(ns, mid, Marker.CUBE)
    m.pose.position = point(*xyz)
    qx, qy, qz, qw = quat_from_yaw(yaw)
    m.pose.orientation.x = qx
    m.pose.orientation.y = qy
    m.pose.orientation.z = qz
    m.pose.orientation.w = qw
    m.scale = Vector3(float(scale[0]), float(scale[1]), float(scale[2]))
    m.color = rgba
    return m


def sphere(ns, mid, xyz, scale, rgba):
    m = base_marker(ns, mid, Marker.SPHERE)
    m.pose.position = point(*xyz)
    m.scale = Vector3(float(scale), float(scale), float(scale))
    m.color = rgba
    return m


def text(ns, mid, label, xyz, size=0.6):
    m = base_marker(ns, mid, Marker.TEXT_VIEW_FACING)
    m.pose.position = point(*xyz)
    m.scale.z = float(size)
    m.color = color(0.04, 0.04, 0.04, 1.0)
    m.text = label
    return m


def line(ns, mid, pts, width, rgba):
    m = base_marker(ns, mid, Marker.LINE_STRIP)
    m.scale.x = float(width)
    m.color = rgba
    m.points = [point(x, y, z) for x, y, z in pts]
    return m


def build_markers(show_labels=True):
    arr = MarkerArray()
    blue = color(0.07, 0.34, 0.78, 0.82)
    green = color(0.04, 0.58, 0.30, 0.92)
    amber = color(0.94, 0.55, 0.08, 0.30)
    red = color(0.86, 0.10, 0.08, 0.78)
    purple = color(0.38, 0.22, 0.68, 0.26)
    cyan = color(0.05, 0.55, 0.68, 0.24)

    arr.markers.append(sphere("arena_start_goal", 0, (0.0, -8.0, 1.5), 0.55, green))
    arr.markers.append(sphere("arena_start_goal", 1, (0.0, 42.0, 1.5), 0.65, red))
    if show_labels:
        arr.markers.append(text("arena_labels", 0, "START", (-1.2, -8.0, 2.6), 0.55))
        arr.markers.append(text("arena_labels", 1, "GOAL", (-1.0, 42.0, 2.8), 0.65))

    guide = [
        (0.0, -8.0, 1.62),
        (1.5, -2.0, 1.62),
        (-1.3, 3.4, 1.62),
        (1.5, 8.8, 1.62),
        (0.0, 14.9, 1.62),
        (-1.0, 22.5, 1.62),
        (1.0, 29.0, 1.62),
        (0.0, 35.0, 1.62),
        (0.0, 42.0, 1.62),
    ]
    arr.markers.append(line("arena_nominal_corridor", 0, guide, 0.065, blue))

    zones = [
        ("S-bend local replanning", (0.0, 4.8, 0.04), (12.8, 14.0, 0.08), 0.0, cyan, 2),
        ("tilted narrow gate", (0.0, 14.9, 0.08), (8.0, 3.0, 0.10), math.radians(24.0), amber, 3),
        ("occluded L-corridor", (0.0, 26.3, 0.06), (5.8, 12.5, 0.10), 0.0, purple, 4),
        ("inspection slot", (0.0, 34.5, 0.07), (5.2, 5.4, 0.10), 0.0, amber, 5),
    ]
    for label, xyz, scale, yaw, rgba, mid in zones:
        arr.markers.append(cube("arena_zones", mid, xyz, scale, yaw, rgba))
        if show_labels:
            arr.markers.append(text("arena_zone_labels", mid, label, (xyz[0] - 2.6, xyz[1], 2.7), 0.42))

    # Visual gate frame: the physical PCD contains the actual obstacle surfaces.
    yaw = math.radians(24.0)
    arr.markers.append(cube("arena_gate_frame", 0, (-0.92, 14.49, 1.55), (0.08, 0.12, 3.1), yaw, red))
    arr.markers.append(cube("arena_gate_frame", 1, (0.92, 15.31, 1.55), (0.08, 0.12, 3.1), yaw, red))
    arr.markers.append(cube("arena_gate_frame", 2, (0.0, 14.9, 3.05), (2.0, 0.10, 0.10), yaw, red))

    return arr


class RescueArenaMarkers:
    def __init__(self):
        global FRAME_ID
        FRAME_ID = rospy.get_param("~frame_id", "world")
        self.show_labels = bool(rospy.get_param("~show_labels", False))
        self.pub_exp = rospy.Publisher("/fsm_node/visualization/exp_traj", MarkerArray, queue_size=1, latch=True)
        self.pub_points = rospy.Publisher("/fsm_node/visualization/points", MarkerArray, queue_size=1, latch=True)
        self.timer = rospy.Timer(rospy.Duration(1.0), self.publish)
        rospy.loginfo("TV-DHOCBF rescue arena semantic markers ready.")

    def publish(self, _event):
        msg = build_markers(self.show_labels)
        self.pub_exp.publish(msg)
        self.pub_points.publish(msg)


def main():
    rospy.init_node("rescue_arena_markers")
    RescueArenaMarkers()
    rospy.spin()


if __name__ == "__main__":
    main()
