#!/usr/bin/env python3
"""Optional lightweight RViz labels for the random tilted-rod field."""

import json
import math
from pathlib import Path

import rospy
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


FRAME_ID = "world"
OUT_PREFIX = "tvdhocbf_rsm_clutter_corridor"


def config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "config"


def color(r: float, g: float, b: float, a: float) -> ColorRGBA:
    return ColorRGBA(float(r), float(g), float(b), float(a))


def point(x: float, y: float, z: float) -> Point:
    return Point(float(x), float(y), float(z))


def quat_from_yaw(yaw: float):
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def base_marker(ns: str, mid: int, marker_type: int) -> Marker:
    m = Marker()
    m.header.frame_id = FRAME_ID
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = int(mid)
    m.type = marker_type
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def cube(ns: str, mid: int, xyz, scale, yaw: float, rgba: ColorRGBA) -> Marker:
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


def sphere(ns: str, mid: int, xyz, scale: float, rgba: ColorRGBA) -> Marker:
    m = base_marker(ns, mid, Marker.SPHERE)
    m.pose.position = point(*xyz)
    m.scale = Vector3(float(scale), float(scale), float(scale))
    m.color = rgba
    return m


def line_strip(ns: str, mid: int, pts, width: float, rgba: ColorRGBA) -> Marker:
    m = base_marker(ns, mid, Marker.LINE_STRIP)
    m.scale.x = float(width)
    m.color = rgba
    m.points = [point(*p) for p in pts]
    return m


def line_list(ns: str, mid: int, segments, width: float, rgba: ColorRGBA) -> Marker:
    m = base_marker(ns, mid, Marker.LINE_LIST)
    m.scale.x = float(width)
    m.color = rgba
    for a, b in segments:
        m.points.append(point(*a))
        m.points.append(point(*b))
    return m


def text(ns: str, mid: int, label: str, xyz, size: float, rgba: ColorRGBA) -> Marker:
    m = base_marker(ns, mid, Marker.TEXT_VIEW_FACING)
    m.pose.position = point(*xyz)
    m.scale.z = float(size)
    m.color = rgba
    m.text = label
    return m


def load_meta(density: str):
    path = config_dir() / ("%s_%s.json" % (OUT_PREFIX, density))
    return json.loads(path.read_text(encoding="utf-8"))


def build_markers(meta, show_labels: bool) -> MarkerArray:
    arr = MarkerArray()

    guide_color = color(0.42, 0.52, 0.62, 0.30)
    green = color(0.58, 0.78, 1.00, 0.88)
    red = color(0.18, 0.42, 0.78, 0.92)
    dark = color(0.04, 0.05, 0.06, 0.92)

    ground = cube("tilted_rod_ground", 0, (0.0, 0.0, -0.025), (16.9, 110.0, 0.028), 0.0, color(0.50, 0.52, 0.54, 0.46))
    arr.markers.append(ground)

    guide = [tuple(p) for p in meta["guide"]]
    arr.markers.append(line_strip("tilted_rod_centerline", 0, guide, 0.030, guide_color))

    arr.markers.append(sphere("tilted_rod_start_goal", 0, meta["start"], 0.58, green))
    arr.markers.append(sphere("tilted_rod_start_goal", 1, meta["goal"], 0.66, red))

    if show_labels:
        arr.markers.append(text("tilted_rod_labels", 0, "Random Tilted-Rod Field", (-5.8, -52.0, 2.8), 0.48, dark))
        arr.markers.append(text("tilted_rod_labels", 1, "Traversability %.1f" % meta["traversability"], (-6.6, 45.0, 2.4), 0.38, dark))

    return arr


class RsmClutterCorridorMarkers:
    def __init__(self):
        global FRAME_ID
        FRAME_ID = rospy.get_param("~frame_id", "world")
        self.density = str(rospy.get_param("~density", "d3"))
        self.show_labels = bool(rospy.get_param("~show_labels", False))
        self.meta = load_meta(self.density)
        self.pub = rospy.Publisher("/tvdhocbf_rsm_clutter_corridor/markers", MarkerArray, queue_size=1, latch=True)
        # Also publish on the existing visualization topic used by the package
        # RViz config, so the scene appears without manual display edits.
        self.pub_compat = rospy.Publisher("/fsm_node/visualization/exp_traj", MarkerArray, queue_size=1, latch=True)
        self.timer = rospy.Timer(rospy.Duration(0.75), self.publish)
        rospy.loginfo("Random tilted-rod labels ready: density=%s", self.density)

    def publish(self, _event):
        msg = build_markers(self.meta, self.show_labels)
        self.pub.publish(msg)
        self.pub_compat.publish(msg)


def main():
    rospy.init_node("rsm_clutter_corridor_markers")
    RsmClutterCorridorMarkers()
    rospy.spin()


if __name__ == "__main__":
    main()
