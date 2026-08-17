#!/usr/bin/env python3
"""Semantic RViz overlays for the custom agile forest benchmark."""

import math
from typing import Iterable, Tuple

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def marker_color(r: float, g: float, b: float, a: float):
    from std_msgs.msg import ColorRGBA

    c = ColorRGBA()
    c.r, c.g, c.b, c.a = r, g, b, a
    return c


def centerline_x(y: float) -> float:
    return 1.05 * math.sin((y + 31.0) / 8.5) + 0.34 * math.sin((y + 8.0) / 3.6)


def point(x: float, y: float, z: float) -> Point:
    p = Point()
    p.x, p.y, p.z = x, y, z
    return p


def line_marker(ns: str, mid: int, pts: Iterable[Tuple[float, float, float]], color, width: float) -> Marker:
    m = Marker()
    m.header.frame_id = "world"
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = mid
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = width
    m.color = color
    m.points = [point(*p) for p in pts]
    return m


def cube_marker(ns: str, mid: int, center, scale, color) -> Marker:
    m = Marker()
    m.header.frame_id = "world"
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = mid
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = center
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = scale
    m.color = color
    return m


def text_marker(ns: str, mid: int, label: str, xyz) -> Marker:
    m = Marker()
    m.header.frame_id = "world"
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = mid
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = xyz
    m.pose.orientation.w = 1.0
    m.scale.z = 1.0
    m.color = marker_color(0.08, 0.08, 0.08, 0.85)
    m.text = label
    return m


def build_markers(show_labels: bool) -> MarkerArray:
    arr = MarkerArray()
    y_vals = [y for y in frange(-50.0, 45.0, 0.8)]
    center_pts = [(centerline_x(y), y, 1.55) for y in y_vals]
    arr.markers.append(line_marker("agile_forest_centerline", 0, center_pts, marker_color(0.05, 0.66, 0.38, 0.95), 0.08))

    sections = [
        ("S-bend recoverable slalom", -28.5, 27.0, marker_color(0.10, 0.45, 0.76, 0.08)),
        ("tilted keyhole gates", -3.5, 14.0, marker_color(0.82, 0.36, 0.12, 0.09)),
        ("occluded comb corridor", 17.0, 20.0, marker_color(0.55, 0.22, 0.68, 0.09)),
        ("exit speed lattice", 40.0, 22.0, marker_color(0.04, 0.60, 0.32, 0.08)),
    ]
    for i, (label, cy, sy, color) in enumerate(sections, start=10):
        arr.markers.append(cube_marker("agile_forest_sections", i, (0.0, cy, 0.02), (15.0, sy, 0.04), color))
        if show_labels:
            arr.markers.append(text_marker("agile_forest_labels", i + 100, label, (-6.7, cy, 3.4)))

    for i, y in enumerate([-10.0, -3.6, 3.4], start=30):
        cx = centerline_x(y) + 0.18 * math.sin(0.7 * (i - 30))
        arr.markers.append(line_marker("agile_forest_gate_axis", i, [(cx - 0.9, y, 1.5), (cx + 0.9, y, 1.5)], marker_color(1.0, 0.68, 0.10, 0.95), 0.055))
    return arr


def frange(start: float, stop: float, step: float):
    x = start
    while x <= stop + 1e-9:
        yield x
        x += step


def main() -> None:
    rospy.init_node("agile_forest_markers")
    show_labels = bool(rospy.get_param("~show_labels", False))
    pub = rospy.Publisher("/tvdhocbf_agile_forest/markers", MarkerArray, queue_size=1, latch=True)
    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        pub.publish(build_markers(show_labels))
        rate.sleep()


if __name__ == "__main__":
    main()
