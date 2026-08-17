#!/usr/bin/env python3
"""RViz rendering overlays for the YOPO-style random forest benchmark."""

import json
from pathlib import Path

import rospy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "config"


def color(r: float, g: float, b: float, a: float) -> ColorRGBA:
    return ColorRGBA(float(r), float(g), float(b), float(a))


def base_marker(ns: str, mid: int, marker_type: int) -> Marker:
    m = Marker()
    m.header.frame_id = "world"
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = mid
    m.type = marker_type
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def load_trees(density: str):
    path = config_dir() / ("tvdhocbf_yopo_forest_%s.json" % density)
    return json.loads(path.read_text(encoding="utf-8"))["trees"]


def build_markers(trees) -> MarkerArray:
    arr = MarkerArray()

    ground = base_marker("forest_ground", 0, Marker.CUBE)
    ground.pose.position.x = 0.0
    ground.pose.position.y = 0.0
    ground.pose.position.z = -0.035
    ground.scale.x = 16.8
    ground.scale.y = 110.0
    ground.scale.z = 0.035
    ground.color = color(0.50, 0.52, 0.54, 0.58)
    arr.markers.append(ground)

    for idx, tree in enumerate(trees):
        x = tree["x"]
        y = tree["y"]
        mix = tree["green_mix"]

        trunk = base_marker("forest_trunks", idx, Marker.CYLINDER)
        trunk.pose.position.x = x + 0.5 * tree["lean_x"]
        trunk.pose.position.y = y + 0.5 * tree["lean_y"]
        trunk.pose.position.z = 0.5 * tree["height"]
        trunk.scale.x = 2.0 * tree["radius"]
        trunk.scale.y = 2.0 * tree["radius"]
        trunk.scale.z = tree["height"]
        trunk.color = color(0.72 - 0.12 * mix, 0.88 - 0.08 * mix, 1.00, 0.62)
        arr.markers.append(trunk)

        canopy = base_marker("forest_canopies", idx, Marker.SPHERE)
        canopy.pose.position.x = x + 0.72 * tree["lean_x"]
        canopy.pose.position.y = y + 0.72 * tree["lean_y"]
        canopy.pose.position.z = tree["canopy_z"]
        canopy.scale.x = 2.0 * tree["canopy_rx"]
        canopy.scale.y = 2.0 * tree["canopy_ry"]
        canopy.scale.z = 2.0 * tree["canopy_rz"]
        canopy.color = color(0.88 - 0.18 * mix, 0.96 - 0.12 * mix, 1.00, 0.36)
        arr.markers.append(canopy)

        shadow = base_marker("forest_shadows", idx, Marker.CYLINDER)
        shadow.pose.position.x = x
        shadow.pose.position.y = y
        shadow.pose.position.z = 0.006
        shadow.scale.x = 1.30 * max(tree["canopy_rx"], tree["canopy_ry"])
        shadow.scale.y = 1.30 * max(tree["canopy_rx"], tree["canopy_ry"])
        shadow.scale.z = 0.012
        shadow.color = color(0.16, 0.22, 0.28, 0.055)
        arr.markers.append(shadow)

    start = base_marker("forest_start_goal", 0, Marker.SPHERE)
    start.pose.position.x = 0.0
    start.pose.position.y = -50.0
    start.pose.position.z = 1.5
    start.scale.x = start.scale.y = start.scale.z = 0.55
    start.color = color(0.00, 0.58, 0.34, 0.95)
    arr.markers.append(start)

    goal = base_marker("forest_start_goal", 1, Marker.SPHERE)
    goal.pose.position.x = 0.0
    goal.pose.position.y = 45.0
    goal.pose.position.z = 1.5
    goal.scale.x = goal.scale.y = goal.scale.z = 0.62
    goal.color = color(0.82, 0.14, 0.15, 0.95)
    arr.markers.append(goal)

    return arr


def main() -> None:
    rospy.init_node("yopo_forest_markers")
    density = str(rospy.get_param("~density", "d3"))
    trees = load_trees(density)
    pub = rospy.Publisher("/tvdhocbf_yopo_forest/markers", MarkerArray, queue_size=1, latch=True)
    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        pub.publish(build_markers(trees))
        rate.sleep()


if __name__ == "__main__":
    main()
