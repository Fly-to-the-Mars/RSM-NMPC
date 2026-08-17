#!/usr/bin/env python3
"""Republish the global map cloud with deterministic paper-style colors.

MARSIM/SUPER publishes the collision map as an XYZ PointCloud2.  RViz can color
that cloud in several ways, but the selected transformer is easy to override
accidentally when displays are saved or when a cloud contains extra fields.
This node creates a dedicated RGB cloud for visualization only; the planner
continues to consume /global_pc unchanged.
"""

import math
import struct
from typing import Iterable, Tuple

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField


def pack_rgb(r: int, g: int, b: int) -> float:
    rgb = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", rgb))[0]


def lerp(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round((1.0 - t) * a[i] + t * b[i])) for i in range(3))


class GlobalCloudColorizer:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param("~input", "/global_pc")
        self.output_topic = rospy.get_param("~output", "/tvdhocbf_visual/global_pc_blue")
        self.max_points = int(rospy.get_param("~max_points", 220000))
        self.z_min = float(rospy.get_param("~z_min", -0.10))
        self.z_mid = float(rospy.get_param("~z_mid", 1.55))
        self.z_max = float(rospy.get_param("~z_max", 3.80))
        self.low = tuple(int(v) for v in rospy.get_param("~low_rgb", [248, 252, 255]))
        self.mid = tuple(int(v) for v in rospy.get_param("~mid_rgb", [145, 210, 246]))
        self.high = tuple(int(v) for v in rospy.get_param("~high_rgb", [22, 92, 178]))
        self.frame_id = rospy.get_param("~frame_id", "")
        self.published_once = False

        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1, latch=True)
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.cloud_cb, queue_size=1)
        rospy.loginfo("Global cloud colorizer: %s -> %s", self.input_topic, self.output_topic)

    def color_for_z(self, z: float) -> float:
        if not math.isfinite(z):
            return pack_rgb(*self.mid)
        if z <= self.z_mid:
            rgb = lerp(self.low, self.mid, (z - self.z_min) / max(self.z_mid - self.z_min, 1e-6))
        else:
            rgb = lerp(self.mid, self.high, (z - self.z_mid) / max(self.z_max - self.z_mid, 1e-6))
        return pack_rgb(*rgb)

    def colored_points(self, msg: PointCloud2) -> Iterable[Tuple[float, float, float, float]]:
        total = max(1, int(msg.width) * max(1, int(msg.height)))
        step = max(1, int(math.ceil(total / max(self.max_points, 1))))
        for i, p in enumerate(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if i % step:
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            yield (x, y, z, self.color_for_z(z))

    def cloud_cb(self, msg: PointCloud2) -> None:
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]
        header = msg.header
        if self.frame_id:
            header.frame_id = self.frame_id
        out = point_cloud2.create_cloud(header, fields, list(self.colored_points(msg)))
        self.pub.publish(out)
        if not self.published_once:
            self.published_once = True
            rospy.loginfo("Published colorized global cloud with %d points", out.width)


def main() -> None:
    rospy.init_node("global_cloud_colorizer")
    GlobalCloudColorizer()
    rospy.spin()


if __name__ == "__main__":
    main()
