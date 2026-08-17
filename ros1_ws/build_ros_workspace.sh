#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
else
  echo "ROS Noetic setup.bash not found at /opt/ros/noetic/setup.bash" >&2
  exit 1
fi

cd "$ROOT"
catkin_make -DOpenCV_DIR="${OpenCV_DIR:-/usr/lib/x86_64-linux-gnu/cmake/opencv4}"

echo
echo "Build finished. Source the workspace with:"
echo "  source $ROOT/devel/setup.bash"
