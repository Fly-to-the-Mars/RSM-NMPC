#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DENSITY="${DENSITY:-${1:-d3}}"
CONTROLLER="${CONTROLLER:-${2:-proposed}}"
RVIZ="${RVIZ:-true}"
LOG_DIR="${LOG_DIR:-$ROOT/../01_dense_obstacle_agile_flight/outputs/ros_live}"

mkdir -p "$LOG_DIR"

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
fi
if [[ -f "$ROOT/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/devel/setup.bash"
else
  echo "Missing $ROOT/devel/setup.bash. Run ./build_ros_workspace.sh first." >&2
  exit 1
fi

roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch \
  rviz:="$RVIZ" \
  density:="$DENSITY" \
  controller:="$CONTROLLER" \
  log_csv:="$LOG_DIR/super_density_${DENSITY}_${CONTROLLER}_live.csv"
