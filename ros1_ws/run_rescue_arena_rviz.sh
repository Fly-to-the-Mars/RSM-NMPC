#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROLLER="${CONTROLLER:-${1:-proposed}}"
RVIZ="${RVIZ:-true}"
LOG_DIR="${LOG_DIR:-$ROOT/../02_composite_clutter_arena/outputs/ros_live}"

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

roslaunch tv_dhocbf_super tvdhocbf_complex_arena.launch \
  rviz:="$RVIZ" \
  controller:="$CONTROLLER" \
  log_csv:="$LOG_DIR/rescue_arena_${CONTROLLER}_live.csv"
