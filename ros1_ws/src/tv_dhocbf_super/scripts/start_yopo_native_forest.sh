#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SUPER_WS="$REPO_ROOT/super_ws"
YOPO_WS="$REPO_ROOT/yopo_ws"

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
fi
if [[ -f "$SUPER_WS/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$SUPER_WS/devel/setup.bash"
fi
if [[ -f "$YOPO_WS/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$YOPO_WS/devel/setup.bash"
fi

roslaunch tv_dhocbf_super tvdhocbf_yopo_native_forest.launch "$@"
