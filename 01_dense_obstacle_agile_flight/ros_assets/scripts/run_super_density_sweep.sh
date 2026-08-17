#!/usr/bin/env bash
set -euo pipefail

# Batch runner for the SUPER/MARSIM density benchmark.
# Use RVIZ=true for visual inspection, or keep RVIZ=false for metric sweeps.
#
# This copy is part of the paper code-release folder. By default it assumes the
# full repository layout still contains ./super_ws. Override SUPER_WS when using
# a different ROS workspace.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../../../.." && pwd)}"
SUPER_WS="${SUPER_WS:-$REPO_ROOT/super_ws}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/../../outputs/ros_logs}"
TRIAL_SECONDS="${TRIAL_SECONDS:-35}"
RVIZ="${RVIZ:-false}"

controllers=("${CONTROLLERS:-nominal nmpc_dc dhocbf_fixed proposed}")
densities=("${DENSITIES:-d1 d2 d3 d4}")

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
fi
if [[ -f "$SUPER_WS/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$SUPER_WS/devel/setup.bash"
fi

"$SUPER_WS/src/tv_dhocbf_super/scripts/generate_super_density_pcd.py"

for density in ${densities[*]}; do
  for controller in ${controllers[*]}; do
    log_csv="$OUT_DIR/super_density_${density}_${controller}.csv"
    ros_log="$OUT_DIR/super_density_${density}_${controller}.roslaunch.log"
    echo "[density-sweep] density=$density controller=$controller log=$log_csv"
    timeout "$TRIAL_SECONDS" roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch \
      rviz:="$RVIZ" \
      density:="$density" \
      controller:="$controller" \
      log_csv:="$log_csv" > "$ros_log" 2>&1 || true
    sleep 1
  done
done

echo "[density-sweep] logs written to $OUT_DIR"
