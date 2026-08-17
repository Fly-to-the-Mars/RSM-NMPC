#!/usr/bin/env bash
set -euo pipefail

# Compatibility batch runner for the paper dense-obstacle benchmark.
# The scene now uses SUPER's original dense random-map density variants.

SUPER_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="$(cd "$SUPER_WS/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/sim_validation/results/super_density_ros}"
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
export ROS_PACKAGE_PATH="$SUPER_WS/src:${ROS_PACKAGE_PATH:-}"

"$SUPER_WS/src/tv_dhocbf_super/scripts/generate_super_density_pcd.py"

for density in ${densities[*]}; do
  for controller in ${controllers[*]}; do
    log_csv="$OUT_DIR/super_density_${density}_${controller}.csv"
    ros_log="$OUT_DIR/super_density_${density}_${controller}.roslaunch.log"
    echo "[super-density] density=$density controller=$controller log=$log_csv"
    timeout "$TRIAL_SECONDS" roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch \
      rviz:="$RVIZ" \
      density:="$density" \
      controller:="$controller" \
      log_csv:="$log_csv" > "$ros_log" 2>&1 || true
    sleep 1
  done
done

echo "[super-density] logs written to $OUT_DIR"
