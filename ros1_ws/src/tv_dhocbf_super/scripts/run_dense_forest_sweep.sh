#!/usr/bin/env bash
set -euo pipefail

# Paper-facing dense-obstacle agile-flight benchmark.
# It intentionally uses the same public launch entry as the GUI demo:
#   roslaunch tv_dhocbf_super tvdhocbf_dense_forest.launch ...

SUPER_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="$(cd "$SUPER_WS/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/sim_validation/results/dense_forest_ros}"
TRIAL_SECONDS="${TRIAL_SECONDS:-55}"
RVIZ="${RVIZ:-false}"
BACKEND="${BACKEND:-pcd}"

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

"$SUPER_WS/src/tv_dhocbf_super/scripts/generate_yopo_forest_pcd.py"

for density in ${densities[*]}; do
  for controller in ${controllers[*]}; do
    log_csv="$OUT_DIR/dense_forest_${density}_${controller}.csv"
    ros_log="$OUT_DIR/dense_forest_${density}_${controller}.roslaunch.log"
    echo "[dense-forest] density=$density controller=$controller backend=$BACKEND log=$log_csv"
    timeout "$TRIAL_SECONDS" roslaunch tv_dhocbf_super tvdhocbf_dense_forest.launch \
      rviz:="$RVIZ" \
      backend:="$BACKEND" \
      density:="$density" \
      controller:="$controller" \
      log_csv:="$log_csv" > "$ros_log" 2>&1 || true
    sleep 1
  done
done

echo "[dense-forest] logs written to $OUT_DIR"
