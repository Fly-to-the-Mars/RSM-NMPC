#!/usr/bin/env bash
set -euo pipefail

# Batch runner for the composite rescue-arena validation.  It records ROS CSV
# logs that can be consumed by sim_validation/composite_arena_validation.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/sim_validation/results/rescue_arena_composite}"
mkdir -p "$OUT_DIR"

controllers=("${CONTROLLERS:-nominal nmpc_dc dhocbf_fixed proposed no_perception_lcb fixed_sphere no_recoverability no_tightening instant_rsm}")

source "${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
if [ -f "$REPO_ROOT/super_ws/devel/setup.bash" ]; then
  source "$REPO_ROOT/super_ws/devel/setup.bash"
fi

for controller in ${controllers[*]}; do
  log_csv="$OUT_DIR/rescue_${controller}.csv"
  ros_log="$OUT_DIR/rescue_${controller}.roslaunch.log"
  extra_args=()
  case "$controller" in
    no_perception_lcb)
      extra_args+=(confidence_inflation:=0.0 horizon_inflation_growth:=0.0)
      ;;
    no_tightening|instant_rsm)
      extra_args+=(horizon_inflation_growth:=0.0)
      ;;
    no_recoverability|static_dhocbf|fixed_sphere)
      extra_args+=(v_max:=4.2)
      ;;
  esac
  echo "[rescue-arena] controller=$controller log=$log_csv"
  timeout "${TIMEOUT_SEC:-70}" roslaunch tv_dhocbf_super tvdhocbf_rescue_arena.launch \
    rviz:=false \
    semantic_markers:=false \
    controller:="$controller" \
    log_csv:="$log_csv" \
    "${extra_args[@]}" > "$ros_log" 2>&1 || true
done

python3 "$REPO_ROOT/sim_validation/composite_arena_validation.py" --out "$REPO_ROOT/Figure"
