#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROLLER="${CONTROLLER:-proposed}"
RVIZ="${RVIZ:-true}"

for density in d1 d2 d3 d4; do
  echo
  echo "Launching density=$density controller=$CONTROLLER"
  echo "Close RViz/roslaunch to continue to the next density."
  "$ROOT/run_density_rviz.sh" "$density" "$CONTROLLER"
done
