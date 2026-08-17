#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd "${PKG_DIR}/../.." && pwd)"

source /opt/ros/noetic/setup.bash
source "${WS_DIR}/devel/setup.bash"

exec roslaunch tv_dhocbf_super tv_dhocbf_click.launch "$@"

