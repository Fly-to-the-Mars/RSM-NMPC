"""Compatibility package for ROS launch scripts.

The release workspace keeps the shared implementation in `rsm_sim`. The ROS
node uses the historical `sim_validation.*` import path, so these modules
forward imports to `rsm_sim` without duplicating code.
"""
