cmake_minimum_required(VERSION 3.0.2)
project(tv_dhocbf_super)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  quadrotor_msgs
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/agile_forest_markers.py
  scripts/generate_agile_forest_pcd.py
  scripts/generate_yopo_forest_pcd.py
  scripts/generate_super_density_pcd.py
  scripts/generate_rsm_clutter_corridor_pcd.py
  scripts/generate_rescue_arena_pcd.py
  scripts/global_cloud_colorizer.py
  scripts/rsm_clutter_corridor_markers.py
  scripts/rescue_arena_markers.py
  scripts/tv_dhocbf_super_node.py
  scripts/yopo_native_drone.py
  scripts/yopo_topic_bridge.py
  scripts/yopo_forest_markers.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/run_agile_forest_sweep.sh
  scripts/run_rsm_clutter_corridor_sweep.sh
  scripts/run_super_density_sweep.sh
  scripts/run_yopo_forest_sweep.sh
  scripts/start_yopo_native_forest.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
