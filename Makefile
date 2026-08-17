.PHONY: check reproduce figures sensitivity sensitivity-full ros-build ros-density ros-arena clean

check:
	python3 check_workspace.py

reproduce:
	python3 run_all.py

figures:
	python3 run_all.py --skip-sensitivity

sensitivity:
	python3 03_parameter_sensitivity/run_parameter_sensitivity.py --trials 2 --max-steps 70

sensitivity-full:
	python3 03_parameter_sensitivity/run_parameter_sensitivity.py --trials 20

ros-build:
	cd ros1_ws && ./build_ros_workspace.sh

ros-density:
	cd ros1_ws && ./run_density_rviz.sh d3 proposed

ros-arena:
	cd ros1_ws && ./run_rescue_arena_rviz.sh proposed

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf ros1_ws/build ros1_ws/devel ros1_ws/log
