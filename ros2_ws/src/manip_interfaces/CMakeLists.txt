# Write-once boilerplate. The only line you'll ever edit is the interface
# list below (append new .srv/.msg files as the pose/pointso topic-JSON
# interfaces get promoted).
cmake_minimum_required(VERSION 3.8)
project(manip_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GenerateMesh.srv"
  DEPENDENCIES sensor_msgs shape_msgs geometry_msgs
)

ament_package()