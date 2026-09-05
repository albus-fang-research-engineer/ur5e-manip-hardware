# Write-once boilerplate. The only lines you'll ever edit are the interface
# list below (append new .srv/.msg files) and, when a new interface pulls in
# a message package not yet listed, its find_package + DEPENDENCIES entry.
cmake_minimum_required(VERSION 3.8)
project(manip_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GenerateMesh.srv"
  "srv/Segment.srv"
  "srv/EstimatePose.srv"
  "srv/Orient.srv"
  "srv/Release.srv"
  "msg/TSR.msg"
  "msg/Grasp.msg"
  "msg/GraspArray.msg"
  DEPENDENCIES sensor_msgs shape_msgs geometry_msgs std_msgs
)

ament_package()