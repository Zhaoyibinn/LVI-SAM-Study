# CMake generated Testfile for 
# Source directory: /home/zhaoyibin/slam/opencv-3.4.14/modules/shape
# Build directory: /home/zhaoyibin/slam/opencv-3.4.14/build/modules/shape
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_shape "/home/zhaoyibin/slam/opencv-3.4.14/build/bin/opencv_test_shape" "--gtest_output=xml:opencv_test_shape.xml")
set_tests_properties(opencv_test_shape PROPERTIES  LABELS "Main;opencv_shape;Accuracy" WORKING_DIRECTORY "/home/zhaoyibin/slam/opencv-3.4.14/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVUtils.cmake;1686;add_test;/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVModule.cmake;1290;ocv_add_test_from_target;/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVModule.cmake;1072;ocv_add_accuracy_tests;/home/zhaoyibin/slam/opencv-3.4.14/modules/shape/CMakeLists.txt;2;ocv_define_module;/home/zhaoyibin/slam/opencv-3.4.14/modules/shape/CMakeLists.txt;0;")
