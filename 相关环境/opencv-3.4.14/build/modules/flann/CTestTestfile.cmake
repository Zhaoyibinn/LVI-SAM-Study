# CMake generated Testfile for 
# Source directory: /home/zhaoyibin/slam/opencv-3.4.14/modules/flann
# Build directory: /home/zhaoyibin/slam/opencv-3.4.14/build/modules/flann
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_flann "/home/zhaoyibin/slam/opencv-3.4.14/build/bin/opencv_test_flann" "--gtest_output=xml:opencv_test_flann.xml")
set_tests_properties(opencv_test_flann PROPERTIES  LABELS "Main;opencv_flann;Accuracy" WORKING_DIRECTORY "/home/zhaoyibin/slam/opencv-3.4.14/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVUtils.cmake;1686;add_test;/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVModule.cmake;1290;ocv_add_test_from_target;/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVModule.cmake;1072;ocv_add_accuracy_tests;/home/zhaoyibin/slam/opencv-3.4.14/modules/flann/CMakeLists.txt;2;ocv_define_module;/home/zhaoyibin/slam/opencv-3.4.14/modules/flann/CMakeLists.txt;0;")
