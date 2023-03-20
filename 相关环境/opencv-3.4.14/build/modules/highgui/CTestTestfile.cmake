# CMake generated Testfile for 
# Source directory: /home/zhaoyibin/slam/opencv-3.4.14/modules/highgui
# Build directory: /home/zhaoyibin/slam/opencv-3.4.14/build/modules/highgui
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_highgui "/home/zhaoyibin/slam/opencv-3.4.14/build/bin/opencv_test_highgui" "--gtest_output=xml:opencv_test_highgui.xml")
set_tests_properties(opencv_test_highgui PROPERTIES  LABELS "Main;opencv_highgui;Accuracy" WORKING_DIRECTORY "/home/zhaoyibin/slam/opencv-3.4.14/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVUtils.cmake;1686;add_test;/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVModule.cmake;1290;ocv_add_test_from_target;/home/zhaoyibin/slam/opencv-3.4.14/modules/highgui/CMakeLists.txt;168;ocv_add_accuracy_tests;/home/zhaoyibin/slam/opencv-3.4.14/modules/highgui/CMakeLists.txt;0;")
