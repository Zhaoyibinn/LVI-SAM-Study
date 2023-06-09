# CMake generated Testfile for 
# Source directory: /home/zhaoyibin/slam/opencv-3.4.14/modules/dnn
# Build directory: /home/zhaoyibin/slam/opencv-3.4.14/build/modules/dnn
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_dnn "/home/zhaoyibin/slam/opencv-3.4.14/build/bin/opencv_test_dnn" "--gtest_output=xml:opencv_test_dnn.xml")
set_tests_properties(opencv_test_dnn PROPERTIES  LABELS "Main;opencv_dnn;Accuracy" WORKING_DIRECTORY "/home/zhaoyibin/slam/opencv-3.4.14/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVUtils.cmake;1686;add_test;/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVModule.cmake;1290;ocv_add_test_from_target;/home/zhaoyibin/slam/opencv-3.4.14/modules/dnn/CMakeLists.txt;140;ocv_add_accuracy_tests;/home/zhaoyibin/slam/opencv-3.4.14/modules/dnn/CMakeLists.txt;0;")
add_test(opencv_perf_dnn "/home/zhaoyibin/slam/opencv-3.4.14/build/bin/opencv_perf_dnn" "--gtest_output=xml:opencv_perf_dnn.xml")
set_tests_properties(opencv_perf_dnn PROPERTIES  LABELS "Main;opencv_dnn;Performance" WORKING_DIRECTORY "/home/zhaoyibin/slam/opencv-3.4.14/build/test-reports/performance" _BACKTRACE_TRIPLES "/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVUtils.cmake;1686;add_test;/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVModule.cmake;1201;ocv_add_test_from_target;/home/zhaoyibin/slam/opencv-3.4.14/modules/dnn/CMakeLists.txt;145;ocv_add_perf_tests;/home/zhaoyibin/slam/opencv-3.4.14/modules/dnn/CMakeLists.txt;0;")
add_test(opencv_sanity_dnn "/home/zhaoyibin/slam/opencv-3.4.14/build/bin/opencv_perf_dnn" "--gtest_output=xml:opencv_perf_dnn.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_dnn PROPERTIES  LABELS "Main;opencv_dnn;Sanity" WORKING_DIRECTORY "/home/zhaoyibin/slam/opencv-3.4.14/build/test-reports/sanity" _BACKTRACE_TRIPLES "/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVUtils.cmake;1686;add_test;/home/zhaoyibin/slam/opencv-3.4.14/cmake/OpenCVModule.cmake;1202;ocv_add_test_from_target;/home/zhaoyibin/slam/opencv-3.4.14/modules/dnn/CMakeLists.txt;145;ocv_add_perf_tests;/home/zhaoyibin/slam/opencv-3.4.14/modules/dnn/CMakeLists.txt;0;")
