# Install script for directory: /home/zhaoyibin/slam/opencv-3.4.14/include

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv" TYPE FILE FILES
    "/home/zhaoyibin/slam/opencv-3.4.14/include/opencv/cv.h"
    "/home/zhaoyibin/slam/opencv-3.4.14/include/opencv/cv.hpp"
    "/home/zhaoyibin/slam/opencv-3.4.14/include/opencv/cvaux.h"
    "/home/zhaoyibin/slam/opencv-3.4.14/include/opencv/cvaux.hpp"
    "/home/zhaoyibin/slam/opencv-3.4.14/include/opencv/cvwimage.h"
    "/home/zhaoyibin/slam/opencv-3.4.14/include/opencv/cxcore.h"
    "/home/zhaoyibin/slam/opencv-3.4.14/include/opencv/cxcore.hpp"
    "/home/zhaoyibin/slam/opencv-3.4.14/include/opencv/cxeigen.hpp"
    "/home/zhaoyibin/slam/opencv-3.4.14/include/opencv/cxmisc.h"
    "/home/zhaoyibin/slam/opencv-3.4.14/include/opencv/highgui.h"
    "/home/zhaoyibin/slam/opencv-3.4.14/include/opencv/ml.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2" TYPE FILE FILES "/home/zhaoyibin/slam/opencv-3.4.14/include/opencv2/opencv.hpp")
endif()

