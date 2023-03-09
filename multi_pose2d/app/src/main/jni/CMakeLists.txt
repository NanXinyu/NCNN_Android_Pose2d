project(ncnnbodypose)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.6.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230223-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(ncnnbodypose SHARED posedetncnn.cpp posedet.cpp ndkcamera.cpp)

target_link_libraries(ncnnbodypose ncnn ${OpenCV_LIBS} camera2ndk mediandk)
