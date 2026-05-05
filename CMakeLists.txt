cmake_minimum_required(VERSION 3.14)
project(OptimizationStudy)
enable_testing()

set(CMAKE_CXX_STANDARD 17)

# 헤더 경로 설정
include_directories(include)

# 구글 테스트 설정 (한 번만 해두면 모든 테스트에서 공유)
include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# [1번 브랜치 결과물]
add_executable(test_dual test/test_dual.cpp)
target_link_libraries(test_dual PRIVATE GTest::gtest_main)
gtest_discover_tests(test_dual)

add_executable(test_LevenbergMarquardt test/test_LevenbergMarquardt.cpp)
target_link_libraries(test_LevenbergMarquardt PRIVATE GTest::gtest_main)
gtest_discover_tests(test_LevenbergMarquardt)

add_executable(test_VehicleIntegrator test/test_VehicleIntegrator.cpp)
target_link_libraries(test_VehicleIntegrator PRIVATE GTest::gtest_main)
gtest_discover_tests(test_VehicleIntegrator)

add_executable(test_MultipleShooting test/test_MultipleShooting.cpp)
target_link_libraries(test_MultipleShooting PRIVATE GTest::gtest_main)
gtest_discover_tests(test_MultipleShooting)

add_executable(test_RiccatiSolver test/test_RiccatiSolver.cpp)
target_link_libraries(test_RiccatiSolver PRIVATE GTest::gtest_main)
gtest_discover_tests(test_RiccatiSolver)

add_executable(test_SparseNMPC test/test_SparseNMPC.cpp)
target_link_libraries(test_SparseNMPC PRIVATE GTest::gtest_main)
gtest_discover_tests(test_SparseNMPC)

add_executable(test_SparseNMPC_Battlefield test/test_SparseNMPC_Battlefield.cpp)
target_link_libraries(test_SparseNMPC_Battlefield PRIVATE GTest::gtest_main)
gtest_discover_tests(test_SparseNMPC_Battlefield)

add_executable(test_QPSolver_IPM test/test_QPSolver_IPM.cpp)
target_link_libraries(test_QPSolver_IPM PRIVATE GTest::gtest_main)
gtest_discover_tests(test_QPSolver_IPM)

add_executable(test_SQPSolver test/test_SQPSolver.cpp)
target_link_libraries(test_SQPSolver PRIVATE GTest::gtest_main)
gtest_discover_tests(test_SQPSolver)