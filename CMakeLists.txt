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

# [3번 브랜치 결과물] 나중에 합쳐지면 여기에 추가됨!
add_executable(test_autodiff test/test_autodiff.cpp)
target_link_libraries(test_autodiff PRIVATE GTest::gtest_main)
gtest_discover_tests(test_autodiff)

add_executable(test_bracketing test/test_bracketing.cpp)
target_link_libraries(test_bracketing PRIVATE GTest::gtest_main)
gtest_discover_tests(test_bracketing)

add_executable(test_linesearch test/test_linesearch.cpp)
target_link_libraries(test_linesearch PRIVATE GTest::gtest_main)
gtest_discover_tests(test_linesearch)

# [4번 브랜치 결과물]
add_executable(test_backtracking test/test_backtracking.cpp)
target_link_libraries(test_backtracking PRIVATE GTest::gtest_main)
gtest_discover_tests(test_backtracking)

add_executable(test_strong_backtracking test/test_strong_backtracking.cpp)
target_link_libraries(test_strong_backtracking PRIVATE GTest::gtest_main)
gtest_discover_tests(test_strong_backtracking)

add_executable(test_trust_region test/test_trust_region.cpp)
target_link_libraries(test_trust_region PRIVATE GTest::gtest_main)
gtest_discover_tests(test_trust_region)

add_executable(test_gradient_descent test/test_gradient_descent.cpp)
target_link_libraries(test_gradient_descent PRIVATE GTest::gtest_main)
gtest_discover_tests(test_gradient_descent)

add_executable(test_conjugate_gradient test/test_conjugate_gradient.cpp)
target_link_libraries(test_conjugate_gradient PRIVATE GTest::gtest_main)
gtest_discover_tests(test_conjugate_gradient)

add_executable(test_momentum test/test_momentum.cpp)
target_link_libraries(test_momentum PRIVATE GTest::gtest_main)
gtest_discover_tests(test_momentum)

add_executable(test_nestrovMomentum test/test_nestrovMomentum.cpp)
target_link_libraries(test_nestrovMomentum PRIVATE GTest::gtest_main)
gtest_discover_tests(test_nestrovMomentum)

add_executable(test_adagrad test/test_adagrad.cpp)
target_link_libraries(test_adagrad PRIVATE GTest::gtest_main)
gtest_discover_tests(test_adagrad)

add_executable(test_rmsprop test/test_rmsprop.cpp)
target_link_libraries(test_rmsprop PRIVATE GTest::gtest_main)
gtest_discover_tests(test_rmsprop)

add_executable(test_adam test/test_adam.cpp)
target_link_libraries(test_adam PRIVATE GTest::gtest_main)
gtest_discover_tests(test_adam)

add_executable(test_hyper_gradient_descent test/test_hyper_gradient_descent.cpp)
target_link_libraries(test_hyper_gradient_descent PRIVATE GTest::gtest_main)
gtest_discover_tests(test_hyper_gradient_descent)

add_executable(test_newton_method test/test_newton_method.cpp)
target_link_libraries(test_newton_method PRIVATE GTest::gtest_main)
gtest_discover_tests(test_newton_method)

add_executable(test_secant_method test/test_secant_method.cpp)
target_link_libraries(test_secant_method PRIVATE GTest::gtest_main)
gtest_discover_tests(test_secant_method)

add_executable(test_bfgs test/test_bfgs.cpp)
target_link_libraries(test_bfgs PRIVATE GTest::gtest_main)
gtest_discover_tests(test_bfgs)

add_executable(test_cyclic_coordinate test/test_cyclic_coordinate.cpp)
target_link_libraries(test_cyclic_coordinate PRIVATE GTest::gtest_main)
gtest_discover_tests(test_cyclic_coordinate)

add_executable(test_powell test/test_powell.cpp)
target_link_libraries(test_powell PRIVATE GTest::gtest_main)
gtest_discover_tests(test_powell)