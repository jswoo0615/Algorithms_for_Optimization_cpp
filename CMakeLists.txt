cmake_minimum_required(VERSION 3.14)
project(OptimizationStudy)

set(CMAKE_CXX_STANDARD 17)

# 헤더 경로 설정
include_directories(include)

# 구글 테스트 설정 (한 번만 해두면 모든 테스트에서 공유)
include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
FetchContent_MakeAvailable(googletest)

# [1번 브랜치 결과물]
add_executable(test_dual test/test_dual.cpp)
target_link_libraries(test_dual GTest::gtest_main)

# [3번 브랜치 결과물] 나중에 합쳐지면 여기에 추가됨!
# add_executable(test_bracket test/test_bracket.cpp)
# target_link_libraries(test_bracket GTest::gtest_main)