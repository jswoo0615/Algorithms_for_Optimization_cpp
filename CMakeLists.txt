cmake_minimum_required(VERSION 3.14)
project(OptimizationStudy)

# C++17 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 각 브랜치(알고리즘)의 방으로 연결시켜줍니다.
add_subdirectory(1-automatic_differentiation)
# add_subdirectory(3-bracket-algorithms) # 이건 나중에 3번 브랜치할 때 주석 해제!
