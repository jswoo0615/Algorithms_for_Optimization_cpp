# 1. 헤더 파일 경로 설정 (이 방 안의 include 폴더를 가리킴)
add_library(optimization_dual INTERFACE)
target_include_directories(optimization_dual INTERFACE include)

# 2. 구글 테스트(GTest) 다운로드 준비
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# 3. 테스트 실행 파일 만들기
# 주의: 내 방 기준이므로 test/test_optimization_dual.cpp 로 씁니다.
add_executable(test_autodiff test/test_optimization_dual.cpp)

# 4. 내 코드와 구글 테스트 연결
target_link_libraries(test_autodiff PRIVATE optimization_dual GTest::gtest_main)

# 5. 테스트 등록
include(GoogleTest)
gtest_discover_tests(test_autodiff)