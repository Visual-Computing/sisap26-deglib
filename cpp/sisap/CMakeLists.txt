cmake_minimum_required(VERSION 3.19)

# Build deglib_sisap_task1 target
add_executable(deglib_sisap_task1 task1.cpp)
target_link_libraries(deglib_sisap_task1 PUBLIC DEG_LIB compile-options)
target_compile_definitions(deglib_sisap_task1 PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

# Build deglib_sisap_task2 target
add_executable(deglib_sisap_task2 task2.cpp)
target_link_libraries(deglib_sisap_task2 PUBLIC DEG_LIB compile-options)
target_compile_definitions(deglib_sisap_task2 PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

# Build deglib_sisap combined target
add_executable(deglib_sisap sisap.cpp task1.cpp task2.cpp)
target_link_libraries(deglib_sisap PUBLIC DEG_LIB compile-options)
target_compile_definitions(deglib_sisap PUBLIC "DATA_PATH=\"${DATA_PATH}\"")
target_compile_definitions(deglib_sisap PRIVATE "BUILD_COMBINED_SISAP")


