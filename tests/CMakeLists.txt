# SPDX-FileCopyrightText: 2026 Kdenlive contributors
# SPDX-License-Identifier: BSD-2-Clause

# Emulated CUDA driver: runs the kernels on the CPU (see mockcuda/mockcuda.cpp)
add_library(gpufx_mockcuda MODULE
    mockcuda/mockcuda.cpp
    mockcuda/emulated_kernels.cpp
)
set_target_properties(gpufx_mockcuda PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
)
if(NOT MSVC)
    target_compile_options(gpufx_mockcuda PRIVATE -ffp-contract=off)
endif()

add_executable(gpufx_test test_gpufx.cpp)
set_target_properties(gpufx_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_link_libraries(gpufx_test PRIVATE Mlt7::mlt ${CMAKE_DL_LIBS})
target_compile_definitions(gpufx_test PRIVATE
    GPUFX_TEST_MODULE="$<TARGET_FILE:mltgpufx>"
    GPUFX_TEST_CUDA_LIBRARY="$<TARGET_FILE:gpufx_mockcuda>"
)
add_dependencies(gpufx_test mltgpufx gpufx_mockcuda)

add_test(NAME gpufx COMMAND gpufx_test)
add_test(NAME gpufx_no_gpu COMMAND gpufx_test --no-gpu)
set_tests_properties(gpufx PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
set_tests_properties(gpufx_no_gpu PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen;MOCKCUDA_NO_DEVICE=1")
