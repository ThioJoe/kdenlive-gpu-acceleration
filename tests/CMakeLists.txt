# SPDX-FileCopyrightText: 2026 ThioJoe
# SPDX-License-Identifier: GPL-3.0-only

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
target_include_directories(gpufx_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(gpufx_test PRIVATE
    GPUFX_TEST_MODULE="$<TARGET_FILE:mltgpufx>"
    GPUFX_TEST_CUDA_LIBRARY="$<TARGET_FILE:gpufx_mockcuda>"
)
add_dependencies(gpufx_test mltgpufx gpufx_mockcuda)

add_test(NAME gpufx COMMAND gpufx_test)
add_test(NAME gpufx_no_gpu COMMAND gpufx_test --no-gpu)
add_test(NAME gpufx_settings_off COMMAND gpufx_test --settings-off)
# Without page-locked memory, frames are copied from and to MLT's memory
add_test(NAME gpufx_pageable COMMAND gpufx_test)
set_tests_properties(gpufx gpufx_settings_off PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
set_tests_properties(gpufx_no_gpu PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen;MOCKCUDA_NO_DEVICE=1")
set_tests_properties(gpufx_pageable PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen;MOCKCUDA_NO_PINNED=1")

# The settings page in a settings dialog made like Kdenlive's
if(TARGET gpufx-settings)
    add_executable(gpufx_settings_test test_settings_page.cpp)
    set_target_properties(gpufx_settings_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_include_directories(gpufx_settings_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(gpufx_settings_test PRIVATE Qt6::Widgets KF6::ConfigWidgets KF6::ConfigCore Mlt7::mlt ${CMAKE_DL_LIBS})
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Qt built with reduce-relocations, like Kdenlive's, binds its own
        # uses of its data: code using that data must be compiled with -fPIC
        # (not -fPIE), else it gets copies of its own, and findChild() does
        # not recognize Qt's meta objects. The Qt the test is built with may
        # not ask for it.
        target_compile_options(gpufx_settings_test PRIVATE -fPIC)
    endif()
    target_compile_definitions(gpufx_settings_test PRIVATE
        QT_NO_VERSION_TAGGING
        GPUFX_TEST_MODULE="$<TARGET_FILE:mltgpufx>"
        GPUFX_TEST_CUDA_LIBRARY="$<TARGET_FILE:gpufx_mockcuda>"
    )
    add_dependencies(gpufx_settings_test mltgpufx gpufx-settings gpufx_mockcuda)
    add_test(NAME gpufx_settings_page COMMAND gpufx_settings_test)
    set_tests_properties(gpufx_settings_page PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
endif()
