# SPDX-FileCopyrightText: 2026 Kdenlive contributors
# SPDX-License-Identifier: BSD-2-Clause

# The module's page in Kdenlive's settings dialog (see settingspage.cpp). It
# uses Qt and KDE Frameworks, which Kdenlive has loaded already: the module
# loads it only in Kdenlive's main program, from a gpufx folder next to it,
# which MLT does not look into.

set(targets gpufx-settings)
if(BUILD_TESTING)
    # A variant that opens the page in Kdenlive and takes a screenshot, for
    # the Windows workflow (see selftest.cpp). Not installed.
    list(APPEND targets gpufx-settings-selftest)
endif()
foreach(target ${targets})
    add_library(${target} MODULE settingspage.cpp)
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(${target} PRIVATE Qt6::Widgets KF6::ConfigWidgets KF6::ConfigCore)
    # Kdenlive's Qt can be older than the one the page is built with, which is
    # fine as long as it has every function the page uses. The tests check that
    # by loading the page with Kdenlive's own libraries.
    target_compile_definitions(${target} PRIVATE QT_NO_VERSION_TAGGING)
    set_target_properties(${target} PROPERTIES
        PREFIX ""
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        # Next to the module in the build folder too, where the module finds it
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/gpufx"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/gpufx"
    )
endforeach()
if(TARGET gpufx-settings-selftest)
    target_sources(gpufx-settings-selftest PRIVATE selftest.cpp)
    target_compile_definitions(gpufx-settings-selftest PRIVATE GPUFX_SETTINGS_SELFTEST)
endif()
install(TARGETS gpufx-settings
    LIBRARY DESTINATION "${gpufx_install_dir}/gpufx"
    RUNTIME DESTINATION "${gpufx_install_dir}/gpufx")
