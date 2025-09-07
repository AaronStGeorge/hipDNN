# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
    find_package(GTest CONFIG REQUIRED)
endif()

include(GoogleTest)
find_package(Threads REQUIRED)

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping binary add toy plugin integration tests.")
    return()
endif()

add_executable(binary_add_plugin_integration_test
    main.cpp
    ReluFwdInferenceIntegrationTest.cpp
)

target_compile_options(binary_add_plugin_integration_test PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_compile_definitions(binary_add_plugin_integration_test PRIVATE
    COMPONENT_NAME="binary_add_integration_test"
    PLUGIN_DIR="${BINARY_ADD_PLUGIN_DIR}/"
)

target_link_libraries(binary_add_plugin_integration_test PUBLIC
    GTest::gtest
    hip::host
    miopen_legacy_plugin
    hipdnn_frontend
    hipdnn_sdk
    Threads::Threads
)


if(BUILD_PLUGIN_AS_DEPENDENCY)
    clang_tidy_check(binary_add_plugin_integration_test)

    add_integration_test_target(binary_add_plugin_integration_test ${CMAKE_CURRENT_BINARY_DIR})
endif()
