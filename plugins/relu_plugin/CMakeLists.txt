# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

enable_language(HIP)
find_package(hip REQUIRED)

# RELU kernel
set_source_files_properties(ReluKernel.cpp PROPERTIES LANGUAGE HIP)
add_library(relu_kernel
    ReluKernel.cpp
)
target_link_libraries(relu_kernel
    PRIVATE
        hip::host
        hip::device
    )

# RELU plugin
add_library(relu_plugin SHARED
    ReluPlugin.cpp
)
target_link_libraries(relu_plugin PUBLIC hipdnn_sdk hip::host)
target_link_libraries(relu_plugin PRIVATE hip::host relu_kernel)
target_compile_definitions(relu_plugin PRIVATE
    COMPONENT_NAME="relu_plugin"
)

# used in tests
set(RELU_PLUGIN_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "miopen legacy plugin dir" FORCE)

add_subdirectory(integration_tests)