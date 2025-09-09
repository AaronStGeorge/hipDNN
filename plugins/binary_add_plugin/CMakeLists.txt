# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

message(STATUS "hello from binary add toy plugin!")

enable_language(HIP)

find_package(hip REQUIRED)



set_source_files_properties(reluKernel.cpp PROPERTIES LANGUAGE HIP)

add_library(relu_kernel
    reluKernel.cpp
)
target_link_libraries(relu_kernel
    PRIVATE
        hip::host
        hip::device
    )

add_library(binary_add_plugin SHARED
    BinaryAddPlugin.cpp
)
target_link_libraries(binary_add_plugin PUBLIC hipdnn_sdk hip::host)
target_link_libraries(binary_add_plugin PRIVATE hip::host relu_kernel)
target_compile_definitions(binary_add_plugin PRIVATE
    COMPONENT_NAME="binary_add_plugin"
)

# TODO: why
set(BINARY_ADD_PLUGIN_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "miopen legacy plugin dir" FORCE)


add_subdirectory(integration_tests)