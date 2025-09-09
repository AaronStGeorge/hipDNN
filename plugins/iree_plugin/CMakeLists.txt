# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

message(STATUS "hello from iree plugin!")

# dependencies
find_package(hip REQUIRED)
find_package(IREERuntime CONFIG REQUIRED)
find_package(fusilli CONFIG REQUIRED)
message(STATUS "fusilli_DIR = ${fusilli_DIR}")

# fusilli proof of life
add_executable(fusilli-connection-proof-of-life main.cpp)
target_link_libraries(fusilli-connection-proof-of-life fusilli::fusilli)

# iree plugin using fusilli

add_library(iree_plugin SHARED
    IreePlugin.cpp
)
target_link_libraries(iree_plugin PUBLIC hipdnn_sdk hip::host)
target_link_libraries(iree_plugin PRIVATE hip::host fusilli::fusilli)
target_compile_definitions(iree_plugin PRIVATE
    COMPONENT_NAME="iree_plugin"
)

# used in tests
set(IREE_PLUGIN_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "iree plugin dir" FORCE)

# add_subdirectory(integration_tests)