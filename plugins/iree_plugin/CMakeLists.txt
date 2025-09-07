message(STATUS "hello from iree plugin!")

find_package(IREERuntime CONFIG REQUIRED)
find_package(fusilli CONFIG REQUIRED)
message(STATUS "fusilli_DIR = ${fusilli_DIR}")

add_executable(iree-plugin-stub main.cpp)

get_target_property(interface_includes fusilli::fusilli INTERFACE_INCLUDE_DIRECTORIES)
if(interface_includes)
		message(STATUS "INTERFACE_INCLUDE_DIRECTORIES: ${interface_includes}")
else()
		message(STATUS "INTERFACE_INCLUDE_DIRECTORIES: (not set)")
endif()

target_link_libraries(iree-plugin-stub fusilli::fusilli)