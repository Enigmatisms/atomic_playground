cmake_minimum_required(VERSION 3.18)
project(atomic_multiply CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

find_package(Python REQUIRED COMPONENTS Development.Module Interpreter)

message(STATUS "Python include dir: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")
message(STATUS "Python executable: ${Python_EXECUTABLE}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CUDA_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_compile_options(
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>"  # CUDA PIC
    "$<$<COMPILE_LANGUAGE:CXX>:-fPIC>"               # C++ PIC
)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps")

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind/CMakeLists.txt")
    message(STATUS "Cloning nanobind submodule...")
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive -- external/nanobind
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

add_subdirectory(external/nanobind)
include_directories(
    ./include
    external/nanobind/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

aux_source_directory(./src ATOMIC_SRC)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bindings)
nanobind_add_module(atomic_ops_py 
    bindings/bind.cu
    ${ATOMIC_SRC}
)
target_link_libraries(atomic_ops_py
    PRIVATE 
    ${Python_LIBRARIES}
)
target_include_directories(atomic_ops_py PRIVATE
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    set(MODULE_SUFFIX ".pyd")
else()
    set(MODULE_SUFFIX ".so")
endif()

set_target_properties(atomic_ops_py PROPERTIES
    OUTPUT_NAME atomic_ops_py
    PREFIX ""
    SUFFIX ${MODULE_SUFFIX}
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_custom_target(test_module
    COMMAND ${Python_EXECUTABLE} -c "import sys; sys.path.insert(0, '${CMAKE_BINARY_DIR}/bindings'); import atomic_ops_py; print('Module imported successfully')"
    DEPENDS atomic_ops_py
    COMMENT "Testing module import"
)