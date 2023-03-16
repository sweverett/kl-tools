cmake_minimum_required(VERSION 3.16)
project(grism_module)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(/usr/local/include)
# add extra include directories
#include_directories(/Users/jiachuanxu/Workspace/GalSim/include)
#include_directories(../galsim/include)
#link_directories(../galsim/lib)
find_package(OpenMP REQUIRED)

add_subdirectory(pybind11)
pybind11_add_module(kltools_grism_module_2 grism_module.cpp)
target_compile_definitions(kltools_grism_module_2
        PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(kltools_grism_module_2 PRIVATE "${OpenMP_CXX_FLAGS}")
target_compile_options(kltools_grism_module_2 PRIVATE "${OpenMP_CXX_FLAGS}")
#target_include_directories(gs PRIVATE galsim/include)
#include(GNUInstallDirs)
#install(TARGETS gs
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#add_executable(test
#        test_grism.cpp)
#target_link_libraries(test galsim)
