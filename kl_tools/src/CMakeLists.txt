cmake_minimum_required(VERSION 3.16)
project(grism_module)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
# add extra include directories
#include_directories(/Users/jiachuanxu/Workspace/GalSim/include)
include_directories(../galsim/include)
link_directories(../galsim/lib)

add_subdirectory(pybind11)
pybind11_add_module(kltools_grism_module grism_module.cpp)
target_compile_definitions(kltools_grism_module
        PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
#target_include_directories(gs PRIVATE galsim/include)
#include(GNUInstallDirs)
#install(TARGETS gs
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#add_executable(test
#        test_grism.cpp)
#target_link_libraries(test galsim)