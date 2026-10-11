cmake_minimum_required(VERSION 3.15)
project(pi_mpi_ext LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(MPI REQUIRED COMPONENTS CXX)

# nanobind_DIR is passed on the command line (from python -c "import nanobind; ...").
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(pi_mpi_ext pi_mpi.cpp)
target_link_libraries(pi_mpi_ext PRIVATE MPI::MPI_CXX)
