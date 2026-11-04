cmake_minimum_required(VERSION 3.15)
project(integrate_par LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(OpenMP REQUIRED COMPONENTS C)
find_package(nanobind CONFIG REQUIRED)

# ROUTINE_DIR points at any work directory holding the converted OpenMP integrate.c (a
# passing agent run, or the reference solution). MPI_ROUTINE_DIR likewise for the MPI
# routine; either may be omitted to skip that module. Passed on the cmake command line.
if(DEFINED ROUTINE_DIR)
  nanobind_add_module(integrate_par integrate_binding.cpp ${ROUTINE_DIR}/integrate.c)
  target_include_directories(integrate_par PRIVATE ${ROUTINE_DIR})
  target_link_libraries(integrate_par PRIVATE OpenMP::OpenMP_C)
endif()

if(DEFINED MPI_ROUTINE_DIR)
  find_package(MPI REQUIRED COMPONENTS C)
  nanobind_add_module(integrate_par_mpi integrate_mpi_binding.cpp
                      ${MPI_ROUTINE_DIR}/integrate.c)
  target_include_directories(integrate_par_mpi PRIVATE ${MPI_ROUTINE_DIR})
  target_link_libraries(integrate_par_mpi PRIVATE MPI::MPI_C)
endif()

if(DEFINED MATMUL_ROUTINE_DIR)
  nanobind_add_module(matmul_par matmul_binding.cpp ${MATMUL_ROUTINE_DIR}/matmul.c)
  target_include_directories(matmul_par PRIVATE ${MATMUL_ROUTINE_DIR})
  target_link_libraries(matmul_par PRIVATE OpenMP::OpenMP_C)
endif()

if(DEFINED JACOBI_ROUTINE_DIR)
  nanobind_add_module(jacobi_par jacobi_binding.cpp ${JACOBI_ROUTINE_DIR}/jacobi.c)
  target_include_directories(jacobi_par PRIVATE ${JACOBI_ROUTINE_DIR})
  target_link_libraries(jacobi_par PRIVATE OpenMP::OpenMP_C)
endif()

# CUDA: the routine (.cu) is compiled by nvcc via CMake's CUDA language; the same scalar
# binding is reused -- the host signature is identical, only the implementation differs.
if(DEFINED CUDA_ROUTINE_DIR)
  enable_language(CUDA)
  nanobind_add_module(integrate_par_cuda integrate_cuda_binding.cpp
                      ${CUDA_ROUTINE_DIR}/integrate.cu)
  target_include_directories(integrate_par_cuda PRIVATE ${CUDA_ROUTINE_DIR})
endif()

# The routines are exported C symbols, and a module's internal reference to its own
# exported symbol goes through the PLT -- so ANY earlier-loaded object exporting the
# same name captures the call (observed: a uv-built CPython exports a global
# 'integrate', and the module silently called CPython's function instead of its own
# routine, returning nan). -Bsymbolic makes each module bind internal references to its
# own definitions at link time.
foreach(_mod integrate_par integrate_par_mpi matmul_par jacobi_par integrate_par_cuda)
  if(TARGET ${_mod})
    target_link_options(${_mod} PRIVATE "-Wl,-Bsymbolic")
  endif()
endforeach()

if(NOT DEFINED ROUTINE_DIR AND NOT DEFINED MPI_ROUTINE_DIR
   AND NOT DEFINED MATMUL_ROUTINE_DIR AND NOT DEFINED JACOBI_ROUTINE_DIR
   AND NOT DEFINED CUDA_ROUTINE_DIR)
  message(FATAL_ERROR "pass -DROUTINE_DIR / -DMPI_ROUTINE_DIR / -DMATMUL_ROUTINE_DIR / -DJACOBI_ROUTINE_DIR / -DCUDA_ROUTINE_DIR")
endif()
