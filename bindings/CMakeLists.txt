cmake_minimum_required(VERSION 3.15)
project(integrate_par LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(OpenMP REQUIRED COMPONENTS C)
find_package(nanobind CONFIG REQUIRED)

# KERNEL_DIR points at any work directory holding the converted OpenMP integrate.c (a
# passing agent run, or the reference solution). MPI_KERNEL_DIR likewise for the MPI
# kernel; either may be omitted to skip that module. Passed on the cmake command line.
if(DEFINED KERNEL_DIR)
  nanobind_add_module(integrate_par integrate_binding.cpp ${KERNEL_DIR}/integrate.c)
  target_include_directories(integrate_par PRIVATE ${KERNEL_DIR})
  target_link_libraries(integrate_par PRIVATE OpenMP::OpenMP_C)
endif()

if(DEFINED MPI_KERNEL_DIR)
  find_package(MPI REQUIRED COMPONENTS C)
  nanobind_add_module(integrate_par_mpi integrate_mpi_binding.cpp
                      ${MPI_KERNEL_DIR}/integrate.c)
  target_include_directories(integrate_par_mpi PRIVATE ${MPI_KERNEL_DIR})
  target_link_libraries(integrate_par_mpi PRIVATE MPI::MPI_C)
endif()

if(DEFINED MATMUL_KERNEL_DIR)
  nanobind_add_module(matmul_par matmul_binding.cpp ${MATMUL_KERNEL_DIR}/matmul.c)
  target_include_directories(matmul_par PRIVATE ${MATMUL_KERNEL_DIR})
  target_link_libraries(matmul_par PRIVATE OpenMP::OpenMP_C)
endif()

if(DEFINED JACOBI_KERNEL_DIR)
  nanobind_add_module(jacobi_par jacobi_binding.cpp ${JACOBI_KERNEL_DIR}/jacobi.c)
  target_include_directories(jacobi_par PRIVATE ${JACOBI_KERNEL_DIR})
  target_link_libraries(jacobi_par PRIVATE OpenMP::OpenMP_C)
endif()

# CUDA: the kernel (.cu) is compiled by nvcc via CMake's CUDA language; the same scalar
# binding is reused -- the host signature is identical, only the implementation differs.
if(DEFINED CUDA_KERNEL_DIR)
  enable_language(CUDA)
  nanobind_add_module(integrate_par_cuda integrate_cuda_binding.cpp
                      ${CUDA_KERNEL_DIR}/integrate.cu)
  target_include_directories(integrate_par_cuda PRIVATE ${CUDA_KERNEL_DIR})
endif()

# The kernels are exported C symbols, and a module's internal reference to its own
# exported symbol goes through the PLT -- so ANY earlier-loaded object exporting the
# same name captures the call (observed: a uv-built CPython exports a global
# 'integrate', and the module silently called CPython's function instead of its own
# kernel, returning nan). -Bsymbolic makes each module bind internal references to its
# own definitions at link time.
foreach(_mod integrate_par integrate_par_mpi matmul_par jacobi_par integrate_par_cuda)
  if(TARGET ${_mod})
    target_link_options(${_mod} PRIVATE "-Wl,-Bsymbolic")
  endif()
endforeach()

if(NOT DEFINED KERNEL_DIR AND NOT DEFINED MPI_KERNEL_DIR
   AND NOT DEFINED MATMUL_KERNEL_DIR AND NOT DEFINED JACOBI_KERNEL_DIR
   AND NOT DEFINED CUDA_KERNEL_DIR)
  message(FATAL_ERROR "pass -DKERNEL_DIR / -DMPI_KERNEL_DIR / -DMATMUL_KERNEL_DIR / -DJACOBI_KERNEL_DIR / -DCUDA_KERNEL_DIR")
endif()
