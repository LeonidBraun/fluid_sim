cmake_minimum_required(VERSION 3.24)

project(RectangularFluidSim VERSION 0.1.0 LANGUAGES CXX)

include(CheckLanguage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(HDF5 REQUIRED COMPONENTS C)

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_executable(fluid_sim
    src/main.cpp
    src/io/IO.cpp
    src/io/grid.cpp
    src/io/hdf5_common.cpp
    src/io/hdf5_writer.cpp
    src/simulation/simulation.cu
  )

  target_include_directories(fluid_sim
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${PROJECT_SOURCE_DIR}/libs/json/include
      ${HDF5_INCLUDE_DIRS}
  )

  target_link_libraries(fluid_sim
    PRIVATE
      CUDA::cudart
      ${HDF5_LIBRARIES}
  )

  target_compile_features(fluid_sim PRIVATE cxx_std_17)

  set_target_properties(fluid_sim PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
  )

  target_compile_options(fluid_sim
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
else()
  message(WARNING
    "CUDA compiler not found. Install the CUDA toolkit (nvcc) inside WSL to build the fluid_sim target."
  )
endif()
