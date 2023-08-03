## Created by dwwcqu

if(DEFINED HIPDNN_ENABLED)
    set(CUTLASS_ENABLE_HIPDNN ${CUDNN_ENABLED} CACHE BOOL "Enable CUTLASS to build with hipDNN library.")
endif()

if(DEFINED CUTLASS_ENABLE_HIPDNN AND NOT CUTLASS_ENABLE_HIPDNN)
  return()
endif()

message(STATUS "Configuring hipDNN ...")

if(NOT DEFINED ENV{MIOPEN_PATH})
  set(MIOPEN_PATH ${ROCM_PATH}/miopen)
else()
  set(MIOPEN_PATH $ENV{MIOPEN_PATH})
endif()

find_package(miopen REQUIRED PATHS ${MIOPEN_PATH})

if(miopen_FOUND)

    message(STATUS "hipDNN: ${miopen_LIBRARIES}")
    message(STATUS "hipDNN: ${miopen_INCLUDE_DIRS}")
    
    set(HIPDNN_FOUND ON CACHE INTERNAL "hipDNN Library Found")

else()

    message(STATUS "MIOPEN not found.")
    set(HIPDNN_FOUND OFF CACHE INTERNAL "MIOPEN Library Found")

endif()

if(CUTLASS_ENABLE_HIPDNN AND NOT HIPDNN_FOUND)
  message(FATAL_ERROR "CUTLASS_ENABLE_HIPDNN enabled but MIOPEN library could not be found.")
endif()

message(STATUS "Configuring hipDNN ... done.")