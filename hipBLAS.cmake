## Created by dwwcqu

message(STATUS "Configuring hipblas ...")

# HIPBLAS Path
if(NOT DEFINED ENV{HIPBLAS_PATH})
  set(HIPBLAS_PATH ${ROCM_PATH}/hipblas)
else()
  set(HIPBLAS_PATH $ENV{HIPBLAS_PATH})
endif()

if((DEFINED CUTLASS_ENABLE_HIPBLAS AND NOT CUTLASS_ENABLE_HIPBLAS) OR
   (DEFINED HIPBLAS_ENABLED AND NOT HIPBLAS_ENABLED))
   set(HIPBLAS_FOUND OFF)
   message(STATUS "hipBLAS Disabled.")
else()
  find_package(hipblas REQUIRED PATH ${HIPBLAS_PATH})
	if(hipblas_FOUND)
		message(STATUS "hipBLAS: ${hipBLAS_LIBRARIES}")
		message(STATUS "hipBLAS: ${hipBLAS_INCLUDE_DIRS}")
		set(HIPBLAS_FOUND ON)
	endif()
endif()

if(CUTLASS_ENABLE_HIPBLAS AND NOT HIPBLAS_FOUND)
  message(FATAL_ERROR "CUTLASS_ENABLE_HIPBLAS enabled but hipBLAS library could not be found.")
endif()

message(STATUS "Configuring hipBLAS ... done.")