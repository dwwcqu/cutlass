## Created by dwwcqu

if(CMAKE_VERSION VERSION_LESS 3.11)
  message(FATAL_ERROR "cmake version must greater than 3.11")
endif()

# ROCm Path
if(NOT DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH "/opt/rocm")
else()
  set(ROCM_PATH $ENV{ROCM_PATH})
endif()

# HIP PATH
if(NOT DEFINED ENV{HIP_PATH})
  set(HIP_PATH ${ROCM_PATH}/hip)
else()
  set(HIP_PATH $ENV{HIP_PATH})
endif()

if(NOT EXISTS ${HIP_PATH})
  message(FATAL_ERROR "ROCm HIP not exist")
  return()
endif()

# add the .cmake module into CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

find_package(hip REQUIRED PATHS ${HIP_PATH})
if(NOT hip_FOUND)
	message(FATAL_ERROR "HIP runtime include and library not found")
endif()

add_library(amd::hiprt ALIAS hip)

find_library(
  HIPRTC_LIBRARY hiprtc
  PATHS
  ${ROCM_PATH}
  PATH_SUFFIXES
  lib
  )

if(NOT TARGET hiprtc AND HIPRTC_LIBRARY)

  message(STATUS "HIPRTC: ${HIPRTC_LIBRARY}")

  add_library(amd::hiprtc ALIAS hiprtc)

  set_property(
    TARGET hiprtc
    PROPERTY IMPORTED_LOCATION
    ${HIPRTC_LIBRARY}
    )

elseif(TARGET hiprtc)

  message(STATUS "hiprtc: Already Found")

else()

  message(FATAL_ERROR "hiprtc: Not Found")

endif()

include_directories(SYSTEM ${hip_INCLUDE_DIRS})

function(cutlass_correct_source_file_language_property)
  if(CMAKE_CXX_COMPILER MATCHES "[Cc]lang")
    foreach(File ${ARGN})
      if(File MATCHES ".*\.cpp$")
        set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
      endif()
    endforeach()
  endif()
endfunction()

if (MSVC OR CUTLASS_LIBRARY_KERNELS MATCHES "all")
  set(CUTLASS_UNITY_BUILD_ENABLED_INIT ON)
else()
  set(CUTLASS_UNITY_BUILD_ENABLED_INIT OFF)
endif()

set(CUTLASS_UNITY_BUILD_ENABLED ${CUTLASS_UNITY_BUILD_ENABLED_INIT} CACHE BOOL "Enable combined source compilation")
set(CUTLASS_UNITY_BUILD_BATCH_SIZE 16 CACHE STRING "Batch size for unified source files")

function(cutlass_unify_source_files TARGET_ARGS_VAR)

  set(options)
  set(oneValueArgs BATCH_SOURCES BATCH_SIZE)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED TARGET_ARGS_VAR)
    message(FATAL_ERROR "TARGET_ARGS_VAR parameter is required")
  endif()

  if (__BATCH_SOURCES AND NOT DEFINED __BATCH_SIZE)
    set(__BATCH_SIZE ${CUTLASS_UNITY_BUILD_BATCH_SIZE})
  endif()

  if (CUTLASS_UNITY_BUILD_ENABLED AND DEFINED __BATCH_SIZE AND __BATCH_SIZE GREATER 1)

    set(CUDA_FILE_ARGS)
    set(TARGET_SOURCE_ARGS)

    foreach(ARG ${__UNPARSED_ARGUMENTS})
      if(${ARG} MATCHES ".*\.cu$")
        list(APPEND CUDA_FILE_ARGS ${ARG})
      else()
        list(APPEND TARGET_SOURCE_ARGS ${ARG})
      endif()
    endforeach()

    list(LENGTH CUDA_FILE_ARGS NUM_CUDA_FILE_ARGS)
    while(NUM_CUDA_FILE_ARGS GREATER 0)
      list(SUBLIST CUDA_FILE_ARGS 0 ${__BATCH_SIZE} CUDA_FILE_BATCH)
      string(SHA256 CUDA_FILE_BATCH_HASH "${CUDA_FILE_BATCH}")
      string(SUBSTRING ${CUDA_FILE_BATCH_HASH} 0 12 CUDA_FILE_BATCH_HASH)
      set(BATCH_FILE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.unity.${CUDA_FILE_BATCH_HASH}.cu)
      message(STATUS "Generating ${BATCH_FILE}")
      file(WRITE ${BATCH_FILE} "// Unity File - Auto Generated!\n")
      foreach(CUDA_FILE ${CUDA_FILE_BATCH})
        get_filename_component(CUDA_FILE_ABS_PATH ${CUDA_FILE} ABSOLUTE)
        file(APPEND ${BATCH_FILE} "#include \"${CUDA_FILE_ABS_PATH}\"\n")
      endforeach()
      list(APPEND TARGET_SOURCE_ARGS ${BATCH_FILE})
      if (NUM_CUDA_FILE_ARGS LESS_EQUAL __BATCH_SIZE)
        break()
      endif()
      list(SUBLIST CUDA_FILE_ARGS ${__BATCH_SIZE} -1 CUDA_FILE_ARGS)
      list(LENGTH CUDA_FILE_ARGS NUM_CUDA_FILE_ARGS)
    endwhile()

  else()

    set(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  endif()

  set(${TARGET_ARGS_VAR} ${TARGET_SOURCE_ARGS} PARENT_SCOPE)

endfunction()

function(cutlass_add_library NAME)

  set(options)
  set(oneValueArgs EXPORT_NAME)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  if(CMAKE_CXX_COMPILER MATCHES "clang")
    cutlass_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
    add_library(${NAME} ${TARGET_SOURCE_ARGS})
  endif()

  cutlass_apply_standard_compile_options(${NAME})
  cutlass_apply_cuda_gencode_flags(${NAME})

  target_compile_features(
   ${NAME}
   INTERFACE
   cxx_std_11
   )

  if(__EXPORT_NAME)
    add_library(nvidia::cutlass::${__EXPORT_NAME} ALIAS ${NAME})
    set_target_properties(${NAME} PROPERTIES EXPORT_NAME ${__EXPORT_NAME})
  endif()

endfunction()


function(cutlass_add_executable NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  if(CMAKE_CXX_COMPILER MATCHES "clang")
    cutlass_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
    add_executable(${NAME} ${TARGET_SOURCE_ARGS})
  endif()

  cutlass_apply_standard_compile_options(${NAME})
  cutlass_apply_cuda_gencode_flags(${NAME})

  target_compile_features(
   ${NAME}
   INTERFACE
   cxx_std_11
   )

endfunction()

function(cutlass_target_sources NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})
  cutlass_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
  target_sources(${NAME} ${TARGET_SOURCE_ARGS})

endfunction()