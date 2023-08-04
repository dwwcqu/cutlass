# Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

add_custom_target(
  cutlass_test_unit_gemm_device
  DEPENDS
  cutlass_test_unit_gemm_device_simt
  cutlass_test_unit_gemm_device_tensorop_sm70
  cutlass_test_unit_gemm_device_tensorop_sm75
  cutlass_test_unit_gemm_device_tensorop_f16_sm80
  cutlass_test_unit_gemm_device_tensorop_f32_sm80
  cutlass_test_unit_gemm_device_tensorop_f32_tf32_sm80
  cutlass_test_unit_gemm_device_tensorop_f64
  cutlass_test_unit_gemm_device_tensorop_s32_sm80
  cutlass_test_unit_gemm_device_wmma
  cutlass_test_unit_gemm_device_tensorop_planar_complex
  cutlass_test_unit_gemm_device_sparse_tensorop_sm80
  cutlass_test_unit_gemv_device
)

add_custom_target(
  test_unit_gemm_device
  DEPENDS
  test_unit_gemm_device_simt
  test_unit_gemm_device_tensorop_sm70
  test_unit_gemm_device_tensorop_sm75
  test_unit_gemm_device_tensorop_f16_sm80
  test_unit_gemm_device_tensorop_f32_sm80
  test_unit_gemm_device_tensorop_f32_tf32_sm80
  test_unit_gemm_device_tensorop_f64
  test_unit_gemm_device_tensorop_s32_sm80
  test_unit_gemm_device_wmma
  test_unit_gemm_device_tensorop_planar_complex
  test_unit_gemm_device_sparse_tensorop_sm80
  test_unit_gemv_device
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_simt

  BATCH_SOURCES ON
  BATCH_SIZE 4

  simt_sgemm_nt_sm80.cpp
  simt_sgemm_tn_sm80.cpp
  
  simt_cgemm_nn_sm50.cpp
  simt_cgemm_nt_sm50.cpp
  simt_cgemm_tn_sm50.cpp
  simt_cgemm_tt_sm50.cpp

  simt_qgemm_nn_sm50.cpp
  simt_qgemm_nt_sm50.cpp
  simt_qgemm_tn_sm50.cpp
  simt_qgemm_tt_sm50.cpp

  simt_dgemm_nn_sm50.cpp
  simt_dgemm_nt_sm50.cpp
  simt_dgemm_tn_sm50.cpp
  simt_dgemm_tt_sm50.cpp

  simt_hgemm_nn_sm50.cpp
  simt_hgemm_nt_sm50.cpp
  simt_hgemm_tn_sm50.cpp
  simt_hgemm_tt_sm50.cpp

  simt_igemm_nn_sm50.cpp
  simt_igemm_nt_sm50.cpp
  simt_igemm_tn_sm50.cpp
  simt_igemm_tt_sm50.cpp

  simt_int8_igemm_sm61_sliced_k.cpp
  simt_int8_igemm_sm61.cpp

  simt_sgemm_nn_sm50.cpp
  simt_sgemm_nt_sm50.cpp
  simt_sgemm_tn_sm50.cpp
  simt_sgemm_tt_sm50.cpp

  simt_zgemm_nn_sm50.cpp
  simt_zgemm_nt_sm50.cpp
  simt_zgemm_tn_sm50.cpp
  simt_zgemm_tt_sm50.cpp

  gemm_splitk_simt_sm50.cpp
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm70

  BATCH_SOURCES ON
  BATCH_SIZE 4


  gemm_f16n_f16n_f32t_volta_tensor_op_f32_sm70.cpp
  gemm_f16n_f16t_f32t_volta_tensor_op_f32_sm70.cpp
  gemm_f16t_f16n_f32t_volta_tensor_op_f32_sm70.cpp
  gemm_f16t_f16t_f32t_volta_tensor_op_f32_sm70.cpp

  gemm_f16n_f16n_f16t_volta_tensor_op_f32_sm70.cpp

  gemm_f16n_f16t_f16t_volta_tensor_op_f16_sm70.cpp
  gemm_f16t_f16n_f16t_volta_tensor_op_f16_sm70.cpp

  gemm_splitk_tensor_op_sm70.cpp
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_sm75

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_universal_f16n_f16t_f32n_tensor_op_f32_sm75.cpp
  gemm_universal_f16n_f16t_f32t_tensor_op_f32_sm75.cpp

  gemm_f16t_f16n_f16t_tensor_op_f16_sm75.cpp
  gemm_f16n_f16t_f16t_tensor_op_f16_sm75.cpp
  gemm_f16n_f16t_f16t_tensor_op_f16_slicedk_sm75.cpp
  gemm_f16t_f16n_f16t_tensor_op_f16_slicedk_sm75.cpp

  gemm_f16n_f16n_f16t_tensor_op_f32_sm75.cpp

  gemm_f16n_f16n_f32t_tensor_op_f32_sm75.cpp
  gemm_f16n_f16t_f32t_tensor_op_f32_sm75.cpp
  gemm_f16t_f16n_f32t_tensor_op_f32_sm75.cpp
  gemm_f16t_f16t_f32t_tensor_op_f32_sm75.cpp

  gemm_f16n_f16n_f32n_tensor_op_f32_sm75.cpp
  gemm_f16t_f16t_f32n_tensor_op_f32_sm75.cpp

  gemm_s8n_s8t_s8n_tensor_op_s32_sm75.cpp
  gemm_s8t_s8n_s32t_tensor_op_s32_sm75.cpp
  gemm_s8t_s8n_s32n_tensor_op_s32_sm75.cpp
  gemm_s8t_s8n_s8t_tensor_op_s32_sm75.cpp
  gemm_s8t_s8n_s8n_tensor_op_s32_sm75.cpp

  gemm_s4n_s4t_s4n_tensor_op_s32_sm75.cpp
  gemm_s4t_s4n_s32t_tensor_op_s32_sm75.cpp
  gemm_s4t_s4n_s32n_tensor_op_s32_sm75.cpp
  gemm_s4t_s4n_s4n_tensor_op_s32_sm75.cpp 
  gemm_s4t_s4n_s4t_tensor_op_s32_sm75.cpp

  gemm_b1t_b1n_s32t_tensor_op_s32_sm75.cpp
  gemm_b1t_b1n_s32n_tensor_op_s32_sm75.cpp

  gemm_splitk_serial_tensor_op_sm75.cpp
  gemm_splitk_tensor_op_sm75.cpp

)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_f16_sm80

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_f16t_f16n_f16t_tensor_op_f16_slicedk_sm80.cpp
  gemm_f16n_f16t_f16t_tensor_op_f16_slicedk_sm80.cpp
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_f32_sm80

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_f16n_f16n_f16t_tensor_op_f32_sm80.cpp
  gemm_f16n_f16n_f32n_tensor_op_f32_sm80.cpp
  gemm_f16n_f16n_f32t_tensor_op_f32_sm80.cpp
  gemm_f16n_f16t_f16t_tensor_op_f16_sm80.cpp
  gemm_f16n_f16t_f32t_tensor_op_f32_sm80.cpp
  gemm_f16t_f16n_f16t_tensor_op_f16_sm80.cpp
  gemm_f16t_f16n_f32t_tensor_op_f32_sm80.cpp
  gemm_f16t_f16t_f32n_tensor_op_f32_sm80.cpp
  gemm_f16t_f16t_f32t_tensor_op_f32_sm80.cpp
  gemm_bf16n_bf16n_f32t_tensor_op_f32_sm80.cpp
  gemm_bf16t_bf16t_bf16t_tensor_op_f32_sm80.cpp
  gemm_f16n_f16n_f16n_direct_store_tensor_op_f32_sm80.cpp
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_f32_tf32_sm80

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_tf32t_tf32n_f32t_tensor_op_f32_sm80.cpp
  gemm_tf32n_tf32t_f32t_tensor_op_f32_sm80.cpp
  gemm_tf32n_tf32n_f32t_tensor_op_f32_sm80.cpp
  gemm_tf32t_tf32t_f32t_tensor_op_f32_sm80.cpp                 
  gemm_universal_cf32n_cf32n_cf32n_tensor_op_f32_sm80.cpp
  gemm_cf32n_cf32t_cf32t_tensor_op_tf32_f32_sm80.cpp
  gemm_cf32t_cf32n_cf32t_tensor_op_tf32_f32_sm80.cpp

  gemm_f32n_f32n_f32t_tensor_op_f32_sm80.cpp
  gemm_f32n_f32n_f32t_tensor_op_bf16_f32_sm80.cpp

)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_f64

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_f64n_f64t_f64t_tensor_op_f64_sm80.cpp
  gemm_f64t_f64n_f64t_tensor_op_f64_sm80.cpp

  gemm_universal_cf64n_cf64t_cf64t_tensor_op_f64_sm80.cpp                             
  gemm_universal_cf64n_cf64t_cf64t_tensor_op_f64_gaussian_sm80.cpp
  gemm_cf64n_cf64t_cf64t_tensor_op_f64_sm80.cpp
  gemm_cf64t_cf64n_cf64t_tensor_op_f64_sm80.cpp
  gemm_cf64n_cf64t_cf64t_tensor_op_f64_gaussian_sm80.cpp
  gemm_cf64t_cf64n_cf64t_tensor_op_f64_gaussian_sm80.cpp

)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_s32_sm80

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_s8t_s8n_s32t_tensor_op_s32_sm80.cpp
  gemm_s8t_s8n_s32n_tensor_op_s32_sm80.cpp
  gemm_s8t_s8n_s8n_tensor_op_s32_sm80.cpp
  gemm_s8t_s8n_s8t_tensor_op_s32_sm80.cpp
  gemm_s4t_s4n_s32n_tensor_op_s32_sm80.cpp
  gemm_s4t_s4n_s32t_tensor_op_s32_sm80.cpp
  gemm_s4t_s4n_s4n_tensor_op_s32_sm80.cpp 
  gemm_s4t_s4n_s4t_tensor_op_s32_sm80.cpp
  gemm_b1t_b1n_s32n_tensor_op_s32_sm80.cpp
  gemm_b1t_b1n_s32t_tensor_op_s32_sm80.cpp

  gemm_s8n_s8t_s8n_tensor_op_s32_sm80.cpp
  gemm_s4n_s4t_s4n_tensor_op_s32_sm80.cpp
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_wmma

  BATCH_SOURCES ON
  BATCH_SIZE 4

  # wmma floating point tests
  gemm_f16t_f16n_f16t_wmma_tensor_op_f16_sm70.cpp
  gemm_f16n_f16t_f16t_wmma_tensor_op_f16_sm70.cpp
  gemm_f16t_f16t_f16t_wmma_tensor_op_f16_sm70.cpp
  gemm_f16n_f16n_f16t_wmma_tensor_op_f16_sm70.cpp
  gemm_f16t_f16n_f16n_wmma_tensor_op_f16_sm70.cpp
  gemm_f16n_f16t_f16n_wmma_tensor_op_f16_sm70.cpp
  gemm_f16t_f16t_f16n_wmma_tensor_op_f16_sm70.cpp
  gemm_f16n_f16n_f16n_wmma_tensor_op_f16_sm70.cpp

  gemm_f16t_f16n_f32t_wmma_tensor_op_f32_sm70.cpp
  gemm_f16n_f16t_f32t_wmma_tensor_op_f32_sm70.cpp
  gemm_f16t_f16t_f32t_wmma_tensor_op_f32_sm70.cpp
  gemm_f16n_f16n_f32t_wmma_tensor_op_f32_sm70.cpp
  gemm_f16t_f16n_f32n_wmma_tensor_op_f32_sm70.cpp
  gemm_f16n_f16t_f32n_wmma_tensor_op_f32_sm70.cpp
  gemm_f16t_f16t_f32n_wmma_tensor_op_f32_sm70.cpp
  gemm_f16n_f16n_f32n_wmma_tensor_op_f32_sm70.cpp

  gemm_f16t_f16n_f16t_wmma_tensor_op_f32_sm70.cpp
  gemm_f16n_f16t_f16t_wmma_tensor_op_f32_sm70.cpp
  gemm_f16t_f16t_f16t_wmma_tensor_op_f32_sm70.cpp
  gemm_f16n_f16n_f16t_wmma_tensor_op_f32_sm70.cpp
  gemm_f16t_f16n_f16n_wmma_tensor_op_f32_sm70.cpp
  gemm_f16n_f16t_f16n_wmma_tensor_op_f32_sm70.cpp
  gemm_f16t_f16t_f16n_wmma_tensor_op_f32_sm70.cpp
  gemm_f16n_f16n_f16n_wmma_tensor_op_f32_sm70.cpp

  # wmma int8 tests
  gemm_s8t_s8n_s32t_wmma_tensor_op_s32_sm72.cpp
  gemm_s8t_s8n_s32n_wmma_tensor_op_s32_sm72.cpp

  gemm_s8t_s8n_s8t_wmma_tensor_op_s32_sm72.cpp
  gemm_s8t_s8n_s8n_wmma_tensor_op_s32_sm72.cpp

  # wmma uint8 tests
  gemm_u8t_u8n_s32t_wmma_tensor_op_s32_sm72.cpp

  # wmma sub byptes (s4 and b1) tests
  gemm_s4t_s4n_s32n_wmma_tensor_op_s32_sm75.cpp
  gemm_s4t_s4n_s32t_wmma_tensor_op_s32_sm75.cpp

  gemm_b1t_b1n_s32n_wmma_tensor_op_s32_sm75.cpp
  gemm_b1t_b1n_s32t_wmma_tensor_op_s32_sm75.cpp

  # wmma floating point tests (using singestage pipeline)
  gemm_f16t_f16n_f16t_singlestage_wmma_tensor_op_f16_sm70.cpp
  gemm_f16t_f16n_f16n_singlestage_wmma_tensor_op_f16_sm70.cpp

  gemm_f16t_f16n_f32t_singlestage_wmma_tensor_op_f32_sm70.cpp
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_tensorop_planar_complex

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_planar_complex_f16_f16_f32_tensor_op_sm70.cpp                              
  gemm_planar_complex_f16_f16_f32_tensor_op_sm75.cpp                                         
  gemm_planar_complex_f16_f16_f32_tensor_op_sm80.cpp
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_grouped

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_grouped_sm80.cpp
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_sparse_tensorop_sm80

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_f16n_f16n_f16t_tensor_op_f32_sparse_sm80.cpp
  gemm_f16n_f16n_f32t_tensor_op_f32_sparse_sm80.cpp
  gemm_f16n_f16t_f16t_tensor_op_f16_sparse_sm80.cpp
  gemm_f16n_f16t_f32t_tensor_op_f32_sparse_sm80.cpp
  gemm_f16t_f16n_f16t_tensor_op_f16_sparse_sm80.cpp
  gemm_f16t_f16n_f32t_tensor_op_f32_sparse_sm80.cpp
  gemm_f16t_f16t_f32t_tensor_op_f32_sparse_sm80.cpp
  gemm_f32t_f32n_f32t_tensor_op_f32_sparse_sm80.cpp
  gemm_f32n_f32t_f32t_tensor_op_f32_sparse_sm80.cpp
  gemm_f32t_f32t_f32t_tensor_op_f32_sparse_sm80.cpp
  gemm_f32n_f32n_f32t_tensor_op_f32_sparse_sm80.cpp
  gemm_s8t_s8n_s32t_tensor_op_s32_sparse_sm80.cpp
  gemm_s4t_s4n_s32t_tensor_op_s32_sparse_sm80.cpp
)


cutlass_test_unit_add_executable(
  cutlass_test_unit_gemv_device

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemv.cpp
)

if (NOT CUDA_COMPILER MATCHES "[Cc]lang")

add_dependencies(
  cutlass_test_unit_gemm_device
  cutlass_test_unit_gemm_device_gemm_with_fused_epilogue_tensorop
  )

add_dependencies(
  test_unit_gemm_device
  test_unit_gemm_device_gemm_with_fused_epilogue_tensorop
  )

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_gemm_with_fused_epilogue_tensorop
  
  gemm_with_reduction_f16n_f16n_f16n_tensorop_f32_sm75.cpp
  gemm_with_broadcast_f16n_f16n_f16n_tensorop_f32_sm75.cpp

  gemm_with_reduction_f16t_f16n_f16n_tensorop_f32_sm80.cpp
)

endif()

if (NOT CUDA_COMPILER MATCHES "[Cc]lang")

add_dependencies(
  cutlass_test_unit_gemm_device
  cutlass_test_unit_gemm_device_blas3
  )

add_dependencies(
  test_unit_gemm_device
  test_unit_gemm_device_blas3
  )

cutlass_test_unit_add_executable(
  cutlass_test_unit_gemm_device_blas3

  BATCH_SOURCES ON
  BATCH_SIZE 4

  ## SYRK 
  # Syrk SM80 f64 tests
  syrk_f64n_f64t_tensor_op_f64_sm80.cpp 
  syrk_f64t_f64n_tensor_op_f64_sm80.cpp 

  # Syrk SM80 f32 tests
  syrk_tf32n_f32t_tensor_op_f32_sm80.cpp
  syrk_tf32t_f32t_tensor_op_f32_sm80.cpp
  syrk_f32n_f32t_tensor_op_fast_f32_sm80.cpp
  syrk_f32t_f32t_tensor_op_fast_f32_sm80.cpp

  # Syrk SM80 complex f64 tests
  syrk_cf64n_cf64t_tensor_op_f64_sm80.cpp
  syrk_cf64n_cf64n_tensor_op_f64_sm80.cpp
  syrk_cf64n_cf64t_tensor_op_f64_gaussian_sm80.cpp

  # Syrk SM80 complex f32 tests
  syrk_cf32n_cf32t_tensor_op_f32_sm80.cpp
  syrk_cf32n_cf32n_tensor_op_f32_sm80.cpp
  syrk_cf32n_cf32t_tensor_op_fast_f32_sm80.cpp
  syrk_cf32n_cf32n_tensor_op_fast_f32_sm80.cpp

  ## HERK 
  # Herk SM80 complex f64 tests
  herk_cf64h_cf64n_tensor_op_f64_sm80.cpp

  # Herk SM80 complex f32 tests
  herk_cf32h_cf32n_tensor_op_f32_sm80.cpp
  herk_cf32h_cf32n_tensor_op_fast_f32_sm80.cpp

  ## TRMM
  # Trmm SM80 f64 tests
  trmm_f64n_f64n_f64t_tensor_op_f64_ls_sm80.cpp
  trmm_f64n_f64n_f64t_tensor_op_f64_rs_sm80.cpp
  trmm_f64t_f64t_f64n_tensor_op_f64_ls_sm80.cpp
  trmm_f64t_f64t_f64n_tensor_op_f64_rs_sm80.cpp
  trmm_f64n_f64t_f64t_tensor_op_f64_rs_sm80.cpp

  # Trmm SM80 f32 tests
  trmm_tf32t_tf32n_f32t_tensor_op_f32_ls_sm80.cpp
  trmm_tf32n_tf32t_f32t_tensor_op_f32_ls_sm80.cpp
  trmm_tf32n_tf32t_f32t_tensor_op_f32_rs_sm80.cpp
  trmm_tf32t_tf32n_f32n_tensor_op_f32_ls_sm80.cpp
  trmm_f32t_f32n_f32t_tensor_op_fast_f32_ls_sm80.cpp
  trmm_f32n_f32t_f32t_tensor_op_fast_f32_ls_sm80.cpp
  trmm_f32n_f32t_f32t_tensor_op_fast_f32_rs_sm80.cpp
  trmm_f32t_f32n_f32n_tensor_op_fast_f32_ls_sm80.cpp

  # Trmm SM80 complex f64 tests
  trmm_cf64n_cf64n_cf64t_tensor_op_f64_sm80.cpp
  trmm_cf64n_cf64n_cf64t_tensor_op_f64_gaussian_sm80.cpp

  # Trmm SM80 complex f32 tests
  trmm_cf32n_cf32n_cf32t_tensor_op_f32_sm80.cpp
  trmm_cf32n_cf32n_cf32t_tensor_op_fast_f32_sm80.cpp

  ## SYR2K
  # Syr2k SM80 f64 tests
  syr2k_f64n_f64t_tensor_op_f64_sm80.cpp
  syr2k_f64n_f64n_tensor_op_f64_sm80.cpp
  syr2k_f64t_f64n_tensor_op_f64_sm80.cpp

  # Syr2k SM80 f32 tests
  syr2k_tf32n_f32n_tensor_op_f32_sm80.cpp
  syr2k_tf32t_f32n_tensor_op_f32_sm80.cpp
  syr2k_f32n_f32n_tensor_op_fast_f32_sm80.cpp
  syr2k_f32t_f32n_tensor_op_fast_f32_sm80.cpp

  # Syr2k SM80 complex f64 tests
  syr2k_cf64n_cf64t_tensor_op_f64_sm80.cpp
  syr2k_cf64n_cf64n_tensor_op_f64_sm80.cpp

  # Syr2k SM80 complex f32 tests
  syr2k_cf32n_cf32n_tensor_op_f32_sm80.cpp
  syr2k_cf32n_cf32t_tensor_op_f32_sm80.cpp
  syr2k_cf32n_cf32n_tensor_op_fast_f32_sm80.cpp
  syr2k_cf32n_cf32t_tensor_op_fast_f32_sm80.cpp

  ## HER2K
  # Her2k SM80 complex f64 tests
  her2k_cf64n_cf64n_tensor_op_f64_sm80.cpp

  # Her2k SM80 complex f32 tests
  her2k_cf32h_cf32n_tensor_op_f32_sm80.cpp
  her2k_cf32h_cf32n_tensor_op_fast_f32_sm80.cpp

  ## SYMM
  # Symm SM80 f64 tests
  symm_f64n_f64n_tensor_op_f64_ls_sm80.cpp
  symm_f64n_f64n_tensor_op_f64_rs_sm80.cpp
  symm_f64n_f64t_tensor_op_f64_ls_sm80.cpp
  symm_f64n_f64t_tensor_op_f64_rs_sm80.cpp
  symm_f64t_f64n_tensor_op_f64_ls_sm80.cpp
  symm_f64t_f64n_tensor_op_f64_rs_sm80.cpp
  symm_f64t_f64t_tensor_op_f64_ls_sm80.cpp
  symm_f64t_f64t_tensor_op_f64_rs_sm80.cpp

  # Symm SM80 f32 tests
  symm_tf32n_f32n_tensor_op_f32_ls_sm80.cpp
  symm_tf32n_f32n_tensor_op_f32_rs_sm80.cpp
  symm_tf32t_f32t_tensor_op_f32_ls_sm80.cpp
  symm_f32n_f32n_tensor_op_fast_f32_ls_sm80.cpp
  symm_f32n_f32n_tensor_op_fast_f32_rs_sm80.cpp
  symm_f32t_f32t_tensor_op_fast_f32_ls_sm80.cpp

  # Symm SM80 complex f64 tests
  symm_cf64n_cf64n_cf64n_tensor_op_ls_f64_sm80.cpp
  symm_cf64n_cf64n_cf64n_tensor_op_rs_f64_sm80.cpp
  symm_cf64n_cf64n_cf64n_tensor_op_ls_f64_gaussian_sm80.cpp

  # Symm SM80 complex f32 tests
  symm_cf32n_cf32n_tensor_op_f32_ls_sm80.cpp
  symm_cf32n_cf32n_tensor_op_f32_rs_sm80.cpp
  symm_cf32n_cf32n_tensor_op_fast_f32_ls_sm80.cpp
  symm_cf32n_cf32n_tensor_op_fast_f32_rs_sm80.cpp

  # Hemm SM80 complex f64 tests
  hemm_cf64h_cf64n_cf64n_tensor_op_ls_f64_sm80.cpp
  hemm_cf64h_cf64n_cf64n_tensor_op_rs_f64_sm80.cpp
  hemm_cf64h_cf64n_cf64n_tensor_op_ls_f64_gaussian_sm80.cpp
  
  # Hemm SM80 complex f32 tests
  hemm_cf32h_cf32n_tensor_op_f32_ls_sm80.cpp
  hemm_cf32h_cf32n_tensor_op_f32_rs_sm80.cpp
  hemm_cf32h_cf32n_tensor_op_fast_f32_ls_sm80.cpp
  hemm_cf32h_cf32n_tensor_op_fast_f32_rs_sm80.cpp
  )

endif()
