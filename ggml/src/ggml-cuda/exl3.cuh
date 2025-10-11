#include "common.cuh"
#include "ggml.h"

void exl3_mmq
(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * A, // input, should be FP16
    const ggml_tensor * B, // weight
    ggml_tensor * C, // output
    const half * suh_ptr,
    half * A_had_ptr,
    const half * svh_ptr,
    uint32_t mcg_mult,
    uint32_t mul1_mult
);

int exl3_mmvq
(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * A, // input
    const ggml_tensor * B, // weight
    ggml_tensor * C, // output
    const half * suh_ptr,
    half * A_had_ptr,
    const half * svh_ptr,
    uint32_t mcg_mult,
    uint32_t mul1_mult
);
