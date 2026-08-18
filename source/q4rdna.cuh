#pragma once

#include "common.cuh"

bool ggml_cuda_q4rdna_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);

bool ggml_cuda_q4rdna_mul_mat_add(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * add,
        ggml_tensor * dst);

bool ggml_cuda_q4rdna_mul_mat_glu(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * up,
        const ggml_tensor * gate,
        const ggml_tensor * activation,
        enum ggml_glu_op glu_op,
        ggml_tensor * dst);
