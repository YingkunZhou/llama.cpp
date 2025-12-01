#include <set>
#include <cuda_fp16.h>
#include <mutex>
#include "exl3.cuh"
#include "exl3-util.cuh"
#include "exl3-util.h"
#include "exl3-hadamard_inner.cuh"
#include <cooperative_groups.h>
#include "exl3-dq.cuh"
namespace cg = cooperative_groups;

#define MAX_DEVICES 32
#define CC_OLD        1
#define CC_AMPERE     2
#define CC_ADA        3
#define CC_HOPPER     4
#define CC_BLACKWELL  5
#define EXL3_GEMM_BASE_THREADS 256
#define SMEM_MAX (90 * 1024)
#define MAX_TILES_C (1024 * 1024)

#define EXL3_GEMM_TILESIZE_K  0, 16, 32, 32, 16
#define EXL3_GEMM_TILESIZE_N  0, 128, 128, 256, 512
#define EXL3_GEMM_BLOCKDIM  0, 256, 512, 512, 256

#define EXL3_GEMM_SHAPE_1     16,     16,    128,     6,     5
#define EXL3_GEMM_SHAPE_2     16,     32,    128,     4,     3
#define EXL3_GEMM_SHAPE_3     16,     32,    256,     4,     3
#define EXL3_GEMM_SHAPE_4     16,     16,    512,     4,     3

#define EXL3_GEMM_NUM_SHAPES 4

#define EXL3_GEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t* __restrict__ B, \
    void* __restrict__ C, \
    int size_m, \
    int size_k, \
    int size_n, \
    int* __restrict__ locks, \
    const half* __restrict__ suh, \
    half* __restrict__ A_had, \
    const half* __restrict__ svh, \
    uint32_t mult

#define EXL3_GEMM_T_ARGS \
    int bits, \
    bool c_fp32, \
    int cb, \
    int TILESIZE_M, \
    int TILESIZE_K, \
    int TILESIZE_N, \
    int SH_STAGES, \
    int FRAG_STAGES

typedef void (*fp_exl3_gemm_kernel) (EXL3_GEMM_ARGS);

std::set<void*> kernel_attr_set;

static int select_gemm_shape(int cc, int size_k, int size_n, int bits, bool multi)
{
    bool mod_256 = (size_n % 256 == 0);
    bool mod_512 = (size_n % 512 == 0);

    switch(cc)
    {
        case CC_OLD:
        case CC_AMPERE:
            if (mod_256 && bits <= 4)
            {
                if (size_n <= 2048 || size_k <= 2048) return 2;
                return 3;
            }
            if (mod_256 && size_n < 4096) return size_k > 8192 ? 3 : 2;
            if (mod_512 && (size_n * size_k) > (4096 * 4096) && bits <= 6) return 4;
            if (mod_256) return 3;
            return 2;

        case CC_ADA:
            if (mod_256 && bits <= 3)
            {
                if (size_k <= 2048 && !multi) return 2;
                if (size_n < 4096 && size_k <= 12288) return 2;
                return 3;
            }
            if (size_n <= 16384) return 2;
            if (mod_512 && size_n >= 32768) return 4;
            if (mod_256) return 3;
            return 2;

        case CC_HOPPER:
        case CC_BLACKWELL:
            if ((bits == 4 || bits == 2) && !multi)
            {
                if (size_k <= 2048) return 1;
            }
            if (bits >= 7)
            {
                if (mod_256 && size_n <= 8192) return size_k > 32768 ? 3 : 2;
                if (mod_512 && size_n > 32768) return 4;
                return 2;
            }
            if (mod_256 && size_n <= 4096) return size_k > 8192 && bits >= 3 ? 3 : 2;
            if (mod_512 && size_n > 16384) return 4;
            if (mod_256) return 3;
            return 2;
    }
    return 0;
}

template<EXL3_GEMM_T_ARGS>
inline __device__
static void exl3_gemm_kernel_inner
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    int size_m,
    int size_k,
    int size_n,
    int* __restrict__ locks,
    uint32_t mult
)
{
    const int TILEBLOCKS_M = TILESIZE_M / 16;
    const int TILEBLOCKS_K = TILESIZE_K / 16;
    const int TILEBLOCKS_N = TILESIZE_N / 16;
    const int FRAGS_M = TILEBLOCKS_M;
    const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);

    const int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                         // in halfs
    const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;   // in uint16s
    const int sh_c_size = 4 * EXL3_GEMM_BASE_THREADS;                            // in floats

    // Sanity checks
    static_assert(EXL3_GEMM_BASE_THREADS == 256);
    static_assert(TILESIZE_M % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_K % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
    static_assert
    (
        SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
        "Invalid kernel params (insufficient shared memory for shape)"
    );

    // Shared memory
    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);

    // Thread index
    int t = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    int sub_k = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    int warp_id = t / 32;
    int lane_id = t % 32;

    // Dimensions
    Dim3 size = { size_m, size_k, size_n };
    Dim3 tiles = { CEIL_DIVIDE(size_m, TILESIZE_M), size_k / TILESIZE_K, size_n / TILESIZE_N };
    Dim3 blocks = { 1, tiles.k * TILEBLOCKS_K, tiles.n * TILEBLOCKS_N };

    // Start and end index of current slice, must span at least one tile
    int num_slices = gridDim.x;
    int slice_beg = tiles.numel_b() * blockIdx.x / num_slices;
    int slice_end = tiles.numel_b() * (blockIdx.x + 1) / num_slices;
    int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    auto index_m = [&] (int slice_i) { return 0; }; //blockIdx.y; };
    auto index_k = [&] (int slice_i) { return (slice_i % tiles.k); };
    auto index_n = [&] (int slice_i) { return (slice_i / tiles.k); };

    // Batch dimension
    int slice_m = index_m(slice_beg);
    int max_m = MIN1(size_m - slice_m * TILESIZE_M, TILESIZE_M);

    // Pipe 0, global A, B tile and shared A, B tile
    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_len;

    int gl_a_stride_m = TILESIZE_M * size_k;
    const int gl_a_stride_k = TILESIZE_K;
    const int sh0_a_stride_m = TILESIZE_M * TILESIZE_K;
    const half* gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
    half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    const int load_a_iters = CEIL_DIVIDE(sh0_a_stride_m / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i)
    {
        int k = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_a_stride_k / 8);
        int m = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        pred_a_gl[i] = m < max_m;
    }

    int gl_b_stride_k = blocks.n * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    const int load_b_iters = CEIL_DIVIDE(sh0_b_stride_k / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i)
    {
        int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * blocks.n * 256 / 16 * bits / 8 * k + n;
        pred_b_gl[i] = i * EXL3_GEMM_BASE_THREADS + t < sh0_b_stride_k / 8;
    }

    auto advance0 = [&] ()
    {
        slice0_k++;
        slice0_iters--;

        int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice0_k >= tiles.k)
        {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
        }
        else
        {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    // Pipe 1, shared A, B tile and registers
    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;

    half* sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    auto advance1 = [&] ()
    {
        slice1_k++;
        slice1_iters--;

        int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice1_k >= tiles.k)
        {
            slice1_k = 0;
            slice1_n++;
        }
    };

    // Pipe 2
    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    int gl_c_stride_n = TILESIZE_N;
    int gl_c_stride_m = TILESIZE_M * size_n;

    half* gl_c_ptr_16 = ((half*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;
    float* gl_c_ptr_32 = ((float*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;

    register FragA frag_a[FRAG_STAGES][FRAGS_M];
    register FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    register FragC frag_c[FRAGS_M][FRAGS_N_PER_WARP];

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;

        if (slice2_k >= tiles.k)
        {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            if constexpr (c_fp32)
                gl_c_ptr_32 += gl_c_stride_n;
            else
                gl_c_ptr_16 += gl_c_stride_n;
        }
    };

    // Schedule load of the next A, B tiles to shared memory and advance the pipeline
    auto async_load_gl = [&] ()
    {
        if (sub_k)
        {
            cp_async_fence();
            return;
        }

        if (slice0_iters)
        {
            // Copy tile from row-major A matrix
            {
                const int4* gl = (const int4*) gl_a_ptr;
                int4* sh = (int4*) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                {
                    // TODO: Rearrange into ldmatrix friendly layout while loading?
                    // cp_async_pred(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_a_gl[i], pred_a_gl[i]);
                    if (pred_a_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_a_gl[i]);
                }
            }

            // Copy tile of 256-element blocks from quantized B matrix
            {
                const int4* gl = (const int4*) gl_b_ptr;
                int4* sh = (int4*) sh0_b_ptr;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i)
                {
                    // cp_async_pred(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i], pred_b_gl[i]);
                    if (pred_b_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i]);
                }
            }
            advance0();
        }

        // Sync and advance
        cp_async_fence();
    };

    // Load fragments
    // Ref. for fragment layout:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    auto load_frags = [&] (int buf)
    {
        if (!slice1_iters) return;

        // A fragments
        {
            // TODO: Resolve bank conflicts
            int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
            int c = lane_id / 16;
            int4* sha = (int4*) sh1_a_ptr + r * TILESIZE_K / 8 + c;
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
                ldsm4(frag_a[buf][m], sha + (m * 16) * TILESIZE_K / 8 + sub_k * 16 / 8);
        }

        // B fragments
        int r0 = lane_id / 2;
        int c0 = (lane_id % 2) * 8;

        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2)
        {
            int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
            const uint32_t* shb = (const uint32_t*) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);

            dq_dispatch<bits, cb>(shb, r0 * 16 + c0, frag_b[buf][n2], frag_b[buf][n2 + 1], mult);
        }

        __syncthreads();
        advance1();
    };

    // Clear C fragments
    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int m = 0; m < FRAGS_M; ++m)
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                frag_c[m][n] = {};
    };

    // Threadblock reduction
    auto threadblock_reduce = [&] ()
    {
        auto store = [&] (int i, int m, int n)
        {
            // TODO: Shuffle to avoid bank conflicts here? Doesn't seem to be a bottleneck
            // TODO: Always accumulates entire C fragment, could be limited when size_m < 16
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                for (int i = 0; i < 4; ++i)
                    *sh_red++ = frag_c[m][n][i];
            }
            __syncthreads();
        };

        auto add = [&] (int i, int m, int n)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                for (int i = 0; i < 4; ++i)
                    frag_c[m][n][i] += *sh_red++;
            }
            __syncthreads();
        };

        for (int m = 0; m < FRAGS_M; ++m)
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            if constexpr (TILEBLOCKS_K == 2)
            {
                store(1, m, n);
                add(0, m, n);
            }
            if constexpr (TILEBLOCKS_K == 3)
            {
                store(1, m, n);
                add(0, m, n);
                store(2, m, n);
                add(0, m, n);
            }
            if constexpr (TILEBLOCKS_K == 4)
            {
                store(3, m, n);
                add(2, m, n);
                store(1, m, n);
                add(0, m, n);
                store(2, m, n);
                add(0, m, n);
            }
        }
    };

    // Output reduction
    auto reduce = [&] ()
    {
        // First reduce all partial sums along k for the current slice
        threadblock_reduce();

        // Process (partial) slices within column in reverse order so the threadblock doing the bottom slice is
        // free to proceed to the next column right away
        int lock_i = tiles.k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice_m * blocks.n + slice2_n];

        barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles.k;

        int n0 = warp_id * FRAGS_N_PER_WARP;

        // Second and subsequent threadblocks in column read back the intermediate sum from global memory
        // TODO: Use an intermediate layout to make these writes coalesce
        if (!sub_k && !first)
        {
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                for (int m = 0; m < FRAGS_M; ++m)
                {
                    int r0 = lane_id / 4 + 16 * m;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                            frag_c[m][n][0] += *c_ptr++;
                            frag_c[m][n][1] += *c_ptr++;
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                            float2 interm = __half22float2(*c_ptr);
                            frag_c[m][n][0] += interm.x;
                            frag_c[m][n][1] += interm.y;
                        }
                    }
                    if (r1 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                            frag_c[m][n][2] += *c_ptr++;
                            frag_c[m][n][3] += *c_ptr++;
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                            float2 interm = __half22float2(*c_ptr);
                            frag_c[m][n][2] += interm.x;
                            frag_c[m][n][3] += interm.y;
                        }
                    }
                }
            }
        }

        // All but last threadblock in column threadblocks write the intermediate result to global memory
        if (!sub_k && !last)
        {
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                for (int m = 0; m < FRAGS_M; ++m)
                {
                    int r0 = lane_id / 4 + 16 * m;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][0];
                            *c_ptr++ = frag_c[m][n][1];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                            *c_ptr = sum;
                        }
                    }
                    if (r1 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][2];
                            *c_ptr++ = frag_c[m][n][3];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
                            *c_ptr = sum;
                        }
                    }
                }
            }
        }

        // Last block writes in row-major format
        if (!sub_k && last)
        {
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                for (int m = 0; m < FRAGS_M; ++m)
                {
                    int r0 = lane_id / 4 + 16 * m;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][0];
                            *c_ptr++ = frag_c[m][n][1];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                            *c_ptr = sum;
                        }
                    }
                    if (r1 < max_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][2];
                            *c_ptr++ = frag_c[m][n][3];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
                            *c_ptr = sum;
                        }
                    }
                }
            }
        }

        barrier_release(lock, lock_d, last);

        clear_frag_c();
    };

    // Wait until there are at most SH_STAGES - 2 async copies pending, i.e. at least one stage has finished loading
    auto wait_stage = [&] ()
    {
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
    };

    // Perform tensor core matmul on current tile
    auto matmul = [&] (int buf)
    {
        for (int m = 0; m < FRAGS_M; ++m)
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                ptx_mma_m16n8k16(frag_a[buf][m], frag_b[buf][n], frag_c[m][n]);
    };

    // Start global to shared pipeline
    for (int i = 0; i < SH_STAGES - 1; ++i)
        async_load_gl();
    wait_stage();

    // Start shared to register pipeline.
    clear_frag_c();
    if constexpr (FRAG_STAGES > 1)
        load_frags(0);

    // Main loop. Fragments are double buffered to allow more interleaving. This is especially important to hide the
    // dequantization overhead, but we need two different iterations of the main loop to avoid confusing the compiler
    // and making it (sometimes) place the fragment arrays in local memory

    #define FSTAGE(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        load_frags(_load); \
        matmul(_mul); \
        if (slice2_k == tiles.k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \

    if constexpr (FRAG_STAGES == 1)
    {
        while (true)
        {
            FSTAGE(0, 0);
        }
    }

    if constexpr (FRAG_STAGES == 2)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(0, 1);
        }
    }

    if constexpr (FRAG_STAGES == 3)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(0, 2);
        }
    }

    if constexpr (FRAG_STAGES == 4)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(0, 3);
        }
    }

    if constexpr (FRAG_STAGES == 5)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(4, 3);
            FSTAGE(0, 4);
        }
    }
}


template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
static void exl3_gemm_kernel(EXL3_GEMM_ARGS)
{
    auto grid = cg::this_grid();

    if (suh)
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                nullptr,
                0.088388347648f  // 1/sqrt(128)
            );

        grid.sync();
        A = A_had;
    }

    int size_m_ = size_m;
    const half* A_ = A;
    void* C_ = C;

    while (size_m_ > 0)
    {
        exl3_gemm_kernel_inner
        <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES>
        (A_, B, C_, size_m_, size_k, size_n, locks, mult);

        A_ += 16 * size_k;
        if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n);
        else                  C_ = (void*) (((half*) C_) + 16 * size_n);
        size_m_ -= 16;
        grid.sync();
    }

    if (svh)
    {
        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            if constexpr (c_fp32)
                had_ff_r_128_inner
                (
                    ((const float*) C) + this_warp * 128,
                    ((float*) C) + this_warp * 128,
                    nullptr,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
            else
                had_hf_r_128_inner
                (
                    ((const half*) C) + this_warp * 128,
                    ((half*) C) + this_warp * 128,
                    nullptr,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
    }
}


int exl3_gemm_tilesize_k[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim[] = {EXL3_GEMM_BLOCKDIM};


fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b2[] = {
    // cb = 0
    nullptr,
    exl3_gemm_kernel<2, true, 0, EXL3_GEMM_SHAPE_1>,
    exl3_gemm_kernel<2, true, 0, EXL3_GEMM_SHAPE_2>,
    exl3_gemm_kernel<2, true, 0, EXL3_GEMM_SHAPE_3>,
    exl3_gemm_kernel<2, true, 0, EXL3_GEMM_SHAPE_4>,

    // cb = 1
    nullptr,
    exl3_gemm_kernel<2, true, 1, EXL3_GEMM_SHAPE_1>,
    exl3_gemm_kernel<2, true, 1, EXL3_GEMM_SHAPE_2>,
    exl3_gemm_kernel<2, true, 1, EXL3_GEMM_SHAPE_3>,
    exl3_gemm_kernel<2, true, 1, EXL3_GEMM_SHAPE_4>,

    // cb = 2
    nullptr,
    exl3_gemm_kernel<2, true, 2, EXL3_GEMM_SHAPE_1>,
    exl3_gemm_kernel<2, true, 2, EXL3_GEMM_SHAPE_2>,
    exl3_gemm_kernel<2, true, 2, EXL3_GEMM_SHAPE_3>,
    exl3_gemm_kernel<2, true, 2, EXL3_GEMM_SHAPE_4>
};

fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b3[] = {
    // cb = 0
    nullptr,
    exl3_gemm_kernel<3, true, 0, EXL3_GEMM_SHAPE_1>,
    exl3_gemm_kernel<3, true, 0, EXL3_GEMM_SHAPE_2>,
    exl3_gemm_kernel<3, true, 0, EXL3_GEMM_SHAPE_3>,
    exl3_gemm_kernel<3, true, 0, EXL3_GEMM_SHAPE_4>,

    // cb = 1
    nullptr,
    exl3_gemm_kernel<3, true, 1, EXL3_GEMM_SHAPE_1>,
    exl3_gemm_kernel<3, true, 1, EXL3_GEMM_SHAPE_2>,
    exl3_gemm_kernel<3, true, 1, EXL3_GEMM_SHAPE_3>,
    exl3_gemm_kernel<3, true, 1, EXL3_GEMM_SHAPE_4>,

    // cb = 2
    nullptr,
    exl3_gemm_kernel<3, true, 2, EXL3_GEMM_SHAPE_1>,
    exl3_gemm_kernel<3, true, 2, EXL3_GEMM_SHAPE_2>,
    exl3_gemm_kernel<3, true, 2, EXL3_GEMM_SHAPE_3>,
    exl3_gemm_kernel<3, true, 2, EXL3_GEMM_SHAPE_4>
};

fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b4[] = {
    // cb = 0
    nullptr,
    exl3_gemm_kernel<4, true, 0, EXL3_GEMM_SHAPE_1>,
    exl3_gemm_kernel<4, true, 0, EXL3_GEMM_SHAPE_2>,
    exl3_gemm_kernel<4, true, 0, EXL3_GEMM_SHAPE_3>,
    exl3_gemm_kernel<4, true, 0, EXL3_GEMM_SHAPE_4>,

    // cb = 1
    nullptr,
    exl3_gemm_kernel<4, true, 1, EXL3_GEMM_SHAPE_1>,
    exl3_gemm_kernel<4, true, 1, EXL3_GEMM_SHAPE_2>,
    exl3_gemm_kernel<4, true, 1, EXL3_GEMM_SHAPE_3>,
    exl3_gemm_kernel<4, true, 1, EXL3_GEMM_SHAPE_4>,

    // cb = 2
    nullptr,
    exl3_gemm_kernel<4, true, 2, EXL3_GEMM_SHAPE_1>,
    exl3_gemm_kernel<4, true, 2, EXL3_GEMM_SHAPE_2>,
    exl3_gemm_kernel<4, true, 2, EXL3_GEMM_SHAPE_3>,
    exl3_gemm_kernel<4, true, 2, EXL3_GEMM_SHAPE_4>
};

static fp_exl3_gemm_kernel select_exl3_gemm_kernel
(
    int cc,
    int size_k,
    int size_n,
    int bits,
    int* out_block_dim,
    int* out_shape_idx,
    int* num_sms,
    int cb
)
{
    int shape_idx = select_gemm_shape(cc, size_k, size_n, bits, false);

    GGML_ASSERT(shape_idx > 0); // "exl3_gemm: no compatible kernel"
    if (out_shape_idx) *out_shape_idx = shape_idx;
    if (out_block_dim) *out_block_dim = exl3_gemm_blockdim[shape_idx];

    // Avoid empty blocks
    if (num_sms)
    {
        int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
        int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
        int max_slices = size_k / tilesize_k * size_n / tilesize_n;
        *num_sms = MAX1(MIN1(max_slices, *num_sms), 1);
    }

    int kernel_idx = shape_idx + (EXL3_GEMM_NUM_SHAPES + 1) * cb;

    switch (bits)
    {
        case 2: return tfp_exl3_gemm_kernel_fp32_b2[kernel_idx];
        case 3: return tfp_exl3_gemm_kernel_fp32_b3[kernel_idx];
        case 4: return tfp_exl3_gemm_kernel_fp32_b4[kernel_idx];
        default: GGML_ASSERT(false && "No kernel for GEMM shape");
    }
}

class DevCtx
{
private:
    void* locks = nullptr;
    int cc = 0;
    std::mutex mtx;

public:
    static DevCtx& instance() {
        static DevCtx ctx;
        return ctx;
    }
    int get_cc();
    int* get_locks();

private:
    DevCtx() = default;
    DevCtx(const DevCtx&) = delete;
    DevCtx& operator=(const DevCtx&) = delete;
};

int* DevCtx::get_locks()
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!locks)
    {
        cudaMalloc(&locks, MAX_TILES_C * sizeof(int));
        cudaMemset(locks, 0, MAX_TILES_C * sizeof(int));
    }
    return (int*) locks;
}

int DevCtx::get_cc()
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!cc)
    {
        cudaDeviceProp prop;
        cuda_check(cudaGetDeviceProperties(&prop, 0));
        if (prop.major >= 10) cc = CC_BLACKWELL;
        else if (prop.major >= 9) cc = CC_HOPPER;
        else if (prop.major >= 8 && prop.minor >= 9) cc = CC_ADA;
        else if (prop.major >= 8 && prop.minor >= 6) cc = CC_AMPERE;
        else cc = CC_OLD;
    }
    return cc;
}

/*
EXL3 matmul, A @ B -> C

- A: row-major A tensor, shape (m, k), dtype float32, contiguous
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*bits), dtype uint16
- C: empty row-major C tensor, shape (m, n), dtype float23, contiguous. Does not need to be zero-initialized
- suh: optional, packed input scales/flips, shape (k//16), dtype float16
- A_had: required if suh given, may be reference to A, temporary storage for input transform, size and dtype as A
- svh: optional, packed output scales/flips, shape (n//16), dtype float16

limitations:
- k % 16 == 0
- n % 128 == 0
*/

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
)
{
    int device = ggml_cuda_get_device();
    GGML_ASSERT(device == 0);
    GGML_ASSERT(A->type == GGML_TYPE_F16);
    //const at::cuda::OptionalCUDAGuard device_guard(A.device());
    //cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaStream_t stream = ctx.stream();

    // Device properties
    int num_sms = ggml_cuda_info().devices[device].nsm;
    int cc = DevCtx::instance().get_cc();
    int* locks = DevCtx::instance().get_locks();

    // Dispatch
    int bits = B->ne[2] / 16;
    const half* A_ptr = (const half*) A->data;
    const uint16_t* B_ptr = (const uint16_t*) B->data;
    void* C_ptr = C->data;

    int size_m = A->ne[1];  // A.size(0)
    int size_k = A->ne[0];  // A.size(1)
    int size_n = B->ne[1];  // B.size(1)

    // Select kernel
    GGML_ASSERT(!(mcg_mult && mul1_mult)); // "Specified both mcg_mult and mul1_mult"
    int cb = 0;
    uint32_t mult = 0;
    if (mcg_mult)  { cb = 1; mult = mcg_mult; }
    if (mul1_mult) { cb = 2; mult = mul1_mult; }

    int selected_shape;
    int block_dim;
    fp_exl3_gemm_kernel kernel = select_exl3_gemm_kernel
    (
        cc, size_k, size_n, bits,
        &block_dim, &selected_shape,
        &num_sms, cb
    );
    if (!kernel) return 0;

    // Launch
    if (kernel_attr_set.find((void*)kernel) == kernel_attr_set.end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
        kernel_attr_set.insert((void*)kernel);
    }
    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr,
        (void*)& mult
    };
    cudaLaunchCooperativeKernel
    (
        (void*)kernel,
        num_sms,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );
    CUDA_CHECK(cudaPeekAtLastError());
    return selected_shape;
}
