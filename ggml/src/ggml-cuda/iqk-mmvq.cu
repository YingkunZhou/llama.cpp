#include "mmvq.cuh"
#include "iqk-common.cuh"

typedef void (*iqk_vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float *);

template <ggml_type type, int vdr, iqk_vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y, int n_interleaved = 1>
__device__ void iqk_mul_mat_vec_q_impl(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst, const int64_t row_size) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int nwarps              = 1;
    constexpr int rows_per_cuda_block = n_interleaved;
#else
    constexpr int nwarps              = n_interleaved == 1 ? ncols_y <= 4 ? 4 : 2 : 1;
    constexpr int rows_per_cuda_block = n_interleaved == 1 ? ncols_y == 1 ? 1 : 2 : n_interleaved;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

// partial sum for each thread
    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
            if constexpr (n_interleaved == 1) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    vec_dot_q_cuda((const void *)((const char *)vx + (row0 + i)*row_size),
                            &y[j*blocks_per_col_y + kby], kbx, kqs, &tmp[j][i]);
                }
            } else {
                vec_dot_q_cuda((const void *)((const char *)vx + row0*row_size),
                    &y[j*blocks_per_col_y + kby], kbx, kqs, tmp[j]);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < nrows_dst)) {
            dst[j*nrows_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

template <ggml_type type, int vdr, iqk_vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y, int n_interleaved = 1>
static __global__ void iqk_mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst, const char * __restrict__ ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst, const int64_t row_size,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0) {
    int i2 = blockIdx.y;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) return;
    const char * cx = (const char *)vx + i02*nb02;
    const char * cy = (const char *)vy + i2*nb12;
    char * cdst = (char *)dst + i2*nb2;
    iqk_mul_mat_vec_q_impl<type, vdr, vec_dot_q_cuda, ncols_y, n_interleaved>(cx, cy, (float *)cdst, ncols_x, nrows_x, nrows_y, nrows_dst, row_size);
}

template <ggml_type type, int vdr, iqk_vec_dot_q_cuda_t vec_dot_q_cuda, int n_interleaved = 1>
static void iqk_mul_mat_vec_q_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    //GGML_ASSERT(ncols_y <= MMVQ_MAX_BATCH_SIZE);

    int id = ggml_cuda_get_device();

    int64_t nwarps = 1;
    int64_t rows_per_cuda_block = n_interleaved;

    if (ggml_cuda_info().devices[id].cc < GGML_CUDA_CC_RDNA2) { // NVIDIA and AMD older than RDNA2
        switch(ncols_y) {
            case 1:
                nwarps = n_interleaved == 1 ? 4 : 1;
                rows_per_cuda_block = n_interleaved == 1 ? 1 : n_interleaved;
                break;
            case 2:
            case 3:
            case 4:
                nwarps = n_interleaved == 1 ? 4 : 1;
                rows_per_cuda_block = n_interleaved == 1 ? 2 : n_interleaved;
                break;
            case 5:
            case 6:
            case 7:
            case 8:
                nwarps = n_interleaved == 1 ? 2 : 1;
                rows_per_cuda_block = n_interleaved == 1 ? 2 : n_interleaved;
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
    }
    const int64_t nblocks = (nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, ne2, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    const int64_t row_size = ggml_row_size(type, ncols_x);

    switch (ncols_y) {
        case 1:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 1, n_interleaved><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, row_size, nb02, nb12, nb2, ids_nb0);
            break;
        case 2:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 2, n_interleaved><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, row_size, nb02, nb12, nb2, ids_nb0);
            break;
        case 3:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 3, n_interleaved><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, row_size, nb02, nb12, nb2, ids_nb0);
            break;
        case 4:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 4, n_interleaved><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, row_size, nb02, nb12, nb2, ids_nb0);
            break;
        case 5:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 5, n_interleaved><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, row_size, nb02, nb12, nb2, ids_nb0);
            break;
        case 6:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 6, n_interleaved><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, row_size, nb02, nb12, nb2, ids_nb0);
            break;
        case 7:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 7, n_interleaved><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, row_size, nb02, nb12, nb2, ids_nb0);
            break;
        case 8:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 8, n_interleaved><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, row_size, nb02, nb12, nb2, ids_nb0);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }
}

#define VDR_IQ2_K_Q8_1_MMVQ 4
#define VDR_IQ2_K_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq2_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    // iqs is 0, 4, 8, 12, 16, 20, 24, 28
    // we have 16 packed quants (when cast to int)

    int i4 = iqs/4;  // 0...7. We will process q8 blocks 4*(i4/4), 4*(i4/4)+1, 4*(i4/4)+2, 4*(i4/4)+3
    const int32_t  * q8_1 = (const int *)bq8_1[4*(i4/4)+0].qs + 2*(i4%4);
    const int32_t  * q8_2 = (const int *)bq8_1[4*(i4/4)+1].qs + 2*(i4%4);
    const int32_t  * q8_3 = (const int *)bq8_1[4*(i4/4)+2].qs + 2*(i4%4);
    const int32_t  * q8_4 = (const int *)bq8_1[4*(i4/4)+3].qs + 2*(i4%4);

    const block_iq2_k * bq2 = (const block_iq2_k *) vbq + kbx;
    const uint32_t * q2 = (const uint32_t *)bq2->qs + 8*(i4/4) + 2*(i4%4);
    const uint16_t extra = bq2->extra >> (8*(i4/4) + (i4%4)/2);

    const int * all_values = (const int *)iq2k_table;
    const int * values;

    uint32_t val1 = q2[0], val2 = q2[1];

    uint32_t aux32[2];
    int v1, v2;

    // Block of 16: (32*(4*(i4/4)+k)+8*(i4%4))/16 = 8*(i4/4) + 2*k + (i4%4)/2
    // -> scales_l[4*(i4/4) + k] >> 4*(((i4%4)/2)%2)

    const uint32_t * scales = (const uint32_t *)bq2->scales;
    uint32_t s32 = __vsub4((scales[i4/4] >> 4*(((i4%4)/2)%2)) & 0x0f0f0f0f, 0x08080808);
    const int8_t * s8 = (const int8_t *)&s32;

    aux32[0] = ((val1 >> 0) & 0x03030303); aux32[1] = ((val2 >> 0) & 0x03030303); values = all_values + ((extra & 0x01) << 8);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi1 = ggml_cuda_dp4a(v2, q8_1[1], ggml_cuda_dp4a(v1, q8_1[0], 0)) * s8[0];

    aux32[0] = ((val1 >> 2) & 0x03030303); aux32[1] = ((val2 >> 2) & 0x03030303); values = all_values + ((extra & 0x04) << 6);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi2 = ggml_cuda_dp4a(v2, q8_2[1], ggml_cuda_dp4a(v1, q8_2[0], 0)) * s8[1];

    aux32[0] = ((val1 >> 4) & 0x03030303); aux32[1] = ((val2 >> 4) & 0x03030303); values = all_values + ((extra & 0x10) << 4);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi3 = ggml_cuda_dp4a(v2, q8_3[1], ggml_cuda_dp4a(v1, q8_3[0], 0)) * s8[2];

    aux32[0] = ((val1 >> 6) & 0x03030303); aux32[1] = ((val2 >> 6) & 0x03030303); values = all_values + ((extra & 0x40) << 2);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi4 = ggml_cuda_dp4a(v2, q8_4[1], ggml_cuda_dp4a(v1, q8_4[0], 0)) * s8[3];

    *result += __half2float(bq2->d) * (__low2float(bq8_1[4*(i4/4)+0].ds) * sumi1
                                    +  __low2float(bq8_1[4*(i4/4)+1].ds) * sumi2
                                    +  __low2float(bq8_1[4*(i4/4)+2].ds) * sumi3
                                    +  __low2float(bq8_1[4*(i4/4)+3].ds) * sumi4);
}

static void iqk_mul_mat_vec_iq2_k_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_K, VDR_IQ2_K_Q8_1_MMVQ, vec_dot_iq2_k_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

#define VDR_IQ2_KS_Q8_1_MMVQ 4
#define VDR_IQ2_KS_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq2_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const half *)vbq;
    const block_iq2_ks * bq2 = (const block_iq2_ks *)((const char *)vbq + sizeof(half)) + kbx;

    int i4 = iqs/4;  // 0...7. We will process q8 blocks 4*(i4/4), 4*(i4/4)+1, 4*(i4/4)+2, 4*(i4/4)+3
    const int32_t  * q8_1 = (const int *)bq8_1[4*(i4/4)+0].qs + 2*(i4%4);
    const int32_t  * q8_2 = (const int *)bq8_1[4*(i4/4)+1].qs + 2*(i4%4);
    const int32_t  * q8_3 = (const int *)bq8_1[4*(i4/4)+2].qs + 2*(i4%4);
    const int32_t  * q8_4 = (const int *)bq8_1[4*(i4/4)+3].qs + 2*(i4%4);

    const uint16_t * q2 = (const uint16_t *)bq2->qs + 16*(i4/4) + 4*(i4%4);
    const uint16_t extra = bq2->extra >> 4*(i4/4);

    const int * all_values = (const int *)iq2k_table;
    const int * values;

    uint32_t val1 = q2[0] | (q2[1] << 16), val2 = q2[2] | (q2[3] << 16);

    uint32_t aux32[2];
    int v1, v2;

    int32_t scales32;
    const uint16_t * scales16 = (const uint16_t *)bq2->scales;
    scales32 = __vsub4((scales16[i4/4] | (scales16[i4/4] << 12)) & 0x0f0f0f0f, 0x10101010);
    int8_t * s8 = (int8_t *)&scales32;
    s8[0] += ((extra >> 4) & 0x10);
    s8[1] += ((extra >> 6) & 0x10);
    s8[2] += ((extra >> 5) & 0x10);
    s8[3] += ((extra >> 7) & 0x10);

    aux32[0] = ((val1 >> 0) & 0x03030303); aux32[1] = ((val2 >> 0) & 0x03030303); values = all_values + ((extra & 0x01) << 8);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi1 = ggml_cuda_dp4a(v2, q8_1[1], ggml_cuda_dp4a(v1, q8_1[0], 0)) * s8[0];

    aux32[0] = ((val1 >> 2) & 0x03030303); aux32[1] = ((val2 >> 2) & 0x03030303); values = all_values + ((extra & 0x02) << 7);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi2 = ggml_cuda_dp4a(v2, q8_2[1], ggml_cuda_dp4a(v1, q8_2[0], 0)) * s8[2];

    aux32[0] = ((val1 >> 4) & 0x03030303); aux32[1] = ((val2 >> 4) & 0x03030303); values = all_values + ((extra & 0x04) << 6);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi3 = ggml_cuda_dp4a(v2, q8_3[1], ggml_cuda_dp4a(v1, q8_3[0], 0)) * s8[1];

    aux32[0] = ((val1 >> 6) & 0x03030303); aux32[1] = ((val2 >> 6) & 0x03030303); values = all_values + ((extra & 0x08) << 5);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi4 = ggml_cuda_dp4a(v2, q8_4[1], ggml_cuda_dp4a(v1, q8_4[0], 0)) * s8[3];

    *result += scale * (__low2float(bq8_1[4*(i4/4)+0].ds) * sumi1
                     +  __low2float(bq8_1[4*(i4/4)+1].ds) * sumi2
                     +  __low2float(bq8_1[4*(i4/4)+2].ds) * sumi3
                     +  __low2float(bq8_1[4*(i4/4)+3].ds) * sumi4);
}

static void iqk_mul_mat_vec_iq2_ks_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_KS, VDR_IQ2_KS_Q8_1_MMVQ, vec_dot_iq2_ks_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

#define VDR_IQ3_K_Q8_1_MMVQ 4
#define VDR_IQ3_K_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq3_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs, float * result) {
    const block_iq3_k * bq3 = (const block_iq3_k *) vbq + kbx;

    int iqs = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1. 0 works on quants 0...127, 1 on quants 128...255
                              // Each thread processes 8 quants in each of the 4 32-blocks
    const int il8   = iqs%4;  // 0...3. 0 works on quants 0...7, 1 on quants 8...15, 2 on 16...23, 3 on 24...31
    const int shift = 4*(il8/2);

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    uint32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    const int hshift = 4*(1-ib128);
    const uint16_t sh = bq3->scales_h >> (8*ib128 + il8/2);

    const uint8_t extra = bq3->extra >> (8*ib128 + il8/2);
    const uint16_t * values1 = iq3k_table + ((extra << 6) & 0x40);
    const uint16_t * values2 = iq3k_table + ((extra << 5) & 0x40);
    const uint16_t * values3 = iq3k_table + ((extra << 4) & 0x40);
    const uint16_t * values4 = iq3k_table + ((extra << 3) & 0x40);

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
    int v;
    for (int i = 0; i < 2; ++i) {
        uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
        uint32_t vh = ((qh[2*i+0] | (qh[2*i+1] << 16)) << hshift) >> 2;

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        aux32 = (vl & 0x03030303) | (vh & 0x04040404);
        v = int_from_table_2(aux8, values1);
        sumi[0] = ggml_cuda_dp4a(v, q8[i], sumi[0]);
        vl >>= 2; vh >>= 1;

        q8 += sizeof(block_q8_1)/4;
        aux32 = (vl & 0x03030303) | (vh & 0x04040404);
        v = int_from_table_2(aux8, values2);
        sumi[1] = ggml_cuda_dp4a(v, q8[i], sumi[1]);
        vl >>= 2; vh >>= 1;

        q8 += sizeof(block_q8_1)/4;
        aux32 = (vl & 0x03030303) | (vh & 0x04040404);
        v = int_from_table_2(aux8, values3);
        sumi[2] = ggml_cuda_dp4a(v, q8[i], sumi[2]);
        vl >>= 2; vh >>= 1;

        q8 += sizeof(block_q8_1)/4;
        aux32 = (vl & 0x03030303) | (vh & 0x04040404);
        v = int_from_table_2(aux8, values4);
        sumi[3] = ggml_cuda_dp4a(v, q8[i], sumi[3]);

    }
    const float d = __half2float(bq3->d);
    const uint16_t * sl16 = (const uint16_t *)bq3->scales_l + 2*ib128;
    aux32 = ((((sl16[0] | (sl16[1] << 16)) >> shift) & 0x0f0f0f0f) << 1) | 0x01010101;
    *result += d * (__low2float(bq8_1[4*ib128+0].ds) * aux8[0] * (sh & 0x01 ? -1 : 1) * sumi[0] +
                    __low2float(bq8_1[4*ib128+1].ds) * aux8[1] * (sh & 0x04 ? -1 : 1) * sumi[1] +
                    __low2float(bq8_1[4*ib128+2].ds) * aux8[2] * (sh & 0x10 ? -1 : 1) * sumi[2] +
                    __low2float(bq8_1[4*ib128+3].ds) * aux8[3] * (sh & 0x40 ? -1 : 1) * sumi[3]);

}

static void iqk_mul_mat_vec_iq3_k_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_K, VDR_IQ3_K_Q8_1_MMVQ, vec_dot_iq3_k_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

#define VDR_IQ3_KS_Q8_1_MMVQ 4
#define VDR_IQ3_KS_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq3_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs, float * result) {

    float d = __half2float(*(const half *)vbq);
    const block_iq3_ks * bq3 = (const block_iq3_ks *)((const char *)vbq + sizeof(half)) + kbx;

    int iqs = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1. 0 works on quants 0...127, 1 on quants 128...255
                              // Each thread processes 8 quants in each of the 4 32-blocks
    const int il8   = iqs%4;  // 0...3. 0 works on quants 0...7, 1 on quants 8...15, 2 on 16...23, 3 on 24...31

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    int32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    uint16_t extra = bq3->extra >> 4*ib128;
    uint16_t extra_v = extra >> 8;

    const uint16_t * values1 = iq3k_table + ((extra_v << 6) & 0x40);
    const uint16_t * values2 = iq3k_table + ((extra_v << 5) & 0x40);
    const uint16_t * values3 = iq3k_table + ((extra_v << 4) & 0x40);
    const uint16_t * values4 = iq3k_table + ((extra_v << 3) & 0x40);

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
    int v;
    for (int i = 0; i < 2; ++i) {
        uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
        uint32_t vh = ((qh[2*i+0] | (qh[2*i+1] << 16)) >> 4*ib128) << 2;

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        aux32 = (vl & 0x03030303) | (vh & 0x04040404);
        v = int_from_table_2(aux8, values1);
        sumi[0] = ggml_cuda_dp4a(v, q8[i], sumi[0]);
        vl >>= 2; vh >>= 1;

        q8 += sizeof(block_q8_1)/4;
        aux32 = (vl & 0x03030303) | (vh & 0x04040404);
        v = int_from_table_2(aux8, values2);
        sumi[1] = ggml_cuda_dp4a(v, q8[i], sumi[1]);
        vl >>= 2; vh >>= 1;

        q8 += sizeof(block_q8_1)/4;
        aux32 = (vl & 0x03030303) | (vh & 0x04040404);
        v = int_from_table_2(aux8, values3);
        sumi[2] = ggml_cuda_dp4a(v, q8[i], sumi[2]);
        vl >>= 2; vh >>= 1;

        q8 += sizeof(block_q8_1)/4;
        aux32 = (vl & 0x03030303) | (vh & 0x04040404);
        v = int_from_table_2(aux8, values4);
        sumi[3] = ggml_cuda_dp4a(v, q8[i], sumi[3]);

    }
    const uint16_t * sl16 = (const uint16_t *)bq3->scales;
    aux32 = __vsub4(((sl16[0] | (sl16[1] << 16)) >> 4*ib128) & 0x0f0f0f0f, 0x10101010);
    const int8_t * a8 = (const int8_t *)&aux32;
    *result += d * (__low2float(bq8_1[4*ib128+0].ds) * (a8[0] + ((extra << 4) & 0x10)) * sumi[0] +
                    __low2float(bq8_1[4*ib128+1].ds) * (a8[1] + ((extra << 3) & 0x10)) * sumi[1] +
                    __low2float(bq8_1[4*ib128+2].ds) * (a8[2] + ((extra << 2) & 0x10)) * sumi[2] +
                    __low2float(bq8_1[4*ib128+3].ds) * (a8[3] + ((extra << 1) & 0x10)) * sumi[3]);

}

static void iqk_mul_mat_vec_iq3_ks_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_KS, VDR_IQ3_KS_Q8_1_MMVQ, vec_dot_iq3_ks_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

__device__ __forceinline__ void get_int_from_table_16_shift(const uint32_t & q4, uint16_t shift, const uint8_t * all_values,
        int & val1, int & val2) {

    uint32_t aux32; const uint8_t * q8 = (const uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    const uint8_t * values = all_values + 16*(shift & 1);
    uint16_t v1 = values[q8[0]] | (values[q8[1]] << 8);
    uint16_t v2 = values[q8[2]] | (values[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    values = all_values + 8*(shift & 2);
    v1 = values[q8[0]] | (values[q8[1]] << 8);
    v2 = values[q8[2]] | (values[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

#define VDR_IQ4_K_Q8_1_MMVQ 4
#define VDR_IQ4_K_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq4_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq4_k * bq4 = (const block_iq4_k *) vbq + kbx;
    const uint8_t * all_values = (const uint8_t *)iq4k_values;

    // iqs is 0...28
    const int ib32 = iqs/4;
    // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint16_t * q4 = (const uint16_t *)bq4->qs + 8*ib32;
    const uint16_t extra = bq4->extra >> 2*ib32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        const uint32_t aux32 = q4[2*j+0] | (q4[2*j+1] << 16);
        get_int_from_table_16_shift(aux32, extra, all_values, v1, v2);
        sumi1 = ggml_cuda_dp4a(v1, q8[j+0], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8[j+4], sumi2);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1[ib32].ds);
    const uint8_t sh = bq4->scales_h[ib32/2] >> 4*(ib32%2);
    const int ls1 = ((bq4->scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32;
    const int ls2 = ((bq4->scales_l[ib32] >>  4) | ((sh << 2) & 0x30)) - 32;
    *result += d * (sumi1 * ls1 + sumi2 * ls2);
}

static void iqk_mul_mat_vec_iq4_k_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_K, VDR_IQ4_K_Q8_1_MMVQ, vec_dot_iq4_k_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

#define VDR_IQ4_KS_Q8_1_MMVQ 4
#define VDR_IQ4_KS_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq4_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq4_ks * bq4 = (const block_iq4_ks *)((const char *)vbq + sizeof(float)) + kbx;
    const uint8_t * all_values = (const uint8_t *)iq4k_values;

    // iqs is 0...28
    const int ib32 = iqs/4; // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint32_t * q4 = (const uint32_t *)bq4->qs + 4*ib32;
    const float dl = scale * ((bq4->scales[ib32] & 254) - 127);
    int v1, v2;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        get_int_from_table_16_shift(q4[j], bq4->scales[ib32] & 1, all_values, v1, v2);
        sumi = ggml_cuda_dp4a(v1, q8[j+0], sumi);
        sumi = ggml_cuda_dp4a(v2, q8[j+4], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

static void iqk_mul_mat_vec_iq4_ks_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KS, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq4_ks_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

#define VDR_IQ4_KSS_Q8_1_MMVQ 4
#define VDR_IQ4_KSS_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq4_kss_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq4_kss * bq4 = (const block_iq4_kss *)((const char *)vbq + sizeof(float)) + kbx;
    const uint8_t * all_values = (const uint8_t *)iq4k_values;

    // iqs is 0...28
    const int ib32 = iqs/4; // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint32_t * q4 = (const uint32_t *)bq4->qs + 4*ib32;
    uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
    uint8_t ls = (s32 | (s32 >> 15)) & 0xff;
    const float dl = scale * ((ls & 254) - 127);
    int v1, v2;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t aux32 = q4[j] & 0xfffefffe;
        aux32 ^= (aux32 >> 1);
        get_int_from_table_16_shift(aux32, ls & 1, all_values, v1, v2);
        sumi = ggml_cuda_dp4a(v1, q8[j+0], sumi);
        sumi = ggml_cuda_dp4a(v2, q8[j+4], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

static void iqk_mul_mat_vec_iq4_kss_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KSS, VDR_IQ4_KSS_Q8_1_MMVQ, vec_dot_iq4_kss_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

#define VDR_IQ5_K_Q8_1_MMVQ 4
#define VDR_IQ5_K_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq5_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq5_k * bq5 = (const block_iq5_k *) vbq + kbx;
    const uint8_t * all_values = (const uint8_t *)iq5nl_values;

    int i4 = iqs/4;  // 0...7.  Blocks of 16 index is 4*(i4/2) + (i4%2) + (0 and 2)

    const int32_t  * q8_1 = (const int *)bq8_1[2*(i4/2)+0].qs + 4*(i4%2);
    const int32_t  * q8_2 = (const int *)bq8_1[2*(i4/2)+1].qs + 4*(i4%2);
    const uint32_t * q4 = (const uint32_t *)bq5->qs + 8*(i4/2) + 4*(i4%2);
    const uint32_t * qh = (const uint32_t *)bq5->qh + 4*(i4%2);
    const uint16_t extra = bq5->extra >> (4*(i4/2) + (i4%2));
    const uint8_t * values1 = all_values + 32*(extra & 1);
    const uint8_t * values2 = all_values +  8*(extra & 4);
    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t h = qh[j] >> 2*(i4/2);
        aux32[0] = ((q4[j] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x10101010);
        aux32[1] = ((q4[j] >> 4) & 0x0f0f0f0f) | ((h << 3) & 0x10101010);
        v1 = int_from_table(a8+0, values1);
        v2 = int_from_table(a8+4, values2);
        sumi1 = ggml_cuda_dp4a(v1, q8_1[j], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8_2[j], sumi2);
    }
    const float d5 = __half2float(bq5->d);
    const uint8_t sh = bq5->scales_h[i4/2] >> 2*(i4%2);
    const int ls1 = (((bq5->scales_l[2*(i4/2)+0] >> 4*(i4%2)) & 0xf) | ((sh << 4) & 0x30)) - 32;
    const int ls2 = (((bq5->scales_l[2*(i4/2)+1] >> 4*(i4%2)) & 0xf) | ((sh << 0) & 0x30)) - 32;
    *result += d5 * (__low2float(bq8_1[2*(i4/2)+0].ds) * sumi1 * ls1 + __low2float(bq8_1[2*(i4/2)+1].ds) * sumi2 * ls2);
}

static void iqk_mul_mat_vec_iq5_k_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ5_K, VDR_IQ5_K_Q8_1_MMVQ, vec_dot_iq5_k_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}


#define VDR_IQ5_KS_Q8_1_MMVQ 4
#define VDR_IQ5_KS_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq5_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq5_ks * bq5 = (const block_iq5_ks *)((const char *)vbq + sizeof(float)) + kbx;
    const uint8_t * all_values = (const uint8_t *)iq5nl_values;

    int i4 = iqs/4;  // 0...7.  Blocks of 16 index is 4*(i4/2) + (i4%2) + (0 and 2)

    const int32_t  * q8_1 = (const int *)bq8_1[2*(i4/2)+0].qs + 4*(i4%2);
    const int32_t  * q8_2 = (const int *)bq8_1[2*(i4/2)+1].qs + 4*(i4%2);
    const uint32_t * q4 = (const uint32_t *)bq5->qs + 8*(i4/2) + 4*(i4%2);
    const uint32_t * qh = (const uint32_t *)bq5->qh + 4*(i4%2);
    const uint8_t * values1 = all_values + ((bq5->scales[2*(i4/2)+0] & 1) << 5);
    const uint8_t * values2 = all_values + ((bq5->scales[2*(i4/2)+1] & 1) << 5);
    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t h = qh[j] >> 2*(i4/2);
        aux32[0] = ((q4[j] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x10101010);
        aux32[1] = ((q4[j] >> 4) & 0x0f0f0f0f) | ((h << 3) & 0x10101010);
        v1 = int_from_table(a8+0, values1);
        v2 = int_from_table(a8+4, values2);
        sumi1 = ggml_cuda_dp4a(v1, q8_1[j], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8_2[j], sumi2);
    }
    const int ls1 = (bq5->scales[2*(i4/2)+0] & 254) - 127;
    const int ls2 = (bq5->scales[2*(i4/2)+1] & 254) - 127;
    *result += scale * (__low2float(bq8_1[2*(i4/2)+0].ds) * sumi1 * ls1 + __low2float(bq8_1[2*(i4/2)+1].ds) * sumi2 * ls2);
}

static void mul_mat_vec_iq5_ks_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ5_KS, VDR_IQ5_KS_Q8_1_MMVQ, vec_dot_iq5_ks_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

#define VDR_IQ6_K_Q8_1_MMVQ 4
#define VDR_IQ6_K_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq6_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq6_k * bq6 = (const block_iq6_k *) vbq + kbx;
    const uint8_t * all_values = (const uint8_t *)iq6nl_values;

    int i4 = iqs/4;  // 0...7.  Blocks of 16 index is 4*(i4/2) + (i4%2) + (0 and 2)
                     //         Blocks of 32 index is 2*(i4/2) + 0 or 1

    const int32_t  * q8_1 = (const int *)bq8_1[2*(i4/2)+0].qs + 4*(i4%2);
    const int32_t  * q8_2 = (const int *)bq8_1[2*(i4/2)+1].qs + 4*(i4%2);
    const uint32_t * q4 = (const uint32_t *)bq6->qs + 8*(i4/2) + 4*(i4%2);
    const uint32_t * qh = (const uint32_t *)bq6->qh + 8*(i4/4) + 4*(i4%2);
    const uint16_t extra = bq6->extra >> (4*(i4/2) + (i4%2));
    const uint8_t * values1 = all_values + 64*(extra & 1);
    const uint8_t * values2 = all_values + 16*(extra & 4);
    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t h = qh[j] >> 4*((i4/2)%2);
        aux32[0] = ((q4[j] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x30303030);
        aux32[1] = ((q4[j] >> 4) & 0x0f0f0f0f) | ((h << 2) & 0x30303030);
        v1 = int_from_table(a8+0, values1);
        v2 = int_from_table(a8+4, values2);
        sumi1 = ggml_cuda_dp4a(v1, q8_1[j], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8_2[j], sumi2);
    }
    const float d6 = __half2float(bq6->d);
    *result += d6 * (__low2float(bq8_1[2*(i4/2)+0].ds) * sumi1 * bq6->scales[4*(i4/2)+(i4%2)] + __low2float(bq8_1[2*(i4/2)+1].ds) * sumi2 * bq6->scales[4*(i4/2)+(i4%2)+2]);
}

static void iqk_mul_mat_vec_iq6_k_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ6_K, VDR_IQ6_K_Q8_1_MMVQ, vec_dot_iq6_k_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

__device__ __forceinline__ void vec_dot_iq2_kt_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

    float scale = *(const float *)vbq;
    const block_iq2_kt * bq2 = (const block_iq2_kt *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4;
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const int ls = iq4k_values[(bq2->scales[ib32%4] >> 4*(ib32/4)) & 0xf];
    const float dl = scale * ls * 1.05f;
    auto ql = (const uint16_t *)bq2->ql;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t val = ql[4*ib32+j] + 4096;
        int v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v4 |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        sumi = ggml_cuda_dp4a(v4, q8[2*j+0], sumi);
        v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v4 |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        sumi = ggml_cuda_dp4a(v4, q8[2*j+1], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

static void iqk_mul_mat_vec_iq2_kt_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq2_kt_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

__device__ __forceinline__ void vec_dot_iq3_kt_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

    float scale = *(const float *)vbq;
    const block_iq3_kt * bq3 = (const block_iq3_kt *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4;
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const int ls = (bq3->scales[ib32%4] >> 4*(ib32/4)) & 0xf;
    const float dl = scale * ls * 1.015f;
    auto ql = (const uint16_t *)bq3->ql;
    uint32_t mask = 0x01010101 << ib32;
    const uint32_t * qh = (const uint32_t *)bq3->qh;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t val = ql[4*ib32+j] + 4096;
        int v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            int8_t q = std::abs(ggml_cuda_dp4a(val & km, 0x01010101, -126));
            v4 |= q << 8*k;
        }
        uint32_t signs = __vcmpne4(qh[2*j+0] & mask, 0);
        v4 = __vsub4(v4 ^ signs, signs);
        sumi = ggml_cuda_dp4a(v4, q8[2*j+0], sumi);
        v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            int8_t q = std::abs(ggml_cuda_dp4a(val & km, 0x01010101, -126));
            v4 |= q << 8*k;
        }
        signs = __vcmpne4(qh[2*j+1] & mask, 0);
        v4 = __vsub4(v4 ^ signs, signs);
        sumi = ggml_cuda_dp4a(v4, q8[2*j+1], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

static void iqk_mul_mat_vec_iq3_kt_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq3_kt_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

__device__ __forceinline__ void vec_dot_iq4_kt_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

    float scale = *(const float *)vbq;
    const block_iq4_kt * bq4 = (const block_iq4_kt *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4; // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    //const int8_t  * q8 = bq8_1[ib32].qs;
    const int ls = (bq4->qs[ib32] & 0xff) >> 1;
    const float dl = scale * (ls - 64);
    const uint32_t idx0 = ((bq4->qs[ib32] & 1) << 15) + 4096;
    auto ql = (const uint8_t *)(bq4->qs + 8);
    auto qh = ql + 64;
    ql += 8*ib32;
    qh += 8*(ib32%4);
    const int shift1 = 8 - 4*(ib32/4);
    int sumi = 0;
    for (int j = 0; j < 8; ++j) {
        const uint32_t sh = bq4->qs[ib32] >> (8 + 3*j);
        uint32_t val = ql[j] + ((qh[j] << shift1) & 0xf00) + ((sh & 7) << 12) + idx0;
        int v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            //int s = val & km;
            //sumi += q8[4*j+k] * ggml_cuda_dp4a(s, 0x01010101, -126);
            v4 |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        sumi = ggml_cuda_dp4a(v4, q8[j], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

static void iqk_mul_mat_vec_iq4_kt_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq4_kt_q8_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

bool iqk_ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context & ctx, ggml_type type,
        const int64_t ne00, const int64_t ne0, const int64_t ne2,
        const int64_t nb02, const int64_t nb12, const int64_t nb2, const int64_t ids_nb0,
        const char * src0_dd_i, const char * src1_ddq_i, float * dst_dd_i, const char * ids_data,
        const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
        const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    switch (type) {
        case GGML_TYPE_IQ2_K:
            iqk_mul_mat_vec_iq2_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        case GGML_TYPE_IQ2_KS:
            iqk_mul_mat_vec_iq2_ks_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        case GGML_TYPE_IQ3_K:
            iqk_mul_mat_vec_iq3_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        case GGML_TYPE_IQ3_KS:
            iqk_mul_mat_vec_iq3_ks_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        case GGML_TYPE_IQ4_K:
            iqk_mul_mat_vec_iq4_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        case GGML_TYPE_IQ4_KS:
            iqk_mul_mat_vec_iq4_ks_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        case GGML_TYPE_IQ4_KSS:
            iqk_mul_mat_vec_iq4_kss_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        case GGML_TYPE_IQ5_K:
            iqk_mul_mat_vec_iq5_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        case GGML_TYPE_IQ5_KS:
            mul_mat_vec_iq5_ks_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        case GGML_TYPE_IQ6_K:
            iqk_mul_mat_vec_iq6_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        case GGML_TYPE_IQ2_KT:
            iqk_mul_mat_vec_iq2_kt_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        case GGML_TYPE_IQ3_KT:
            iqk_mul_mat_vec_iq3_kt_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        case GGML_TYPE_IQ4_KT:
            iqk_mul_mat_vec_iq4_kt_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            return true;
        default:
            return false;
    }
}
