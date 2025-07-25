#include "ggml-cpu.h"
#include "repack.h"

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

struct DataInfo {
    float       * s;
    const char  * cy;
    size_t        bs;
    size_t        by;
    int           cur_y;
    int           ne11;
    size_t        bs2;
};

struct BlockInfo {
    //BaseDequantizer
    float d;
    const void * x;
    __m256i values[4];
    union {
        __m256i hbits; //Q3_K, Q5_K
        uint16_t aux[8]; //IQ4_KSS
    };
    __m256i data[4];
};

template <int nrc_y>
static inline void process_mins_16(const __m256i all_scales, const block_q8_K **q8, int i, float d, __m256 * accm) {
    for (int iy = 0; iy < nrc_y; ++iy) {
        // const __m256i prod  = _mm256_madd_epi16(all_scales, q8.load_bsums(iy, i));
        const __m256i prod  = _mm256_madd_epi16(all_scales, _mm256_loadu_si256((const __m256i*)q8[iy][i].bsums));
        accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d * q8[iy][i].d), _mm256_cvtepi32_ps(prod), accm[iy]);
    }
}
static inline void prepare_scales_16(const __m256i all_scales, __m256i * scales) {
    scales[0] = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(all_scales, 0));
    scales[1] = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(all_scales, 1));
}
template <int nrc_y>
static inline void process_mins_and_scales_16(const __m128i scales128, const block_q8_K **q8, int i, float d, __m256 * accm, __m256i * scales) {
    const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
    process_mins_16<nrc_y>(all_scales, q8, i, d, accm);
    prepare_scales_16(all_scales, scales);
}
static inline __m256i iqk_get_scale_shuffle_16(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline void iqk_set_scales_16(const __m256i all_scales, __m256i* scales) {
    scales[0] = _mm256_shuffle_epi8(all_scales, iqk_get_scale_shuffle_16(0));
    scales[1] = _mm256_shuffle_epi8(all_scales, iqk_get_scale_shuffle_16(1));
    scales[2] = _mm256_shuffle_epi8(all_scales, iqk_get_scale_shuffle_16(2));
    scales[3] = _mm256_shuffle_epi8(all_scales, iqk_get_scale_shuffle_16(3));
}
static inline __m256i iqk_get_scale_shuffle_8(int i) {
    return _mm256_set1_epi16((2*i) | ((2*i+1) << 8));
}
static inline void iqk_set_scales_8(const __m256i all_scales, int j, __m256i * scales) {
    scales[0] = _mm256_shuffle_epi8(all_scales, iqk_get_scale_shuffle_8(4*j+0));
    scales[1] = _mm256_shuffle_epi8(all_scales, iqk_get_scale_shuffle_8(4*j+1));
    scales[2] = _mm256_shuffle_epi8(all_scales, iqk_get_scale_shuffle_8(4*j+2));
    scales[3] = _mm256_shuffle_epi8(all_scales, iqk_get_scale_shuffle_8(4*j+3));
}

template <int nrc_y>
static inline void iqk_multiply_add(const struct BlockInfo deq, const __m256i * scales, int j, int i, const block_q8_K **q8, __m256i * sumi) {
    if (j == 0) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            // const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(bits.values[0], q8.load_quants(iy, i, 0)));
            const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(deq.values[0], _mm256_loadu_si256((const __m256i*)q8[iy][i].qs + 0)));
            const __m256i p2 = _mm256_madd_epi16(scales[1], _mm256_maddubs_epi16(deq.values[1], _mm256_loadu_si256((const __m256i*)q8[iy][i].qs + 1)));
            const __m256i p3 = _mm256_madd_epi16(scales[2], _mm256_maddubs_epi16(deq.values[2], _mm256_loadu_si256((const __m256i*)q8[iy][i].qs + 2)));
            const __m256i p4 = _mm256_madd_epi16(scales[3], _mm256_maddubs_epi16(deq.values[3], _mm256_loadu_si256((const __m256i*)q8[iy][i].qs + 3)));
            sumi[iy] = _mm256_add_epi32(_mm256_add_epi32(p1, p3), _mm256_add_epi32(p2, p4));
        }
    } else {
        for (int iy = 0; iy < nrc_y; ++iy) {
            // const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(bits.values[0], q8.load_quants(iy, i, 4)));
            const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(deq.values[0], _mm256_loadu_si256((const __m256i*)q8[iy][i].qs + 4)));
            const __m256i p2 = _mm256_madd_epi16(scales[1], _mm256_maddubs_epi16(deq.values[1], _mm256_loadu_si256((const __m256i*)q8[iy][i].qs + 5)));
            const __m256i p3 = _mm256_madd_epi16(scales[2], _mm256_maddubs_epi16(deq.values[2], _mm256_loadu_si256((const __m256i*)q8[iy][i].qs + 6)));
            const __m256i p4 = _mm256_madd_epi16(scales[3], _mm256_maddubs_epi16(deq.values[3], _mm256_loadu_si256((const __m256i*)q8[iy][i].qs + 7)));
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p1, p3));
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p2, p4));
        }
    }
}
template <int nrc_y>
static inline void iqk_multiply_add_avx2(const struct BlockInfo deq, const __m256i * scales, int j, int i, const block_q8_K **q8, __m256i * sumi) {
    __m256i p[4];
    if (j == 0) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            for (int k = 0; k < 4; ++k) {
                __m256i s = _mm256_sign_epi8(deq.values[k], deq.values[k]);
                // p[k] = _mm256_madd_epi16(scales[k], _mm256_maddubs_epi16(s, _mm256_sign_epi8(q8.load_quants(iy, i, k), deq.values[k])));
                p[k] = _mm256_madd_epi16(scales[k], _mm256_maddubs_epi16(s, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)q8[iy][i].qs + k), deq.values[k])));
            }
            sumi[iy] = _mm256_add_epi32(_mm256_add_epi32(p[0], p[1]), _mm256_add_epi32(p[2], p[3]));
        }
    } else {
        for (int iy = 0; iy < nrc_y; ++iy) {
            for (int k = 0; k < 4; ++k) {
                __m256i s = _mm256_sign_epi8(deq.values[k], deq.values[k]);
                p[k] = _mm256_madd_epi16(scales[k], _mm256_maddubs_epi16(s, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)q8[iy][i].qs + 4+k), deq.values[k])));
            }
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p[0], p[2]));
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p[1], p[3]));
        }
    }
}
static inline float iqk_hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
static inline float iqk_hsum_float_8(__m256 x) {
    return iqk_hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}
static inline __m256i iqk_dequant16(const uint8_t * qs, __m256i ml) {
    const __m128i aux128 = _mm_loadu_si128((const __m128i *)qs);
    const __m256i aux256 = MM256_SET_M128I(_mm_srli_epi16(aux128, 4), aux128);
    return _mm256_and_si256(ml, aux256);
}
static inline void iqk_make_q4_scales(const uint8_t * scales8, uint32_t * aux32) {
    const uint16_t * scales = (const uint16_t *)scales8;
    const uint32_t a0 = scales[0] | (scales[1] << 16);
    const uint32_t a1 = scales[2] | (scales[3] << 16);
    const uint32_t a2 = scales[4] | (scales[5] << 16);
    aux32[3] = ((a2 >> 4) & 0x0f0f0f0f) | ((a1 >> 2) & 0x30303030);
    aux32[1] = ((a2 >> 0) & 0x0f0f0f0f) | ((a0 >> 2) & 0x30303030);
    aux32[2] = a1 & 0x3f3f3f3f;
    aux32[0] = a0 & 0x3f3f3f3f;
}

struct ConstInfo
{
    __m256i ml;
    union
    {
        __m256i iqk_values;
        __m256i mh1; //IQ6_K
    };
    union {
        __m256i mh; //Q3_K
        __m256i min; //IQ2_K
        __m256i hmask256; //IQ3_K
        __m256i m1_256; //IQ4_KSS
        __m256i mh2; //IQ6_K
        __m128i shuffles[2]; //IQ4_XS
    };
    __m256i mh3; //IQ6_K
    union {
        __m128i m32;
        __m128i m4;
        __m128i m8;
        __m128i m1;
    };
    union {
        __m128i hshift; //IQ4_XS
        __m128i m5; //IQ2_KS
        __m128i eshuffle; //IQ2_K
        __m128i mask; //IQ4_KSS
        __m128i m2; //IQ5_KS
    };
    union {
        __m128i lshift; //IQ4_XS
        __m128i shift; //IQ2_KS
        __m128i eshift; //IQ2_K
        __m128i m127; //IQ4_KSS
    };
    union {
        __m128i hmask; //IQ4_XS, IQ2_KS
        __m128i emask; //IQ2_K
        __m128i m128; //IQ4_KSS
        __m128i maskh; //IQ4_K
    };
    union {
        __m128i lmask; //IQ4_XS
        __m128i m16; //IQ2_KS
        __m128i maskl; //IQ2_K
        __m128i sign_mask; //IQ3_K
    };
    union {
        __m128i lshuffle; //IQ4_XS
        __m128i shuffle; //IQ2_KS
        __m128i hshuff; //IQ2_K
    };
    union {
        __m256i iqk_value[4];
        __m256i sbmask[2]; //0:smask; 1:bmask
    };
};

struct ConstInfo cl;

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    ggml_half d;    // super-block scale for quantized scales
    ggml_half dmin; // super-block scale for quantized mins
} iqk_block_q2_K;
template <int nrc_y>
static inline void q2k_deq_new_block(int i, const block_q8_K **q8, struct BlockInfo * deq, __m256 * accm, __m256i * scales) {
    const iqk_block_q2_K * x = (const iqk_block_q2_K *) deq->x;
    deq->d = GGML_FP16_TO_FP32(x[i].d);
    const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
    const __m128i scales8 = _mm_and_si128(mins_and_scales, cl.m4);
    const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), cl.m4);
    process_mins_16<nrc_y>(_mm256_cvtepi8_epi16(mins8), q8, i, -GGML_FP16_TO_FP32(x[i].dmin), accm);
    prepare_scales_16(_mm256_cvtepi8_epi16(scales8), scales);
}

static inline void q2k_prepare(int i, int j, struct BlockInfo * deq) {
    const uint8_t * qs = ((const block_q2_K *)deq->x)[i].qs;
    // bits.prepare(x[i].qs, j);
    __m256i q2bits = _mm256_loadu_si256((const __m256i *) qs + j);
    deq->values[0] = _mm256_and_si256(q2bits, cl.ml);
    deq->values[1] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), cl.ml);
    deq->values[2] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), cl.ml);
    deq->values[3] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), cl.ml);
}

// struct DequantizerQ2K final : public BaseDequantizer<block_q2_K>
template <int nrc_y>
static void q2k_mul_mat(int n, const void * vx, size_t bx, struct DataInfo * info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;
    //template <int nrc, typename block_q8 = block_q8_K> struct Q8
    // Q8<nrc_y> q8(info);
    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }

    __m256i all_scales[2];
    __m256i scales[4];
    __m256  accd[nrc_y];

    //Dequantizer deq(vx, bx);
    struct BlockInfo deq;
    for (int ix = 0; ix < nrc_x; ++ix) {
        // deq.new_row(ix);
        deq.x = (const void *)((const char *)vx + bx*ix);
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            // deq.new_block(i, q8, accd, all_scales);
            q2k_deq_new_block<nrc_y>(i, q8, &deq, accd, all_scales);
            __m256i sumi[nrc_y];
            for (int j = 0; j < QK_K/128; ++j) {
                // deq.prepare(i, j);
                q2k_prepare(i, j, &deq);
                iqk_set_scales_16(all_scales[j], scales);
                // multiply_add(deq.bits, scales, j, i, q8, sumi);
                iqk_multiply_add<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                // accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8.scale(iy, i)), _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8[iy][i].d), _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            // info.store(ix, iy, hsum_float_8(accd[iy]));
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

template <int nrc_y>
static inline void q3k_deq_new_block(int i, const block_q8_K **q8, struct BlockInfo * deq, __m256 * accm, __m256i * scales) {
    const block_q3_K * x = (const block_q3_K *) deq->x;
    deq->d = GGML_FP16_TO_FP32(x[i].d);
    // hbits.load();
    deq->hbits = _mm256_loadu_si256((const __m256i *)x[i].hmask);
    // sc3.make_scales((const uint16_t *)x[i].scales)
    const uint16_t * scales16 = (const uint16_t *)x[i].scales;
    uint32_t aux0 = scales16[0] | (scales16[1] << 16);
    uint32_t aux1 = scales16[2] | (scales16[3] << 16);
    uint32_t aux2 = scales16[4] | (scales16[5] << 16);
    __m128i scales128 = _mm_set_epi32(
        ((aux1 >> 4) & 0x0f0f0f0f) | ((aux2 >> 2) & 0x30303030),
        ((aux0 >> 4) & 0x0f0f0f0f) | ((aux2 >> 0) & 0x30303030),
         (aux1       & 0x0f0f0f0f) | ((aux2 << 2) & 0x30303030),
         (aux0       & 0x0f0f0f0f) | ((aux2 << 4) & 0x30303030));
    process_mins_and_scales_16<nrc_y>(_mm_add_epi8(scales128, cl.m32), q8, i, -4.f*deq->d, accm, scales);
}

static inline void q3k_prepare(int i, int j, struct BlockInfo * deq) {
    const uint8_t * qs = ((const block_q3_K *)deq->x)[i].qs;
    __m256i q2bits = _mm256_loadu_si256((const __m256i *) qs + j);
    deq->values[0] = _mm256_and_si256(q2bits, cl.ml);
    deq->values[1] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), cl.ml);
    deq->values[2] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), cl.ml);
    deq->values[3] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), cl.ml);
    // hbits.apply(bits, j == 0);
    deq->values[0] = _mm256_or_si256(deq->values[0], _mm256_and_si256(_mm256_slli_epi16(deq->hbits, 2), cl.mh));
    deq->values[1] = _mm256_or_si256(deq->values[1], _mm256_and_si256(_mm256_slli_epi16(deq->hbits, 1), cl.mh));
    deq->values[2] = _mm256_or_si256(deq->values[2], _mm256_and_si256(deq->hbits, cl.mh));
    deq->values[3] = _mm256_or_si256(deq->values[3], _mm256_and_si256(_mm256_srli_epi16(deq->hbits, 1), cl.mh));
    if (j == 0) deq->hbits = _mm256_srli_epi16(deq->hbits, 4);
}

// struct DequantizerQ3K final : public BaseDequantizer<block_q3_K>
template <int nrc_y>
static void q3k_mul_mat(int n, const void * vx, size_t bx, struct DataInfo * info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;
    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }

    __m256i all_scales[2];
    __m256i scales[4];
    __m256  accd[nrc_y];

    struct BlockInfo deq;
    for (int ix = 0; ix < nrc_x; ++ix) {
        deq.x = (const void *)((const char *)vx + bx*ix);
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            q3k_deq_new_block<nrc_y>(i, q8, &deq, accd, all_scales);
            __m256i sumi[nrc_y];
            for (int j = 0; j < QK_K/128; ++j) {
                q3k_prepare(i, j, &deq);
                iqk_set_scales_16(all_scales[j], scales);
                iqk_multiply_add<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8[iy][i].d), _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

static inline void q4k_avx2_prepare(int i, int j, struct BlockInfo * deq) {
    const uint8_t * qs = ((const block_q4_K *)deq->x)[i].qs;
    // bits.prepare(x[i].qs, j);
    __m256i q4bits = _mm256_loadu_si256((const __m256i*)qs + 2*j+0);
    deq->values[0] = _mm256_and_si256(q4bits, cl.ml);
    deq->values[1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), cl.ml);
    q4bits = _mm256_loadu_si256((const __m256i*)qs + 2*j+1);
    deq->values[2] = _mm256_and_si256(q4bits, cl.ml);
    deq->values[3] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), cl.ml);
}

typedef struct {
    ggml_half d;    // super-block scale for quantized scales
    ggml_half dmin; // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} iqk_block_q4_K;
// struct DequantizerQ4K_AVX2 final : public BaseDequantizer<block_q4_K>
template <int nrc_y>
static void q4k_mul_mat_X4(int n, const void * vx, size_t bx, struct DataInfo * info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    // Q8<nrc_y, block_q8_2_x4> q8(info);
    const block_q8_2_x4 * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_2_x4 *)(info->cy + (info->cur_y + iy)*info->by);
    }

    struct BlockInfo deq;

    uint32_t utmp[4];
    __m256  accd[nrc_y];
    __m256  scales[2];
    float   d8[8*nrc_y];

    for (int ix = 0; ix < nrc_x; ++ix) {
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        deq.x = (const void *)((const char *)vx + bx*ix);
        for (int i = 0; i < nb; ++i) {
            const iqk_block_q4_K * deq_x = (const iqk_block_q4_K *)deq.x;
            deq.d = GGML_FP16_TO_FP32(deq_x[i].d);
            __m256 vm = _mm256_cvtph_ps(_mm_set1_epi16(deq_x[i].dmin));
            iqk_make_q4_scales(deq_x[i].scales, utmp);
            __m256 mins = _mm256_mul_ps(vm, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(utmp + 2)))));
            mins = _mm256_mul_ps(_mm256_set1_ps(-1.f), mins);
            for (int iy = 0; iy < nrc_y; ++iy) {
                __m128i d4_1 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q8[iy][2*i+0].d)));
                __m128i d4_2 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q8[iy][2*i+1].d)));
                __m256 dy = _mm256_castsi256_ps(_mm256_slli_epi32(MM256_SET_M128I(d4_2, d4_1), 16));
                _mm256_storeu_ps(d8 + 8*iy, dy);
                __m128i m4_1 = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8[iy][2*i+0].d+4)));
                __m128i m4_2 = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8[iy][2*i+1].d+4)));
                __m256i myi  = MM256_SET_M128I(m4_2, m4_1);
                __m256 my   = _mm256_mul_ps(dy, _mm256_cvtepi32_ps(myi));
                accd[iy]  = _mm256_fmadd_ps(my, mins, accd[iy]);
            }

            __m256 all_scales = _mm256_mul_ps(_mm256_set1_ps(deq.d), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)utmp))));
            scales[0] = _mm256_set_m128(_mm256_castps256_ps128(all_scales), _mm256_castps256_ps128(all_scales));
            __m128 scales_h = _mm256_extractf128_ps(all_scales, 1);
            scales[1] = _mm256_set_m128(scales_h, scales_h);

            for (int j = 0; j < QK_K/128; ++j) {
                // deq.prepare(i, j);
                q4k_avx2_prepare(i, j, &deq);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    // const block_q8_2_x4& y = q8.y[iy][2*i+j];
                    const block_q8_2_x4* y = &q8[iy][2 * i + j];
                    __m256i sumi1 = _mm256_maddubs_epi16(deq.values[0], _mm256_loadu_si256((const __m256i*)y->qs+0));
                    __m256i sumi2 = _mm256_maddubs_epi16(deq.values[1], _mm256_loadu_si256((const __m256i*)y->qs+1));
                    __m256i sumi3 = _mm256_maddubs_epi16(deq.values[2], _mm256_loadu_si256((const __m256i*)y->qs+2));
                    __m256i sumi4 = _mm256_maddubs_epi16(deq.values[3], _mm256_loadu_si256((const __m256i*)y->qs+3));
                    sumi1 = _mm256_add_epi16(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2));
                    sumi3 = _mm256_add_epi16(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4));
                    sumi1 = _mm256_add_epi16(_mm256_unpacklo_epi64(sumi1, sumi3), _mm256_unpackhi_epi64(sumi1, sumi3));
                    sumi1 = _mm256_madd_epi16(_mm256_set1_epi16(1), sumi1);
                    __m128 dy4 = _mm_loadu_ps(d8 + 8*iy + 4*j);
                    __m256 d4d8 = _mm256_mul_ps(scales[j], _mm256_set_m128(dy4, dy4));
                    accd[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi1), accd[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

static inline void q5k_avx2_prepare(int i, int j, struct BlockInfo * deq) {
    const block_q5_K * deq_x = (const block_q5_K *)deq->x;
    const uint8_t * qs = deq_x[i].qs;
    __m256i q4bits = _mm256_loadu_si256((const __m256i*)qs + 2*j+0);
    deq->values[0] = _mm256_and_si256(q4bits, cl.ml);
    deq->values[1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), cl.ml);
    q4bits = _mm256_loadu_si256((const __m256i*)qs + 2*j+1);
    deq->values[2] = _mm256_and_si256(q4bits, cl.ml);
    deq->values[3] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), cl.ml);
    deq->hbits = j == 0 ? _mm256_loadu_si256((const __m256i *)deq_x[i].qh) : _mm256_srli_epi16(deq->hbits, 4);
    // apply_hbits();
    deq->values[0] = _mm256_or_si256(deq->values[0], _mm256_and_si256(_mm256_slli_epi16(deq->hbits, 4), cl.mh));
    deq->values[1] = _mm256_or_si256(deq->values[1], _mm256_and_si256(_mm256_slli_epi16(deq->hbits, 3), cl.mh));
    deq->values[2] = _mm256_or_si256(deq->values[2], _mm256_and_si256(_mm256_slli_epi16(deq->hbits, 2), cl.mh));
    deq->values[3] = _mm256_or_si256(deq->values[3], _mm256_and_si256(_mm256_slli_epi16(deq->hbits, 1), cl.mh));
}

typedef struct {
    ggml_half d;    // super-block scale for quantized scales
    ggml_half dmin; // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} iqk_block_q5_K;
// struct DequantizerQ5K_AVX2 final : public BaseDequantizer<block_q5_K>
template <int nrc_y>
static void q5k_mul_mat_X4(int n, const void * vx, size_t bx, struct DataInfo * info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    const block_q8_2_x4 * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_2_x4 *)(info->cy + (info->cur_y + iy)*info->by);
    }

    struct BlockInfo deq;

    uint32_t utmp[4];
    __m256  accd[nrc_y];
    __m256  scales[2];
    float   d8[8*nrc_y];

    for (int ix = 0; ix < nrc_x; ++ix) {
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        deq.x = (const void *)((const char *)vx + bx*ix);
        for (int i = 0; i < nb; ++i) {
            const iqk_block_q5_K * deq_x = (const iqk_block_q5_K *)deq.x;
            deq.d = GGML_FP16_TO_FP32(deq_x[i].d);
            __m256 vm = _mm256_cvtph_ps(_mm_set1_epi16(deq_x[i].dmin));
            iqk_make_q4_scales(deq_x[i].scales, utmp);
            __m256 mins = _mm256_mul_ps(vm, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(utmp + 2)))));
            mins = _mm256_mul_ps(_mm256_set1_ps(-1.f), mins);
            for (int iy = 0; iy < nrc_y; ++iy) {
                __m128i d4_1 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q8[iy][2*i+0].d)));
                __m128i d4_2 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q8[iy][2*i+1].d)));
                __m256 dy = _mm256_castsi256_ps(_mm256_slli_epi32(MM256_SET_M128I(d4_2, d4_1), 16));
                _mm256_storeu_ps(d8 + 8*iy, dy);
                __m128i m4_1 = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8[iy][2*i+0].d+4)));
                __m128i m4_2 = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8[iy][2*i+1].d+4)));
                __m256i myi  = MM256_SET_M128I(m4_2, m4_1);
                __m256 my   = _mm256_mul_ps(dy, _mm256_cvtepi32_ps(myi));
                accd[iy]  = _mm256_fmadd_ps(my, mins, accd[iy]);
            }

            __m256 all_scales = _mm256_mul_ps(_mm256_set1_ps(deq.d), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)utmp))));
            scales[0] = _mm256_set_m128(_mm256_castps256_ps128(all_scales), _mm256_castps256_ps128(all_scales));
            __m128 scales_h = _mm256_extractf128_ps(all_scales, 1);
            scales[1] = _mm256_set_m128(scales_h, scales_h);

            for (int j = 0; j < QK_K/128; ++j) {
                q5k_avx2_prepare(i, j, &deq);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    // const block_q8_2_x4& y = q8.y[iy][2*i+j];
                    const block_q8_2_x4* y = &q8[iy][2 * i + j];
                    __m256i sumi1 = _mm256_maddubs_epi16(deq.values[0], _mm256_loadu_si256((const __m256i*)y->qs+0));
                    __m256i sumi2 = _mm256_maddubs_epi16(deq.values[1], _mm256_loadu_si256((const __m256i*)y->qs+1));
                    __m256i sumi3 = _mm256_maddubs_epi16(deq.values[2], _mm256_loadu_si256((const __m256i*)y->qs+2));
                    __m256i sumi4 = _mm256_maddubs_epi16(deq.values[3], _mm256_loadu_si256((const __m256i*)y->qs+3));
                    sumi1 = _mm256_add_epi16(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2));
                    sumi3 = _mm256_add_epi16(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4));
                    sumi1 = _mm256_add_epi16(_mm256_unpacklo_epi64(sumi1, sumi3), _mm256_unpackhi_epi64(sumi1, sumi3));
                    sumi1 = _mm256_madd_epi16(_mm256_set1_epi16(1), sumi1);
                    __m128 dy4 = _mm_loadu_ps(d8 + 8*iy + 4*j);
                    __m256 d4d8 = _mm256_mul_ps(scales[j], _mm256_set_m128(dy4, dy4));
                    accd[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi1), accd[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

static inline void q6k_avx2_prepare(int i, int j, struct BlockInfo * deq, __m256i * us) {
    const block_q6_K * deq_x = (const block_q6_K *)deq->x;
    __m256i lbits1 = _mm256_loadu_si256((const __m256i *)deq_x[i].ql + 2*j+0);
    __m256i lbits2 = _mm256_loadu_si256((const __m256i *)deq_x[i].ql + 2*j+1);
    __m256i hbits  = _mm256_loadu_si256((const __m256i *)deq_x[i].qh + j);
    deq->values[0] = _mm256_or_si256(_mm256_and_si256(lbits1, cl.ml), _mm256_and_si256(_mm256_slli_epi16(hbits, 4), cl.mh));
    deq->values[1] = _mm256_or_si256(_mm256_and_si256(lbits2, cl.ml), _mm256_and_si256(_mm256_slli_epi16(hbits, 2), cl.mh));
    deq->values[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits1, 4), cl.ml), _mm256_and_si256(hbits, cl.mh));
    deq->values[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits2, 4), cl.ml), _mm256_and_si256(_mm256_srli_epi16(hbits, 2), cl.mh));
    for (int k = 0; k < 4; ++k) {
        deq->values[k] = _mm256_add_epi8(deq->values[k], _mm256_set1_epi8(-32));
        us[k] = _mm256_sign_epi8(deq->values[k], deq->values[k]);
    }
}

// struct DequantizerQ6K_AVX2 final : public BaseDequantizer<block_q6_K>
template <int nrc_y>
static void q6k_mul_mat_X4(int n, const void * vx, size_t bx, struct DataInfo * info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    const block_q8_2_x4 * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_2_x4 *)(info->cy + (info->cur_y + iy)*info->by);
    }

    struct BlockInfo deq;

    __m256  accd[nrc_y];
    __m256  scales[2];
    float   d8[8*nrc_y];
    __m256i us[4];

    uint8_t k_shuff[32] = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};
    __m256i shuff = _mm256_loadu_si256((const __m256i *)k_shuff);

    for (int ix = 0; ix < nrc_x; ++ix) {
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        deq.x = (const void *)((const char *)vx + bx*ix);
        for (int i = 0; i < nb; ++i) {
            const block_q6_K * deq_x = (const block_q6_K *)deq.x;
            deq.d = GGML_FP16_TO_FP32(deq_x[i].d);
            __m256 vd = _mm256_set1_ps(deq.d);
            // auto sc16 = (deq.make_scales(i), shuff);
            __m256i sc16 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)deq_x[i].scales)), shuff);
            scales[0] = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(sc16))));
            scales[1] = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sc16, 1))));
            for (int iy = 0; iy < nrc_y; ++iy) {
                __m128i d4_1 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q8[iy][2*i+0].d)));
                __m128i d4_2 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q8[iy][2*i+1].d)));
                __m256 dy = _mm256_castsi256_ps(_mm256_slli_epi32(MM256_SET_M128I(d4_2, d4_1), 16));
                if constexpr (nrc_y == 1) {
                    __m128 dyh = _mm256_extractf128_ps(dy, 1);
                    scales[0] = _mm256_mul_ps(scales[0], _mm256_set_m128(_mm256_castps256_ps128(dy), _mm256_castps256_ps128(dy)));
                    scales[1] = _mm256_mul_ps(scales[1], _mm256_set_m128(dyh, dyh));
                } else {
                    _mm256_storeu_ps(d8 + 8*iy, dy);
                }
            }
            for (int j = 0; j < QK_K/128; ++j) {
                // prepare(i, j);
                q6k_avx2_prepare(i, j, &deq, us);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const int8_t * qs = q8[iy][2*i+j].qs;
                    __m256i sumi1 = _mm256_maddubs_epi16(us[0], _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)qs+0), deq.values[0]));
                    __m256i sumi2 = _mm256_maddubs_epi16(us[1], _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)qs+1), deq.values[1]));
                    __m256i sumi3 = _mm256_maddubs_epi16(us[2], _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)qs+2), deq.values[2]));
                    __m256i sumi4 = _mm256_maddubs_epi16(us[3], _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)qs+3), deq.values[3]));
                    sumi1 = _mm256_add_epi16(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2));
                    sumi3 = _mm256_add_epi16(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4));
                    sumi1 = _mm256_add_epi16(_mm256_unpacklo_epi64(sumi1, sumi3), _mm256_unpackhi_epi64(sumi1, sumi3));
                    sumi1 = _mm256_madd_epi16(_mm256_set1_epi16(1), sumi1);
                    if constexpr (nrc_y > 1) {
                        __m128 dy4 = _mm_loadu_ps(d8 + 8*iy + 4*j);
                        __m256 d4d8 = _mm256_mul_ps(scales[j], _mm256_set_m128(dy4, dy4));
                        accd[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi1), accd[iy]);
                    } else {
                        accd[iy] = _mm256_fmadd_ps(scales[j], _mm256_cvtepi32_ps(sumi1), accd[iy]);
                    }
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

template <int nrc_y>
static inline __m256i iq4xs_deq_new_block(int i, const block_q8_K **q8, struct BlockInfo * deq, __m256 * accm) {
    const block_iq4_xs * x = (const block_iq4_xs *) deq->x;
    deq->d = GGML_FP16_TO_FP32(x[i].d);
    // auto scales128 = siq4.make_scales(*(const uint32_t *)x[i].scales_l, x[i].scales_h);
    uint32_t tmp32 = x[i].scales_h | (x[i].scales_h << 14);
    const __m128i sh = _mm_slli_epi16(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(tmp32), cl.hshift), cl.hmask), 4);
    const __m128i sl = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(*(const uint32_t *)x[i].scales_l), cl.lshift), cl.lmask);
    __m128i scales128 = _mm_add_epi16(_mm_or_si128(sh, _mm_cvtepi8_epi16(_mm_shuffle_epi8(sl, cl.lshuffle))), cl.m32);
    // s8k.accum_mins(scales128, q8, i, -128.f*d, accd);
    const __m256i mins = MM256_SET_M128I(_mm_shuffle_epi8(scales128, cl.shuffles[1]), _mm_shuffle_epi8(scales128, cl.shuffles[0]));
    for (int iy = 0; iy < nrc_y; ++iy) {
        const __m256i q8s = _mm256_loadu_si256((const __m256i*)q8[iy][i].bsums);
        const __m256i prod = _mm256_madd_epi16(mins, q8s);
        accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(-128.f*deq->d*q8[iy][i].d), _mm256_cvtepi32_ps(prod), accm[iy]);
    }
    return _mm256_broadcastsi128_si256(scales128);
}

static inline __m256i load_iq4nl_values(void) {
    static const uint8_t kvalues_iq4nl[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};
    __m128i val128 = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
    return _mm256_broadcastsi128_si256(val128);
}

static inline void iq4xs_prepare(int i, int j, struct BlockInfo * deq) {
    const uint8_t * qs = ((const block_iq4_xs *)deq->x)[i].qs;
    deq->values[0] = _mm256_shuffle_epi8(cl.iqk_values, iqk_dequant16(qs + 64*j +  0, cl.ml));
    deq->values[1] = _mm256_shuffle_epi8(cl.iqk_values, iqk_dequant16(qs + 64*j + 16, cl.ml));
    deq->values[2] = _mm256_shuffle_epi8(cl.iqk_values, iqk_dequant16(qs + 64*j + 32, cl.ml));
    deq->values[3] = _mm256_shuffle_epi8(cl.iqk_values, iqk_dequant16(qs + 64*j + 48, cl.ml));
}

// struct DequantizerIQ4XS final : public BaseDequantizer<block_iq4_xs>
// DequantizerIQ4XS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_iq4nl_values_256()) {}
template <int nrc_y>
static void iq4xs_mul_mat(int n, const void * vx, size_t bx, struct DataInfo * info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }
    struct BlockInfo deq;

    __m256  accd[nrc_y];
    __m256i scales[4];

    for (int ix = 0; ix < nrc_x; ++ix) {
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        deq.x = (const void *)((const char *)vx + bx*ix);
        for (int i = 0; i < nb; ++i) {
            // auto all_scales = deq.new_block(i, q8, accd);
            __m256i all_scales = iq4xs_deq_new_block<nrc_y>(i, q8, &deq, accd);
            __m256i sumi[nrc_y];
            for (int j = 0; j < QK_K/128; ++j) {
                // bits.prepare16(x[i].qs, j);
                iq4xs_prepare(i, j, &deq);
                // set_scales_8(all_scales, j, scales);
                iqk_set_scales_8(all_scales, j, scales);
                iqk_multiply_add<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(deq.d*q8[iy][i].d);
                accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

static inline __m256i load_iq2nl_values(void) {
    static const uint8_t kvalues_iq2nl[16] = {1, 19, 33, 49, 0, 0, 0, 0,  6, 24, 38, 54, 0, 0, 0, 0};
    __m128i val128 = _mm_loadu_si128((const __m128i *)kvalues_iq2nl);
    return _mm256_broadcastsi128_si256(val128);
}

static inline __m128i iq2ks_make_scales(const uint8_t * scales_l, uint8_t scales_h) {
    const uint16_t * scales = (const uint16_t *)scales_l;
    uint32_t aux32 = scales[0] | ((uint32_t)scales[1] << 16);
    __m128i scl = _mm_srlv_epi32(_mm_set1_epi32(aux32), cl.shift);
    scl = _mm_and_si128(_mm_shuffle_epi8(scl, cl.shuffle), _mm_set1_epi8(0xf));
    __m128i sch = _mm_set1_epi8(scales_h);
    sch = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(sch, cl.hmask), _mm_setzero_si128()), cl.m16);
    return _mm_cvtepi8_epi16(_mm_add_epi8(scl, sch));
}

template <int nrc_y>
static inline __m256i iq2ks_deq_new_block(int i, const block_q8_K **q8, struct BlockInfo * deq, __m256 * accm) {
    const block_iq2_ks * x = (const block_iq2_ks *) deq->x;
    __m128i scales128 = iq2ks_make_scales(x[i].scales, x[i].extra >> 8);
    __m128i shifts = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi8(x[i].extra), cl.hmask), cl.hmask), cl.m5);
    __m128i scales_s = _mm_mullo_epi16(scales128, _mm_cvtepi8_epi16(_mm_add_epi8(cl.m32, shifts)));
    // s8k.accum_mins(scales_s, q8, i, deq->d, accm);
    const __m256i mins = MM256_SET_M128I(_mm_shuffle_epi8(scales_s, cl.shuffles[1]), _mm_shuffle_epi8(scales_s, cl.shuffles[0]));
    for (int iy = 0; iy < nrc_y; ++iy) {
        const __m256i q8s = _mm256_loadu_si256((const __m256i*)q8[iy][i].bsums);
        const __m256i prod = _mm256_madd_epi16(mins, q8s);
        accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq->d*q8[iy][i].d), _mm256_cvtepi32_ps(prod), accm[iy]);
    }
    return _mm256_broadcastsi128_si256(scales128);
}

static inline void iq2ks_prepare(int i, int j, struct BlockInfo * deq) {
    const uint8_t * qs = ((const block_iq2_ks *)deq->x)[i].qs;
    __m256i q2bits = _mm256_loadu_si256((const __m256i *)qs + j);
    deq->values[0] = _mm256_shuffle_epi8(cl.iqk_values, _mm256_and_si256(q2bits, cl.ml));
    deq->values[1] = _mm256_shuffle_epi8(cl.iqk_values, _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), cl.ml));
    deq->values[2] = _mm256_shuffle_epi8(cl.iqk_values, _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), cl.ml));
    deq->values[3] = _mm256_shuffle_epi8(cl.iqk_values, _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), cl.ml));
}

// struct DequantizerIQ2KS final : public BaseDequantizer<block_iq2_ks, true, true>
// DequantizerIQ2KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
template <int nrc_y>
static void iq2ks_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }

    struct BlockInfo deq;
    __m256  accd[nrc_y];
    __m256i scales[4];

    for (int ix = 0; ix < nrc_x; ++ix) {
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        // deq.new_row(ix);
        const ggml_half * dptr = (const ggml_half *)((const char *)vx + bx*ix);
        deq.d = GGML_FP16_TO_FP32(*dptr);
        deq.x = (const void *)(dptr + 1);
        for (int i = 0; i < nb; ++i) {
            // auto all_scales = deq.new_block(i, q8, accd);
            __m256i all_scales = iq2ks_deq_new_block<nrc_y>(i, q8, &deq, accd);
            __m256i sumi[nrc_y];
            for (int j = 0; j < QK_K/128; ++j) {
                iq2ks_prepare(i, j, &deq);
                iqk_set_scales_8(all_scales, j, scales);
                iqk_multiply_add<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(deq.d*q8[iy][i].d);
                accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

static inline __m128i iq2k_make_scales(const uint8_t * scales_l) {
    uint64_t aux64; memcpy(&aux64, scales_l, 8);
    __m128i scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), cl.maskl);
    return _mm_add_epi8(scl, cl.m8);
}

template <int nrc_y>
static inline void iq2k_deq_new_block(int i, const block_q8_K **q8, struct BlockInfo * deq, __m256 * accm, __m256i * scales) {
    const block_iq2_k * x = (const block_iq2_k *) deq->x;
    deq->d = GGML_FP16_TO_FP32(x[i].d);
    // iqxk.process(i, deq->d, x[i].extra, make_scales(x[i].scales), q8, accm, scales);
    __m256i scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(iq2k_make_scales(x[i].scales), cl.hshuff));
    __m128i extra128 = _mm_set1_epi16(x[i].extra);
    extra128 = _mm_cmpeq_epi8(_mm_and_si128(extra128, cl.emask), cl.emask);
    extra128 = _mm_and_si128(extra128, cl.eshift);
    extra128 = _mm_shuffle_epi8(extra128, cl.eshuffle);
    __m256i scales_s = _mm256_mullo_epi16(scales16, _mm256_add_epi16(cl.min, _mm256_cvtepi8_epi16(extra128)));
    for (int iy = 0; iy < nrc_y; ++iy) {
        const __m256i prod  = _mm256_madd_epi16(scales_s, _mm256_loadu_si256((const __m256i*)q8[iy][i].bsums));
        accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq->d * q8[iy][i].d), _mm256_cvtepi32_ps(prod), accm[iy]);
    }
    prepare_scales_16(scales16, scales);
}

static inline void iq2k_prepare(int i, int j, struct BlockInfo * deq) {
    const uint8_t * qs = ((const block_iq2_k *)deq->x)[i].qs;
    __m256i q2bits = _mm256_loadu_si256((const __m256i *)qs + j);
    deq->values[0] = _mm256_shuffle_epi8(cl.iqk_values, _mm256_and_si256(q2bits, cl.ml));
    deq->values[1] = _mm256_shuffle_epi8(cl.iqk_values, _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), cl.ml));
    deq->values[2] = _mm256_shuffle_epi8(cl.iqk_values, _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), cl.ml));
    deq->values[3] = _mm256_shuffle_epi8(cl.iqk_values, _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), cl.ml));
}

// struct DequantizerIQ2K final : public BaseDequantizer<block_iq2_k>
// DequantizerIQ2K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(5, -32), values(load_values()) {}
template <int nrc_y>
static void iq2k_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }

    __m256i all_scales[2];
    __m256i scales[4];
    __m256  accd[nrc_y];

    struct BlockInfo deq;

    for (int ix = 0; ix < nrc_x; ++ix) {
        deq.x = (const void *)((const char *)vx + bx*ix);
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            // deq.new_block(i, q8, accd, all_scales);
            iq2k_deq_new_block<nrc_y>(i, q8, &deq, accd, all_scales);
            __m256i sumi[nrc_y];
            for (int j = 0; j < QK_K/128; ++j) {
                iq2k_prepare(i, j, &deq);
                iqk_set_scales_16(all_scales[j], scales);
                iqk_multiply_add<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8[iy][i].d), _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

static inline __m128i iq3k_make_scales(uint16_t signs, const uint8_t * scales_l) {
    uint64_t aux64; memcpy(&aux64, scales_l, 8);
    __m128i scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
    scl = _mm_add_epi8(_mm_slli_epi16(scl, 1), cl.m1);
    const __m128i sc_signs = _mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi16(signs), cl.sign_mask), cl.sign_mask);
    const __m128i sch = _mm_shuffle_epi8(_mm_or_si128(sc_signs, _mm_set1_epi8(1)), cl.hshuff);
    return _mm_sign_epi8(scl, sch);
}

template <int nrc_y>
static inline void iq3k_deq_new_block(int i, const block_q8_K **q8, struct BlockInfo * deq, __m256 * accm, __m256i * scales) {
    const block_iq3_k * x = (const block_iq3_k *) deq->x;
    deq->d = GGML_FP16_TO_FP32(x[i].d);
    // iqxk.process(i, d, x[i].extra, make_scales(), q8, accm, scales);
    __m256i scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(iq3k_make_scales(x[i].scales_h, x[i].scales_l), cl.hshuff));
    __m128i extra128 = _mm_set1_epi16(x[i].extra);
    extra128 = _mm_cmpeq_epi8(_mm_and_si128(extra128, cl.emask), cl.emask);
    extra128 = _mm_and_si128(extra128, cl.eshift);
    extra128 = _mm_shuffle_epi8(extra128, cl.eshuffle);
    __m256i scales_s = _mm256_mullo_epi16(scales16, _mm256_add_epi16(cl.min, _mm256_cvtepi8_epi16(extra128)));
    for (int iy = 0; iy < nrc_y; ++iy) {
        const __m256i prod  = _mm256_madd_epi16(scales_s, _mm256_loadu_si256((const __m256i*)q8[iy][i].bsums));
        accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq->d * q8[iy][i].d), _mm256_cvtepi32_ps(prod), accm[iy]);
    }
    prepare_scales_16(scales16, scales);
    deq->hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
}

static inline __m256i iq3k_load_values(void) {
    static const uint8_t kvalues_iq3nl[16] = {1, 24, 41, 54, 65, 77, 92, 111, 5, 28, 45, 58, 69, 81, 96, 115};
    __m128i val128 = _mm_loadu_si128((const __m128i *)kvalues_iq3nl);
    return _mm256_broadcastsi128_si256(val128);
}

static inline void iq3k_prepare(int i, int j, struct BlockInfo * deq) {
    const uint8_t * qs = ((const block_iq3_k *)deq->x)[i].qs;
    __m256i q2bits = _mm256_loadu_si256((const __m256i *)qs + j);
    deq->values[0] = _mm256_and_si256(q2bits, cl.ml);
    deq->values[1] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), cl.ml);
    deq->values[2] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), cl.ml);
    deq->values[3] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), cl.ml);

    __m256i h256 = j == 0 ? deq->hbits : _mm256_srli_epi16(deq->hbits, 4);
    deq->values[0] = _mm256_or_si256(deq->values[0], _mm256_and_si256(_mm256_slli_epi16(h256, 2), cl.hmask256));
    deq->values[1] = _mm256_or_si256(deq->values[1], _mm256_and_si256(_mm256_slli_epi16(h256, 1), cl.hmask256));
    deq->values[2] = _mm256_or_si256(deq->values[2], _mm256_and_si256(h256, cl.hmask256));
    deq->values[3] = _mm256_or_si256(deq->values[3], _mm256_and_si256(_mm256_srli_epi16(h256, 1), cl.hmask256));
    deq->values[0] = _mm256_shuffle_epi8(cl.iqk_values, deq->values[0]);
    deq->values[1] = _mm256_shuffle_epi8(cl.iqk_values, deq->values[1]);
    deq->values[2] = _mm256_shuffle_epi8(cl.iqk_values, deq->values[2]);
    deq->values[3] = _mm256_shuffle_epi8(cl.iqk_values, deq->values[3]);
}

// struct DequantizerIQ3K final : public BaseDequantizer<block_iq3_k>
// DequantizerIQ3K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(4, -64), values(load_values()) {}
template <int nrc_y>
static void iq3k_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }

    __m256i all_scales[2];
    __m256i scales[4];
    __m256  accd[nrc_y];

    struct BlockInfo deq;

    for (int ix = 0; ix < nrc_x; ++ix) {
        deq.x = (const void *)((const char *)vx + bx*ix);
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            iq3k_deq_new_block<nrc_y>(i, q8, &deq, accd, all_scales);

            __m256i sumi[nrc_y];

            for (int j = 0; j < QK_K/128; ++j) {
                iq3k_prepare(i, j, &deq);
                iqk_set_scales_16(all_scales[j], scales);
                iqk_multiply_add<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8[iy][i].d), _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

static inline __m256i iq3ks_deq_new_block(int i, struct BlockInfo * deq) {
    const block_iq3_ks * x = (const block_iq3_ks *) deq->x;
    uint32_t aux32; memcpy(&aux32, x[i].scales, 4);
    __m128i scl = _mm_cvtepi8_epi16(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(aux32), _mm_set_epi32(0, 0, 4, 0)), _mm_set1_epi8(0xf)));
    __m128i sch = _mm_cmpeq_epi16(_mm_and_si128(_mm_set1_epi16(x[i].extra), cl.mask), cl.mask);
    __m128i scales128 = _mm_add_epi16(scl, _mm_and_si128(sch, _mm_set1_epi16(16)));
    scales128 = _mm_sub_epi16(scales128, _mm_set1_epi16(16));
    return _mm256_broadcastsi128_si256(scales128);
}

static const int8_t iq3nl_values[16] = {-63, -40, -23, -10, 1, 13, 28,  47, -59, -36, -19,  -6, 5, 17, 32,  51};

static inline __m256i load_iq3nl_values(void) {
    __m128i val128 = _mm_loadu_si128((const __m128i *)iq3nl_values);
    return _mm256_broadcastsi128_si256(val128);
}

static inline void iq3ks_prepare(int i, int j, struct BlockInfo * deq) {
    const block_iq3_ks * x = (const block_iq3_ks *)deq->x;
    uint8_t extra = x[i].extra >> (8 + 4*j);
    deq->hbits = j == 0 ? _mm256_loadu_si256((const __m256i *)x[i].qh) : _mm256_srli_epi16(deq->hbits, 4);

    __m256i q2bits = _mm256_loadu_si256((const __m256i *)x[i].qs + j);
    deq->values[0] = _mm256_and_si256(q2bits, cl.ml);
    deq->values[1] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), cl.ml);
    deq->values[2] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), cl.ml);
    deq->values[3] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), cl.ml);

    deq->values[0] = _mm256_add_epi8(_mm256_set1_epi8((extra << 3) & 8), _mm256_or_si256(deq->values[0], _mm256_and_si256(_mm256_slli_epi16(deq->hbits, 2), cl.mh)));
    deq->values[1] = _mm256_add_epi8(_mm256_set1_epi8((extra << 2) & 8), _mm256_or_si256(deq->values[1], _mm256_and_si256(_mm256_slli_epi16(deq->hbits, 1), cl.mh)));
    deq->values[2] = _mm256_add_epi8(_mm256_set1_epi8((extra << 1) & 8), _mm256_or_si256(deq->values[2], _mm256_and_si256(deq->hbits, cl.mh)));
    deq->values[3] = _mm256_add_epi8(_mm256_set1_epi8((extra << 0) & 8), _mm256_or_si256(deq->values[3], _mm256_and_si256(_mm256_srli_epi16(deq->hbits, 1), cl.mh)));
    for (int k = 0; k < 4; ++k) deq->values[k] = _mm256_shuffle_epi8(cl.iqk_values, deq->values[k]);
}

// struct DequantizerIQ3KS final : public BaseDequantizer<block_iq3_ks, true, true>
// DequantizerIQ3KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
template <int nrc_y>
static void iq3ks_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }

    struct BlockInfo deq;

    __m256  accd[nrc_y];
    __m256i scales[4];

    for (int ix = 0; ix < nrc_x; ++ix) {
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        const ggml_half * dptr = (const ggml_half *)((const char *)vx + bx*ix);
        deq.d = GGML_FP16_TO_FP32(*dptr);
        deq.x = (const void *)(dptr + 1);
        for (int i = 0; i < nb; ++i) {
            __m256i all_scales = iq3ks_deq_new_block(i, &deq);
            __m256i sumi[nrc_y];
            for (int j = 0; j < QK_K/128; ++j) {
                iq3ks_prepare(i, j, &deq);
                iqk_set_scales_8(all_scales, j, scales);
                iqk_multiply_add_avx2<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(deq.d*q8[iy][i].d);
                accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

template <int nrc_y>
static inline __m256i iq4kss_deq_new_block(int i, const block_q8_K **q8, struct BlockInfo * deq, __m256 * accm) {
    // inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
    const block_iq4_kss * x = (const block_iq4_kss *)deq->x;
        union { __m256i vec; uint16_t val[16]; } helper;
        for (int k = 0; k < 4; ++k) {
            deq->data[k] = _mm256_loadu_si256((const __m256i *)x[i].qs + k);
            __m256i p = _mm256_and_si256(_mm256_cmpeq_epi16(_mm256_and_si256(deq->data[k], cl.m1_256), cl.m1_256), cl.sbmask[0]);
            p = _mm256_add_epi32(_mm256_unpackhi_epi64(p, p), p);
            p = _mm256_add_epi32(_mm256_shuffle_epi32(p, _MM_SHUFFLE(2, 3, 0, 1)), p);
            helper.vec = _mm256_hadd_epi16(p, p);
            deq->aux[2*k+0] = helper.val[0];
            deq->aux[2*k+1] = helper.val[8];
            deq->data[k] = _mm256_and_si256(deq->data[k], cl.sbmask[1]);
            deq->data[k] = _mm256_xor_si256(deq->data[k], _mm256_srli_epi16(deq->data[k], 1));
        }
        __m128i scales128 = _mm_loadu_si128((const __m128i *)deq->aux);
        __m128i shifts = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(scales128, _mm256_castsi256_si128(cl.m1_256)), _mm256_castsi256_si128(cl.m1_256)), cl.m4);
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, cl.mask), cl.m127);
        __m128i scales_s = _mm_mullo_epi16(scales128, _mm_add_epi16(cl.m128, shifts));

        const __m256i mins = MM256_SET_M128I(_mm_shuffle_epi8(scales_s, cl.shuffles[1]), _mm_shuffle_epi8(scales_s, cl.shuffles[0]));
        for (int iy = 0; iy < nrc_y; ++iy) {
            const __m256i q8s = _mm256_loadu_si256((const __m256i*)q8[iy][i].bsums);
            const __m256i prod = _mm256_madd_epi16(mins, q8s);
            accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq->d*q8[iy][i].d), _mm256_cvtepi32_ps(prod), accm[iy]);
        }
        return _mm256_broadcastsi128_si256(scales128);
}

static inline void iq4kss_prepare(int j, struct BlockInfo * deq) {
    for (int k = 0; k < 2; ++k) {
        __m128i p1 = _mm256_castsi256_si128(deq->data[2*j+k]);
        __m128i p2 = _mm256_extractf128_si256(deq->data[2*j+k], 1);
        deq->values[2*k+0] = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(p1, 4), p1), cl.ml);
        deq->values[2*k+0] = _mm256_shuffle_epi8(cl.iqk_values, deq->values[2*k+0]);
        deq->values[2*k+1] = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(p2, 4), p2), cl.ml);
        deq->values[2*k+1] = _mm256_shuffle_epi8(cl.iqk_values, deq->values[2*k+1]);
    }
}

// struct DequantizerIQ4KSS final : public BaseDequantizer<block_iq4_kss, true>
// DequantizerIQ4KSS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_iq4nl_values_256()) {}
template <int nrc_y>
static void iq4kss_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }

    struct BlockInfo deq;

    __m256  accd[nrc_y];
    __m256i scales[4];

    for (int ix = 0; ix < nrc_x; ++ix) {
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        const float * dptr = (const float *)((const char *)vx + bx*ix);
        deq.d = *dptr;
        deq.x = (const void *)(dptr + 1);
        for (int i = 0; i < nb; ++i) {
            __m256i all_scales = iq4kss_deq_new_block<nrc_y>(i, q8, &deq, accd);
            __m256i sumi[nrc_y];
            for (int j = 0; j < QK_K/128; ++j) {
                iq4kss_prepare(j, &deq);
                iqk_set_scales_8(all_scales, j, scales);
                iqk_multiply_add<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(deq.d*q8[iy][i].d);
                accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

static inline __m256i iq4ks_deq_new_block(int i, struct BlockInfo * deq) {
    const block_iq4_ks * x = (const block_iq4_ks *) deq->x;
    __m128i scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
    scales128 = _mm_add_epi16(_mm_and_si128(scales128, cl.mask), cl.m127);
    return _mm256_broadcastsi128_si256(scales128);
}

static const int8_t iq4k_values[32] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113, -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117};

static void iq4ks_load_values(__m256i * values) {
    __m128i v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
    __m128i v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
    values[0] = _mm256_broadcastsi128_si256(v1);
    values[1] = _mm256_broadcastsi128_si256(v2);
}

static inline void iq4ks_prepare(int i, int j, struct BlockInfo * deq) {
    const block_iq4_ks * x = (const block_iq4_ks *)deq->x;
    deq->values[0] = _mm256_shuffle_epi8(cl.iqk_value[x[i].scales[4*j+0] & 1], iqk_dequant16(x[i].qs + 64*j + 0, cl.ml));
    deq->values[1] = _mm256_shuffle_epi8(cl.iqk_value[x[i].scales[4*j+1] & 1], iqk_dequant16(x[i].qs + 64*j +16, cl.ml));
    deq->values[2] = _mm256_shuffle_epi8(cl.iqk_value[x[i].scales[4*j+2] & 1], iqk_dequant16(x[i].qs + 64*j +32, cl.ml));
    deq->values[3] = _mm256_shuffle_epi8(cl.iqk_value[x[i].scales[4*j+3] & 1], iqk_dequant16(x[i].qs + 64*j +48, cl.ml));
}

// struct DequantizerIQ4KS final : public BaseDequantizer<block_iq4_ks, true>
// DequantizerIQ4KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(); }
template <int nrc_y>
static void iq4ks_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }

    struct BlockInfo deq;

    __m256  accd[nrc_y];
    __m256i scales[4];

    for (int ix = 0; ix < nrc_x; ++ix) {
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        const float * dptr = (const float *)((const char *)vx + bx*ix);
        deq.d = *dptr;
        deq.x = (const void *)(dptr + 1);
        for (int i = 0; i < nb; ++i) {
            __m256i all_scales = iq4ks_deq_new_block(i, &deq);
            __m256i sumi[nrc_y];
            for (int j = 0; j < QK_K/128; ++j) {
                iq4ks_prepare(i, j, &deq);
                iqk_set_scales_8(all_scales, j, scales);
                iqk_multiply_add_avx2<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(deq.d*q8[iy][i].d);
                accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

static inline __m128i iq4k_make_scales(const uint8_t * scales_l, const uint16_t * scales_h) {
    uint64_t aux64;
    memcpy(&aux64, scales_l, 8);
    __m128i scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), cl.maskl);
    const uint32_t aux32 = scales_h[0] | (scales_h[1] << 16);
    __m128i aux = _mm_and_si128(_mm_set_epi32(aux32 >> 2, aux32, aux32 << 2, aux32 << 4), cl.maskh);
    __m128i sch = _mm_shuffle_epi8(aux, cl.hshuff);
    return _mm_add_epi8(_mm_or_si128(scl, sch), cl.m32);
}

static inline void iq4k_deq_new_block(int i, struct BlockInfo * deq, __m256i * scales) {
    const block_iq4_k * x = (const block_iq4_k *) deq->x;
    deq->d = GGML_FP16_TO_FP32(x[i].d);
    __m128i scales8 = iq4k_make_scales(x[i].scales_l, (const uint16_t *)x[i].scales_h);
    __m256i scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales8, cl.hshuff));
    prepare_scales_16(scales16, scales);
}

static void iq4k_load_values(__m256i * values) {
    __m128i v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
    __m128i v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
    values[0] = _mm256_broadcastsi128_si256(v1);
    values[1] = MM256_SET_M128I(v1, v2);
    values[2] = MM256_SET_M128I(v2, v1);
    values[3] = _mm256_broadcastsi128_si256(v2);
}

static inline void iq4k_prepare(int i, int j, struct BlockInfo * deq) {
    const block_iq4_k * x = (const block_iq4_k *)deq->x;
        uint16_t extra = x[i].extra >> 8*j;
        deq->values[0] = _mm256_shuffle_epi8(cl.iqk_value[extra & 3], iqk_dequant16(x[i].qs + 64*j + 0, cl.ml)); extra >>= 2;
        deq->values[1] = _mm256_shuffle_epi8(cl.iqk_value[extra & 3], iqk_dequant16(x[i].qs + 64*j +16, cl.ml)); extra >>= 2;
        deq->values[2] = _mm256_shuffle_epi8(cl.iqk_value[extra & 3], iqk_dequant16(x[i].qs + 64*j +32, cl.ml)); extra >>= 2;
        deq->values[3] = _mm256_shuffle_epi8(cl.iqk_value[extra & 3], iqk_dequant16(x[i].qs + 64*j +48, cl.ml));
}

// struct DequantizerIQ4K final : public BaseDequantizer<block_iq4_k>
// DequantizerIQ4K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(); }
template <int nrc_y>
static void iq4k_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }

    __m256i all_scales[2];
    __m256i scales[4];
    __m256  accd[nrc_y];

    struct BlockInfo deq;

    for (int ix = 0; ix < nrc_x; ++ix) {
        deq.x = (const void *)((const char *)vx + bx*ix);
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            iq4k_deq_new_block(i, &deq, all_scales);
            __m256i sumi[nrc_y];
            for (int j = 0; j < QK_K/128; ++j) {
                iq4k_prepare(i, j, &deq);
                iqk_set_scales_16(all_scales[j], scales);
                iqk_multiply_add_avx2<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8[iy][i].d), _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

template <int nrc_y>
static inline __m256i iq5ks_deq_new_block(int i, const block_q8_K **q8, struct BlockInfo * deq, __m256 * accm) {
    const block_iq5_ks * x = (const block_iq5_ks *)deq->x;
    deq->hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
    __m128i scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
    __m128i shifts = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(scales128, cl.m1), cl.m1), cl.m2);
    scales128 = _mm_add_epi16(_mm_and_si128(scales128, cl.mask), cl.m127);
    __m128i scales_s = _mm_mullo_epi16(scales128, _mm_add_epi16(cl.m128, shifts));
    const __m256i mins = MM256_SET_M128I(_mm_shuffle_epi8(scales_s, cl.shuffles[1]), _mm_shuffle_epi8(scales_s, cl.shuffles[0]));
    for (int iy = 0; iy < nrc_y; ++iy) {
        const __m256i q8s = _mm256_loadu_si256((const __m256i*)q8[iy][i].bsums);
        const __m256i prod = _mm256_madd_epi16(mins, q8s);
        accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq->d*q8[iy][i].d), _mm256_cvtepi32_ps(prod), accm[iy]);
    }
    return _mm256_broadcastsi128_si256(scales128);
}

static void iq5ks_load_values(__m256i * values) {
    static const uint8_t kvalues_iq5nl[32] = {
        2,  14,  25,  36,  45,  54,  63,  71,  78,  85,  92,  98, 104, 110, 116, 122, 127,
        133, 139, 145, 151, 157, 164, 171, 179, 187, 196, 205, 215, 225, 237, 249,
    };
    __m128i values128_1 = _mm_loadu_si128((const __m128i *)kvalues_iq5nl + 0);
    __m128i values128_2 = _mm_loadu_si128((const __m128i *)kvalues_iq5nl + 1);
    values[0] = _mm256_broadcastsi128_si256(values128_1);
    values[1] = _mm256_broadcastsi128_si256(values128_2);
}

static inline void iq5ks_prepare(int i, int j, struct BlockInfo * deq) {
    const uint8_t * qs = ((const block_iq5_ks *)deq->x)[i].qs;
    __m256i q4bits = _mm256_loadu_si256((const __m256i*)qs + 2*j+0);
    deq->values[0] = _mm256_and_si256(q4bits, cl.ml);
    deq->values[1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), cl.ml);
    q4bits = _mm256_loadu_si256((const __m256i*)qs + 2*j+1);
    deq->values[2] = _mm256_and_si256(q4bits, cl.ml);
    deq->values[3] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), cl.ml);
    __m256i h = j == 0 ? deq->hbits : _mm256_srli_epi16(deq->hbits, 4);
    for (int k = 0; k < 4; ++k) {
        __m256i qh = _mm256_and_si256(_mm256_slli_epi16(h, 7-k), cl.mh);
        __m256i q5vl = _mm256_or_si256(deq->values[k], qh);
        __m256i q5vh = _mm256_or_si256(deq->values[k], _mm256_xor_si256(qh, cl.mh));
        deq->values[k] = _mm256_or_si256(_mm256_shuffle_epi8(cl.iqk_value[0], q5vl), _mm256_shuffle_epi8(cl.iqk_value[1], q5vh));
    }
}

// struct DequantizerIQ5KS final : public BaseDequantizer<block_iq5_ks, true>
// DequantizerIQ5KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(values); }
template <int nrc_y>
static void iq5ks_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }

    struct BlockInfo deq;

    __m256  accd[nrc_y];
    __m256i scales[4];

    for (int ix = 0; ix < nrc_x; ++ix) {
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        const float * dptr = (const float *)((const char *)vx + bx*ix);
        deq.d = *dptr;
        deq.x = (const void *)(dptr + 1);
        for (int i = 0; i < nb; ++i) {
            __m256i all_scales = iq5ks_deq_new_block<nrc_y>(i, q8, &deq, accd);
            __m256i sumi[nrc_y];
            for (int j = 0; j < QK_K/128; ++j) {
                iq5ks_prepare(i, j, &deq);
                iqk_set_scales_8(all_scales, j, scales);
                iqk_multiply_add<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(deq.d*q8[iy][i].d);
                accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

static const int8_t iq5nl_values[64] = {
    -126, -114, -103, -92, -83, -74, -65, -57, -50, -43, -36, -30, -24, -18, -12, -6, -1, 5, 11, 17, 23, 29, 36, 43, 51, 59, 68, 77, 87, 97, 109, 121,
    -124, -112, -101, -90, -81, -72, -63, -55, -48, -41, -34, -28, -22, -16, -10, -4,  1, 7, 13, 19, 25, 31, 38, 45, 53, 61, 70, 79, 89, 99, 111, 123,
};

static void load_iq5nl_values(__m256i * values) {
    __m128i values128_1 = _mm_loadu_si128((const __m128i *)iq5nl_values + 0);
    __m128i values128_2 = _mm_loadu_si128((const __m128i *)iq5nl_values + 1);
    values[0] = _mm256_broadcastsi128_si256(values128_1);
    values[1] = _mm256_broadcastsi128_si256(values128_2);
}

static inline __m128i iq5k_make_scales(const uint8_t * scales_l, const uint16_t * scales_h) {
    uint64_t aux64;
    memcpy(&aux64, scales_l, 8);
    __m128i scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), cl.maskl);
    const uint32_t aux32 = scales_h[0] | (scales_h[1] << 16);
    __m128i aux = _mm_and_si128(_mm_set_epi32(aux32 >> 2, aux32, aux32 << 2, aux32 << 4), cl.maskh);
    __m128i sch = _mm_shuffle_epi8(aux, cl.hshuff);
    return _mm_add_epi8(_mm_or_si128(scl, sch), cl.m32);
}

template <int nrc_y>
static inline void iq5k_deq_new_block(int i, const block_q8_K **q8, struct BlockInfo * deq, __m256 * accm, __m256i * scales) {
    const block_iq5_k * x = (const block_iq5_k *) deq->x;
    deq->d = GGML_FP16_TO_FP32(x[i].d);
    __m256i scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(iq5k_make_scales(x[i].scales_l, (const uint16_t *)x[i].scales_h), cl.hshuff));
    __m128i extra128 = _mm_set1_epi16(x[i].extra);
    extra128 = _mm_cmpeq_epi8(_mm_and_si128(extra128, cl.emask), cl.emask);
    extra128 = _mm_and_si128(extra128, cl.eshift);
    extra128 = _mm_shuffle_epi8(extra128, cl.eshuffle);
    __m256i scales_s = _mm256_mullo_epi16(scales16, _mm256_add_epi16(cl.min, _mm256_cvtepi8_epi16(extra128)));
    for (int iy = 0; iy < nrc_y; ++iy) {
        const __m256i prod  = _mm256_madd_epi16(scales_s, _mm256_loadu_si256((const __m256i*)q8[iy][i].bsums));
        accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq->d * q8[iy][i].d), _mm256_cvtepi32_ps(prod), accm[iy]);
    }
    prepare_scales_16(scales16, scales);
    deq->hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
}

static inline void iq5k_prepare(int i, int j, struct BlockInfo * deq) {
    const uint8_t * qs = ((const block_iq5_k *)deq->x)[i].qs;
    __m256i q4bits = _mm256_loadu_si256((const __m256i*)qs + 2*j+0);
    deq->values[0] = _mm256_and_si256(q4bits, cl.ml);
    deq->values[1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), cl.ml);
    q4bits = _mm256_loadu_si256((const __m256i*)qs + 2*j+1);
    deq->values[2] = _mm256_and_si256(q4bits, cl.ml);
    deq->values[3] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), cl.ml);
    __m256i h = j == 0 ? deq->hbits : _mm256_srli_epi16(deq->hbits, 4);
    for (int k = 0; k < 4; ++k) {
        __m256i qh = _mm256_and_si256(_mm256_slli_epi16(h, 7-k), cl.mh);
        __m256i q5vl = _mm256_or_si256(deq->values[k], qh);
        __m256i q5vh = _mm256_or_si256(deq->values[k], _mm256_xor_si256(qh, cl.mh));
        deq->values[k] = _mm256_or_si256(_mm256_shuffle_epi8(cl.iqk_value[0], q5vl), _mm256_shuffle_epi8(cl.iqk_value[1], q5vh));
    }
}

// struct DequantizerIQ5K final : public BaseDequantizer<block_iq5_k>
// DequantizerIQ5K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(2, 0) { load_values(values); }
template <int nrc_y>
static void iq5k_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }

    __m256i all_scales[2];
    __m256i scales[4];
    __m256  accd[nrc_y];

    struct BlockInfo deq;

    for (int ix = 0; ix < nrc_x; ++ix) {
        deq.x = (const void *)((const char *)vx + bx*ix);
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            iq5k_deq_new_block<nrc_y>(i, q8, &deq, accd, all_scales);
            __m256i sumi[nrc_y];
            for (int j = 0; j < QK_K/128; ++j) {
                iq5k_prepare(i, j, &deq);
                iqk_set_scales_16(all_scales[j], scales);
                iqk_multiply_add_avx2<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8[iy][i].d), _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

static const int8_t iq6nl_values[128] = {-127, -121, -115, -109, -104,  -98,  -93,  -88,  -84,  -79,  -74,  -70,  -66,  -62,  -58,  -54,
    -51,  -47,  -44,  -40,  -37,  -34,  -31,  -28,  -25,  -22,  -19,  -16,  -13,  -11,   -8,   -5,
    -2,    0,    3,    6,    9,   12,   14,   17,   20,   23,   27,   30,   33,   36,   40,   44,
    47,   51,   55,   59,   63,   68,   72,   77,   82,   87,   92,   98,  103,  109,  115,  121,
-126, -120, -114, -108, -103,  -97,  -92,  -87,  -83,  -78,  -73,  -69,  -65,  -61,  -57,  -53,
    -50,  -46,  -43,  -39,  -36,  -33,  -30,  -27,  -24,  -21,  -18,  -15,  -12,  -10,   -7,   -4,
    -1,    1,    4,    7,   10,   13,   15,   18,   21,   24,   28,   31,   34,   37,   41,   45,
    48,   52,   56,   60,   64,   69,   73,   78,   83,   88,   93,   99,  104,  110,  116,  122,};

static void load_iq6nl_values(__m256i * values) {
    for (int k = 0; k < 4; ++k) {
        __m128i values128 = _mm_loadu_si128((const __m128i *)iq6nl_values + k);
        values[k] = _mm256_broadcastsi128_si256(values128);
    }
}

template <int nrc_y>
static inline void iq6k_deq_new_block(int i, const block_q8_K **q8, struct BlockInfo * deq, __m256 * accm, __m256i * scales) {
    const block_iq6_k * x = (const block_iq6_k *)deq->x;
    deq->d = GGML_FP16_TO_FP32(x[i].d);
    __m128i scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
    __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
    __m128i extra128 = _mm_set1_epi16(x[i].extra);
    extra128 = _mm_cmpeq_epi8(_mm_and_si128(extra128, cl.emask), cl.emask);
    extra128 = _mm_and_si128(extra128, cl.eshift);
    extra128 = _mm_shuffle_epi8(extra128, cl.eshuffle);
    __m256i scales_s = _mm256_mullo_epi16(scales16, _mm256_add_epi16(cl.min, _mm256_cvtepi8_epi16(extra128)));
    for (int iy = 0; iy < nrc_y; ++iy) {
        const __m256i prod  = _mm256_madd_epi16(scales_s, _mm256_loadu_si256((const __m256i*)q8[iy][i].bsums));
        accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq->d * q8[iy][i].d), _mm256_cvtepi32_ps(prod), accm[iy]);
    }
    prepare_scales_16(scales16, scales);
}

static inline void iq6k_prepare(int i, int j, struct BlockInfo * deq) {
    const block_iq6_k * x = (const block_iq6_k *)deq->x;
    const uint8_t * qs = x[i].qs;
    __m256i q4bits = _mm256_loadu_si256((const __m256i*)qs + 2*j+0);
    deq->values[0] = _mm256_and_si256(q4bits, cl.ml);
    deq->values[1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), cl.ml);
    q4bits = _mm256_loadu_si256((const __m256i*)qs + 2*j+1);
    deq->values[2] = _mm256_and_si256(q4bits, cl.ml);
    deq->values[3] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), cl.ml);

    __m256i hbits = _mm256_loadu_si256((const __m256i *)x[i].qh + j);
    for (int k = 0; k < 4; ++k) {
        __m256i l = deq->values[k];
        __m256i mask4 = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, cl.mh3), cl.mh3);
        __m256i h1 = _mm256_andnot_si256(mask4, hbits);
        __m256i mask2 = _mm256_cmpeq_epi8(_mm256_and_si256(h1, cl.mh1), cl.mh1);
        __m256i mask3 = _mm256_cmpeq_epi8(_mm256_and_si256(h1, cl.mh2), cl.mh2);
        __m256i mask1 = _mm256_andnot_si256(_mm256_or_si256(mask4, _mm256_or_si256(mask2, mask3)), _mm256_set1_epi8(-1)); // 0xff;
        deq->values[k] = _mm256_or_si256(
            _mm256_or_si256(_mm256_and_si256(mask1, _mm256_shuffle_epi8(cl.iqk_value[0], l)),
                            _mm256_and_si256(mask2, _mm256_shuffle_epi8(cl.iqk_value[1], l))),
            _mm256_or_si256(_mm256_and_si256(mask3, _mm256_shuffle_epi8(cl.iqk_value[2], l)),
                            _mm256_and_si256(mask4, _mm256_shuffle_epi8(cl.iqk_value[3], l)))
        );

        hbits = _mm256_srli_epi16(hbits, 2);
    }
}

// struct DequantizerIQ6K final : public BaseDequantizer<block_iq6_k>
// DequantizerIQ6K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(1, 0) { load_values(values); }
template <int nrc_y>
static void iq6k_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }

    __m256i all_scales[2];
    __m256i scales[4];
    __m256  accd[nrc_y];

    struct BlockInfo deq;

    for (int ix = 0; ix < nrc_x; ++ix) {
        deq.x = (const void *)((const char *)vx + bx*ix);
        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            iq6k_deq_new_block<nrc_y>(i, q8, &deq, accd, all_scales);
            __m256i sumi[nrc_y];
            for (int j = 0; j < QK_K/128; ++j) {
                iq6k_prepare(i, j, &deq);
                iqk_set_scales_16(all_scales[j], scales);
                iqk_multiply_add_avx2<nrc_y>(deq, scales, j, i, q8, sumi);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8[iy][i].d), _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

typedef struct {
    __m256i mka;
    __m256i shuffle;
    uint32_t ka3;
} Trellis3;

static inline Trellis3 make_Trellis3(int is_8) {
    Trellis3 t;
    static const uint32_t ka  = 0xCBAC1FEDu;
    static const uint32_t ka1 = ka  * ka;
    static const uint32_t ka2 = ka1 * ka;
    static const uint32_t ka3 = ka2 * ka;
    static const uint32_t ka4 = ka3 * ka;
    static const uint32_t ka5 = ka4 * ka;
    static const uint32_t ka6 = ka5 * ka;
    static const uint32_t ka7 = ka6 * ka;
    t.mka = is_8 ? _mm256_setr_epi32(ka, ka1, ka2, ka3, ka4, ka5, ka6, ka7) : _mm256_setr_epi32(ka, ka1, ka2, ka3, ka, ka1, ka2, ka3);
    t.shuffle = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    t.ka3 = ka3;
    return t;
}

__attribute__((__always_inline__)) inline void iq4kt_next_128(const uint32_t * val, __m256i * result, Trellis3 t) {
    // Even though we only have 16 vector registers nn AVX2, this is still faster
    __m256i aux[16];
    __m256i perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    for (int k = 0; k < 4; ++k) {
        __m256i v = _mm256_loadu_si256((const __m256i *)val + k);
        v = _mm256_permutevar8x32_epi32(v, perm);
        aux[4*k+0] = _mm256_shuffle_epi32(v, 0x00);
        aux[4*k+1] = _mm256_shuffle_epi32(v, 0x55);
        aux[4*k+2] = _mm256_shuffle_epi32(v, 0xaa);
        aux[4*k+3] = _mm256_shuffle_epi32(v, 0xff);
    }
    for (int i = 0; i < 16; ++i) {
        aux[i] = _mm256_mullo_epi32(aux[i], t.mka);
    }
    __m256i mask = _mm256_set1_epi32(0x3f3f3f3f);
    for (int i = 0; i < 16; ++i) {
        aux[i] = _mm256_and_si256(aux[i], mask);
    }
    __m256i offset = _mm256_set1_epi32(-126);
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__)
    __m256i m1     = _mm256_set1_epi32(0x01010101);
#endif
    for (int i = 0; i < 16; ++i) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__)
        aux[i] = _mm256_dpbusd_epi32(offset, aux[i], m1);
#else
        __m256i dot = _mm256_maddubs_epi16(aux[i], _mm256_set1_epi32(0x01010101));
        aux[i] = _mm256_add_epi32(offset, _mm256_madd_epi16(dot, _mm256_set1_epi16(1)));
#endif
    }
    for (int k = 0; k < 4; ++k) {
        __m256i v1 = _mm256_packs_epi32(aux[4*k+0], aux[4*k+1]);
        __m256i v2 = _mm256_packs_epi32(aux[4*k+2], aux[4*k+3]);
        result[k] = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(v1, v2), t.shuffle);
    }
}

__attribute__((__always_inline__)) inline void iq23kt_next_128(const uint16_t * val, uint32_t v0, __m256i * result, Trellis3 t) {
    // Even though we only have 16 vector registers nn AVX2, this is still faster
    __m256i aux[16];
    for (int k = 0; k < 4; ++k) {
        __m128i v128 = _mm_add_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(val + 4*k))), _mm_set1_epi32(v0));
        __m256i v = _mm256_broadcastsi128_si256(v128);
        aux[4*k+0] = _mm256_shuffle_epi32(v, 0x00);
        aux[4*k+1] = _mm256_shuffle_epi32(v, 0x55);
        aux[4*k+2] = _mm256_shuffle_epi32(v, 0xaa);
        aux[4*k+3] = _mm256_shuffle_epi32(v, 0xff);
    }
    for (int i = 0; i < 16; ++i) {
        aux[i] = _mm256_mullo_epi32(aux[i], t.mka);
    }
    __m256i mask = _mm256_set1_epi32(0x3f3f3f3f);
    for (int i = 0; i < 16; ++i) {
        aux[i] = _mm256_and_si256(aux[i], mask);
    }
    __m256i offset = _mm256_set1_epi32(-126);
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__)
    __m256i m1     = _mm256_set1_epi32(0x01010101);
#endif
    for (int i = 0; i < 16; ++i) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__)
        aux[i] = _mm256_dpbusd_epi32(offset, aux[i], m1);
#else
        __m256i dot = _mm256_maddubs_epi16(aux[i], _mm256_set1_epi32(0x01010101));
        aux[i] = _mm256_add_epi32(offset, _mm256_madd_epi16(dot, _mm256_set1_epi16(1)));
#endif
    }
    for (int k = 0; k < 4; ++k) {
        __m256i v1 = _mm256_packs_epi32(aux[4*k+0], aux[4*k+1]);
        __m256i v2 = _mm256_packs_epi32(aux[4*k+2], aux[4*k+3]);
        result[k] = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(v1, v2), t.shuffle);
    }
}

// void mul_mat_iq2_kt_q8_2_x4_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x)
//Trellis3<true, false> trellis;
template <int nrc_y>
static void iq2kt_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis3 trellis = make_Trellis3(true);

    __m128i shifts = _mm_set_epi32(0, 0, 4, 0);
    __m128i values = _mm_loadu_si128((const __m128i *)iq4k_values);

    __m256  accd[nrc_y];
    const block_q8_2_x4 * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = (const block_q8_2_x4 *)info->cy + (info->cur_y + iy)*info->by;
    }

    __m256i  xv[4], dot[4];
    __m256   scales[2];

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        __m256 d = _mm256_set1_ps(dptr[0] * 1.05f);
        const block_iq2_kt * x = (const block_iq2_kt *)(dptr + 1);

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            const uint16_t * ql = (const uint16_t *)x[i].ql;
            __m128i s8 = _mm_set1_epi32(*(const uint32_t *)x[i].scales);
            s8 = _mm_and_si128(_mm_srlv_epi32(s8, shifts), _mm_set1_epi8(0xf));
            s8 = _mm_shuffle_epi8(values, s8);
            __m256i s32 = _mm256_cvtepi8_epi32(s8);
            __m256 all_scales = _mm256_mul_ps(d, _mm256_cvtepi32_ps(s32));
            __m128 scales_l = _mm256_castps256_ps128(all_scales);
            __m128 scales_h = _mm256_extractf128_ps(all_scales, 1);
            scales[0] = _mm256_set_m128(scales_l, scales_l);
            scales[1] = _mm256_set_m128(scales_h, scales_h);
            for (int i128 = 0; i128 < 2; ++i128) {
                iq23kt_next_128(ql + 16*i128, 4096, xv, trellis);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const block_q8_2_x4 yb = y[iy][2*i+i128];
                    __m128 dy4 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)yb.d)), 16));
                    __m256 dy8 = _mm256_mul_ps(scales[i128], _mm256_set_m128(dy4, dy4));
                    for (int k = 0; k < 4; ++k) {
                        __m256i yv = _mm256_loadu_si256((const __m256i *)yb.qs + k);
                        __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(xv[k], xv[k]), _mm256_sign_epi8(yv, xv[k]));
                        dot[k] = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
                    }
                    // dot[k] has 8 values from block k
                    // 0 1 0 1 0 1 0 1
                    dot[0] = _mm256_add_epi32(_mm256_unpacklo_epi32(dot[0], dot[1]), _mm256_unpackhi_epi32(dot[0], dot[1]));
                    // 2 3 2 3 2 3 2 3
                    dot[2] = _mm256_add_epi32(_mm256_unpacklo_epi32(dot[2], dot[3]), _mm256_unpackhi_epi32(dot[2], dot[3]));
                    // 0 1 2 3 0 1 2 3
                    dot[0] = _mm256_add_epi32(_mm256_unpacklo_epi64(dot[0], dot[2]), _mm256_unpackhi_epi64(dot[0], dot[2]));
                    accd[iy] = _mm256_fmadd_ps(dy8, _mm256_cvtepi32_ps(dot[0]), accd[iy]);
                }
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

// void mul_mat_iq3_kt_q8_2_x4_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x)
//Trellis3<true, true> trellis;
template <int nrc_y>
static void iq3kt_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis3 trellis = make_Trellis3(true);

    __m128i shifts = _mm_set_epi32(0, 0, 4, 0);

    __m256  accd[nrc_y];
    const block_q8_2_x4 * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = (const block_q8_2_x4 *)info->cy + (info->cur_y + iy)*info->by;
    }

    __m256i  xv[4], sv[4], dot[4];
    __m256   scales[2];

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        __m256 d = _mm256_set1_ps(dptr[0] * 1.01f);
        const block_iq3_kt * x = (const block_iq3_kt *)(dptr + 1);

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            const uint16_t *ql = (const uint16_t *)x[i].ql;
            __m256i sign_bits = _mm256_loadu_si256((const __m256i *)x[i].qh);
            __m128i s8 = _mm_set1_epi32(*(const uint32_t *)x[i].scales);
            s8 = _mm_and_si128(_mm_srlv_epi32(s8, shifts), _mm_set1_epi8(0xf));
            __m256i s32 = _mm256_cvtepi8_epi32(s8);
            __m256 all_scales = _mm256_mul_ps(d, _mm256_cvtepi32_ps(s32));
            __m128 scales_l = _mm256_castps256_ps128(all_scales);
            __m128 scales_h = _mm256_extractf128_ps(all_scales, 1);
            scales[0] = _mm256_set_m128(scales_l, scales_l);
            scales[1] = _mm256_set_m128(scales_h, scales_h);
            __m256i mask = _mm256_set1_epi8(1);
            for (int i128 = 0; i128 < 2; ++i128) {
                iq23kt_next_128(ql + 16*i128, 4096, xv, trellis);
                for (int k = 0; k < 4; ++k) {
                    xv[k] = _mm256_sign_epi8(xv[k], xv[k]);
                }
                for (int k = 0; k < 4; ++k) {
                    sv[k] = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sign_bits, mask), mask), mask);
                    sign_bits = _mm256_srli_epi16(sign_bits, 1);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const block_q8_2_x4 yb = y[iy][2*i+i128];
                    __m128 dy4 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)yb.d)), 16));
                    __m256 dy8 = _mm256_mul_ps(scales[i128], _mm256_set_m128(dy4, dy4));
                    for (int k = 0; k < 4; ++k) {
                        __m256i yv = _mm256_loadu_si256((const __m256i *)yb.qs + k);
                        __m256i p = _mm256_maddubs_epi16(xv[k], _mm256_sign_epi8(yv, sv[k]));
                        dot[k] = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
                    }
                    // dot[k] has 8 values from block k
                    // 0 1 0 1 0 1 0 1
                    dot[0] = _mm256_add_epi32(_mm256_unpacklo_epi32(dot[0], dot[1]), _mm256_unpackhi_epi32(dot[0], dot[1]));
                    // 2 3 2 3 2 3 2 3
                    dot[2] = _mm256_add_epi32(_mm256_unpacklo_epi32(dot[2], dot[3]), _mm256_unpackhi_epi32(dot[2], dot[3]));
                    // 0 1 2 3 0 1 2 3
                    dot[0] = _mm256_add_epi32(_mm256_unpacklo_epi64(dot[0], dot[2]), _mm256_unpackhi_epi64(dot[0], dot[2]));
                    accd[iy] = _mm256_fmadd_ps(dy8, _mm256_cvtepi32_ps(dot[0]), accd[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

// void mul_mat_iq4_kt_q8_2_x4_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x)
//Trellis3<bool is_8 = false, bool is_abs = false> trellis;
template <int nrc_y>
static void iq4kt_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;
    const int kNumGroups = 64;

    Trellis3 trellis = make_Trellis3(false);

    union { __m256i vec; uint32_t val[8]; } o_helper;

    __m256  accd[nrc_y];
    const block_q8_2_x4 * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = (const block_q8_2_x4 *)info->cy + (info->cur_y + iy)*info->by;
    }

    uint32_t values[64];
    __m256i  xv[4], dot[4];
    __m256   scales[2];

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        __m256 d = _mm256_set1_ps(dptr[0]);
        const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 1);

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            __m256i vshb = _mm256_loadu_si256((const __m256i *)x[i].qs);
            const uint32_t * shb = x[i].qs;
            const uint8_t * ql = (const uint8_t *)(shb + 8);
            const uint8_t * qh = ql + kNumGroups;
            __m256i iscales = _mm256_srli_epi32(_mm256_and_si256(vshb, _mm256_set1_epi32(0xff)), 1);
            iscales = _mm256_sub_epi32(iscales, _mm256_set1_epi32(64));
            __m256 all_scales = _mm256_mul_ps(d, _mm256_cvtepi32_ps(iscales));
            __m128 scales_l = _mm256_castps256_ps128(all_scales);
            __m128 scales_h = _mm256_extractf128_ps(all_scales, 1);
            scales[0] = _mm256_set_m128(scales_l, scales_l);
            scales[1] = _mm256_set_m128(scales_h, scales_h);
            o_helper.vec = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(vshb, _mm256_set1_epi32(1)), 15), _mm256_set1_epi32(4096));
            for (int ib = 0; ib < 4; ++ib) {
                for (int j = 0; j < 2; ++j) {
                    const uint32_t sh1 = shb[ib+0] >> (8 + 12*j);
                    const uint32_t sh2 = shb[ib+4] >> (8 + 12*j);
                    values[8*ib+4*j+ 0] = ql[8*ib+4*j+ 0] + ((qh[8*ib+4*j+0] << 8) & 0xf00) + ((sh1 << 12) & 0x7000) + o_helper.val[ib+0];
                    values[8*ib+4*j+ 1] = ql[8*ib+4*j+ 1] + ((qh[8*ib+4*j+1] << 8) & 0xf00) + ((sh1 <<  9) & 0x7000) + o_helper.val[ib+0];
                    values[8*ib+4*j+ 2] = ql[8*ib+4*j+ 2] + ((qh[8*ib+4*j+2] << 8) & 0xf00) + ((sh1 <<  6) & 0x7000) + o_helper.val[ib+0];
                    values[8*ib+4*j+ 3] = ql[8*ib+4*j+ 3] + ((qh[8*ib+4*j+3] << 8) & 0xf00) + ((sh1 <<  3) & 0x7000) + o_helper.val[ib+0];
                    values[8*ib+4*j+32] = ql[8*ib+4*j+32] + ((qh[8*ib+4*j+0] << 4) & 0xf00) + ((sh2 << 12) & 0x7000) + o_helper.val[ib+4];
                    values[8*ib+4*j+33] = ql[8*ib+4*j+33] + ((qh[8*ib+4*j+1] << 4) & 0xf00) + ((sh2 <<  9) & 0x7000) + o_helper.val[ib+4];
                    values[8*ib+4*j+34] = ql[8*ib+4*j+34] + ((qh[8*ib+4*j+2] << 4) & 0xf00) + ((sh2 <<  6) & 0x7000) + o_helper.val[ib+4];
                    values[8*ib+4*j+35] = ql[8*ib+4*j+35] + ((qh[8*ib+4*j+3] << 4) & 0xf00) + ((sh2 <<  3) & 0x7000) + o_helper.val[ib+4];
                }
            }
            for (int i128 = 0; i128 < 2; ++i128) {
                iq4kt_next_128(values + 32*i128, xv, trellis);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const block_q8_2_x4 yb = y[iy][2*i+i128];
                    __m128 dy4 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)yb.d)), 16));
                    __m256 dy8 = _mm256_mul_ps(scales[i128], _mm256_set_m128(dy4, dy4));
                    for (int k = 0; k < 4; ++k) {
                        __m256i yv = _mm256_loadu_si256((const __m256i *)yb.qs + k);
                        __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(xv[k], xv[k]), _mm256_sign_epi8(yv, xv[k]));
                        dot[k] = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
                    }
                    // dot[k] has 8 values from block k
                    // 0 1 0 1 0 1 0 1
                    dot[0] = _mm256_add_epi32(_mm256_unpacklo_epi32(dot[0], dot[1]), _mm256_unpackhi_epi32(dot[0], dot[1]));
                    // 2 3 2 3 2 3 2 3
                    dot[2] = _mm256_add_epi32(_mm256_unpacklo_epi32(dot[2], dot[3]), _mm256_unpackhi_epi32(dot[2], dot[3]));
                    // 0 1 2 3 0 1 2 3
                    dot[0] = _mm256_add_epi32(_mm256_unpacklo_epi64(dot[0], dot[2]), _mm256_unpackhi_epi64(dot[0], dot[2]));
                    accd[iy] = _mm256_fmadd_ps(dy8, _mm256_cvtepi32_ps(dot[0]), accd[iy]);
                }
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info->s[(info->cur_y + iy)*info->bs + ix] = iqk_hsum_float_8(accd[iy]);
        }
    }
}

template <int nrc_y>
static void q8_1_r8_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    const block_q8_2_x4 * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_2_x4 *)(info->cy + (info->cur_y + iy)*info->by);
    }
    int nb = n / QK8_0;
    __m256 acc[nrc_y];
    for (int i = 0; i < nrc_y; i++) acc[i] = _mm256_setzero_ps();
    float d8[4*nrc_y];

    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q8_1_r8 * iq8 = (const block_q8_1_r8 *)((const char *)vx + ix*bx);
        for (int i4 = 0; i4 < nb/4; ++i4) {
            {
                __m256 mx[4];
                for (int ib32 = 0; ib32 < 4; ++ib32) mx[ib32] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*i4+ib32].d+1));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    __m128 scales = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8[iy][i4].d)), 16));
                    _mm_storeu_ps(d8 + 4*iy + 0, scales);
                    __m128 bsums4 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8[iy][i4].d+4))));
                    bsums4 = _mm_mul_ps(bsums4, scales);
                    __m256 bsums  = _mm256_set_m128(bsums4, bsums4);
                    acc[iy] = _mm256_fmadd_ps(mx[0], _mm256_shuffle_ps(bsums, bsums, 0x00), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(mx[1], _mm256_shuffle_ps(bsums, bsums, 0x55), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(mx[2], _mm256_shuffle_ps(bsums, bsums, 0xaa), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(mx[3], _mm256_shuffle_ps(bsums, bsums, 0xff), acc[iy]);
                }
            }
            for (int ib32 = 0; ib32 < 4; ++ib32) {
                __m256 scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*i4+ib32].d));
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*i4+ib32].qs+j);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    __m128i y128 = _mm_loadu_si128((const __m128i*)(q8[iy][i4].qs+32*ib32));
                    __m256i y = _mm256_broadcastsi128_si256(y128);
                    __m256i sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    __m256i sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    __m256i sumi = _mm256_add_epi32(_mm256_madd_epi16(_mm256_set1_epi16(1), sumi1), _mm256_madd_epi16(_mm256_set1_epi16(1), sumi2));
                    __m256 d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+ib32]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*i4+ib32].qs+4+j);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    __m128i y128 = _mm_loadu_si128((const __m128i*)(q8[iy][i4].qs+32*ib32+16));
                    __m256i y = _mm256_broadcastsi128_si256(y128);
                    __m256i sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    __m256i sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    __m256i sumi = _mm256_add_epi32(_mm256_madd_epi16(_mm256_set1_epi16(1), sumi1), _mm256_madd_epi16(_mm256_set1_epi16(1), sumi2));

                    __m256 d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+ib32]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            // info.store(ix, iy, acc[iy]);
            _mm256_storeu_ps(info->s + (info->cur_y + iy)*info->bs + ix, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

template <int nrc_y>
static void q8_0_r8_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    const block_q8_2_x4 * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_2_x4 *)(info->cy + (info->cur_y + iy)*info->by);
    }

    __m256i m1 = _mm256_set1_epi16(1);
    int nb = n / QK8_0;
    __m256 acc[nrc_y] = {}; //acc[i] = _mm256_setzero_ps();
    // for (int i = 0; i < nrc_y; i++)
    float d8[4*nrc_y];
    __m256i qx[4], sx[4];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q8_0_r8 * iq8 = (const block_q8_0_r8 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                __m128 scales = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8[iy][ib4].d)), 16));
                _mm_storeu_ps(d8 + 4*iy, scales);
            }
            for (int k = 0; k < 4; ++k) {
                __m256 scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*ib4+k].d));
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*ib4+k].qs+j);
                    sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    __m128i y128 = _mm_loadu_si128((const __m128i*)(q8[iy][ib4].qs+32*k));
                    __m256i y = _mm256_broadcastsi128_si256(y128);
                    __m256i sumi1 = _mm256_add_epi32(
                            _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[0], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]))),
                            _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[1], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1])))
                    );
                    __m256i sumi2 = _mm256_add_epi32(
                            _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[2], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]))),
                            _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[3], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3])))
                    );
                    __m256i sumi = _mm256_add_epi32(sumi1, sumi2);
                    __m256 d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*ib4+k].qs+4+j);
                    sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    __m128i y128 = _mm_loadu_si128((const __m128i*)(q8[iy][ib4].qs+32*k+16));
                    __m256i y = _mm256_broadcastsi128_si256(y128);
                    __m256i sumi1 = _mm256_add_epi32(
                            _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[0], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]))),
                            _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[1], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1])))
                    );
                    __m256i sumi2 = _mm256_add_epi32(
                            _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[2], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]))),
                            _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[3], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3])))
                    );
                    __m256i sumi = _mm256_add_epi32(sumi1, sumi2);
                    __m256 d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            __m256 scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[ib].d));
            for (int j = 0; j < 4; ++j) {
                qx[j] = _mm256_loadu_si256((const __m256i *)iq8[ib].qs+j);
                sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                const block_q8_2 * qy = (const block_q8_2 *)q8[iy];
                __m128i y128 = _mm_loadu_si128((const __m128i*)qy[ib].qs);
                __m256i y = _mm256_broadcastsi128_si256(y128);
                __m256i sumi1 = _mm256_add_epi32(
                        _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[0], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]))),
                        _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[1], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1])))
                );
                __m256i sumi2 = _mm256_add_epi32(
                        _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[2], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]))),
                        _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[3], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3])))
                );
                __m256i sumi = _mm256_add_epi32(sumi1, sumi2);
                ggml_bf16_t tmp = {qy[ib].d};
                __m256 d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(tmp)));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
            }
            for (int j = 0; j < 4; ++j) {
                qx[j] = _mm256_loadu_si256((const __m256i *)iq8[ib].qs+4+j);
                sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                const block_q8_2 * qy = (const block_q8_2 *)q8[iy];
                __m128i y128 = _mm_loadu_si128((const __m128i*)(qy[ib].qs+16));
                __m256i y = _mm256_broadcastsi128_si256(y128);
                __m256i sumi1 = _mm256_add_epi32(
                        _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[0], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]))),
                        _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[1], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1])))
                );
                __m256i sumi2 = _mm256_add_epi32(
                        _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[2], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]))),
                        _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[3], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3])))
                );
                __m256i sumi = _mm256_add_epi32(sumi1, sumi2);
                ggml_bf16_t tmp = {qy[ib].d};
                __m256 d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(tmp)));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            _mm256_storeu_ps(info->s + (info->cur_y + iy)*info->bs + ix, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

template <int nrc_y>
static void q8k_r8_mul_mat(int n, const void * vx, size_t bx, struct DataInfo *info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    const block_q8_K * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const block_q8_K *)(info->cy + (info->cur_y + iy)*info->by);
    }
#ifndef HAVE_FANCY_SIMD
    __m256i m1 = _mm256_set1_epi16(1);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y];
    __m256i isum[nrc_y];
    for (int i = 0; i < nrc_y; i++) {
        acc[i] = _mm256_setzero_ps();
        isum[i] = _mm256_setzero_si256();
    }
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q8_k_r8 * iq8 = (const block_q8_k_r8 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            __m256 d4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[ibl].d));
            for (int ib = 0; ib < QK_K/16; ++ib) {
                qx[0] = _mm256_loadu_si256((const __m256i *)iq8[ibl].qs+4*ib+0);
                qx[1] = _mm256_loadu_si256((const __m256i *)iq8[ibl].qs+4*ib+1);
                qx[2] = _mm256_loadu_si256((const __m256i *)iq8[ibl].qs+4*ib+2);
                qx[3] = _mm256_loadu_si256((const __m256i *)iq8[ibl].qs+4*ib+3);
#ifndef HAVE_FANCY_SIMD
                __m256i s0 = _mm256_sign_epi8(qx[0], qx[0]);
                __m256i s1 = _mm256_sign_epi8(qx[1], qx[1]);
                __m256i s2 = _mm256_sign_epi8(qx[2], qx[2]);
                __m256i s3 = _mm256_sign_epi8(qx[3], qx[3]);
#else
                qx[0] = _mm256_add_epi8(qx[0], _mm256_set1_epi8(127));
                qx[1] = _mm256_add_epi8(qx[1], _mm256_set1_epi8(127));
                qx[2] = _mm256_add_epi8(qx[2], _mm256_set1_epi8(127));
                qx[3] = _mm256_add_epi8(qx[3], _mm256_set1_epi8(127));
#endif
                for (int iy = 0; iy < nrc_y; ++iy) {
                    __m128i y128 = _mm_loadu_si128((const __m128i*)q8[iy][ibl].qs+ib);
                    __m256i y = _mm256_broadcastsi128_si256(y128);
#ifdef HAVE_FANCY_SIMD
                    isum[iy] = _mm256_dpbusd_epi32(isum[iy], qx[0], _mm256_shuffle_epi32(y, 0x00));
                    isum[iy] = _mm256_dpbusd_epi32(isum[iy], qx[1], _mm256_shuffle_epi32(y, 0x55));
                    isum[iy] = _mm256_dpbusd_epi32(isum[iy], qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    isum[iy] = _mm256_dpbusd_epi32(isum[iy], qx[3], _mm256_shuffle_epi32(y, 0xff));
#else
                    __m256i sumi1 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0])));
                    __m256i sumi2 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1])));
                    __m256i sumi3 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2])));
                    __m256i sumi4 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3])));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(sumi1, sumi2));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(sumi3, sumi4));
#endif
                }
            }
#ifdef HAVE_FANCY_SIMD
            __m256 m4 = _mm256_mul_ps(d4, _mm256_set1_ps(-127.f));
#endif
            for (int iy = 0; iy < nrc_y; ++iy) {
                __m256 d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8[iy][ibl].d));
                acc[iy] = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
#ifdef HAVE_FANCY_SIMD
                acc[iy] = _mm256_fmadd_ps(m4, _mm256_set1_ps(q8.y[iy][ibl].sum), acc[iy]);
#endif
                isum[iy] = _mm256_setzero_si256();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            _mm256_storeu_ps(info->s + (info->cur_y + iy)*info->bs + ix, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

static inline enum ggml_type iqk_is_dequant_better(enum ggml_type type, int nrc_y) {
#ifdef __AVX2__
    switch (type) {
        case GGML_TYPE_IQ4_XS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_Q2_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_Q3_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_Q4_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
        case GGML_TYPE_Q5_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
        case GGML_TYPE_Q6_K   : return nrc_y >= 64 ? GGML_TYPE_Q8_0_R8 : type;
        case GGML_TYPE_IQ2_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ2_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ3_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ3_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ4_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ4_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ5_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ5_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ6_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ2_KT : return nrc_y >= 16 ? GGML_TYPE_Q8_0_R8 : type;
        case GGML_TYPE_IQ3_KT : return nrc_y >= 16 ? GGML_TYPE_Q8_0_R8 : type;
        case GGML_TYPE_IQ4_KT : return nrc_y >= 24 ? GGML_TYPE_Q8_0_R8 : type;
        default: break;
    }
#else
    switch (type) {
        case GGML_TYPE_Q2_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_Q3_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_Q4_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
        case GGML_TYPE_Q5_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
        case GGML_TYPE_Q6_K   : return nrc_y >= 64 ? GGML_TYPE_Q8_0_R8 : type;
        case GGML_TYPE_IQ4_XS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_Q4_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
        case GGML_TYPE_Q4_1   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
        case GGML_TYPE_Q5_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
        case GGML_TYPE_Q5_1   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
        case GGML_TYPE_Q8_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
        case GGML_TYPE_IQ4_NL : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
        case GGML_TYPE_IQ2_KT : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
        case GGML_TYPE_IQ3_KT : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
        case GGML_TYPE_IQ4_KT : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
        case GGML_TYPE_IQ2_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ3_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ4_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ5_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ2_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ3_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ4_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ5_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        case GGML_TYPE_IQ6_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
        default: break;
    }
#endif
    return type;
}

static inline int iqk_num_rows(enum ggml_type type) {
#ifdef HAVE_FANCY_SIMD
    switch (type) {
        case GGML_TYPE_Q2_K_R4:
        case GGML_TYPE_Q3_K_R4:
        case GGML_TYPE_Q6_K_R4:
        case GGML_TYPE_IQ2_K_R4:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ5_K_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ2_XXS_R4:
        case GGML_TYPE_IQ2_XS_R4:
        case GGML_TYPE_IQ2_S_R4:
        case GGML_TYPE_IQ3_XXS_R4:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:
        case GGML_TYPE_IQ3_S_R4: return 4;
        case GGML_TYPE_IQ4_NL_R4:
        case GGML_TYPE_Q5_0_R4:
        case GGML_TYPE_Q6_0_R4:
        case GGML_TYPE_IQ2_BN_R4:
        case GGML_TYPE_IQ4_XS_R8:
        case GGML_TYPE_Q4_K_R4:
        case GGML_TYPE_Q5_K_R4:
        case GGML_TYPE_Q8_KV:
        case GGML_TYPE_Q8_KV_R8:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q8_K_R8: return 8;
        case GGML_TYPE_Q4_0_R8:
        case GGML_TYPE_Q8_0_R8:
        case GGML_TYPE_BF16_R16: return 16;
        default: return 1;
    }
#else
    switch (type) {
        case GGML_TYPE_Q8_0_R8:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q8_K_R8: return 8;
        default: return 1;
    }
#endif
}

typedef void (*mul_mat_t)(int n, const void * vx, size_t bx, struct DataInfo * info, int nrc_x);

static const uint8_t k_shuff[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

#include <array>
std::array<mul_mat_t, 8> funcs = {};
#define IQK_SET_MUL_MAT_FUNCTIONS_T(kernel) \
            funcs[0] = kernel<1>;\
            funcs[1] = kernel<2>;\
            funcs[2] = kernel<3>;\
            funcs[3] = kernel<4>;\
            funcs[4] = kernel<5>;\
            funcs[5] = kernel<6>;\
            funcs[6] = kernel<7>;\
            funcs[7] = kernel<8>;\
            break;\

static void iqk_prepare(int typeA, int typeB, int ne00) {
    assert(ne00%QK_K == 0);
    switch (typeA) {
        //kquants
        case GGML_TYPE_Q2_K: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q2Bits
            cl.m4 = _mm_set1_epi8(0xf);
            //
            cl.ml = _mm256_set1_epi8(0x03);
            IQK_SET_MUL_MAT_FUNCTIONS_T(q2k_mul_mat);
        }
        case GGML_TYPE_Q3_K: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q2Bits
            cl.ml = _mm256_set1_epi8(0x03);
            //HighBit3
            cl.mh = _mm256_set1_epi8(0x04);
            //(ScaleQ3)
            cl.m32 = _mm_set1_epi8(-32);
            IQK_SET_MUL_MAT_FUNCTIONS_T(q3k_mul_mat);
        }
        case GGML_TYPE_Q4_K: {
            assert(typeB == GGML_TYPE_Q8_2_X4);
            //Q4Bits_AVX2
            cl.ml = _mm256_set1_epi8(0xf);
            IQK_SET_MUL_MAT_FUNCTIONS_T(q4k_mul_mat_X4);
        }
        case GGML_TYPE_Q5_K: {
            assert(typeB == GGML_TYPE_Q8_2_X4);
            //Q4Bits_AVX2
            cl.ml = _mm256_set1_epi8(0xf);
            //
            cl.mh = _mm256_set1_epi8(0x10);
            IQK_SET_MUL_MAT_FUNCTIONS_T(q5k_mul_mat_X4);
        }
        case GGML_TYPE_Q6_K: {
            assert(typeB == GGML_TYPE_Q8_2_X4);
            //Q4Bits_AVX2
            cl.ml = _mm256_set1_epi8(0xf);
            //
            cl.mh = _mm256_set1_epi8(0x30);
            IQK_SET_MUL_MAT_FUNCTIONS_T(q6k_mul_mat_X4);
        }
        case GGML_TYPE_IQ4_XS: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q4Bits
            cl.ml = _mm256_set1_epi8(0xf);
            //Scales8K -- Scales8KBase
            cl.shuffles[0] = _mm_set_epi32(0x07060706, 0x05040504, 0x03020302, 0x01000100);
            cl.shuffles[1] = _mm_set_epi32(0x0f0e0f0e, 0x0d0c0d0c, 0x0b0a0b0a, 0x09080908);
            //ScaleIQ4XS
            cl.hshift = _mm_set_epi32(12, 8, 4, 0);
            cl.lshift = _mm_set_epi32(4, 0, 4, 0);
            cl.hmask  = _mm_set1_epi16(0x03);
            cl.lmask  = _mm_set1_epi8(0xf);
            cl.lshuffle = _mm_set_epi32(0x07030602, 0x05010400, 0x07030602, 0x05010400);
            cl.m32 = _mm_set1_epi16(-32);
            //
            cl.iqk_values = load_iq4nl_values();
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq4xs_mul_mat);
        }
        //iqk_quants
        case GGML_TYPE_IQ2_KS: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q2Bits
            cl.ml = _mm256_set1_epi8(0x03);
            //Scales8KBase
            cl.shuffles[0] = _mm_set_epi32(0x07060706, 0x05040504, 0x03020302, 0x01000100);
            cl.shuffles[1] = _mm_set_epi32(0x0f0e0f0e, 0x0d0c0d0c, 0x0b0a0b0a, 0x09080908);
            //
            cl.iqk_values = load_iq2nl_values();
            cl.m16 = _mm_set1_epi8(-16);
            cl.m5  = _mm_set1_epi8(5);
            cl.m32 = _mm_set1_epi8(-32);
            cl.hmask = _mm_set1_epi64x(0x8040201008040201);
            cl.shuffle = _mm_set1_epi64x(0x0703060205010400);
            cl.shift = _mm_set_epi32(0, 0, 4, 0);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq2ks_mul_mat);
        }
        case GGML_TYPE_IQ2_K: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q2Bits
            cl.ml = _mm256_set1_epi8(0x03);
            // IQXKScales
            cl.min = _mm256_set1_epi16(-32);
            cl.eshift = _mm_set1_epi8(5);
            cl.hshuff   = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
            cl.emask    = _mm_set_epi32(0x80804040, 0x20201010, 0x08080404, 0x02020101);
            cl.eshuffle = _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
            //
            cl.iqk_values = load_iq2nl_values();
            cl.m8       = _mm_set1_epi8(-8);
            cl.maskl    = _mm_set1_epi8(0xf);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq2k_mul_mat);
        }
        case GGML_TYPE_IQ3_KS: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q2Bits
            cl.ml = _mm256_set1_epi8(0x03);
            //
            cl.iqk_values = load_iq3nl_values();
            cl.mh   = _mm256_set1_epi8(4);
            cl.mask = _mm_setr_epi16(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq3ks_mul_mat);
        }
        case GGML_TYPE_IQ3_K: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q2Bits
            cl.ml = _mm256_set1_epi8(0x03);
            //IQXKScales
            cl.min = _mm256_set1_epi16(-64);
            cl.eshift = _mm_set1_epi8(4);
            cl.hshuff   = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
            cl.emask    = _mm_set_epi32(0x80804040, 0x20201010, 0x08080404, 0x02020101);
            cl.eshuffle = _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
            //
            cl.iqk_values = iq3k_load_values();
            cl.hmask256  = _mm256_set1_epi8(4);
            cl.m1 = _mm_set1_epi8(1);
            cl.sign_mask = _mm_set_epi64x(0x8080404020201010, 0x0808040402020101);
            cl.hshuff = _mm_loadu_si128((const __m128i*)k_shuff);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq3k_mul_mat);
        }
        case GGML_TYPE_IQ4_KSS: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q4Bits
            cl.ml = _mm256_set1_epi8(0xf);
            //Scales8KBase
            cl.shuffles[0] = _mm_set_epi32(0x07060706, 0x05040504, 0x03020302, 0x01000100);
            cl.shuffles[1] = _mm_set_epi32(0x0f0e0f0e, 0x0d0c0d0c, 0x0b0a0b0a, 0x09080908);
            //
            cl.iqk_values = load_iq4nl_values();
            cl.sbmask[0]    = _mm256_set_epi64x(0x0080004000200010, 0x0008000400020001, 0x0080004000200010, 0x0008000400020001);
            cl.sbmask[1]    = _mm256_set1_epi16(-2); // 0xfffe;
            cl.mask     = _mm_set1_epi16(254);
            cl.m127     = _mm_set1_epi16(-127);
            cl.m128     = _mm_set1_epi16(-128);
            cl.m1_256   = _mm256_set1_epi16(1);
            cl.m4       = _mm_set1_epi16(4);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq4kss_mul_mat);
        }
        case GGML_TYPE_IQ4_KS: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q4Bits
            cl.ml = _mm256_set1_epi8(0xf);
            //
            iq4ks_load_values(cl.iqk_value);
            cl.mask     = _mm_set1_epi16(254);
            cl.m127     = _mm_set1_epi16(-127);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq4ks_mul_mat);
        }
        case GGML_TYPE_IQ4_K: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q4Bits
            cl.ml = _mm256_set1_epi8(0xf);
            //
            cl.maskl    = _mm_set1_epi8(0xf);
            cl.maskh    = _mm_set1_epi8(0x30);
            cl.m32      = _mm_set1_epi8(-32);
            cl.hshuff   = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
            iq4k_load_values(cl.iqk_value);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq4k_mul_mat);
        }
        case GGML_TYPE_IQ5_KS: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q4Bits
            cl.ml = _mm256_set1_epi8(0xf);
            //Scales8KBase
            cl.shuffles[0] = _mm_set_epi32(0x07060706, 0x05040504, 0x03020302, 0x01000100);
            cl.shuffles[1] = _mm_set_epi32(0x0f0e0f0e, 0x0d0c0d0c, 0x0b0a0b0a, 0x09080908);
            //
            iq5ks_load_values(cl.iqk_value);
            cl.maskl    = _mm_set1_epi8(0xf);
            cl.maskh    = _mm_set1_epi8(0x30);
            cl.mh       = _mm256_set1_epi8(-128);
            cl.mask     = _mm_set1_epi16(254);
            cl.m127     = _mm_set1_epi16(-127);
            cl.m128     = _mm_set1_epi16(-128);
            cl.m1       = _mm_set1_epi16(1);
            cl.m2       = _mm_set1_epi16(2);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq5ks_mul_mat);
        }
        case GGML_TYPE_IQ5_K: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q4Bits
            cl.ml = _mm256_set1_epi8(0xf);
            //Scales8KBase
            cl.shuffles[0] = _mm_set_epi32(0x07060706, 0x05040504, 0x03020302, 0x01000100);
            cl.shuffles[1] = _mm_set_epi32(0x0f0e0f0e, 0x0d0c0d0c, 0x0b0a0b0a, 0x09080908);
            //
            load_iq5nl_values(cl.iqk_value);
            cl.maskl    = _mm_set1_epi8(0xf);
            cl.maskh    = _mm_set1_epi8(0x30);
            cl.m32      = _mm_set1_epi8(-32);
            cl.mh       = _mm256_set1_epi8(-128);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq5k_mul_mat);
        }
        case GGML_TYPE_IQ6_K: {
            assert(typeB == GGML_TYPE_Q8_K);
            //Q4Bits
            cl.ml = _mm256_set1_epi8(0xf);
            //IQXKScales
            cl.min = _mm256_set1_epi16(0);
            cl.eshift = _mm_set1_epi8(1);
            cl.hshuff   = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
            cl.emask    = _mm_set_epi32(0x80804040, 0x20201010, 0x08080404, 0x02020101);
            cl.eshuffle = _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
            //
            load_iq6nl_values(cl.iqk_value);
            cl.mh1 = _mm256_set1_epi8(1);
            cl.mh2 = _mm256_set1_epi8(2);
            cl.mh3 = _mm256_set1_epi8(3);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq6k_mul_mat);
        }
        // ktquants assert(typeB == GGML_TYPE_Q8_2_X4)
        case GGML_TYPE_IQ2_KT: {
            assert(typeB == GGML_TYPE_Q8_2_X4);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq2kt_mul_mat);
        }
        case GGML_TYPE_IQ3_KT: {
            assert(typeB == GGML_TYPE_Q8_2_X4);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq3kt_mul_mat);
        }
        case GGML_TYPE_IQ4_KT: {
            assert(typeB == GGML_TYPE_Q8_2_X4);
            IQK_SET_MUL_MAT_FUNCTIONS_T(iq4kt_mul_mat);
        }
        case GGML_TYPE_Q8_1:
            assert(typeB == GGML_TYPE_Q8_2_X4);
            IQK_SET_MUL_MAT_FUNCTIONS_T(q8_1_r8_mul_mat);
        case GGML_TYPE_Q8_0_R8:
            assert(typeB == GGML_TYPE_Q8_2_X4);
            IQK_SET_MUL_MAT_FUNCTIONS_T(q8_0_r8_mul_mat);
        case GGML_TYPE_Q8_K_R8:
            assert(typeB == GGML_TYPE_Q8_K);
            IQK_SET_MUL_MAT_FUNCTIONS_T(q8k_r8_mul_mat);
        default:
            GGML_ABORT("Fatal error");
    }
    GGML_UNUSED(typeB);
    GGML_UNUSED(ne00);
}

static bool iqk_convert_q2_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const iqk_block_q2_K * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    float    f_values[QK_K];
    uint32_t block[8];

    __m256i xv[4];

    __m256i ml = _mm256_set1_epi8(0x03);
    __m256 sign_bit = _mm256_set1_ps(-0.0f);
    __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const iqk_block_q2_K *)((const char *)vx + (ix + k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                __m256 vd = _mm256_set1_ps(GGML_FP16_TO_FP32(x8[k][i].d));
                __m256 vm = _mm256_mul_ps(_mm256_set1_ps(GGML_FP16_TO_FP32(x8[k][i].dmin)), _mm256_set1_ps(-1.f));
                __m256 block_max = _mm256_setzero_ps();
                for (int i128 = 0; i128 < 2; ++i128) {
                    __m256i bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+i128);
                    xv[0] = _mm256_and_si256(bits, ml);
                    xv[1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), ml);
                    xv[2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), ml);
                    xv[3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), ml);
                    for (int l = 0; l < 4; ++l) {
                        __m256i q1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv[l]));
                        __m256i q2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv[l], 1));
                        q1 = _mm256_mullo_epi16(q1, _mm256_set1_epi16(x8[k][i].scales[8*i128 + 2*l + 0] & 0xf));
                        q2 = _mm256_mullo_epi16(q2, _mm256_set1_epi16(x8[k][i].scales[8*i128 + 2*l + 1] & 0xf));
                        __m256 m1 = _mm256_mul_ps(vm, _mm256_set1_ps(x8[k][i].scales[8*i128 + 2*l + 0] >> 4));
                        __m256 m2 = _mm256_mul_ps(vm, _mm256_set1_ps(x8[k][i].scales[8*i128 + 2*l + 1] >> 4));
                        __m256 v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q1))), vd, m1);
                        __m256 v1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(q1, 1))), vd, m1);
                        __m256 v2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q2))), vd, m2);
                        __m256 v3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(q2, 1))), vd, m2);
                        __m256 max = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign_bit, v0), _mm256_andnot_ps(sign_bit, v1)),
                                                 _mm256_max_ps(_mm256_andnot_ps(sign_bit, v2), _mm256_andnot_ps(sign_bit, v3)));
                        block_max = _mm256_max_ps(block_max, max);
                        _mm256_storeu_ps(f_values + 128*i128 + 32*l +  0, v0);
                        _mm256_storeu_ps(f_values + 128*i128 + 32*l +  8, v1);
                        _mm256_storeu_ps(f_values + 128*i128 + 32*l + 16, v2);
                        _mm256_storeu_ps(f_values + 128*i128 + 32*l + 24, v3);
                    }
                }
                __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(block_max, 1), _mm256_castps256_ps128(block_max));
                max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
                max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
                float d = _mm_cvtss_f32(max4)/127.f;
                __m256 id = _mm256_set1_ps(d != 0.0f ? 1/d : 0.0f);
                y[i].d[k] = GGML_FP32_TO_FP16(d);
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    __m256 v0 = _mm256_loadu_ps(f_values + 32*ib32 +  0);
                    __m256 v1 = _mm256_loadu_ps(f_values + 32*ib32 +  8);
                    __m256 v2 = _mm256_loadu_ps(f_values + 32*ib32 + 16);
                    __m256 v3 = _mm256_loadu_ps(f_values + 32*ib32 + 24);
                    __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v0, id), _MM_ROUND_NEAREST));
                    __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v1, id), _MM_ROUND_NEAREST));
                    __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v2, id), _MM_ROUND_NEAREST));
                    __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v3, id), _MM_ROUND_NEAREST));
                    i0 = _mm256_packs_epi32(i0, i1);
                    i2 = _mm256_packs_epi32(i2, i3);
                    i0 = _mm256_packs_epi16(i0, i2);
                    i0 = _mm256_permutevar8x32_epi32(i0, perm);

                    _mm256_storeu_si256((__m256i *)block, i0);
                    uint32_t * q8 = (uint32_t *)y[i].qs + 64*ib32;
                    for (int l = 0; l < 4; ++l) {
                        q8[8*l + k +  0] = block[l + 0];
                        q8[8*l + k + 32] = block[l + 4];
                    }
                }
            }
        }
        y += nb;
    }
    return true;
}

static bool iqk_convert_q3_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_q3_K * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    uint32_t block[8];
    __m256i values[8];

    const __m128i m32 = _mm_set1_epi8(-32);
    __m256i ml  = _mm256_set1_epi8(0x03);
    __m256i mh  = _mm256_set1_epi8(0x04);

    union { __m256i vec; int16_t val[16]; } helper;

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_q3_K *)((const char *)vx + (ix + k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                __m256i hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].hmask);
                // sc3.make_scales((const uint16_t *)x8[k][i].scales)
                const uint16_t * scales16 = (const uint16_t *)x8[k][i].scales;
                uint32_t aux0 = scales16[0] | (scales16[1] << 16);
                uint32_t aux1 = scales16[2] | (scales16[3] << 16);
                uint32_t aux2 = scales16[4] | (scales16[5] << 16);
                __m128i scales128 = _mm_set_epi32(
                    ((aux1 >> 4) & 0x0f0f0f0f) | ((aux2 >> 2) & 0x30303030),
                    ((aux0 >> 4) & 0x0f0f0f0f) | ((aux2 >> 0) & 0x30303030),
                     (aux1       & 0x0f0f0f0f) | ((aux2 << 2) & 0x30303030),
                     (aux0       & 0x0f0f0f0f) | ((aux2 << 4) & 0x30303030));
                helper.vec = _mm256_cvtepi8_epi16(_mm_add_epi8(scales128, m32));
                __m256i max_i16 = _mm256_setzero_si256();
                for (int i128 = 0; i128 < 2; ++i128) {
                    __m256i q2bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs + i128);
                    values[4*i128+0] = _mm256_and_si256(q2bits, ml);
                    values[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), ml);
                    values[4*i128+2] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), ml);
                    values[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), ml);
                    values[4*i128+0] = _mm256_or_si256(values[4*i128+0], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh));
                    values[4*i128+1] = _mm256_or_si256(values[4*i128+1], _mm256_and_si256(_mm256_slli_epi16(hbits, 1), mh));
                    values[4*i128+2] = _mm256_or_si256(values[4*i128+2], _mm256_and_si256(hbits, mh));
                    values[4*i128+3] = _mm256_or_si256(values[4*i128+3], _mm256_and_si256(_mm256_srli_epi16(hbits, 1), mh));
                    values[4*i128+0] = _mm256_sub_epi8(values[4*i128+0], mh);
                    values[4*i128+1] = _mm256_sub_epi8(values[4*i128+1], mh);
                    values[4*i128+2] = _mm256_sub_epi8(values[4*i128+2], mh);
                    values[4*i128+3] = _mm256_sub_epi8(values[4*i128+3], mh);
                    hbits = _mm256_srli_epi16(hbits, 4);

                    for (int l = 0; l < 4; ++l) {
                        __m256i q16_l = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(values[4*i128+l]));
                        __m256i q16_h = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(values[4*i128+l], 1));
                        q16_l = _mm256_mullo_epi16(_mm256_set1_epi16(helper.val[8*i128+2*l+0]), q16_l);
                        q16_h = _mm256_mullo_epi16(_mm256_set1_epi16(helper.val[8*i128+2*l+1]), q16_h);
                        max_i16 = _mm256_max_epi16(max_i16, _mm256_sign_epi16(q16_l, q16_l));
                        max_i16 = _mm256_max_epi16(max_i16, _mm256_sign_epi16(q16_h, q16_h));
                    }
                }
                __m256i max_q32 = _mm256_cvtepi16_epi32(_mm_max_epi16(_mm256_castsi256_si128(max_i16), _mm256_extracti128_si256(max_i16, 1)));
                __m128i imax4 = _mm_max_epi32(_mm256_castsi256_si128(max_q32), _mm256_extracti128_si256(max_q32, 1));
                __m128 max4  = _mm_cvtepi32_ps(imax4);
                max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
                max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
                bool needs_scaling = true;
                float dnew = _mm_cvtss_f32(max4) / 127;
                if (dnew < 1.f) {
                    dnew = 1.f; needs_scaling = false;
                }
                d *= dnew;
                y[i].d[k] = GGML_FP32_TO_FP16(d);
                __m256 scale = _mm256_set1_ps(fabsf(dnew) > 1e-9f ? 1/dnew : 0.f);
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    __m256i q16_l = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(values[ib32]));
                    __m256i q16_h = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(values[ib32], 1));
                    q16_l = _mm256_mullo_epi16(q16_l, _mm256_set1_epi16(helper.val[2*ib32+0]));
                    q16_h = _mm256_mullo_epi16(q16_h, _mm256_set1_epi16(helper.val[2*ib32+1]));
                    if (needs_scaling) {
                        __m256i i0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16_l));
                        __m256i i1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16_l, 1));
                        __m256i i2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16_h));
                        __m256i i3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16_h, 1));
                        i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i0)), _MM_ROUND_NEAREST));
                        i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i1)), _MM_ROUND_NEAREST));
                        i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i2)), _MM_ROUND_NEAREST));
                        i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i3)), _MM_ROUND_NEAREST));
                        i0 = _mm256_packs_epi32(i0, i1);
                        i2 = _mm256_packs_epi32(i2, i3);
                        i0 = _mm256_packs_epi16(i0, i2);
                        i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
                        _mm256_storeu_si256((__m256i *)block, i0);
                    } else {
                        // 0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31
                        __m256i i0 = _mm256_packs_epi16(q16_l, q16_h);
                        __m128i i0_l = _mm256_castsi256_si128(i0);
                        __m128i i0_h = _mm256_extracti128_si256(i0, 1);
                        _mm_storeu_si128((__m128i *)block+0, _mm_unpacklo_epi64(i0_l, i0_h));
                        _mm_storeu_si128((__m128i *)block+1, _mm_unpackhi_epi64(i0_l, i0_h));
                    }
                    uint32_t * qs = (uint32_t *)y[i].qs + 64*ib32;
                    for (int l = 0; l < 8; ++l) {
                        qs[8*l + k] = block[l];
                    }
                }
            }
        }
        y += nb;
    }
    return true;
}

static bool iqk_convert_q4_k_q8_1_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const iqk_block_q4_K * x8[8];

    block_q8_1_r8 * y = (block_q8_1_r8 *)vy;

    ggml_half dh[16];
    uint16_t all_ls[128];

    uint32_t utmp[4];
    const uint8_t * u8 = (const uint8_t *)utmp;
    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const iqk_block_q4_K *)((const char *)vx + (ix + k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                dh[k+0] = x8[k][i].d;
                dh[k+8] = x8[k][i].dmin;
                iqk_make_q4_scales(x8[k][i].scales, utmp);
                const uint8_t* qs  = x8[k][i].qs;
                for (int ib64 = 0; ib64 < 4; ++ib64) {
                    all_ls[8*(2*ib64 + 0) + k     ] = u8[2*ib64+0];
                    all_ls[8*(2*ib64 + 1) + k     ] = u8[2*ib64+1];
                    all_ls[8*(2*ib64 + 0) + k + 64] = u8[2*ib64+8];
                    all_ls[8*(2*ib64 + 1) + k + 64] = u8[2*ib64+9];
                    __m256i bits = _mm256_loadu_si256((const __m256i *)qs+ib64);
                    __m256i values1 = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    __m256i values2 = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    _mm256_storeu_si256((__m256i *)block, values1);
                    uint32_t * q8 = (uint32_t *)y[2*ib64+0].qs;
                    for (int l = 0; l < 4; ++l) {
                        q8[8*l + k +  0] = block[l + 0];
                        q8[8*l + k + 32] = block[l + 4];
                    }
                    _mm256_storeu_si256((__m256i *)block, values2);
                    q8 = (uint32_t *)y[2*ib64+1].qs;
                    for (int l = 0; l < 4; ++l) {
                        q8[8*l + k +  0] = block[l + 0];
                        q8[8*l + k + 32] = block[l + 4];
                    }
                }
            }
            __m256 vd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh+0));
            __m256 vm = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh+1));
            vm = _mm256_mul_ps(_mm256_set1_ps(-1.f), vm);
            for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                __m128i iscales16 = _mm_loadu_si128((const __m128i *)all_ls + ib32);
                __m256i iscales32 = _mm256_cvtepi16_epi32(iscales16);
                __m256 scales = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(iscales32));
                _mm_storeu_si128((__m128i *)y[ib32].d+0, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
                iscales16 = _mm_loadu_si128((const __m128i *)all_ls + ib32 + 8);
                iscales32 = _mm256_cvtepi16_epi32(iscales16);
                scales = _mm256_mul_ps(vm, _mm256_cvtepi32_ps(iscales32));
                _mm_storeu_si128((__m128i *)y[ib32].d+1, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
            }
            y += QK_K/32;
        }
    }
    return true;
}

static bool iqk_convert_q5_k_q8_1_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const iqk_block_q5_K * x8[8];

    block_q8_1_r8 * y = (block_q8_1_r8 *)vy;

    ggml_half dh[16];
    uint16_t all_ls[128];

    uint32_t utmp[4];
    const uint8_t * u8 = (const uint8_t *)utmp;
    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const iqk_block_q5_K *)((const char *)vx + (ix + k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                dh[k+0] = x8[k][i].d;
                dh[k+8] = x8[k][i].dmin;
                iqk_make_q4_scales(x8[k][i].scales, utmp);
                const uint8_t * qs  = x8[k][i].qs;
                __m256i hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                for (int ib64 = 0; ib64 < 4; ++ib64) {
                    all_ls[8*(2*ib64 + 0) + k     ] = u8[2*ib64+0];
                    all_ls[8*(2*ib64 + 1) + k     ] = u8[2*ib64+1];
                    all_ls[8*(2*ib64 + 0) + k + 64] = u8[2*ib64+8];
                    all_ls[8*(2*ib64 + 1) + k + 64] = u8[2*ib64+9];
                    __m256i bits = _mm256_loadu_si256((const __m256i *)qs+ib64);
                    __m256i values1 = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    __m256i values2 = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    values1 = _mm256_or_si256(values1, _mm256_and_si256(_mm256_set1_epi8(0x10), _mm256_slli_epi16(hbits, 4)));
                    values2 = _mm256_or_si256(values2, _mm256_and_si256(_mm256_set1_epi8(0x10), _mm256_slli_epi16(hbits, 3)));
                    hbits = _mm256_srli_epi16(hbits, 2);
                    _mm256_storeu_si256((__m256i *)block, values1);
                    uint32_t * q8 = (uint32_t *)y[2*ib64+0].qs;
                    for (int l = 0; l < 4; ++l) {
                        q8[8*l + k +  0] = block[l + 0];
                        q8[8*l + k + 32] = block[l + 4];
                    }
                    _mm256_storeu_si256((__m256i *)block, values2);
                    q8 = (uint32_t *)y[2*ib64+1].qs;
                    for (int l = 0; l < 4; ++l) {
                        q8[8*l + k +  0] = block[l + 0];
                        q8[8*l + k + 32] = block[l + 4];
                    }
                }
            }
            __m256 vd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh+0));
            __m256 vm = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh+1));
            vm = _mm256_mul_ps(_mm256_set1_ps(-1.f), vm);
            for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                __m128i iscales16 = _mm_loadu_si128((const __m128i *)all_ls + ib32);
                __m256i iscales32 = _mm256_cvtepi16_epi32(iscales16);
                __m256 scales = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(iscales32));
                _mm_storeu_si128((__m128i *)y[ib32].d+0, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
                iscales16 = _mm_loadu_si128((const __m128i *)all_ls + ib32 + 8);
                iscales32 = _mm256_cvtepi16_epi32(iscales16);
                scales = _mm256_mul_ps(vm, _mm256_cvtepi32_ps(iscales32));
                _mm_storeu_si128((__m128i *)y[ib32].d+1, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
            }
            y += QK_K/32;
        }
    }
    return true;
}

static bool iqk_convert_q6_k_q8_0_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_q6_K * x8[8];

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    float all_s[64];
    uint32_t block[8];
    __m256i values[8];

    __m256i ml  = _mm256_set1_epi8(0x0f);
    __m256i mh  = _mm256_set1_epi8(0x30);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_q6_K *)((const char *)vx + (ix + k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                const uint8_t * ql = x8[k][i].ql;
                const uint8_t * qh = x8[k][i].qh;
                for (int i128 = 0; i128 < 2; ++i128) {
                    __m256i lbits1 = _mm256_loadu_si256((const __m256i *)ql + 2*i128 + 0);
                    __m256i lbits2 = _mm256_loadu_si256((const __m256i *)ql + 2*i128 + 1);
                    __m256i hbits  = _mm256_loadu_si256((const __m256i *)qh + i128);
                    values[4*i128+0] = _mm256_or_si256(_mm256_and_si256(lbits1, ml), _mm256_and_si256(_mm256_slli_epi16(hbits, 4), mh));
                    values[4*i128+1] = _mm256_or_si256(_mm256_and_si256(lbits2, ml), _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh));
                    values[4*i128+2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits1, 4), ml), _mm256_and_si256(hbits, mh));
                    values[4*i128+3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits2, 4), ml), _mm256_and_si256(_mm256_srli_epi16(hbits, 2), mh));
                }
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    // We have two blocks of 16 with different scales
                    // We multiply the quants with the scales, find the max value, and convert to 8-bit quants with a single block scale.
                    __m256i q8 = _mm256_add_epi8(values[ib32], _mm256_set1_epi8(-32));
                    __m256i q16_l = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8));
                    __m256i q16_h = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8, 1));
                    q16_l = _mm256_mullo_epi16(q16_l, _mm256_set1_epi16(x8[k][i].scales[2*ib32+0]));
                    q16_h = _mm256_mullo_epi16(q16_h, _mm256_set1_epi16(x8[k][i].scales[2*ib32+1]));
                    __m256i abs_q16_l = _mm256_sign_epi16(q16_l, q16_l);
                    __m256i abs_q16_h = _mm256_sign_epi16(q16_h, q16_h);
                    __m256i max_q16 = _mm256_max_epi16(abs_q16_l, abs_q16_h);
                    __m256i max_q32 = _mm256_cvtepi16_epi32(_mm_max_epi16(_mm256_castsi256_si128(max_q16), _mm256_extracti128_si256(max_q16, 1)));
                    __m128i imax4 = _mm_max_epi32(_mm256_castsi256_si128(max_q32), _mm256_extracti128_si256(max_q32, 1));
                    __m128 max4  = _mm_cvtepi32_ps(imax4);
                    max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
                    max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
                    float max = _mm_cvtss_f32(max4) / 127;
                    all_s[8*ib32+k] = d*max;
                    if (max > 1e-9f) {
                        __m256 scale = _mm256_set1_ps(1/max);
                        __m256i i0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16_l));
                        __m256i i1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16_l, 1));
                        __m256i i2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16_h));
                        __m256i i3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16_h, 1));
                        i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i0)), _MM_ROUND_NEAREST));
                        i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i1)), _MM_ROUND_NEAREST));
                        i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i2)), _MM_ROUND_NEAREST));
                        i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i3)), _MM_ROUND_NEAREST));
                        i0 = _mm256_packs_epi32(i0, i1);
                        i2 = _mm256_packs_epi32(i2, i3);
                        i0 = _mm256_packs_epi16(i0, i2);
                        i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
                        _mm256_storeu_si256((__m256i *)block, i0);
                    } else {
                        _mm256_storeu_si256((__m256i *)block, _mm256_setzero_si256());
                    }
                    uint32_t * qs = (uint32_t *)y[ib32].qs;
                    for (int l = 0; l < 4; ++l) {
                        qs[8*l + k +  0] = block[l + 0];
                        qs[8*l + k + 32] = block[l + 4];
                    }
                }
            }
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                _mm_storeu_si128((__m128i *)y[ib32].d, _mm256_cvtps_ph(_mm256_loadu_ps(all_s + 8*ib32), _MM_FROUND_TO_NEAREST_INT));
            }
            y += QK_K/32;
        }
    }
    return true;
}

static inline float iqk_convert_to_q8_k_r8(int k, float d0, const __m256i * qx, const int16_t * scales, uint32_t * block, int8_t * q8_k) {
    __m256i max_i16 = _mm256_setzero_si256();
    __m256i qs[16];
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        qs[2*ib32+0] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qx[ib32]));
        qs[2*ib32+1] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qx[ib32], 1));
        qs[2*ib32+0] = _mm256_mullo_epi16(qs[2*ib32+0], _mm256_set1_epi16(scales[2*ib32+0]));
        qs[2*ib32+1] = _mm256_mullo_epi16(qs[2*ib32+1], _mm256_set1_epi16(scales[2*ib32+1]));
        max_i16 = _mm256_max_epi16(max_i16, _mm256_sign_epi16(qs[2*ib32+0], qs[2*ib32+0]));
        max_i16 = _mm256_max_epi16(max_i16, _mm256_sign_epi16(qs[2*ib32+1], qs[2*ib32+1]));
    }
    __m256i max_q32 = _mm256_cvtepi16_epi32(_mm_max_epi16(_mm256_castsi256_si128(max_i16), _mm256_extracti128_si256(max_i16, 1)));
    __m128i imax4 = _mm_max_epi32(_mm256_castsi256_si128(max_q32), _mm256_extracti128_si256(max_q32, 1));
    __m128 max4  = _mm_cvtepi32_ps(imax4);
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    bool needs_scaling = true;
    float dnew = _mm_cvtss_f32(max4) * d0;
    if (dnew < 1.f) {
        dnew = 1.f; needs_scaling = false;
    }
    __m256 scale = _mm256_set1_ps(fabsf(dnew) > 1e-9f ? 1/dnew : 0.f);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        if (needs_scaling) {
            __m256i i0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(qs[2*ib32+0]));
            __m256i i1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(qs[2*ib32+0], 1));
            __m256i i2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(qs[2*ib32+1]));
            __m256i i3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(qs[2*ib32+1], 1));
            i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i0)), _MM_ROUND_NEAREST));
            i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i1)), _MM_ROUND_NEAREST));
            i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i2)), _MM_ROUND_NEAREST));
            i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i3)), _MM_ROUND_NEAREST));
            i0 = _mm256_packs_epi32(i0, i1);
            i2 = _mm256_packs_epi32(i2, i3);
            i0 = _mm256_packs_epi16(i0, i2);
            i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256((__m256i *)block, i0);
        } else {
            // 0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31
            __m256i i0 = _mm256_packs_epi16(qs[2*ib32+0], qs[2*ib32+1]);
            __m128i i0_l = _mm256_castsi256_si128(i0);
            __m128i i0_h = _mm256_extracti128_si256(i0, 1);
            _mm_storeu_si128((__m128i *)block+0, _mm_unpacklo_epi64(i0_l, i0_h));
            _mm_storeu_si128((__m128i *)block+1, _mm_unpackhi_epi64(i0_l, i0_h));
        }
        uint32_t * q = (uint32_t *)q8_k + 64*ib32;
        for (int l = 0; l < 8; ++l) {
            q[8*l + k] = block[l];
        }
    }
    return dnew;
}

static bool iqk_convert_iq4_xs_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq4_xs * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m128i values128 = _mm_loadu_si128((const __m128i *)iq4k_values);
    __m256i values = _mm256_broadcastsi128_si256(values128);

    int16_t  ls[16];
    float    dnew[8];
    uint32_t block[8];
    __m256i  xv[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq4_xs *)((const char *)vx + (ix + k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    ls[2*ib32+0] = ls[2*ib32+1] = (((x8[k][i].scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((x8[k][i].scales_h >> 2*ib32) & 3) << 4)) - 32;
                    __m128i bits = _mm_loadu_si128((const __m128i *)x8[k][i].qs + ib32);
                    xv[ib32] = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(bits, 4), bits), _mm256_set1_epi8(0xf));
                    xv[ib32] = _mm256_shuffle_epi8(values, xv[ib32]);
                }
                dnew[k] = d * iqk_convert_to_q8_k_r8(k, 1.f/127, xv, ls, block, y[i].qs);
            }
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(_mm256_loadu_ps(dnew), _MM_ROUND_NEAREST));
        }
        y += nb;
    }
    return true;
}

static const int8_t iq2nl_values[8] = {-31, -13, 1, 17,   -26, -8, 6, 22};

static bool iqk_convert_iq2_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq2_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values;
    {
        __m128i v = _mm_loadl_epi64((const __m128i *)iq2nl_values);
        values = MM256_SET_M128I(v, v);
    }

    ggml_half dh[8];
    float     dnew[8];
    uint32_t  block[8];
    int16_t   ls[16];

    __m256i  xv[8];

    __m256i ml = _mm256_set1_epi8(0x03);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const ggml_half * dptr = (const ggml_half *)((const char *)vx + (ix+k)*bx);
            dh[k] = dptr[0];
            x8[k] = (const block_iq2_ks *)(dptr + 1);
        }
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                uint16_t extra = x8[k][i].extra;
                for (int i128 = 0; i128 < 2; ++i128) {
                    ls[8*i128+0] = ls[8*i128+1] = ((x8[k][i].scales[2*i128+0] & 0xf) | ((extra >> 4) & 0x10)) - 16;
                    ls[8*i128+2] = ls[8*i128+3] = ((x8[k][i].scales[2*i128+0] >>  4) | ((extra >> 5) & 0x10)) - 16;
                    ls[8*i128+4] = ls[8*i128+5] = ((x8[k][i].scales[2*i128+1] & 0xf) | ((extra >> 6) & 0x10)) - 16;
                    ls[8*i128+6] = ls[8*i128+7] = ((x8[k][i].scales[2*i128+1] >>  4) | ((extra >> 7) & 0x10)) - 16;
                    __m256i bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+i128);
                    xv[4*i128+0] = _mm256_and_si256(bits, ml);
                    xv[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), ml);
                    xv[4*i128+2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), ml);
                    xv[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), ml);
                    xv[4*i128+0] = _mm256_add_epi8(xv[4*i128+0], _mm256_set1_epi8((extra << 2) & 0x04));
                    xv[4*i128+1] = _mm256_add_epi8(xv[4*i128+1], _mm256_set1_epi8((extra << 1) & 0x04));
                    xv[4*i128+2] = _mm256_add_epi8(xv[4*i128+2], _mm256_set1_epi8((extra >> 0) & 0x04));
                    xv[4*i128+3] = _mm256_add_epi8(xv[4*i128+3], _mm256_set1_epi8((extra >> 1) & 0x04));
                    xv[4*i128+0] = _mm256_shuffle_epi8(values, xv[4*i128+0]);
                    xv[4*i128+1] = _mm256_shuffle_epi8(values, xv[4*i128+1]);
                    xv[4*i128+2] = _mm256_shuffle_epi8(values, xv[4*i128+2]);
                    xv[4*i128+3] = _mm256_shuffle_epi8(values, xv[4*i128+3]);
                    extra >>= 4;
                }
                dnew[k] = iqk_convert_to_q8_k_r8(k, 1.f/125, xv, ls, block, y[i].qs);
            }
            __m256 vd = _mm256_mul_ps(_mm256_loadu_ps(dnew), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh)));
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(vd, _MM_ROUND_NEAREST));
        }
        y += nb;
    }
    return true;
}

static bool iqk_convert_iq2_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq2_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values;
    {
        __m128i v = _mm_loadl_epi64((const __m128i *)iq2nl_values);
        values = MM256_SET_M128I(v, v);
    }

    __m256i  xv[8];
    uint32_t block[8];

    const __m128i scale_shuffle = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);

    union { __m256i vec; int16_t val[16]; } helper;

    __m256i ml = _mm256_set1_epi8(0x03);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq2_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                uint64_t aux64; memcpy(&aux64, x8[k][i].scales, 8);
                __m128i scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
                scl = _mm_add_epi8(scl, _mm_set1_epi8(-8));
                helper.vec = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scl, scale_shuffle));
                uint16_t extra = x8[k][i].extra;
                for (int i128 = 0; i128 < 2; ++i128) {
                    __m256i bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+i128);
                    xv[4*i128+0] = _mm256_and_si256(bits, ml);
                    xv[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), ml);
                    xv[4*i128+2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), ml);
                    xv[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), ml);
                    __m256i shift1 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x02) << 1), _mm_set1_epi8((extra & 0x01) << 2));
                    __m256i shift2 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x08) >> 1), _mm_set1_epi8((extra & 0x04) >> 0));
                    __m256i shift3 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x20) >> 3), _mm_set1_epi8((extra & 0x10) >> 2));
                    __m256i shift4 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x80) >> 5), _mm_set1_epi8((extra & 0x40) >> 4));
                    xv[4*i128+0] = _mm256_add_epi8(xv[4*i128+0], shift1);
                    xv[4*i128+1] = _mm256_add_epi8(xv[4*i128+1], shift2);
                    xv[4*i128+2] = _mm256_add_epi8(xv[4*i128+2], shift3);
                    xv[4*i128+3] = _mm256_add_epi8(xv[4*i128+3], shift4);
                    xv[4*i128+0] = _mm256_shuffle_epi8(values, xv[4*i128+0]);
                    xv[4*i128+1] = _mm256_shuffle_epi8(values, xv[4*i128+1]);
                    xv[4*i128+2] = _mm256_shuffle_epi8(values, xv[4*i128+2]);
                    xv[4*i128+3] = _mm256_shuffle_epi8(values, xv[4*i128+3]);
                    extra >>= 8;
                }
                float dnew = iqk_convert_to_q8_k_r8(k, 1.f/120, xv, helper.val, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
    return true;
}

static bool iqk_convert_iq3_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq3_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values;
    {
        __m128i v = _mm_loadu_si128((const __m128i *)iq3nl_values);
        values = _mm256_broadcastsi128_si256(v);
    }

    ggml_half drow[8];
    float dnew[8];
    int16_t ls[16];

    __m256i xv[8];
    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const ggml_half * dptr = (const ggml_half *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq3_ks *)(dptr + 1);
        }
        __m256 vd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)drow));
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                __m256i hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                uint16_t extra = x8[k][i].extra;
                uint8_t extra_v = extra >> 8;
                for (int j = 0; j < 4; ++j) {
                    ls[2*j+0] = ls[2*j+1] = ((x8[k][i].scales[j] & 0xf) | ((extra << 4) & 0x10)) - 16;
                    ls[2*j+8] = ls[2*j+9] = ((x8[k][i].scales[j] >>  4) | ((extra << 0) & 0x10)) - 16;
                    extra >>= 1;
                }
                for (int i128 = 0; i128 < QK_K/128; ++i128) {
                    __m256i lbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs + i128);
                    for (int j = 0; j < 4; ++j) {
                        xv[4*i128+j] = _mm256_or_si256(_mm256_and_si256(lbits, _mm256_set1_epi8(3)), _mm256_and_si256(_mm256_slli_epi16(hbits, 2), _mm256_set1_epi8(4)));
                        xv[4*i128+j] = _mm256_add_epi8(xv[4*i128+j], _mm256_set1_epi8((extra_v & 1) << 3));
                        xv[4*i128+j] = _mm256_shuffle_epi8(values, xv[4*i128+j]);
                        extra_v >>= 1;
                        lbits = _mm256_srli_epi16(lbits, 2);
                        hbits = _mm256_srli_epi16(hbits, 1);
                    }
                }
                dnew[k] = iqk_convert_to_q8_k_r8(k, 1.f/127, xv, ls, block, y[i].qs);
            }
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(_mm256_mul_ps(vd, _mm256_loadu_ps(dnew)), _MM_ROUND_NEAREST));
        }
        y += nb;
    }
    return true;
}

static bool iqk_convert_iq3_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq3_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values;
    {
        __m128i v = _mm_loadu_si128((const __m128i *)iq3nl_values);
        values = _mm256_broadcastsi128_si256(v);
    }

    __m256i  xv[8];
    uint32_t block[8];

    const __m128i sign_mask = _mm_set_epi64x(0x8080404020201010, 0x0808040402020101);
    const __m128i hshuff = _mm_loadu_si128((const __m128i*)k_shuff);
    const __m128i scale_shuffle = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);

    union { __m256i vec; int16_t val[16]; } helper;

    __m256i ml = _mm256_set1_epi8(0x03);
    __m256i hmask  = _mm256_set1_epi8(4);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq3_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                uint64_t aux64; memcpy(&aux64, x8[k][i].scales_l, 8);
                __m128i scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
                scl = _mm_add_epi8(_mm_slli_epi16(scl, 1), _mm_set1_epi8(1));
                __m128i sc_signs = _mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi16(x8[k][i].scales_h), sign_mask), sign_mask);
                __m128i sch   = _mm_shuffle_epi8(_mm_or_si128(sc_signs, _mm_set1_epi8(1)), hshuff);
                helper.vec = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(_mm_sign_epi8(scl, sch), scale_shuffle));
                uint16_t extra = x8[k][i].extra;
                __m256i hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                for (int i128 = 0; i128 < 2; ++i128) {
                    __m256i bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+i128);
                    xv[4*i128+0] = _mm256_and_si256(bits, ml);
                    xv[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), ml);
                    xv[4*i128+2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), ml);
                    xv[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), ml);
                    xv[4*i128+0] = _mm256_or_si256(xv[4*i128+0], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), hmask));
                    xv[4*i128+1] = _mm256_or_si256(xv[4*i128+1], _mm256_and_si256(_mm256_slli_epi16(hbits, 1), hmask));
                    xv[4*i128+2] = _mm256_or_si256(xv[4*i128+2], _mm256_and_si256(hbits, hmask));
                    xv[4*i128+3] = _mm256_or_si256(xv[4*i128+3], _mm256_and_si256(_mm256_srli_epi16(hbits, 1), hmask));
                    __m256i shift1 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x02) << 2), _mm_set1_epi8((extra & 0x01) << 3));
                    __m256i shift2 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x08) << 0), _mm_set1_epi8((extra & 0x04) << 1));
                    __m256i shift3 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x20) >> 2), _mm_set1_epi8((extra & 0x10) >> 1));
                    __m256i shift4 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x80) >> 4), _mm_set1_epi8((extra & 0x40) >> 3));
                    xv[4*i128+0] = _mm256_add_epi8(xv[4*i128+0], shift1);
                    xv[4*i128+1] = _mm256_add_epi8(xv[4*i128+1], shift2);
                    xv[4*i128+2] = _mm256_add_epi8(xv[4*i128+2], shift3);
                    xv[4*i128+3] = _mm256_add_epi8(xv[4*i128+3], shift4);
                    xv[4*i128+0] = _mm256_shuffle_epi8(values, xv[4*i128+0]);
                    xv[4*i128+1] = _mm256_shuffle_epi8(values, xv[4*i128+1]);
                    xv[4*i128+2] = _mm256_shuffle_epi8(values, xv[4*i128+2]);
                    xv[4*i128+3] = _mm256_shuffle_epi8(values, xv[4*i128+3]);
                    hbits = _mm256_srli_epi16(hbits, 4);
                    extra >>= 8;
                }
                float dnew = iqk_convert_to_q8_k_r8(k, 1.f/127, xv, helper.val, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
    return true;
}

static bool iqk_convert_iq4_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq4_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values[2];
    {
        __m128i v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
        __m128i v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
        values[0] = _mm256_broadcastsi128_si256(v1);
        values[1] = _mm256_broadcastsi128_si256(v2);
    }

    float drow[8];
    float dnew[8];
    int16_t ls[16];

    __m256i xv[8];
    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq4_ks *)(dptr + 1);
        }
        __m256 vd = _mm256_loadu_ps(drow);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    ls[2*ib32+0] = (x8[k][i].scales[ib32] & 254) - 127;
                    ls[2*ib32+1] = ls[2*ib32+0];
                    __m128i aux128 = _mm_loadu_si128((const __m128i *)x8[k][i].qs+ib32);
                    xv[ib32] = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(aux128, 4), aux128), _mm256_set1_epi8(0xf));
                    xv[ib32] = _mm256_shuffle_epi8(values[x8[k][i].scales[ib32] & 1], xv[ib32]);
                }
                dnew[k] = iqk_convert_to_q8_k_r8(k, 1.f/127, xv, ls, block, y[i].qs);
            }
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(_mm256_mul_ps(vd, _mm256_loadu_ps(dnew)), _MM_ROUND_NEAREST));
        }
        y += nb;
    }
    return true;
}

static bool iqk_convert_iq4_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq4_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values[4];
    {
        __m128i v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
        __m128i v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
        values[0] = _mm256_broadcastsi128_si256(v1);
        values[1] = MM256_SET_M128I(v1, v2);
        values[2] = MM256_SET_M128I(v2, v1);
        values[3] = _mm256_broadcastsi128_si256(v2);
    }

    __m256i  xv[8];
    uint32_t block[8];
    int16_t  ls[16];

    //auto hshuff = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);

    //union { __m256i vec; int16_t val[16]; } helper;

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq4_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                uint16_t extra = x8[k][i].extra;
                //uint64_t aux64;
                //memcpy(&aux64, x8[k][i].scales_l, 8);
                //auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
                //const uint32_t aux32 = *(const uint32_t *)x8[k][i].scales_h;
                //auto aux = _mm_and_si128(_mm_set_epi32(aux32 >> 2, aux32, aux32 << 2, aux32 << 4), _mm_set1_epi8(0x30));
                //auto sch = _mm_shuffle_epi8(aux, hshuff);
                //aux = _mm_add_epi8(_mm_or_si128(scl, sch), _mm_set1_epi8(-32));
                //helper.vec = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(aux, hshuff));
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const uint8_t sh = x8[k][i].scales_h[ib32/2] >> 4*(ib32%2);
                    ls[2*ib32+0] = ((x8[k][i].scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32;
                    ls[2*ib32+1] = ((x8[k][i].scales_l[ib32] >>  4) | ((sh << 2) & 0x30)) - 32;
                    __m128i bits = _mm_loadu_si128((const __m128i *)x8[k][i].qs+ib32);
                    xv[ib32]  = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(bits, 4), bits), _mm256_set1_epi8(0xf));
                    xv[ib32]  = _mm256_shuffle_epi8(values[extra & 3], xv[ib32]); extra >>= 2;
                }
                //float dnew = iqk_convert_to_q8_k_r8(k, 1.f/127, xv, helper.val, block, y[i].qs);
                float dnew = iqk_convert_to_q8_k_r8(k, 1.f/127, xv, ls, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
    return true;
}

static bool iqk_convert_iq5_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq5_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values[2];
    {
        __m128i v1 = _mm_loadu_si128((const __m128i *)iq5nl_values+0);
        __m128i v2 = _mm_loadu_si128((const __m128i *)iq5nl_values+1);
        values[0] = _mm256_broadcastsi128_si256(v1);
        values[1] = _mm256_broadcastsi128_si256(v2);
    }

    float drow[8];
    float dnew[8];
    int16_t ls[16];

    __m256i xv[8];
    uint32_t block[8];

    __m256i mh = _mm256_set1_epi8(-128); // to avoid stupid warning about 0x80 overflowing

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq5_ks *)(dptr + 1);
        }
        __m256 vd = _mm256_loadu_ps(drow);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                __m256i hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                for (int ib64 = 0; ib64 < 4; ++ib64) {
                    ls[4*ib64+0] = (x8[k][i].scales[2*ib64+0] & 254) - 127;
                    ls[4*ib64+1] = ls[4*ib64+0];
                    ls[4*ib64+2] = (x8[k][i].scales[2*ib64+1] & 254) - 127;
                    ls[4*ib64+3] = ls[4*ib64+2];
                    __m256i bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+ib64);
                    xv[2*ib64+0] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[2*ib64+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    __m256i qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 7), mh);
                    __m256i q5vl = _mm256_or_si256(xv[2*ib64+0], qh);
                    __m256i q5vh = _mm256_or_si256(xv[2*ib64+0], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+0] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 6), mh);
                    q5vl = _mm256_or_si256(xv[2*ib64+1], qh);
                    q5vh = _mm256_or_si256(xv[2*ib64+1], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+1] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    __m256i shift1 = _mm256_set1_epi8((x8[k][i].scales[2*ib64+0] & 1) << 1);
                    __m256i shift2 = _mm256_set1_epi8((x8[k][i].scales[2*ib64+1] & 1) << 1);
                    xv[2*ib64+0] = _mm256_add_epi8(xv[2*ib64+0], shift1);
                    xv[2*ib64+1] = _mm256_add_epi8(xv[2*ib64+1], shift2);
                    hbits = _mm256_srli_epi16(hbits, 2);
                }
                dnew[k] = iqk_convert_to_q8_k_r8(k, 1.f/127, xv, ls, block, y[i].qs);
            }
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(_mm256_mul_ps(vd, _mm256_loadu_ps(dnew)), _MM_ROUND_NEAREST));
        }
        y += nb;
    }
    return true;
}

static bool iqk_convert_iq5_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq5_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values[2];
    {
        __m128i v1 = _mm_loadu_si128((const __m128i *)iq5nl_values+0);
        __m128i v2 = _mm_loadu_si128((const __m128i *)iq5nl_values+1);
        values[0] = _mm256_broadcastsi128_si256(v1);
        values[1] = _mm256_broadcastsi128_si256(v2);
    }

    __m256i  xv[8];
    uint32_t block[8];
    int16_t  ls[16];

    __m256i mh = _mm256_set1_epi8(-128); // to avoid stupid warning about 0x80 overflowing

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq5_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                uint16_t extra = x8[k][i].extra;
                __m256i hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                for (int ib64 = 0; ib64 < 4; ++ib64) {
                    ls[4*ib64+0] = ((x8[k][i].scales_l[2*ib64+0] & 0xf) | ((x8[k][i].scales_h[ib64] << 4) & 0x30)) - 32;
                    ls[4*ib64+1] = ((x8[k][i].scales_l[2*ib64+0] >>  4) | ((x8[k][i].scales_h[ib64] << 2) & 0x30)) - 32;
                    ls[4*ib64+2] = ((x8[k][i].scales_l[2*ib64+1] & 0xf) | ((x8[k][i].scales_h[ib64] >> 0) & 0x30)) - 32;
                    ls[4*ib64+3] = ((x8[k][i].scales_l[2*ib64+1] >>  4) | ((x8[k][i].scales_h[ib64] >> 2) & 0x30)) - 32;
                    __m256i bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+ib64);
                    xv[2*ib64+0] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[2*ib64+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    __m256i qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 7), mh);
                    __m256i q5vl = _mm256_or_si256(xv[2*ib64+0], qh);
                    __m256i q5vh = _mm256_or_si256(xv[2*ib64+0], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+0] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 6), mh);
                    q5vl = _mm256_or_si256(xv[2*ib64+1], qh);
                    q5vh = _mm256_or_si256(xv[2*ib64+1], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+1] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    __m256i shift1 = MM256_SET_M128I(_mm_set1_epi8((extra & 2) << 0), _mm_set1_epi8((extra & 1) << 1));
                    __m256i shift2 = MM256_SET_M128I(_mm_set1_epi8((extra & 8) >> 2), _mm_set1_epi8((extra & 4) >> 1));
                    xv[2*ib64+0] = _mm256_add_epi8(xv[2*ib64+0], shift1);
                    xv[2*ib64+1] = _mm256_add_epi8(xv[2*ib64+1], shift2);
                    hbits = _mm256_srli_epi16(hbits, 2);
                    extra >>= 4;
                }
                float dnew = iqk_convert_to_q8_k_r8(k, 1.f/127, xv, ls, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
    return true;
}

static bool iqk_convert_iq6_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq6_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values[4];
    for (int k = 0; k < 4; ++k) {
        __m128i values128 = _mm_loadu_si128((const __m128i *)iq6nl_values + k);
        values[k] = _mm256_broadcastsi128_si256(values128);
    }

    __m256i  xv[8];
    uint32_t block[8];

    union { __m256i vec; int16_t val[16]; } helper;

    __m256i mh1 = _mm256_set1_epi8(1);
    __m256i mh2 = _mm256_set1_epi8(2);
    __m256i mh3 = _mm256_set1_epi8(3);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq6_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                helper.vec = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)x8[k][i].scales));
                uint16_t extra = x8[k][i].extra;
                for (int i128 = 0; i128 < 2; ++i128) {
                    __m256i hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh+i128);
                    __m256i bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+2*i128+0);
                    xv[4*i128+0] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+2*i128+1);
                    xv[4*i128+2] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    for (int j = 0; j < 4; ++j) {
                        // xv[4*i128+k] = make_one(xv[4*i128+k], hbits);
                        __m256i l = xv[4*i128+j];
                        __m256i mask4 = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, mh3), mh3);
                        __m256i h1 = _mm256_andnot_si256(mask4, hbits);
                        __m256i mask2 = _mm256_cmpeq_epi8(_mm256_and_si256(h1, mh1), mh1);
                        __m256i mask3 = _mm256_cmpeq_epi8(_mm256_and_si256(h1, mh2), mh2);
                        __m256i mask1 = _mm256_andnot_si256(_mm256_or_si256(mask4, _mm256_or_si256(mask2, mask3)), _mm256_set1_epi8(-1)); // 0xff;
                        xv[4*i128+j] = _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(mask1, _mm256_shuffle_epi8(values[0], l)),
                                               _mm256_and_si256(mask2, _mm256_shuffle_epi8(values[1], l))),
                               _mm256_or_si256(_mm256_and_si256(mask3, _mm256_shuffle_epi8(values[2], l)),
                                               _mm256_and_si256(mask4, _mm256_shuffle_epi8(values[3], l))));
                        hbits = _mm256_srli_epi16(hbits, 2);
                    }
                    __m256i shift1 = MM256_SET_M128I(_mm_set1_epi8((extra >> 1) & 1), _mm_set1_epi8((extra >> 0) & 1));
                    __m256i shift2 = MM256_SET_M128I(_mm_set1_epi8((extra >> 3) & 1), _mm_set1_epi8((extra >> 2) & 1));
                    __m256i shift3 = MM256_SET_M128I(_mm_set1_epi8((extra >> 5) & 1), _mm_set1_epi8((extra >> 4) & 1));
                    __m256i shift4 = MM256_SET_M128I(_mm_set1_epi8((extra >> 7) & 1), _mm_set1_epi8((extra >> 6) & 1));
                    xv[4*i128+0] = _mm256_add_epi8(xv[4*i128+0], shift1);
                    xv[4*i128+1] = _mm256_add_epi8(xv[4*i128+1], shift2);
                    xv[4*i128+2] = _mm256_add_epi8(xv[4*i128+2], shift3);
                    xv[4*i128+3] = _mm256_add_epi8(xv[4*i128+3], shift4);
                    extra >>= 8;
                }
                float dnew = iqk_convert_to_q8_k_r8(k, 1.f/127, xv, helper.val, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
    return true;
}

static inline void iq23kt_next64(const uint32_t * val, __m256i * result, Trellis3 t) {
    const __m256i offset = _mm256_set1_epi32(-126);
    __m256i vka3 = _mm256_set1_epi32(t.ka3);
    __m256i aux[8];
    for (int i = 0; i < 4; ++i) {
        // auto i8_1 = next8(val[2*i+0], val[2*i+1]);
        __m256i mval = MM256_SET_M128I(_mm_set1_epi32(val[2*i+1]), _mm_set1_epi32(val[2*i+0]));
        __m256i i8_1 = _mm256_mullo_epi32(mval, t.mka);
        __m256i i8_2 = _mm256_mullo_epi32(i8_1, vka3);
        i8_1 = _mm256_and_si256(i8_1, _mm256_set1_epi32(0x3f3f3f3f));
        i8_2 = _mm256_and_si256(i8_2, _mm256_set1_epi32(0x3f3f3f3f));
#ifdef HAVE_FANCY_SIMD
        aux[i+0] = _mm256_dpbusd_epi32(offset, _mm256_set1_epi32(0x01010101), i8_1);
        aux[i+4] = _mm256_dpbusd_epi32(offset, _mm256_set1_epi32(0x01010101), i8_2);
#else
        __m256i dot1 = _mm256_maddubs_epi16(i8_1, _mm256_set1_epi32(0x01010101));
        __m256i dot2 = _mm256_maddubs_epi16(i8_2, _mm256_set1_epi32(0x01010101));
        aux[i+0] = _mm256_add_epi32(offset, _mm256_madd_epi16(dot1, _mm256_set1_epi16(1)));
        aux[i+4] = _mm256_add_epi32(offset, _mm256_madd_epi16(dot2, _mm256_set1_epi16(1)));
#endif
    }
    for (int k = 0; k < 2; ++k) {
        aux[4*k+0] = _mm256_packs_epi32(aux[4*k+0], aux[4*k+1]); //  0,  1,  2,  3,  8,  9, 10, 11,  4,  5,  6,  7, 12, 13, 14, 15
        aux[4*k+2] = _mm256_packs_epi32(aux[4*k+2], aux[4*k+3]); // 16, 17, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 30, 31
        aux[4*k+0] = _mm256_packs_epi16(aux[4*k+0], aux[4*k+2]); //  0,  1,  2,  3,  8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
                                                                    //  4,  5,  6,  7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
        result[k] = _mm256_permutevar8x32_epi32(aux[4*k+0], t.shuffle);
    }
}

static inline __m256i iq4kt_next32(const uint32_t * val, Trellis3 t) {
    const __m256i offset = _mm256_set1_epi32(-126);
    __m256i aux[4];
    for (int i = 0; i < 4; ++i) {
        __m256i mval = MM256_SET_M128I(_mm_set1_epi32(val[2*i+1]), _mm_set1_epi32(val[2*i+0]));
        __m256i i8 = _mm256_and_si256(_mm256_mullo_epi32(mval, t.mka), _mm256_set1_epi32(0x3f3f3f3f));
#ifdef HAVE_FANCY_SIMD
        aux[i] = _mm256_dpbusd_epi32(offset, _mm256_set1_epi32(0x01010101), i8);
#else
        __m256i dot = _mm256_maddubs_epi16(i8, _mm256_set1_epi32(0x01010101));
        aux[i] = _mm256_add_epi32(offset, _mm256_madd_epi16(dot, _mm256_set1_epi16(1)));
#endif
    }
    aux[0] = _mm256_packs_epi32(aux[0], aux[1]); //  0,  1,  2,  3,  8,  9, 10, 11,  4,  5,  6,  7, 12, 13, 14, 15
    aux[2] = _mm256_packs_epi32(aux[2], aux[3]); // 16, 17, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 30, 31
    aux[0] = _mm256_packs_epi16(aux[0], aux[2]); //  0,  1,  2,  3,  8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
                                                    //  4,  5,  6,  7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
    return _mm256_permutevar8x32_epi32(aux[0], t.shuffle);
}

static bool iqk_dequantize_iq2_kt_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);
    const int nb = n/QK_K;

    // Trellis3 trellis;
    Trellis3 trellis = make_Trellis3(false);

    __m128i shifts = _mm_set_epi32(0, 0, 4, 0);
    __m128i values = _mm_loadu_si128((const __m128i *)iq4k_values);

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const block_iq2_kt * x8[8];
    float dkt[8];
    float ls[8];
    float ls_all[64];
    uint32_t idx[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char*)vx + (ix+k)*bx);
            dkt[k] = dptr[0];
            x8[k] = (const block_iq2_kt *)(dptr + 1);
        }
        __m256 vd = _mm256_mul_ps(_mm256_set1_ps(1.05f), _mm256_loadu_ps(dkt));

        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                __m128i s8 = _mm_set1_epi32(*(const uint32_t *)x8[k][i].scales);
                s8 = _mm_and_si128(_mm_srlv_epi32(s8, shifts), _mm_set1_epi8(0xf));
                s8 = _mm_shuffle_epi8(values, s8);
                __m256i s32 = _mm256_cvtepi8_epi32(s8);
                _mm256_storeu_ps(ls_all + 8*k, _mm256_cvtepi32_ps(s32));
            }
            for (int ib = 0; ib < QK_K/32; ++ib) {
                for (int k = 0; k < 8; ++k) ls[k] = ls_all[8*k+ib];
                __m256 scales = _mm256_mul_ps(vd, _mm256_loadu_ps(ls));
                _mm_storeu_si128((__m128i *)y[ib].d, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
                for (int j = 0; j < 4; ++j) {
                    for (int k = 0; k < 8; ++k) {
                        const uint16_t * ql = (const uint16_t *)x8[k][i].ql;
                        idx[k] = ql[4*ib+j] + 4096;
                    }
                    __m256i packed[2];
                    iq23kt_next64(idx, packed, trellis);
                    _mm256_storeu_si256((__m256i *)y[ib].qs+2*j+0, packed[0]);
                    _mm256_storeu_si256((__m256i *)y[ib].qs+2*j+1, packed[1]);
                }
            }
            y += 8; // = QK_K/32;
        }
    }
    return true;
}

static bool iqk_dequantize_iq3_kt_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);
    const int nb = n/QK_K;

    // Trellis3<false, true> trellis;
    Trellis3 trellis = make_Trellis3(false);

    __m128i shifts = _mm_set_epi32(0, 0, 4, 0);

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const block_iq3_kt * x8[8];
    float dkt[8];
    float ls[8];
    float ls_all[64];
    uint32_t idx[8];
    uint32_t sign_bits[16];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char*)vx + (ix+k)*bx);
            dkt[k] = dptr[0];
            x8[k] = (const block_iq3_kt *)(dptr + 1);
        }
        __m256 vd = _mm256_mul_ps(_mm256_set1_ps(1.01f), _mm256_loadu_ps(dkt));

        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                __m128i s8 = _mm_set1_epi32(*(const uint32_t *)x8[k][i].scales);
                s8 = _mm_and_si128(_mm_srlv_epi32(s8, shifts), _mm_set1_epi8(0xf));
                __m256i s32 = _mm256_cvtepi8_epi32(s8);
                _mm256_storeu_ps(ls_all + 8*k, _mm256_cvtepi32_ps(s32));
            }
            __m256i mask = _mm256_set1_epi8(1);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                for (int k = 0; k < 8; ++k) ls[k] = ls_all[8*k+ib];
                __m256 scales = _mm256_mul_ps(vd, _mm256_loadu_ps(ls));
                _mm_storeu_si128((__m128i *)y[ib].d, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
                for (int j = 0; j < 4; ++j) {
                    for (int k = 0; k < 8; ++k) {
                        const uint16_t * ql = (const uint16_t *)x8[k][i].ql;
                        idx[k] = ql[4*ib+j] + 4096;
                        const uint32_t * qh = (const uint32_t *)x8[k][i].qh;
                        sign_bits[k+0] = qh[2*j+0];
                        sign_bits[k+8] = qh[2*j+1];
                    }
                    __m256i packed[2];
                    iq23kt_next64(idx, packed, trellis);
                    for (int k = 0; k < 2; ++k) {
                        packed[k] = _mm256_sign_epi8(packed[k], packed[k]);
                    }
                    __m256i signs1 = _mm256_loadu_si256((const __m256i *)sign_bits+0);
                    __m256i signs2 = _mm256_loadu_si256((const __m256i *)sign_bits+1);
                    signs1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(signs1, mask), mask), _mm256_set1_epi8(1));
                    signs2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(signs2, mask), mask), _mm256_set1_epi8(1));
                    packed[0] = _mm256_sign_epi8(packed[0], signs1);
                    packed[1] = _mm256_sign_epi8(packed[1], signs2);
                    _mm256_storeu_si256((__m256i *)y[ib].qs+2*j+0, packed[0]);
                    _mm256_storeu_si256((__m256i *)y[ib].qs+2*j+1, packed[1]);
                }
                mask = _mm256_slli_epi16(mask, 1);
            }
            y += 8; // = QK_K/32;
        }
    }
    return true;
}

static bool iqk_dequantize_iq4_kt_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);
    const int nb = n/QK_K;
    const int kNumGroups = 64;

    // Trellis3 trellis;
    Trellis3 trellis = make_Trellis3(false);

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const block_iq4_kt * x8[8];
    float dkt[8];
    int32_t ls[8];
    uint32_t idx0[8], idx[16];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char*)vx + (ix+k)*bx);
            dkt[k] = dptr[0];
            x8[k] = (const block_iq4_kt *)(dptr + 1);
        }
        __m256 vd = _mm256_loadu_ps(dkt);

        for (int i = 0; i < nb; ++i) {
            for (int ib = 0; ib < QK_K/32; ++ib) {
                for (int k = 0; k < 8; ++k) {
                    ls[k] = ((x8[k][i].qs[ib] & 0xff) >> 1) - 64;
                    idx0[k] = ((x8[k][i].qs[ib] & 1) << 15) + 4096;
                }
                __m256 scales = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)ls)));
                _mm_storeu_si128((__m128i *)y[ib].d, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
                int shift1 = 8 - 4*(ib/4);
                for (int j = 0; j < 8; ++j) {
                    for (int k = 0; k < 8; ++k) {
                        const uint8_t * ql = (const uint8_t *)(x8[k][i].qs + 8);
                        const uint8_t * qh = ql + kNumGroups;
                        const uint32_t sh = x8[k][i].qs[ib] >> (8 + 3*j);
                        idx[k+0] = ql[8*ib+j] + ((qh[8*(ib%4)+j] << shift1) & 0xf00) + ((sh & 7) << 12) + idx0[k];
                    }
                    _mm256_storeu_si256((__m256i *)y[ib].qs+j, iq4kt_next32(idx, trellis));
                }
            }
            y += 8; // = QK_K/32;
        }
    }
    return true;
}

static bool iqk_convert_repack(int typeA, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    switch (typeA) {
        //kquants
        case GGML_TYPE_Q2_K  : return iqk_convert_q2_k_q8_k_r8  (n, vx, bx, vy, nrc_x);
        case GGML_TYPE_Q3_K  : return iqk_convert_q3_k_q8_k_r8  (n, vx, bx, vy, nrc_x);
        case GGML_TYPE_Q4_K  : return iqk_convert_q4_k_q8_1_r8  (n, vx, bx, vy, nrc_x);
        case GGML_TYPE_Q5_K  : return iqk_convert_q5_k_q8_1_r8  (n, vx, bx, vy, nrc_x);
        case GGML_TYPE_Q6_K  : return iqk_convert_q6_k_q8_0_r8  (n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ4_XS: return iqk_convert_iq4_xs_q8_k_r8(n, vx, bx, vy, nrc_x);
        //iqk_quants
        case GGML_TYPE_IQ2_KS : return iqk_convert_iq2_ks_q8_k_r8(n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ2_K  : return iqk_convert_iq2_k_q8_k_r8 (n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ3_KS : return iqk_convert_iq3_ks_q8_k_r8(n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ3_K  : return iqk_convert_iq3_k_q8_k_r8 (n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ4_KS : return iqk_convert_iq4_ks_q8_k_r8(n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ4_K  : return iqk_convert_iq4_k_q8_k_r8 (n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ5_KS : return iqk_convert_iq5_ks_q8_k_r8(n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ5_K  : return iqk_convert_iq5_k_q8_k_r8 (n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ6_K  : return iqk_convert_iq6_k_q8_k_r8 (n, vx, bx, vy, nrc_x);
        //ktquants
        case GGML_TYPE_IQ2_KT: return iqk_dequantize_iq2_kt_q80_r8(n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ3_KT: return iqk_dequantize_iq3_kt_q80_r8(n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ4_KT: return iqk_dequantize_iq4_kt_q80_r8(n, vx, bx, vy, nrc_x);
        default:
            break;
    }
    return false;
}

static inline void iqk_mul_mat_NxM(int n, const void * vx, size_t bx, struct DataInfo * info, int nrc_x, int nrc_y) {
    const int k_x_step = 64; // This works best on my Ryzen-7950X (but differences to other tile size are small)
    int ny = funcs.size();
    int n_left = nrc_y - info->cur_y;
    int n_step = n_left/ny;
    if (n_step > 0) {
        if (n_step*ny != n_left) {
            ++n_step;
            int ny1 = n_left/n_step;
            int ny2 = ny1 + 1;
            int my1 = n_step*ny2 - n_left;
            int my2 = n_step - my1;
            for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                struct DataInfo this_info = *info;
                this_info.s += ix;
                int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                for (int iy = 0; iy < my1; ++iy) {
                    funcs[ny1-1](n, (const void *)((const char *)vx + ix*bx), bx, &this_info, this_nrc_x);
                    this_info.cur_y += ny1;
                }
                for (int iy = 0; iy < my2; ++iy) {
                    funcs[ny2-1](n, (const void *)((const char *)vx + ix*bx), bx, &this_info, this_nrc_x);
                    this_info.cur_y += ny2;
                }
            }
            info->cur_y += n_left;
        }
        else {
            for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                struct DataInfo this_info = *info;
                this_info.s += ix;
                int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                for (int iy = 0; iy < n_step; ++iy) {
                    funcs[ny-1](n, (const void *)((const char *)vx + ix*bx), bx, &this_info, this_nrc_x);
                    this_info.cur_y += ny;
                }
            }
            info->cur_y += ny * n_step;
        }
    }
    n_left = nrc_y - info->cur_y;
    if (n_left > 0) {
        funcs[n_left-1](n, vx, bx, info, nrc_x);
    }
}

static std::vector<char> & thread_local_work_buffer() {
    thread_local std::vector<char> f;
    return f;
}

extern "C" __attribute__ ((visibility ("default"))) void iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth) {

    auto etypeA = ggml_type(typeA);
    if (auto dequant_type = iqk_is_dequant_better(etypeA, Ny); dequant_type != etypeA) {
        iqk_prepare(dequant_type, typeB, ne00);

        const int k_x_step = 32;

        int num_rows = iqk_num_rows(dequant_type);
        GGML_ASSERT(Nx%num_rows == 0);
        int nrc_x = (Nx/num_rows + nth - 1)/nth;
        int first_x = ith*nrc_x;
        if (first_x + nrc_x > Nx/num_rows) nrc_x = Nx/num_rows - first_x;
        first_x *= num_rows;
        nrc_x   *= num_rows;

        size_t row_size_qx = ggml_row_size(dequant_type, ne00);
        size_t row_size_qy = strideB;

        //printf("Dequant mul mat %s x %s: ne00 = %d, row_size = %d\n", ggml_type_name(dequant_type), ggml_type_name(ggml_type(typeB)), (int)ne00, (int)row_size_qx);

        struct DataInfo info = {C + first_x, (const char *)B, (size_t)stride_C, row_size_qy, 0, 1, 0};

        auto& f = thread_local_work_buffer();

        for (int ix = 0; ix < nrc_x; ix += k_x_step) {
            struct DataInfo this_info = info;
            this_info.s += ix;
            int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
            if (f.size() < row_size_qx*this_nrc_x) f.resize(row_size_qx*this_nrc_x);
            if (!iqk_convert_repack(typeA, ne00, (const char *)A + (first_x + ix)*strideA, strideA, f.data(), this_nrc_x)) {
                GGML_ABORT("Fatal error");
            }
            iqk_mul_mat_NxM(ne00, f.data(), row_size_qx, &this_info, this_nrc_x, Ny);
        }
        return;
    }

    iqk_prepare(typeA, typeB, ne00);

    size_t row_size_qx = strideA; //*ggml_type_size(ggml_type(typeA));
    size_t row_size_qy = strideB; //*ggml_type_size(ggml_type(typeB));
    //if (ith == 0) printf("%s: ne00 = %d, row_size_qx = %d, strideA = %d\n", __func__, int(ne00), int(row_size_qx), int(strideA));

    int num_rows = iqk_num_rows(etypeA);
    GGML_ASSERT(Nx%num_rows == 0);
    int nrc_x = (Nx/num_rows + nth - 1)/nth;
    int first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx/num_rows) nrc_x = Nx/num_rows - first_x;

    struct DataInfo info = {C + first_x*num_rows, (const char *)B, (size_t)stride_C, row_size_qy, 0, 1, 0};
    iqk_mul_mat_NxM(ne00, (const char *)A + row_size_qx*first_x*num_rows, row_size_qx, &info, nrc_x*num_rows, Ny);
}
