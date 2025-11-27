#include "mask.cuh"

const int NUM_BUCKETS = 32;

__global__ static void generate_mask_by_rows(const float* activation, int8_t* device_mask, unsigned int k, unsigned int rows, float b30, float b0) {
    unsigned int group_start = blockIdx.x * 1024;
    unsigned int group_end = (group_start + 1024 < k) ? group_start + 1024 : k;
    unsigned int actual_group_size = group_end - group_start;
    unsigned int top_count = actual_group_size / 2;

    if (group_start >= k) return;
    int8_t* device_mask_group = device_mask + group_start;

    // Calculate the sum of squares
    __shared__ float device_squared_values[1024];
    #pragma unroll
    for(int i = threadIdx.x * 4; i < actual_group_size; i += blockDim.x * 4) {
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float val[4];
        #pragma unroll
        for (int r = 0; r < rows; r++) {
            *(float4 *)val = *(float4 *)(activation + r * k + blockIdx.x * 1024 + i);
            for(int j = 0; j < 4; ++j){
                sum[j] += val[j] * val[j];
            }
        }
        #pragma unroll
        for(int j = 0; j <4; ++j){
            device_squared_values[i + j] = sum[j];
        }
    }
    __syncthreads();

    const float* group_values = device_squared_values;

    // Shared Memory: Bucket Counting
    __shared__ int bucket_counts[NUM_BUCKETS];

    int bucket_count = 0;
    __shared__ int edge_bucket[1];

    if (threadIdx.x < NUM_BUCKETS) {
        float lower_bound, upper_bound;
        if (threadIdx.x == 0 ) {
            upper_bound = INFINITY;
            lower_bound = b0;
        } else {
            if (threadIdx.x == NUM_BUCKETS - 1)
                lower_bound = 0.0f;
            else
                lower_bound = b0 - (b0 - b30)  / (NUM_BUCKETS -2.0f) * threadIdx.x ;
            upper_bound = b0 - (b0 - b30)  / (NUM_BUCKETS -2.0f) * (threadIdx.x - 1.0f);
        }

        // Phase 1: Assign a bucket to each element
        for (int i = 0; i < actual_group_size; i += 4) {
            float value[4];
            *(float4 *)value = *(float4 *)&group_values[i];
            for(int j = 0; j < 4; ++j) {
                if (value[j] >= lower_bound && value[j] < upper_bound) {
                    bucket_count++;
                    device_mask_group[i + j] = threadIdx.x;
                }
            }
        }
        bucket_counts[threadIdx.x] = bucket_count;
    }
    __syncthreads();

    // Phase 2: Count the number of elements in each bucket and identify the critical bucket.
    if (threadIdx.x == 0) {
        int collected = 0;
        int edge_bucket_in = 0;
        while(collected < top_count && edge_bucket_in < NUM_BUCKETS) {
            collected += bucket_counts[edge_bucket_in];
            edge_bucket_in++;
        }
        edge_bucket_in = max(0, edge_bucket_in - 1);
        edge_bucket[0] = edge_bucket_in;
    }
    __syncthreads();

    int8_t threshold = edge_bucket[0];
    // Phase 3: Modify the masked
    if (threshold != 0) {
        for (int i = threadIdx.x ; i < actual_group_size; i += blockDim.x) {
#if DISABLE_FAST_GPU_VERIFY
            device_mask_group[i] = max(0, device_mask_group[i] + 1 - threshold);
#else
            device_mask_group[i] = max(0, device_mask_group[i] - threshold);
#endif
        }
    }
}

void generate_mask(const ggml_tensor * node, int8_t* bitmask, const float b30, const float b0, unsigned int rows, cudaStream_t stream) {
    const float* activation = (const float*) node->data;
    unsigned int k = node->ne[0];
    unsigned int m = node->ne[1];
    int threads_per_block = 256;
    unsigned int block_num = (k + 1023) / 1024;
    unsigned int bias = (m / 512) * 12;
    for (unsigned int base = bias; base < m ; base += rows){
        generate_mask_by_rows<<<block_num, threads_per_block, 0, stream>>>(activation + base * k, bitmask + ((base - bias) / rows) * k, k, rows, b30, b0);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ static void mask_activation_by_rows(float * activation, const int8_t * bitmask, unsigned int k, unsigned int m, unsigned int rows) {
    unsigned int channel_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int bias = (m / 512) * 12;
    for (int base = bias; base < m; base += rows){
        if (channel_id < k) {
            if (bitmask[((base - bias) / rows) * k + channel_id] != 0) {
                // set whole channel zero
                for (int r = 0; r < rows; r++) {
                    activation[(base + r) * k + channel_id] = 0.0f;
                }
            }
        }
    }
}

void mask_activation(ggml_tensor * node, const int8_t* bitmask, unsigned int rows, cudaStream_t stream) {
    float * activation = (float *) node->data;
    unsigned int k = node->ne[0];
    unsigned int m = node->ne[1];
    unsigned int blockSize = 256;
    unsigned int gridSize = (k + blockSize - 1) / blockSize;

    mask_activation_by_rows<<<gridSize, blockSize, 0, stream>>>(activation, bitmask, k, m, rows);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

__global__ static void maskColumnsByColKernel(const float* src, float* dst, const int8_t* bitmask, int k, int m, int rows) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int bias = (m / 512) * 12;

    for (int j = 0; j < bias; j += 1) {
        dst[j * k + col] = src[j * k + col];
    }

    for (int base = bias; base < m ; base += rows) {
        int offset_mask = ((base - bias) / rows) * k;

        if (col < k) {
            if (bitmask[offset_mask + col] != 0) {
                // set channal vaues to zero
                for (int row = 0; row < rows; row++) {
                    dst[(base + row) * k + col] = 0.0f;
                }
            } else {
                // copy channal values
                for (int row = 0; row < rows; row++) {
                    dst[(base + row) * k + col] = src[(base + row) * k + col];
                }
            }
        }
    }
}

void maskColumnsGPUDevice(const ggml_tensor * node, float* masked_act, const int8_t* bitmask, unsigned int rows, cudaStream_t stream) {
    const float * activation = (float *) node->data;
    unsigned int k = node->ne[0];
    unsigned int m = node->ne[1];
    unsigned int blockSize = 256;
    unsigned int gridSize = (k + blockSize - 1) / blockSize;

    maskColumnsByColKernel<<<gridSize, blockSize, 0, stream>>>(activation, masked_act, bitmask, k, m, rows);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
}