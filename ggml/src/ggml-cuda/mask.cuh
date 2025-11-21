#include "common.cuh"
#include "ggml.h"

#include <stdint.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include <algorithm>

void generate_mask(const ggml_tensor * node, uint8_t* bitmask, const float b30, const float b0, unsigned int rows, cudaStream_t stream);

void mask_activation(ggml_tensor * node, const uint8_t* bitmask, unsigned int rows, cudaStream_t stream);

void maskColumnsGPUDevice(const ggml_tensor * node, float* masked_act, const uint8_t* bitmask, unsigned int rows, cudaStream_t stream);
