#pragma once

#include "../../topology/topology.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <algorithm>
#include <cstdio>

struct alignas(16) GpuBox {
    float minX, minY, maxX, maxY;
    int layer;
};

struct GpuBoxComparator {
    __host__ __device__ bool operator()(const GpuBox& a, const GpuBox& b) const {
        return a.minX < b.minX;
    }
};

__global__ void distanceKernelOptimized(
    const GpuBox* __restrict__ boxes,
    int count,
    float minDistance,
    float minDistSq,
    unsigned int* violations
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const GpuBox a = boxes[i];
    int found = 0;

    for (int j = i + 1; j < count; j++) {
        const GpuBox b = boxes[j];

        if (b.minX - a.maxX > minDistance) break;

        if (a.layer == b.layer) {
            float dx = fmaxf(0.0f, fmaxf(a.minX - b.maxX, b.minX - a.maxX));
            float dy = fmaxf(0.0f, fmaxf(a.minY - b.maxY, b.minY - a.maxY));

            if ((dx * dx + dy * dy) < minDistSq) {
                found++;
            }
        }
    }

    if (found > 0) {
        atomicAdd(violations, (unsigned int)found);
    }
}

int runGpuOptimizedDistanceCheck(const Layout& layout, float minDistance) {
    int n = static_cast<int>(layout.polygons.size());
    if (n == 0) return 0;

    std::vector<GpuBox> hostBoxes;
    hostBoxes.reserve(n);
    for (int i = 0; i < n; i++) {
        const auto& poly = layout.polygons[i];
        if (poly.vertices.empty()) continue;
        float x1 = poly.vertices[0].x, x2 = x1;
        float y1 = poly.vertices[0].y, y2 = y1;
        for (const auto& v : poly.vertices) {
            x1 = std::min(x1, v.x); x2 = std::max(x2, v.x);
            y1 = std::min(y1, v.y); y2 = std::max(y2, v.y);
        }
        hostBoxes.push_back({x1, y1, x2, y2, poly.layer});
    }
    int count = static_cast<int>(hostBoxes.size());
    if (count == 0) return 0;

    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int threads = std::min(prop.maxThreadsPerBlock, 256);
    int blocks = (count + threads - 1) / threads;

    GpuBox*       d_boxes      = nullptr;
    unsigned int* d_violations = nullptr;

    if (cudaMalloc(&d_boxes, count * sizeof(GpuBox)) != cudaSuccess) {
        fprintf(stderr, "[gpu_optimized distance] cudaMalloc d_boxes failed\n");
        return -1;
    }
    if (cudaMalloc(&d_violations, sizeof(unsigned int)) != cudaSuccess) {
        fprintf(stderr, "[gpu_optimized distance] cudaMalloc d_violations failed\n");
        cudaFree(d_boxes);
        return -1;
    }

    cudaMemcpy(d_boxes, hostBoxes.data(), count * sizeof(GpuBox), cudaMemcpyHostToDevice);
    cudaMemset(d_violations, 0, sizeof(unsigned int));

    thrust::device_ptr<GpuBox> ptr(d_boxes);
    thrust::sort(ptr, ptr + count, GpuBoxComparator());

    distanceKernelOptimized<<<blocks, threads>>>(
        d_boxes, count, minDistance, minDistance * minDistance, d_violations
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_optimized distance] kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_boxes); cudaFree(d_violations);
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_optimized distance] kernel sync error: %s\n", cudaGetErrorString(err));
        cudaFree(d_boxes); cudaFree(d_violations);
        return -1;
    }

    unsigned int rawResult = 0;
    cudaMemcpy(&rawResult, d_violations, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_boxes);
    cudaFree(d_violations);

    return (rawResult <= (unsigned int)INT_MAX) ? (int)rawResult : INT_MAX;
}
