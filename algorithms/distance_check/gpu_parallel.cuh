#pragma once

#include "../../topology/topology.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdio>

// Структура вирівняна по 16 байт (або 32) для кращого Memory Coalescing
struct alignas(16) DistanceCheckField {
    float minX, minY, maxX, maxY;
    int layer;
};

// Device-функція порівняння (без sqrt)
__device__ inline bool isTooCloseDevice(const DistanceCheckField& a, const DistanceCheckField& b, float minDistSq) {
    float dx = fmaxf(0.0f, fmaxf(a.minX - b.maxX, b.minX - a.maxX));
    float dy = fmaxf(0.0f, fmaxf(a.minY - b.maxY, b.minY - a.maxY));
    return (dx * dx + dy * dy) < minDistSq;
}

__global__ void distanceKernel(const DistanceCheckField* __restrict__ boxes, int count, float minDistSq, unsigned int* violations) {
    // Використовуємо 2D сітку для більш природного відображення на пари (i, j)
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Перевіряємо лише верхній трикутник матриці (i < j)
    if (i < count && j < count && i < j) {
        const DistanceCheckField a = boxes[i];
        const DistanceCheckField b = boxes[j];

        if (a.layer == b.layer && isTooCloseDevice(a, b, minDistSq)) {
            atomicAdd(violations, 1u);
        }
    }
}

int runGpuBruteforceDistanceCheck(const Layout& layout, float minDistance) {
    int n = static_cast<int>(layout.polygons.size());
    if (n == 0) return 0;

    // Компактний масив — порожні полігони пропускаємо,
    // щоб не відправляти некоректні дані на GPU
    std::vector<DistanceCheckField> hostBoxes;
    hostBoxes.reserve(n);
    for (int i = 0; i < n; ++i) {
        const auto& poly = layout.polygons[i];
        if (poly.vertices.empty()) continue;

        float minX = poly.vertices[0].x, maxX = minX;
        float minY = poly.vertices[0].y, maxY = minY;
        for (const auto& v : poly.vertices) {
            minX = std::min(minX, v.x); maxX = std::max(maxX, v.x);
            minY = std::min(minY, v.y); maxY = std::max(maxY, v.y);
        }
        hostBoxes.push_back({minX, minY, maxX, maxY, poly.layer});
    }
    int count = static_cast<int>(hostBoxes.size());
    if (count == 0) return 0;

    // Запитуємо пристрій для оптимальних розмірів блоку та гріду
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    // Для 2D блоку: blockSide² <= maxThreadsPerBlock
    int blockSide = (prop.maxThreadsPerBlock >= 256) ? 16 : 8;

    dim3 threads(blockSide, blockSide);
    dim3 blocks(
        (count + threads.x - 1) / threads.x,
        (count + threads.y - 1) / threads.y
    );

    // Перевіряємо межі гріду пристрою по обох осях
    if (blocks.x > (unsigned)prop.maxGridSize[0] || blocks.y > (unsigned)prop.maxGridSize[1]) {
        fprintf(stderr,
            "[gpu_parallel distance] grid (%u, %u) exceeds device limits (%d, %d)\n",
            blocks.x, blocks.y, prop.maxGridSize[0], prop.maxGridSize[1]);
        return -1;
    }

    DistanceCheckField* d_boxes      = nullptr;
    unsigned int*       d_violations = nullptr;

    if (cudaMalloc(&d_boxes, count * sizeof(DistanceCheckField)) != cudaSuccess) {
        fprintf(stderr, "[gpu_parallel distance] cudaMalloc d_boxes failed\n");
        return -1;
    }
    if (cudaMalloc(&d_violations, sizeof(unsigned int)) != cudaSuccess) {
        fprintf(stderr, "[gpu_parallel distance] cudaMalloc d_violations failed\n");
        cudaFree(d_boxes);
        return -1;
    }

    cudaMemcpy(d_boxes, hostBoxes.data(), count * sizeof(DistanceCheckField), cudaMemcpyHostToDevice);
    cudaMemset(d_violations, 0, sizeof(unsigned int));

    distanceKernel<<<blocks, threads>>>(d_boxes, count, minDistance * minDistance, d_violations);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_parallel distance] kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_boxes); cudaFree(d_violations);
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_parallel distance] kernel sync error: %s\n", cudaGetErrorString(err));
        cudaFree(d_boxes); cudaFree(d_violations);
        return -1;
    }

    unsigned int rawViolations = 0;
    cudaMemcpy(&rawViolations, d_violations, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_boxes);
    cudaFree(d_violations);

    return (rawViolations <= (unsigned int)INT_MAX) ? (int)rawViolations : INT_MAX;
}
