#pragma once

#include "../../topology/topology.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

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

__global__ void distanceKernel(const DistanceCheckField* __restrict__ boxes, int count, float minDistSq, int* violations) {
    // Використовуємо 2D сітку для більш природного відображення на пари (i, j)
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Перевіряємо лише верхній трикутник матриці (i < j)
    if (i < count && j < count && i < j) {
        const DistanceCheckField a = boxes[i];
        const DistanceCheckField b = boxes[j];

        if (a.layer == b.layer && isTooCloseDevice(a, b, minDistSq)) {
            atomicAdd(violations, 1);
        }
    }
}

int runGpuBruteforceDistanceCheck(const Layout& layout, float minDistance) {
    int count = static_cast<int>(layout.polygons.size());
    if (count == 0) return 0;

    // Precompute BBoxes на CPU (це O(n), не впливає на загальний O(n^2))
    std::vector<DistanceCheckField> hostBoxes(count);
    for (int i = 0; i < count; ++i) {
        const auto& poly = layout.polygons[i];
        if (poly.vertices.empty()) continue;
        
        float minX = poly.vertices[0].x, maxX = minX;
        float minY = poly.vertices[0].y, maxY = minY;
        for (const auto& v : poly.vertices) {
            minX = std::min(minX, v.x); maxX = std::max(maxX, v.x);
            minY = std::min(minY, v.y); maxY = std::max(maxY, v.y);
        }
        hostBoxes[i] = {minX, minY, maxX, maxY, poly.layer};
    }

    DistanceCheckField* d_boxes;
    int* d_violations;
    cudaMalloc(&d_boxes, count * sizeof(DistanceCheckField));
    cudaMalloc(&d_violations, sizeof(int));

    cudaMemcpy(d_boxes, hostBoxes.data(), count * sizeof(DistanceCheckField), cudaMemcpyHostToDevice);
    cudaMemset(d_violations, 0, sizeof(int));

    // Налаштовуємо 2D блоки для кращого покриття пар
    dim3 threads(16, 16); 
    dim3 blocks((count + threads.x - 1) / threads.x, (count + threads.y - 1) / threads.y);

    distanceKernel<<<blocks, threads>>>(d_boxes, count, minDistance * minDistance, d_violations);

    int h_violations = 0;
    cudaMemcpy(&h_violations, d_violations, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_boxes);
    cudaFree(d_violations);

    return h_violations;
}