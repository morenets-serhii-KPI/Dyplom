#pragma once

#include "../../topology/topology.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <algorithm>

// Вирівнювання для Memory Coalescing
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
    const GpuBox* __restrict__ boxes, // __restrict__ допомагає використовувати Read-Only Cache
    int count,
    float minDistance,
    float minDistSq, // Передаємо квадрат заздалегідь
    int* violations
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    // Читаємо свій бокс один раз у регістри
    const GpuBox a = boxes[i];
    int found = 0;

    // Sweep-line цикл
    for (int j = i + 1; j < count; j++) {
        const GpuBox b = boxes[j];

        // Ранній вихід (Sweep Line)
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
        atomicAdd(violations, found);
    }
}

int runGpuOptimizedDistanceCheck(const Layout& layout, float minDistance) {
    int count = static_cast<int>(layout.polygons.size());
    if (count == 0) return 0;

    // Обчислення BBox на CPU
    std::vector<GpuBox> hostBoxes(count);
    for (int i = 0; i < count; i++) {
        const auto& poly = layout.polygons[i];
        if (poly.vertices.empty()) continue;
        float x1 = poly.vertices[0].x, x2 = x1;
        float y1 = poly.vertices[0].y, y2 = y1;
        for (const auto& v : poly.vertices) {
            x1 = std::min(x1, (float)v.x); x2 = std::max(x2, (float)v.x);
            y1 = std::min(y1, (float)v.y); y2 = std::max(y2, (float)v.y);
        }
        hostBoxes[i] = {x1, y1, x2, y2, poly.layer};
    }

    GpuBox* d_boxes;
    int* d_violations;
    cudaMalloc(&d_boxes, count * sizeof(GpuBox));
    cudaMalloc(&d_violations, sizeof(int));

    cudaMemcpy(d_boxes, hostBoxes.data(), count * sizeof(GpuBox), cudaMemcpyHostToDevice);
    cudaMemset(d_violations, 0, sizeof(int));

    // Сортування на GPU через Thrust
    thrust::device_ptr<GpuBox> ptr(d_boxes);
    thrust::sort(ptr, ptr + count, GpuBoxComparator());

    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    distanceKernelOptimized<<<blocks, threads>>>(
        d_boxes, count, minDistance, minDistance * minDistance, d_violations
    );

    int result = 0;
    cudaMemcpy(&result, d_violations, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_boxes);
    cudaFree(d_violations);

    return result;
}