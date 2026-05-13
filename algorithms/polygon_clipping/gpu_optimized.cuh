#pragma once

#include "../../topology/topology.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <algorithm>

// Структура для GPU з вирівнюванням для кращого доступу до пам'яті
struct GpuClipBox {
    float minX, minY, maxX, maxY;
    int layer;
};

// Компаратор для Thrust
struct BoxComparator {
    __host__ __device__
    bool operator()(const GpuClipBox& a, const GpuClipBox& b) const {
        if (a.minX != b.minX) return a.minX < b.minX;
        return a.minY < b.minY;
    }
};

__global__
void clippingKernelOptimized(const GpuClipBox* boxes, int count, GpuClipBox* results, int* resultCount, int maxResults) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    GpuClipBox a = boxes[i];

    // Sweep-line logic: перевіряємо лише наступні бокси, поки вони не вийдуть за межу maxX
    for (int j = i + 1; j < count; j++) {
        GpuClipBox b = boxes[j];

        // Ранній вихід: якщо наступний бокс починається правіше, ніж закінчується поточний
        if (b.minX > a.maxX) break;

        if (a.layer == b.layer) {
            // Перевірка перекриття
            if (a.minX < b.maxX && a.maxX > b.minX && a.minY < b.maxY && a.maxY > b.minY) {
                int outIdx = atomicAdd(resultCount, 1);
                if (outIdx < maxResults) {
                    GpuClipBox clipped;
                    clipped.minX = fmaxf(a.minX, b.minX);
                    clipped.minY = fmaxf(a.minY, b.minY);
                    clipped.maxX = fminf(a.maxX, b.maxX);
                    clipped.maxY = fminf(a.maxY, b.maxY);
                    clipped.layer = a.layer;
                    results[outIdx] = clipped;
                }
            }
        }
    }
}

inline Layout runGpuOptimizedPolygonClipping(const Layout& layout) {
    int count = static_cast<int>(layout.polygons.size());
    if (count == 0) return Layout();

    // 1. Підготовка боксів на Host
    std::vector<GpuClipBox> hostBoxes(count);
    for (int i = 0; i < count; i++) {
        const auto& poly = layout.polygons[i];
        float x1 = poly.vertices[0].x, y1 = poly.vertices[0].y;
        float x2 = x1, y2 = y1;
        for (const auto& v : poly.vertices) {
            x1 = std::min(x1, v.x); y1 = std::min(y1, v.y);
            x2 = std::max(x2, v.x); y2 = std::max(y2, v.y);
        }
        hostBoxes[i] = {x1, y1, x2, y2, poly.layer};
    }

    // 2. Виділення пам'яті
    GpuClipBox *d_boxes, *d_results;
    int *d_resultCount;
    int maxResults = std::min<int>(count * 10, 2000000); // Ліміт, щоб не переповнити VRAM

    cudaMalloc(&d_boxes, count * sizeof(GpuClipBox));
    cudaMalloc(&d_results, maxResults * sizeof(GpuClipBox));
    cudaMalloc(&d_resultCount, sizeof(int));

    cudaMemcpy(d_boxes, hostBoxes.data(), count * sizeof(GpuClipBox), cudaMemcpyHostToDevice);
    cudaMemset(d_resultCount, 0, sizeof(int));

    // 3. Сортування Thrust (виконується на GPU)
    thrust::device_ptr<GpuClipBox> ptr(d_boxes);
    thrust::sort(ptr, ptr + count, BoxComparator());

    // 4. Запуск оптимізованого кернела
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    clippingKernelOptimized<<<blocks, threads>>>(d_boxes, count, d_results, d_resultCount, maxResults);
    
    cudaDeviceSynchronize();

    // 5. Копіювання результатів назад
    int finalCount = 0;
    cudaMemcpy(&finalCount, d_resultCount, sizeof(int), cudaMemcpyDeviceToHost);
    int actualCopy = std::min(finalCount, maxResults);

    std::vector<GpuClipBox> hostResults(actualCopy);
    if (actualCopy > 0) {
        cudaMemcpy(hostResults.data(), d_results, actualCopy * sizeof(GpuClipBox), cudaMemcpyDeviceToHost);
    }

    // 6. Формування вихідного Layout
    Layout result;
    result.polygons.reserve(actualCopy);
    for (const auto& r : hostResults) {
        GdsPolygon p;
        p.layer = r.layer;
        p.vertices = {{r.minX, r.minY}, {r.maxX, r.minY}, {r.maxX, r.maxY}, {r.minX, r.maxY}};
        result.polygons.push_back(std::move(p));
    }

    cudaFree(d_boxes); cudaFree(d_results); cudaFree(d_resultCount);
    return result;
}