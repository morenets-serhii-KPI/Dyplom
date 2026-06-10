#pragma once

#include "../../topology/topology.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <algorithm>
#include <climits>
#include <cstdio>

struct GpuClipBox {
    float minX, minY, maxX, maxY;
    int layer;
};

struct BoxComparator {
    __host__ __device__
    bool operator()(const GpuClipBox& a, const GpuClipBox& b) const {
        if (a.minX != b.minX) return a.minX < b.minX;
        return a.minY < b.minY;
    }
};

__global__
void countIntersectionsKernel(const GpuClipBox* boxes, int count, int* perThreadCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    GpuClipBox a = boxes[i];
    int cnt = 0;
    for (int j = i + 1; j < count; j++) {
        GpuClipBox b = boxes[j];
        if (b.minX > a.maxX) break;
        if (a.layer == b.layer &&
            a.minX < b.maxX && a.maxX > b.minX &&
            a.minY < b.maxY && a.maxY > b.minY) {
            cnt++;
        }
    }
    perThreadCount[i] = cnt;
}

__global__
void writeIntersectionsKernel(const GpuClipBox* boxes, int count,
                               const int* offsets, GpuClipBox* results, int maxResults) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    GpuClipBox a = boxes[i];
    int outIdx = offsets[i];
    for (int j = i + 1; j < count; j++) {
        GpuClipBox b = boxes[j];
        if (b.minX > a.maxX) break;
        if (a.layer == b.layer &&
            a.minX < b.maxX && a.maxX > b.minX &&
            a.minY < b.maxY && a.maxY > b.minY) {
            if (outIdx < maxResults) {
                GpuClipBox clipped;
                clipped.minX = fmaxf(a.minX, b.minX);
                clipped.minY = fmaxf(a.minY, b.minY);
                clipped.maxX = fminf(a.maxX, b.maxX);
                clipped.maxY = fminf(a.maxY, b.maxY);
                clipped.layer = a.layer;
                results[outIdx] = clipped;
            }
            outIdx++;
        }
    }
}

static bool gpuClipCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_optimized clipping] %s: %s\n", msg, cudaGetErrorString(err));
        return false;
    }
    return true;
}

inline Layout runGpuOptimizedPolygonClipping(const Layout& layout) {
    int n = static_cast<int>(layout.polygons.size());
    if (n == 0) return Layout();

    std::vector<GpuClipBox> hostBoxes;
    hostBoxes.reserve(n);
    for (int i = 0; i < n; i++) {
        const auto& poly = layout.polygons[i];
        if (poly.vertices.empty()) continue;
        float x1 = poly.vertices[0].x, y1 = poly.vertices[0].y;
        float x2 = x1, y2 = y1;
        for (const auto& v : poly.vertices) {
            x1 = std::min(x1, v.x); y1 = std::min(y1, v.y);
            x2 = std::max(x2, v.x); y2 = std::max(y2, v.y);
        }
        hostBoxes.push_back({x1, y1, x2, y2, poly.layer});
    }
    int count = static_cast<int>(hostBoxes.size());
    if (count == 0) return Layout();

    int deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int threads = std::min(prop.maxThreadsPerBlock, 256);
    int blocks  = (count + threads - 1) / threads;

    GpuClipBox *d_boxes = nullptr, *d_results = nullptr;
    int *d_perThreadCount = nullptr, *d_offsets = nullptr;

    auto cleanup = [&]() {
        if (d_boxes)          cudaFree(d_boxes);
        if (d_perThreadCount) cudaFree(d_perThreadCount);
        if (d_offsets)        cudaFree(d_offsets);
        if (d_results)        cudaFree(d_results);
    };

    if (!gpuClipCheck(cudaMalloc(&d_boxes,          count * sizeof(GpuClipBox)), "malloc d_boxes")    ||
        !gpuClipCheck(cudaMalloc(&d_perThreadCount, count * sizeof(int)),        "malloc perThreadCount") ||
        !gpuClipCheck(cudaMalloc(&d_offsets,        count * sizeof(int)),        "malloc offsets")) {
        cleanup();
        return Layout();
    }

    cudaMemcpy(d_boxes, hostBoxes.data(), count * sizeof(GpuClipBox), cudaMemcpyHostToDevice);

    thrust::device_ptr<GpuClipBox> ptr(d_boxes);
    thrust::sort(ptr, ptr + count, BoxComparator());

    countIntersectionsKernel<<<blocks, threads>>>(d_boxes, count, d_perThreadCount);
    if (!gpuClipCheck(cudaGetLastError(),       "countKernel launch") ||
        !gpuClipCheck(cudaDeviceSynchronize(),  "countKernel sync")) {
        cleanup();
        return Layout();
    }

    {
        thrust::device_ptr<int> pCount(d_perThreadCount);
        thrust::device_ptr<int> pOffsets(d_offsets);
        thrust::exclusive_scan(pCount, pCount + count, pOffsets);
    }

    int lastCount = 0, lastOffset = 0;
    cudaMemcpy(&lastCount,  d_perThreadCount + count - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastOffset, d_offsets        + count - 1, sizeof(int), cudaMemcpyDeviceToHost);
    long long totalResults = (long long)lastOffset + lastCount;

    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    long long maxByVram = (long long)(freeMem * 0.8) / (long long)sizeof(GpuClipBox);
    long long maxByInt  = (long long)INT_MAX;
    int maxResults = (int)std::min({totalResults, maxByVram, maxByInt});

    if (totalResults > (long long)maxResults) {
        fprintf(stderr, "[gpu_optimized clipping] warning: %lld intersections found, capped at %d (VRAM limit)\n",
                totalResults, maxResults);
    }
    if (maxResults == 0) {
        cleanup();
        return Layout();
    }

    if (!gpuClipCheck(cudaMalloc(&d_results, (size_t)maxResults * sizeof(GpuClipBox)), "malloc d_results")) {
        cleanup();
        return Layout();
    }

    writeIntersectionsKernel<<<blocks, threads>>>(d_boxes, count, d_offsets, d_results, maxResults);
    if (!gpuClipCheck(cudaGetLastError(),      "writeKernel launch") ||
        !gpuClipCheck(cudaDeviceSynchronize(), "writeKernel sync")) {
        cleanup();
        return Layout();
    }

    std::vector<GpuClipBox> hostResults((size_t)maxResults);
    cudaMemcpy(hostResults.data(), d_results, (size_t)maxResults * sizeof(GpuClipBox), cudaMemcpyDeviceToHost);

    Layout result;
    result.polygons.reserve((size_t)maxResults);
    for (const auto& r : hostResults) {
        GdsPolygon p;
        p.layer = r.layer;
        p.vertices = {{r.minX, r.minY}, {r.maxX, r.minY}, {r.maxX, r.maxY}, {r.minX, r.maxY}};
        result.polygons.push_back(std::move(p));
    }

    cleanup();
    return result;
}
