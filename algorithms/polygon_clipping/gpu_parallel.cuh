#pragma once

#include "../../topology/topology.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>

struct ClipBoxInternal {
    float minX, minY, maxX, maxY;
    int layer;
};

struct ClippedResultInternal {
    float minX, minY, maxX, maxY;
    int layer;
};

static ClipBoxInternal prepareClipBox(const GdsPolygon& poly) {
    ClipBoxInternal box;
    if (poly.vertices.empty()) return {0.0f, 0.0f, 0.0f, 0.0f, poly.layer};

    box.minX = box.maxX = static_cast<float>(poly.vertices[0].x);
    box.minY = box.maxY = static_cast<float>(poly.vertices[0].y);
    box.layer = poly.layer;

    for (const auto& p : poly.vertices) {
        float px = static_cast<float>(p.x);
        float py = static_cast<float>(p.y);
        box.minX = fminf(box.minX, px);
        box.minY = fminf(box.minY, py);
        box.maxX = fmaxf(box.maxX, px);
        box.maxY = fmaxf(box.maxY, py);
    }
    return box;
}

__device__ bool checkIntersectsDevice(const ClipBoxInternal& a, const ClipBoxInternal& b) {
    return (a.minX < b.maxX && a.maxX > b.minX && a.minY < b.maxY && a.maxY > b.minY);
}

__global__ void clippingKernel(const ClipBoxInternal* boxes, int count, ClippedResultInternal* results, int* resultCount, int maxResults) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)count * count) return;

    int i = idx / count;
    int j = idx % count;

    if (i >= j) return;

    ClipBoxInternal a = boxes[i];
    ClipBoxInternal b = boxes[j];

    if (a.layer == b.layer && checkIntersectsDevice(a, b)) {
        int outIdx = atomicAdd(resultCount, 1);
        if (outIdx < maxResults) {
            ClippedResultInternal clipped;
            clipped.minX = fmaxf(a.minX, b.minX);
            clipped.minY = fmaxf(a.minY, b.minY);
            clipped.maxX = fminf(a.maxX, b.maxX);
            clipped.maxY = fminf(a.maxY, b.maxY);
            clipped.layer = a.layer;
            results[outIdx] = clipped;
        }
    }
}

inline Layout runGpuPolygonClipping(const Layout& layout) {
    int count = static_cast<int>(layout.polygons.size());
    if (count == 0) return Layout();

    std::vector<ClipBoxInternal> hostBoxes;
    hostBoxes.reserve(count);
    for (const auto& poly : layout.polygons) hostBoxes.push_back(prepareClipBox(poly));

    ClipBoxInternal* deviceBoxes;
    ClippedResultInternal* deviceResults;
    int* deviceResultCount;

    // Обмежуємо масив результатів, щоб не "вбити" відеопам'ять (VRAM)
    int maxResults = std::min<int>(count * count, 1000000); 

    cudaMalloc(&deviceBoxes, count * sizeof(ClipBoxInternal));
    cudaMalloc(&deviceResults, maxResults * sizeof(ClippedResultInternal));
    cudaMalloc(&deviceResultCount, sizeof(int));

    cudaMemcpy(deviceBoxes, hostBoxes.data(), count * sizeof(ClipBoxInternal), cudaMemcpyHostToDevice);
    cudaMemset(deviceResultCount, 0, sizeof(int));

    int threads = 256;
    long long totalThreads = (long long)count * count;
    long long blocks = (totalThreads + threads - 1) / threads;

    clippingKernel<<<blocks, threads>>>(deviceBoxes, count, deviceResults, deviceResultCount, maxResults);

    int hostCount = 0;
    cudaMemcpy(&hostCount, deviceResultCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (hostCount > maxResults) hostCount = maxResults;

    std::vector<ClippedResultInternal> hostResults(hostCount);
    if (hostCount > 0) {
        cudaMemcpy(hostResults.data(), deviceResults, hostCount * sizeof(ClippedResultInternal), cudaMemcpyDeviceToHost);
    }

    Layout result;
    result.polygons.reserve(hostCount);
    for (const auto& res : hostResults) {
        GdsPolygon p; 
        p.layer = res.layer;
        p.vertices = {{res.minX, res.minY}, {res.maxX, res.minY}, {res.maxX, res.maxY}, {res.minX, res.maxY}};
        result.polygons.push_back(std::move(p));
    }

    cudaFree(deviceBoxes); cudaFree(deviceResults); cudaFree(deviceResultCount);
    return result;
}