#pragma once

#include "../../topology/topology.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>
#include <climits>
#include <cstdio>

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
        box.minX = fminf(box.minX, px); box.minY = fminf(box.minY, py);
        box.maxX = fmaxf(box.maxX, px); box.maxY = fmaxf(box.maxY, py);
    }
    return box;
}

__device__ bool checkIntersectsDevice(const ClipBoxInternal& a, const ClipBoxInternal& b) {
    return (a.minX < b.maxX && a.maxX > b.minX && a.minY < b.maxY && a.maxY > b.minY);
}

// unsigned int лічильник: уникає знакового переповнення при великій кількості результатів
__global__
void clippingKernel(const ClipBoxInternal* boxes, int count,
                    ClippedResultInternal* results, unsigned int* resultCount, int maxResults) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)count * count) return;

    int i = (int)(idx / count);
    int j = (int)(idx % count);
    if (i >= j) return;

    ClipBoxInternal a = boxes[i];
    ClipBoxInternal b = boxes[j];

    if (a.layer == b.layer && checkIntersectsDevice(a, b)) {
        unsigned int outIdx = atomicAdd(resultCount, 1u);
        if (outIdx < (unsigned int)maxResults) {
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
    int n = static_cast<int>(layout.polygons.size());
    if (n == 0) return Layout();

    std::vector<ClipBoxInternal> hostBoxes;
    hostBoxes.reserve(n);
    for (const auto& poly : layout.polygons) {
        if (!poly.vertices.empty())
            hostBoxes.push_back(prepareClipBox(poly));
    }
    int count = static_cast<int>(hostBoxes.size());
    if (count == 0) return Layout();

    int deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int threads = std::min(prop.maxThreadsPerBlock, 256);

    long long totalThreads = (long long)count * count;
    long long numBlocks    = (totalThreads + threads - 1) / threads;

    if (numBlocks > (long long)prop.maxGridSize[0]) {
        fprintf(stderr,
            "[gpu_parallel clipping] %lld blocks needed, device max %d — input too large\n",
            numBlocks, prop.maxGridSize[0]);
        return Layout();
    }
    int blocks = (int)numBlocks;

    // Динамічний ліміт по вільній VRAM (80%) та INT_MAX
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    long long maxByVram  = (long long)(freeMem * 0.8) / (long long)sizeof(ClippedResultInternal);
    long long maxByInt   = (long long)INT_MAX;
    long long theoretical = (long long)count * (count - 1) / 2;
    int maxResults = (int)std::min({theoretical, maxByVram, maxByInt});

    if (maxResults <= 0) return Layout();

    ClipBoxInternal*      deviceBoxes       = nullptr;
    ClippedResultInternal* deviceResults    = nullptr;
    unsigned int*          deviceResultCount = nullptr;

    auto cleanup = [&]() {
        if (deviceBoxes)        cudaFree(deviceBoxes);
        if (deviceResults)      cudaFree(deviceResults);
        if (deviceResultCount)  cudaFree(deviceResultCount);
    };

    if (cudaMalloc(&deviceBoxes, count * sizeof(ClipBoxInternal)) != cudaSuccess) {
        fprintf(stderr, "[gpu_parallel clipping] cudaMalloc deviceBoxes failed\n");
        return Layout();
    }
    if (cudaMalloc(&deviceResults, (size_t)maxResults * sizeof(ClippedResultInternal)) != cudaSuccess) {
        fprintf(stderr, "[gpu_parallel clipping] cudaMalloc deviceResults failed (%d items)\n", maxResults);
        cleanup(); return Layout();
    }
    if (cudaMalloc(&deviceResultCount, sizeof(unsigned int)) != cudaSuccess) {
        fprintf(stderr, "[gpu_parallel clipping] cudaMalloc deviceResultCount failed\n");
        cleanup(); return Layout();
    }

    cudaMemcpy(deviceBoxes, hostBoxes.data(), count * sizeof(ClipBoxInternal), cudaMemcpyHostToDevice);
    cudaMemset(deviceResultCount, 0, sizeof(unsigned int));

    clippingKernel<<<blocks, threads>>>(deviceBoxes, count, deviceResults, deviceResultCount, maxResults);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_parallel clipping] kernel launch error: %s\n", cudaGetErrorString(err));
        cleanup(); return Layout();
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_parallel clipping] kernel sync error: %s\n", cudaGetErrorString(err));
        cleanup(); return Layout();
    }

    unsigned int rawCount = 0;
    cudaMemcpy(&rawCount, deviceResultCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (rawCount > (unsigned int)maxResults) {
        fprintf(stderr, "[gpu_parallel clipping] warning: %u intersections found, capped at %d (VRAM limit)\n",
                rawCount, maxResults);
    }
    int hostCount = (int)std::min(rawCount, (unsigned int)maxResults);

    std::vector<ClippedResultInternal> hostResults((size_t)hostCount);
    if (hostCount > 0) {
        cudaMemcpy(hostResults.data(), deviceResults,
                   (size_t)hostCount * sizeof(ClippedResultInternal), cudaMemcpyDeviceToHost);
    }

    cleanup();

    Layout result;
    result.polygons.reserve((size_t)hostCount);
    for (const auto& r : hostResults) {
        GdsPolygon p;
        p.layer = r.layer;
        p.vertices = {{r.minX, r.minY}, {r.maxX, r.minY}, {r.maxX, r.maxY}, {r.minX, r.maxY}};
        result.polygons.push_back(std::move(p));
    }
    return result;
}
