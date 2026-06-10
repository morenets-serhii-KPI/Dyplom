#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "../external/json.hpp"
using json = nlohmann::json;

struct alignas(16) GpuBox {
    float minX, minY, maxX, maxY;
    int   layer;
    int   idx;
};

struct GpuViolation {
    int   idxA, idxB;
    float dist;
    float oMinX, oMinY, oMaxX, oMaxY;
    int   hasOverlap;
};

#define MAX_RESULTS 500000

struct GpuBoxCmp {
    __host__ __device__
    bool operator()(const GpuBox& a, const GpuBox& b) const {
        return a.minX < b.minX;
    }
};

__global__ void drcKernel(
    const GpuBox* __restrict__ boxes,
    int count,
    float minDist,
    float minDistSq,
    GpuViolation* results,
    int* resultCount,
    int maxResults)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const GpuBox a = boxes[i];

    for (int j = i + 1; j < count; j++) {
        const GpuBox b = boxes[j];

        if (b.minX - a.maxX > minDist) break;

        if (a.layer != b.layer) continue;

        float dx = fmaxf(0.f, fmaxf(a.minX - b.maxX, b.minX - a.maxX));
        float dy = fmaxf(0.f, fmaxf(a.minY - b.maxY, b.minY - a.maxY));
        float dSq = dx * dx + dy * dy;

        if (dSq >= minDistSq) continue; 

        float dist = sqrtf(dSq);

        float oMinX = fmaxf(a.minX, b.minX);
        float oMinY = fmaxf(a.minY, b.minY);
        float oMaxX = fminf(a.maxX, b.maxX);
        float oMaxY = fminf(a.maxY, b.maxY);
        int hasOverlap = (oMaxX > oMinX && oMaxY > oMinY) ? 1 : 0;

        int slot = atomicAdd(resultCount, 1);
        if (slot < maxResults) {
            results[slot] = {
                a.idx, b.idx,
                dist,
                oMinX, oMinY, oMaxX, oMaxY,
                hasOverlap
            };
        }
    }
}

struct PolyData  {
    int layer;
    std::vector<std::pair<float,float>> pts;
};

static std::vector<PolyData > loadJson(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[gpu_opt_drc_methods] Cannot open: %s\n", path);
        return {};
    }
    json data; f >> data;
    std::vector<PolyData > polys;
    for (const auto& p : data) {
        PolyData  poly;
        poly.layer = p.value("layer", 0);
        for (const auto& pt : p["points"]) {
            if (pt.is_array())
                poly.pts.push_back({ pt[0].get<float>(), pt[1].get<float>() });
            else
                poly.pts.push_back({ pt["x"].get<float>(), pt["y"].get<float>() });
        }
        if (!poly.pts.empty())
            polys.push_back(std::move(poly));
    }
    return polys;
}

static GpuBox toBbox(const PolyData & p, int idx) {
    float x1 = p.pts[0].first,  x2 = x1;
    float y1 = p.pts[0].second, y2 = y1;
    for (const auto& v : p.pts) {
        x1 = std::min(x1, v.first);  x2 = std::max(x2, v.first);
        y1 = std::min(y1, v.second); y2 = std::max(y2, v.second);
    }
    return { x1, y1, x2, y2, p.layer, idx };
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: gpu_opt_drc_methods.exe <topology.json> <min_distance> <output_dir>\n");
        return 1;
    }

    const char* topoPath = argv[1];
    float       minDist  = std::stof(argv[2]);
    std::string outDir   = argv[3];
    if (!outDir.empty() && outDir.back() != '\\' && outDir.back() != '/')
        outDir += '/';

    auto polys = loadJson(topoPath);
    int n = (int)polys.size();
    if (n == 0) { fprintf(stderr, "[gpu_opt_drc_methods] No PolyData s.\n"); return 1; }

    std::vector<GpuBox> hostBoxes;
    hostBoxes.reserve(n);
    for (int i = 0; i < n; i++)
        hostBoxes.push_back(toBbox(polys[i], i));

    int deviceId; cudaGetDevice(&deviceId);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, deviceId);
    int threads = std::min(prop.maxThreadsPerBlock, 256);
    int blocks  = (n + threads - 1) / threads;

    GpuBox*       d_boxes   = nullptr;
    GpuViolation* d_results = nullptr;
    int*          d_cnt     = nullptr;

    cudaMalloc(&d_boxes,   n * sizeof(GpuBox));
    cudaMalloc(&d_results, MAX_RESULTS * sizeof(GpuViolation));
    cudaMalloc(&d_cnt,     sizeof(int));

    cudaMemcpy(d_boxes, hostBoxes.data(), n * sizeof(GpuBox),
               cudaMemcpyHostToDevice);
    cudaMemset(d_cnt, 0, sizeof(int));

    thrust::device_ptr<GpuBox> ptr(d_boxes);
    thrust::sort(ptr, ptr + n, GpuBoxCmp());

    cudaMemcpy(hostBoxes.data(), d_boxes, n * sizeof(GpuBox),
               cudaMemcpyDeviceToHost);

    auto t0 = std::chrono::high_resolution_clock::now();

    drcKernel<<<blocks, threads>>>(
        d_boxes, n, minDist, minDist * minDist,
        d_results, d_cnt, MAX_RESULTS);

    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    long long ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(t1 - t0).count();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_opt_drc_methods] Kernel error: %s\n",
            cudaGetErrorString(err));
        cudaFree(d_boxes); cudaFree(d_results); cudaFree(d_cnt);
        return 1;
    }

    int total = 0;
    cudaMemcpy(&total, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
    int actual = std::min(total, MAX_RESULTS);

    std::vector<GpuViolation> hostResults(actual);
    if (actual > 0)
        cudaMemcpy(hostResults.data(), d_results,
                   actual * sizeof(GpuViolation), cudaMemcpyDeviceToHost);

    cudaFree(d_boxes); cudaFree(d_results); cudaFree(d_cnt);

    json violations = json::array();
    json overlaps   = json::array();

    for (const auto& r : hostResults) {
        const GpuBox& bA = hostBoxes[r.idxA];
        const GpuBox& bB = hostBoxes[r.idxB];

        json entry = {
            {"layer",    bA.layer},
            {"distance", r.dist},
            {"polyA", {
                {"idx",  bA.idx},
                {"minX", bA.minX}, {"minY", bA.minY},
                {"maxX", bA.maxX}, {"maxY", bA.maxY}
            }},
            {"polyB", {
                {"idx",  bB.idx},
                {"minX", bB.minX}, {"minY", bB.minY},
                {"maxX", bB.maxX}, {"maxY", bB.maxY}
            }}
        };

        if (r.hasOverlap) {
            entry["overlap"] = {
                {"minX", r.oMinX}, {"minY", r.oMinY},
                {"maxX", r.oMaxX}, {"maxY", r.oMaxY}
            };
            overlaps.push_back(entry);
        } else {
            violations.push_back(entry);
        }
    }

    std::string violPath = outDir + "violations.json";
    std::string overPath = outDir + "overlaps.json";

    std::ofstream(violPath) << violations.dump(2);
    std::ofstream(overPath) << overlaps.dump(2);

    std::string repPath = outDir + "report.txt";
    std::ofstream rf(repPath);
    rf << "DRC Report\n";
    rf << "==========\n";
    rf << "Input:         " << topoPath << "\n";
    rf << "PolyData s:      " << n << "\n";
    rf << "Min distance:  " << minDist << " um\n";
    rf << "Time:          " << ms << " ms (GPU Optimized)\n";
    rf << "Violations:    " << violations.size() << "\n";
    rf << "Overlaps:      " << overlaps.size() << "\n";

    printf("[gpu_opt_drc_methods] %lld ms  violations=%zu  overlaps=%zu\n",
        ms, violations.size(), overlaps.size());

    return 0;
}
