// benchmark.cu
// Self-contained DRC benchmark for hardware comparison study.
// No configuration needed — just run benchmark.exe
// Results are saved to benchmark_results.csv
//
// Build:
//   nvcc benchmark.cu -o benchmark.exe -O2 -arch=sm_75 ^
//        -Xcompiler "/openmp /std:c++17 /EHsc /Zc:preprocessor" ^
//        -IC:\path\to\project -IC:\path\to\project\external
//
// For volunteers: just run benchmark.exe, send benchmark_results.csv

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstring>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda_runtime.h>

#include "topology/topology.h"

#include "algorithms/distance_check/cpu_naive.h"
#include "algorithms/distance_check/cpu_sweepline.h"
#include "algorithms/distance_check/cpu_parallel.h"
#include "algorithms/distance_check/cpu_optimized_parallel.h"
#include "algorithms/distance_check/gpu_parallel.cuh"
#include "algorithms/distance_check/gpu_optimized.cuh"

#include "algorithms/polygon_clipping/cpu_naive.h"
#include "algorithms/polygon_clipping/cpu_sweepline.h"
#include "algorithms/polygon_clipping/cpu_parallel.h"
#include "algorithms/polygon_clipping/cpu_optimized_parallel.h"
#include "algorithms/polygon_clipping/gpu_parallel.cuh"
#include "algorithms/polygon_clipping/gpu_optimized.cuh"

// =============================================================================
//  CONFIG
// =============================================================================

static const int   RUNS     = 5;       // runs per measurement — median reported
static const int   LAYERS   = 3;
static const float POLY_MIN = 20.f;
static const float POLY_MAX = 150.f;
static const float MIN_DIST = 20.f;
static const unsigned int SEED = 42;

static const std::vector<int> SIZES = {
    5000, 10000, 20000
};

struct Density { const char* id; const char* label; float w, h; };
static const std::vector<Density> DENSITIES = {
    { "LOW",    "Low    (50k x 50k)", 50000.f, 50000.f },
    { "MEDIUM", "Medium (10k x 10k)", 10000.f, 10000.f },
    { "HIGH",   "High   ( 3k x  3k)",  3000.f,  3000.f },
};

static const char* ALGO_IDS[6] = {
    "CPU_Naive", "CPU_Sweepline", "CPU_Parallel_OMP",
    "CPU_Opt_Parallel", "GPU_BruteForce", "GPU_Optimized"
};
static const char* ALGO_LABELS[6] = {
    "CPU Naive", "CPU Sweep-line", "CPU Parallel (OMP)",
    "CPU Opt.Parallel", "GPU Brute-force", "GPU Optimized"
};

static const char* OUT_FILE = "benchmark_results.csv";

// =============================================================================
//  HARDWARE INFO
// =============================================================================

struct HardwareInfo {
    // GPU
    char   gpuName[256];
    int    cudaCores;
    int    smCount;
    int    computeCapMajor;
    int    computeCapMinor;
    float  gpuClockMHz;
    int    gpuMemMB;
    float  memBandwidthGBs;
    char   memType[32];

    // CPU / system
    int    ompThreads;
    char   platform[64];
    char   cpuName[256];
    int    cpuCores;
    int    cpuThreads;
    int    cpuMaxMHz;
};

static int cudaCoresPerSM(int major, int minor) {
    // CUDA cores per SM for each architecture
    struct { int major, minor, cores; } table[] = {
        {3,0,192},{3,2,192},{3,5,192},{3,7,192},
        {5,0,128},{5,2,128},
        {6,0,64},{6,1,128},{6,2,128},
        {7,0,64},{7,2,64},{7,5,64},
        {8,0,64},{8,6,128},{8,7,128},{8,9,128},
        {9,0,128},
    };
    for (auto& e : table)
        if (e.major == major && e.minor == minor) return e.cores;
    return 128; // default
}

static HardwareInfo collectHardwareInfo() {
    HardwareInfo h{};

    // GPU
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    strncpy(h.gpuName, prop.name, sizeof(h.gpuName)-1);
    h.smCount           = prop.multiProcessorCount;
    h.computeCapMajor   = prop.major;
    h.computeCapMinor   = prop.minor;
    h.gpuClockMHz       = (float)prop.maxThreadsPerBlock; // placeholder, clockRate removed in CUDA 13
    h.gpuMemMB          = (int)(prop.totalGlobalMem / (1024*1024));
    h.cudaCores         = prop.multiProcessorCount * cudaCoresPerSM(prop.major, prop.minor);

    // Memory bandwidth
    h.memBandwidthGBs = 0.f;

    // Memory type (approximate by compute capability)
    if (prop.major >= 8)      strcpy(h.memType, "GDDR6/HBM");
    else if (prop.major >= 7) strcpy(h.memType, "GDDR5/GDDR6");
    else                      strcpy(h.memType, "GDDR5");

    // CPU threads
#ifdef _OPENMP
    h.ompThreads = omp_get_max_threads();
#else
    h.ompThreads = 1;
#endif

#ifdef _WIN32
    strcpy(h.platform, "Windows");

    // CPU info via WMIC
    strcpy(h.cpuName, "Unknown");
    h.cpuCores   = 0;
    h.cpuThreads = 0;
    h.cpuMaxMHz  = 0;

    // Run WMIC and capture output
    FILE* pipe = _popen(
        "wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed /format:list 2>nul",
        "r");
    if (pipe) {
        char line[512];
        while (fgets(line, sizeof(line), pipe)) {
            // Remove \r\n
            char* nl = strchr(line, '\r'); if (nl) *nl = 0;
            nl = strchr(line, '\n'); if (nl) *nl = 0;

            if (strncmp(line, "Name=", 5) == 0)
                strncpy(h.cpuName, line + 5, sizeof(h.cpuName) - 1);
            else if (strncmp(line, "NumberOfCores=", 14) == 0)
                h.cpuCores = atoi(line + 14);
            else if (strncmp(line, "NumberOfLogicalProcessors=", 26) == 0)
                h.cpuThreads = atoi(line + 26);
            else if (strncmp(line, "MaxClockSpeed=", 14) == 0)
                h.cpuMaxMHz = atoi(line + 14);
        }
        _pclose(pipe);
    }
#else
    strcpy(h.platform, "Linux");
#endif

    return h;
}

static void printHardwareInfo(const HardwareInfo& h) {
    printf("\n");
    printf("  GPU:          %s\n", h.gpuName);
    printf("  CUDA cores:   %d  (%d SM x %d cores/SM)\n",
        h.cudaCores, h.smCount,
        h.cudaCores / (h.smCount ? h.smCount : 1));
    printf("  Compute CC:   %d.%d\n", h.computeCapMajor, h.computeCapMinor);
    printf("  GPU memory:   %d MB  (%s)\n", h.gpuMemMB, h.memType);
    printf("  CPU:          %s\n", h.cpuName);
    printf("  CPU cores:    %d cores / %d threads / %.1f GHz\n",
        h.cpuCores, h.cpuThreads, h.cpuMaxMHz / 1000.f);
    printf("  OMP threads:  %d\n", h.ompThreads);
    printf("  Platform:     %s\n", h.platform);
}

static void writeHardwareToCSV(std::ofstream& f, const HardwareInfo& h) {
    f << "# HARDWARE INFO\n";
    f << "# GPU," << h.gpuName << "\n";
    f << "# CUDA_cores," << h.cudaCores << "\n";
    f << "# SM_count," << h.smCount << "\n";
    f << "# Compute_CC," << h.computeCapMajor << "." << h.computeCapMinor << "\n";
    f << "# GPU_memory_MB," << h.gpuMemMB << "\n";
    f << "# Memory_type," << h.memType << "\n";
    f << "# CPU," << h.cpuName << "\n";
    f << "# CPU_cores," << h.cpuCores << "\n";
    f << "# CPU_threads," << h.cpuThreads << "\n";
    f << "# CPU_max_MHz," << h.cpuMaxMHz << "\n";
    f << "# OMP_threads," << h.ompThreads << "\n";
    f << "# Platform," << h.platform << "\n";
    f << "#\n";
}

// =============================================================================
//  MEASUREMENT
// =============================================================================

template<typename F, typename... A>
long long medMs(int runs, F fn, A&&... args) {
    std::vector<long long> t(runs);
    for (int r = 0; r < runs; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn(args...);
        t[r] = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
    std::sort(t.begin(), t.end());
    return t[runs / 2];
}

template<typename F, typename... A>
long long medLMs(int runs, F fn, A&&... args) {
    std::vector<long long> t(runs);
    for (int r = 0; r < runs; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        Layout l = fn(args...);
        (void)l;
        t[r] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
    std::sort(t.begin(), t.end());
    return t[runs / 2];
}

struct MeasResult {
    long long ms[6];
    long long ref;
    bool      ok[6];
};

static MeasResult benchDC(const Layout& lay) {
    MeasResult r{};
    r.ref   = runNaiveDistanceCheck(lay, MIN_DIST);
    r.ok[0] = true;
    r.ms[0] = medMs(RUNS, runNaiveDistanceCheck, lay, MIN_DIST);
    r.ms[1] = medMs(RUNS, runSweepLineDistanceCheck, lay, MIN_DIST);
    r.ok[1] = (runSweepLineDistanceCheck(lay, MIN_DIST) == r.ref);
    r.ms[2] = medMs(RUNS, runParallelDistanceCheck, lay, MIN_DIST);
    r.ok[2] = (runParallelDistanceCheck(lay, MIN_DIST) == r.ref);
    r.ms[3] = medMs(RUNS, runOptimizedParallelDistanceCheck, lay, MIN_DIST);
    r.ok[3] = (runOptimizedParallelDistanceCheck(lay, MIN_DIST) == r.ref);
    r.ms[4] = medMs(RUNS, runGpuBruteforceDistanceCheck, lay, MIN_DIST);
    r.ok[4] = (runGpuBruteforceDistanceCheck(lay, MIN_DIST) == r.ref);
    r.ms[5] = medMs(RUNS, runGpuOptimizedDistanceCheck, lay, MIN_DIST);
    r.ok[5] = (runGpuOptimizedDistanceCheck(lay, MIN_DIST) == r.ref);
    return r;
}

static MeasResult benchPC(const Layout& lay) {
    MeasResult r{};
    r.ref   = (long long)runNaivePolygonClipping(lay).polygons.size();
    r.ok[0] = true;
    r.ms[0] = medLMs(RUNS, runNaivePolygonClipping, lay);
    r.ms[1] = medLMs(RUNS, runSweepLinePolygonClipping, lay);
    r.ok[1] = ((long long)runSweepLinePolygonClipping(lay).polygons.size() == r.ref);
    r.ms[2] = medLMs(RUNS, runParallelPolygonClipping, lay);
    r.ok[2] = ((long long)runParallelPolygonClipping(lay).polygons.size() == r.ref);
    r.ms[3] = medLMs(RUNS, runOptimizedParallelPolygonClipping, lay);
    r.ok[3] = ((long long)runOptimizedParallelPolygonClipping(lay).polygons.size() == r.ref);
    r.ms[4] = medLMs(RUNS, runGpuPolygonClipping, lay);
    r.ok[4] = ((long long)runGpuPolygonClipping(lay).polygons.size() == r.ref);
    r.ms[5] = medLMs(RUNS, runGpuOptimizedPolygonClipping, lay);
    r.ok[5] = ((long long)runGpuOptimizedPolygonClipping(lay).polygons.size() == r.ref);
    return r;
}

// =============================================================================
//  CSV ROW
// =============================================================================

static void writeRow(
    std::ofstream& csv,
    const char* op,
    const char* densityId,
    const char* densityLabel,
    int n,
    const MeasResult& r)
{
    long long naiveMs = r.ms[0];
    for (int a = 0; a < 6; a++) {
        double sp = (naiveMs > 0 && r.ms[a] > 0)
            ? (double)naiveMs / r.ms[a] : 0.0;
        csv << op << ","
            << densityId << ","
            << densityLabel << ","
            << n << ","
            << ALGO_IDS[a] << ","
            << ALGO_LABELS[a] << ","
            << r.ms[a] << ","
            << r.ref << ","
            << (r.ok[a] ? "OK" : "ERR") << ","
            << std::fixed << std::setprecision(2) << sp << "\n";
    }
}

// =============================================================================
//  PROGRESS BAR
// =============================================================================

static void progress(int done, int total, const char* label) {
    const int W = 28;
    int filled = (total > 0) ? (done * W) / total : 0;
    fprintf(stderr, "\r  [");
    for (int i = 0; i < W; i++) fputc(i < filled ? '#' : '.', stderr);
    fprintf(stderr, "] %2d/%-2d  %s     ", done, total, label);
    fflush(stderr);
    if (done == total) fprintf(stderr, "\n");
}

// =============================================================================
//  MAIN
// =============================================================================

int main() {

#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(hOut, &mode))
        SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif

    fprintf(stderr, "\n");
    fprintf(stderr, "================================================================\n");
    fprintf(stderr, "  DRC Hardware Benchmark\n");
    fprintf(stderr, "  Collecting performance data for hardware comparison study\n");
    fprintf(stderr, "================================================================\n");

    // Collect hardware info
    HardwareInfo hw = collectHardwareInfo();
    printHardwareInfo(hw);

    fprintf(stderr, "\n  Settings:\n");
    fprintf(stderr, "  Runs per measurement (median): %d\n", RUNS);
    fprintf(stderr, "  Polygon sizes: ");
    for (int s : SIZES) fprintf(stderr, "%d ", s);
    fprintf(stderr, "\n  Densities: LOW / MEDIUM / HIGH\n");
    fprintf(stderr, "  Operations: Distance Check + Polygon Clipping\n");
    fprintf(stderr, "  Output: %s\n", OUT_FILE);
    fprintf(stderr, "\n  This will take approximately 5-15 minutes.\n");
    fprintf(stderr, "  Please do not close this window.\n");
    fprintf(stderr, "================================================================\n\n");

    // Open CSV
    std::ofstream csv(OUT_FILE);
    if (!csv.is_open()) {
        fprintf(stderr, "[ERROR] Cannot open %s for writing.\n", OUT_FILE);
        return 1;
    }

    writeHardwareToCSV(csv, hw);

    csv << "operation,density_id,density_label,n,algo_id,algo_label,"
       "time_us,result,status,speedup_vs_naive\n";

    int total = (int)(DENSITIES.size() * SIZES.size() * 2);
    int done  = 0;

    for (const auto& d : DENSITIES) {
        for (int n : SIZES) {

            char label[64];
            snprintf(label, sizeof(label), "%-8s n=%d", d.id, n);
            progress(done, total, label);

            Scene sc;
            sc.generateRandomLayout(n, LAYERS, d.w, d.h, POLY_MIN, POLY_MAX, SEED);
            const Layout& lay = sc.layouts[0];

            // Distance Check
            MeasResult dc = benchDC(lay);
            writeRow(csv, "DistanceCheck", d.id, d.label, n, dc);
            done++;

            MeasResult pc = benchPC(lay);
            writeRow(csv, "PolygonClipping", d.id, d.label, n, pc);
            done++;

            progress(done, total, label);
        }
    }

    csv.close();

    // Summary to stderr
    fprintf(stderr, "\n================================================================\n");
    fprintf(stderr, "  DONE\n");
    fprintf(stderr, "  Results saved to: %s\n", OUT_FILE);
    fprintf(stderr, "  Please send this file to the researcher.\n");
    fprintf(stderr, "================================================================\n\n");

    // Also print summary to stdout for console visibility
    printf("\n");
    printf("Hardware: %s  |  %d CUDA cores  |  CC %d.%d\n",
        hw.gpuName, hw.cudaCores, hw.computeCapMajor, hw.computeCapMinor);
    printf("CPU OMP threads: %d  |  Platform: %s\n",
        hw.ompThreads, hw.platform);
    printf("Results saved to: %s\n", OUT_FILE);
    printf("Please send this file.\n\n");

    return 0;
}
