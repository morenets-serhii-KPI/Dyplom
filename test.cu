// test_table58.cu
// Тестування всіх 6 алгоритмів: 3 щільності x 3 розміри
// Вивід у форматі CSV для вставки у таблицю 5.8

#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdio>

#ifdef _WIN32
#include <windows.h>
#endif

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

// ─── Параметри ───────────────────────────────────────────────

static const int   RUNS     = 3;
static const int   LAYERS   = 3;
static const float POLY_MIN = 20.f;
static const float POLY_MAX = 150.f;
static const float MIN_DIST = 20.f;
static const unsigned int SEED = 42;

static const std::vector<int> SIZES = { 5000, 10000, 20000 };

struct Density { const char* id; const char* label; float w, h; };
static const std::vector<Density> DENSITIES = {
    { "LOW",    "Низька",   50000.f, 50000.f },
    { "MEDIUM", "Середня",  10000.f, 10000.f },
    { "HIGH",   "Висока",    3000.f,  3000.f },
};

// ─── Вимірювання (медіана) ───────────────────────────────────

template<typename F, typename... A>
long long med(F fn, A&&... args) {
    std::vector<long long> t(RUNS);
    for (int r = 0; r < RUNS; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn(args...);
        t[r] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
    std::sort(t.begin(), t.end());
    return t[RUNS / 2];
}

template<typename F, typename... A>
long long medL(F fn, A&&... args) {
    std::vector<long long> t(RUNS);
    for (int r = 0; r < RUNS; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        Layout l = fn(args...);
        (void)l;
        t[r] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
    std::sort(t.begin(), t.end());
    return t[RUNS / 2];
}

// speedup: якщо час = 0 — позначаємо як "<1ms"
static void printSp(long long base, long long cur) {
    if (cur <= 0)       printf(",>%.0fx", (double)base);
    else                printf(",%.1fx", (double)base / cur);
}

// ─── main ─────────────────────────────────────────────────────

int main() {

#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    GetConsoleMode(hOut, &mode);
    SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif

    // ── CSV-заголовок ────────────────────────────────────────
    printf("operation,density,n,"
           "naive_ms,sweep_ms,omp_ms,opt_ms,gpubf_ms,gpuopt_ms,"
           "sp_sweep,sp_omp,sp_opt,sp_gpubf,sp_gpuopt\n");

    for (const auto& d : DENSITIES) {
        for (int n : SIZES) {

            // Прогрес у stderr, щоб не змішувати з CSV
            fprintf(stderr, "  [%s | n=%d] генерація...", d.id, n);
            Scene sc;
            sc.generateRandomLayout(n, LAYERS, d.w, d.h, POLY_MIN, POLY_MAX, SEED);
            const Layout& lay = sc.layouts[0];
            fprintf(stderr, " тестування");

            // ── Distance Check ────────────────────────────
            fprintf(stderr, " DC");
            long long ref = runNaiveDistanceCheck(lay, MIN_DIST);

            long long t0 = med(runNaiveDistanceCheck,             lay, MIN_DIST);
            long long t1 = med(runSweepLineDistanceCheck,         lay, MIN_DIST);
            long long t2 = med(runParallelDistanceCheck,          lay, MIN_DIST);
            long long t3 = med(runOptimizedParallelDistanceCheck, lay, MIN_DIST);
            long long t4 = med(runGpuBruteforceDistanceCheck,     lay, MIN_DIST);
            long long t5 = med(runGpuOptimizedDistanceCheck,      lay, MIN_DIST);

            printf("DistanceCheck,%s,%d,%lld,%lld,%lld,%lld,%lld,%lld",
                d.label, n, t0, t1, t2, t3, t4, t5);
            printSp(t0, t1);
            printSp(t0, t2);
            printSp(t0, t3);
            printSp(t0, t4);
            printSp(t0, t5);
            printf("\n");

            // ── Polygon Clipping ──────────────────────────
            fprintf(stderr, " PC");
            long long refpc = (long long)runNaivePolygonClipping(lay).polygons.size();

            long long p0 = medL(runNaivePolygonClipping,             lay);
            long long p1 = medL(runSweepLinePolygonClipping,         lay);
            long long p2 = medL(runParallelPolygonClipping,          lay);
            long long p3 = medL(runOptimizedParallelPolygonClipping, lay);
            long long p4 = medL(runGpuPolygonClipping,               lay);
            long long p5 = medL(runGpuOptimizedPolygonClipping,      lay);

            printf("PolygonClipping,%s,%d,%lld,%lld,%lld,%lld,%lld,%lld",
                d.label, n, p0, p1, p2, p3, p4, p5);
            printSp(p0, p1);
            printSp(p0, p2);
            printSp(p0, p3);
            printSp(p0, p4);
            printSp(p0, p5);
            printf("\n");

            fprintf(stderr, " OK\n");
        }
    }

    fprintf(stderr, "\nГотово. Скопіюй вивід вище у .csv файл.\n");
    return 0;
}
