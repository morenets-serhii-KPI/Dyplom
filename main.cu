// main.cu
// DRC Benchmark — entry point
// Reads config.json, runs all enabled algorithms
// for every (density x polygon_count) combination,
// prints results to console and/or CSV file.

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstring>

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

#include "external/json.hpp"
using json = nlohmann::json;

// =============================================================================
//  CONFIG
// =============================================================================

static const char* CONFIG_FILE = "config.json";

struct DensityConfig {
    std::string name, id;
    float worldW, worldH;
    bool enabled;
};

struct AlgoConfig {
    bool enabled;
    std::string label;
};

struct Config {
    bool        saveToFile     = true;
    std::string outputFilename = "results.csv";
    bool        printToConsole = true;
    std::string format         = "both";    // "csv" | "table" | "both"

    unsigned int seed    = 42;
    float polyMin        = 20.f;
    float polyMax        = 150.f;
    int   layers         = 3;

    std::vector<int>           polygonCounts;
    std::vector<DensityConfig> densities;

    float minDistance        = 20.f;
    int   runsPerMeasurement = 3;

    AlgoConfig dc[6];   // 0=naive 1=sweep 2=omp 3=opt 4=gpubf 5=gpuopt
    AlgoConfig pc[6];

    static const char* DEFAULT_LABELS[6];
};

const char* Config::DEFAULT_LABELS[6] = {
    "CPU Naive",
    "CPU Sweep-line",
    "CPU Parallel (OMP)",
    "CPU Opt.Parallel",
    "GPU Brute-force",
    "GPU Optimized",
};

static void parseAlgos(const json& j, AlgoConfig cfg[6], const char* section) {
    static const char* keys[6] = {
        "cpu_naive", "cpu_sweepline", "cpu_parallel",
        "cpu_opt_parallel", "gpu_bruteforce", "gpu_optimized"
    };
    if (!j.contains(section)) return;
    const auto& sec = j[section];
    for (int i = 0; i < 6; i++) {
        cfg[i].enabled = sec.contains(keys[i])
            ? sec[keys[i]].value("enabled", true) : true;
        cfg[i].label   = sec.contains(keys[i])
            ? sec[keys[i]].value("label", Config::DEFAULT_LABELS[i])
            : Config::DEFAULT_LABELS[i];
    }
}

static Config loadConfig(const char* path) {
    Config cfg;

    for (int i = 0; i < 6; i++) {
        cfg.dc[i] = { true, Config::DEFAULT_LABELS[i] };
        cfg.pc[i] = { true, Config::DEFAULT_LABELS[i] };
    }
    cfg.polygonCounts = { 1000, 3000, 5000, 10000, 20000 };
    cfg.densities = {
        { "Low",    "LOW",    50000.f, 50000.f, true },
        { "Medium", "MEDIUM", 10000.f, 10000.f, true },
        { "High",   "HIGH",    3000.f,  3000.f, true },
    };

    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[WARN] %s not found — using defaults.\n", path);
        return cfg;
    }

    try {
        json j = json::parse(f, nullptr, true, true);

        if (j.contains("output")) {
            auto& o = j["output"];
            cfg.saveToFile     = o.value("save_to_file",     cfg.saveToFile);
            cfg.outputFilename = o.value("filename",         cfg.outputFilename);
            cfg.printToConsole = o.value("print_to_console", cfg.printToConsole);
            cfg.format         = o.value("format",           cfg.format);
        }

        if (j.contains("topology")) {
            auto& t = j["topology"];
            cfg.seed    = t.value("seed",          cfg.seed);
            cfg.polyMin = t.value("poly_min_size", cfg.polyMin);
            cfg.polyMax = t.value("poly_max_size", cfg.polyMax);
            cfg.layers  = t.value("layers",        cfg.layers);

            if (t.contains("polygon_counts") && t["polygon_counts"].is_array()) {
                cfg.polygonCounts.clear();
                for (auto& v : t["polygon_counts"])
                    cfg.polygonCounts.push_back(v.get<int>());
            }

            if (t.contains("densities") && t["densities"].is_array()) {
                cfg.densities.clear();
                for (auto& d : t["densities"])
                    cfg.densities.push_back({
                        d.value("name",         "?"),
                        d.value("id",           "?"),
                        d.value("world_width",  10000.f),
                        d.value("world_height", 10000.f),
                        d.value("enabled",      true)
                    });
            }
        }

        if (j.contains("check"))
            cfg.minDistance = j["check"].value("min_distance", cfg.minDistance);

        if (j.contains("algorithms")) {
            auto& a = j["algorithms"];
            cfg.runsPerMeasurement = a.value("runs_per_measurement",
                                              cfg.runsPerMeasurement);
            parseAlgos(a, cfg.dc, "distance_check");
            parseAlgos(a, cfg.pc, "polygon_clipping");
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Failed to parse %s: %s\n", path, e.what());
        fprintf(stderr, "[WARN]  Using defaults.\n");
    }

    return cfg;
}

// =============================================================================
//  MEASUREMENT
// =============================================================================

struct Result {
    long long ms;
    long long value;
    bool      ok;
};

template<typename F, typename... A>
long long medMs(int runs, F fn, A&&... args) {
    std::vector<long long> t(runs);
    for (int r = 0; r < runs; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn(args...);
        t[r] = std::chrono::duration_cast<std::chrono::milliseconds>(
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

static Result runDC(int idx, const Layout& lay, float d, int runs, long long ref) {
    Result r{};
    switch (idx) {
        case 0: r.value=runNaiveDistanceCheck(lay,d);
                r.ms=medMs(runs,runNaiveDistanceCheck,lay,d); break;
        case 1: r.value=runSweepLineDistanceCheck(lay,d);
                r.ms=medMs(runs,runSweepLineDistanceCheck,lay,d); break;
        case 2: r.value=runParallelDistanceCheck(lay,d);
                r.ms=medMs(runs,runParallelDistanceCheck,lay,d); break;
        case 3: r.value=runOptimizedParallelDistanceCheck(lay,d);
                r.ms=medMs(runs,runOptimizedParallelDistanceCheck,lay,d); break;
        case 4: r.value=runGpuBruteforceDistanceCheck(lay,d);
                r.ms=medMs(runs,runGpuBruteforceDistanceCheck,lay,d); break;
        case 5: r.value=runGpuOptimizedDistanceCheck(lay,d);
                r.ms=medMs(runs,runGpuOptimizedDistanceCheck,lay,d); break;
    }
    r.ok = (ref < 0 || r.value == ref);
    return r;
}

static Result runPC(int idx, const Layout& lay, int runs, long long ref) {
    Result r{};
    switch (idx) {
        case 0: r.value=(long long)runNaivePolygonClipping(lay).polygons.size();
                r.ms=medLMs(runs,runNaivePolygonClipping,lay); break;
        case 1: r.value=(long long)runSweepLinePolygonClipping(lay).polygons.size();
                r.ms=medLMs(runs,runSweepLinePolygonClipping,lay); break;
        case 2: r.value=(long long)runParallelPolygonClipping(lay).polygons.size();
                r.ms=medLMs(runs,runParallelPolygonClipping,lay); break;
        case 3: r.value=(long long)runOptimizedParallelPolygonClipping(lay).polygons.size();
                r.ms=medLMs(runs,runOptimizedParallelPolygonClipping,lay); break;
        case 4: r.value=(long long)runGpuPolygonClipping(lay).polygons.size();
                r.ms=medLMs(runs,runGpuPolygonClipping,lay); break;
        case 5: r.value=(long long)runGpuOptimizedPolygonClipping(lay).polygons.size();
                r.ms=medLMs(runs,runGpuOptimizedPolygonClipping,lay); break;
    }
    r.ok = (ref < 0 || r.value == ref);
    return r;
}

// =============================================================================
//  OUTPUT
// =============================================================================

// Accumulated output — printed all at once after benchmarks complete
static std::ostringstream OUTPUT;

static void hline(int w, char c = '-') {
    for (int i = 0; i < w; i++) OUTPUT << c;
    OUTPUT << '\n';
}

static std::string fmtMs(long long ms) {
    return ms <= 0 ? std::string("<1") : std::to_string(ms);
}

static std::string fmtSpeedup(long long base, long long cur) {
    char b[24];
    if (cur <= 0) snprintf(b, sizeof(b), ">%.0fx", (double)base + 1);
    else          snprintf(b, sizeof(b), "%.1fx", (double)base / cur);
    return b;
}

static void printTable(
    const char* op, const DensityConfig& dens, int n,
    const AlgoConfig algos[6], const Result res[6], long long naiveMs)
{
    const int NW = 22, CW = 12;
    int W = NW + CW * 3 + 2;

    hline(W, '=');
    OUTPUT << "  " << op << "  |  " << dens.name << "  |  n = " << n << "\n";
    hline(W, '-');
    OUTPUT << "  " << std::left << std::setw(NW-2) << "Algorithm"
           << std::right << std::setw(CW) << "Time (ms)"
           << std::setw(CW) << "Result"
           << std::setw(CW) << "Speedup" << "\n";
    hline(W, '-');

    for (int i = 0; i < 6; i++) {
        if (!algos[i].enabled) continue;
        const auto& r = res[i];
        std::string sp = (i == 0) ? "base" : fmtSpeedup(naiveMs, r.ms);
        OUTPUT << "  " << std::left  << std::setw(NW-2) << algos[i].label
               << std::right << std::setw(CW) << fmtMs(r.ms)
               << std::setw(CW) << r.value
               << std::setw(CW) << sp;
        if (!r.ok) OUTPUT << "  [ERR]";
        OUTPUT << "\n";
    }
    hline(W, '=');
    OUTPUT << "\n";
}

static void writeCsv(
    std::ofstream& csv,
    const char* op, const DensityConfig& dens, int n,
    const AlgoConfig algos[6], const Result res[6], long long naiveMs)
{
    for (int i = 0; i < 6; i++) {
        if (!algos[i].enabled) continue;
        const auto& r = res[i];
        double sp = (naiveMs > 0 && r.ms > 0) ? (double)naiveMs / r.ms : 0.0;
        csv << op << ";"
            << dens.id << ";" << dens.name << ";"
            << n << ";"
            << algos[i].label << ";"
            << r.ms << ";"
            << r.value << ";"
            << (r.ok ? "OK" : "ERR") << ";"
            << std::fixed << std::setprecision(2) << sp << "\n";
    }
}

// Progress bar printed to stderr (separate from results)
static void showProgress(int done, int total, const char* what) {
    const int BAR = 30;
    int filled = (done * BAR) / total;
    fprintf(stderr, "\r  [");
    for (int i = 0; i < BAR; i++) fputc(i < filled ? '#' : '.', stderr);
    fprintf(stderr, "] %d/%d  %s          ", done, total, what);
    if (done == total) fputc('\n', stderr);
    fflush(stderr);
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

    fprintf(stderr, "DRC Benchmark\n");
    fprintf(stderr, "Reading %s ...\n", CONFIG_FILE);

    Config cfg = loadConfig(CONFIG_FILE);

    // Active densities
    std::vector<DensityConfig> active;
    for (auto& d : cfg.densities)
        if (d.enabled) active.push_back(d);

    if (active.empty()) {
        fprintf(stderr, "[ERROR] All densities disabled in config.\n");
        return 1;
    }

    int total = (int)(active.size() * cfg.polygonCounts.size());
    int done  = 0;

    fprintf(stderr, "Running %d combinations (density x polygon_count)...\n\n",
        total);

    // CSV
    std::ofstream csvFile;
    bool doCSV   = cfg.saveToFile    && (cfg.format=="csv"   || cfg.format=="both");
    bool doTable = cfg.printToConsole && (cfg.format=="table" || cfg.format=="both");

    if (doCSV) {
        csvFile.open(cfg.outputFilename);
        if (!csvFile.is_open()) {
            fprintf(stderr, "[ERROR] Cannot open %s\n", cfg.outputFilename.c_str());
            doCSV = false;
        } else {
            csvFile << "operation,density_id,density_name,n,algorithm,"
                       "time_ms,result,status,speedup_vs_naive\n";
        }
    }

    // Build config header for output buffer
    {
        const int W = 64;
        hline(W, '=');
        OUTPUT << "  DRC BENCHMARK RESULTS\n";
        hline(W, '-');
        OUTPUT << "  Config:  " << CONFIG_FILE << "\n";
        OUTPUT << "  Seed:    " << cfg.seed
               << "   Poly size: " << cfg.polyMin << "-" << cfg.polyMax << " um"
               << "   Layers: " << cfg.layers << "\n";
        OUTPUT << "  Min distance: " << cfg.minDistance << " um"
               << "   Runs/median: " << cfg.runsPerMeasurement << "\n";
        OUTPUT << "  Polygon counts:";
        for (int n : cfg.polygonCounts) OUTPUT << " " << n;
        OUTPUT << "\n";
        OUTPUT << "  Densities:";
        for (auto& d : active) OUTPUT << "  " << d.id
            << "(" << d.worldW << "x" << d.worldH << ")";
        OUTPUT << "\n";
        hline(W, '=');
        OUTPUT << "\n";
    }

    // Main benchmark loop
    for (const auto& dens : active) {
        for (int n : cfg.polygonCounts) {

            char what[64];
            snprintf(what, sizeof(what), "%s n=%-6d", dens.id.c_str(), n);
            showProgress(done, total, what);

            // Generate topology
            Scene sc;
            sc.generateRandomLayout(n, cfg.layers,
                dens.worldW, dens.worldH,
                cfg.polyMin, cfg.polyMax, cfg.seed);
            const Layout& lay = sc.layouts[0];

            // --- Distance Check ---
            {
                Result res[6]{};
                long long ref     = runNaiveDistanceCheck(lay, cfg.minDistance);
                long long naiveMs = medMs(cfg.runsPerMeasurement,
                    runNaiveDistanceCheck, lay, cfg.minDistance);
                res[0] = { naiveMs, ref, true };

                for (int i = 1; i < 6; i++) {
                    if (!cfg.dc[i].enabled) continue;
                    res[i] = runDC(i, lay, cfg.minDistance,
                                   cfg.runsPerMeasurement, ref);
                }

                if (doTable)
                    printTable("Distance Check", dens, n, cfg.dc, res, naiveMs);
                if (doCSV)
                    writeCsv(csvFile, "DistanceCheck", dens, n,
                             cfg.dc, res, naiveMs);
            }

            // --- Polygon Clipping ---
            {
                Result res[6]{};
                long long ref     = (long long)runNaivePolygonClipping(lay).polygons.size();
                long long naiveMs = medLMs(cfg.runsPerMeasurement,
                    runNaivePolygonClipping, lay);
                res[0] = { naiveMs, ref, true };

                for (int i = 1; i < 6; i++) {
                    if (!cfg.pc[i].enabled) continue;
                    res[i] = runPC(i, lay, cfg.runsPerMeasurement, ref);
                }

                if (doTable)
                    printTable("Polygon Clipping", dens, n, cfg.pc, res, naiveMs);
                if (doCSV)
                    writeCsv(csvFile, "PolygonClipping", dens, n,
                             cfg.pc, res, naiveMs);
            }

            done++;
            showProgress(done, total, what);
        }
    }

    if (csvFile.is_open()) {
        csvFile.close();
        OUTPUT << "\nResults saved to: " << cfg.outputFilename << "\n";
    }

    OUTPUT << "\nDone. " << total << " combinations completed.\n";

    // Print all results at once — clear console first
    printf("%s", OUTPUT.str().c_str());

    return 0;
}