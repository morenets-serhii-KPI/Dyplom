# DRC GPU Accelerator

Benchmark research project for the bachelor's thesis:
**"Accelerating DRC Geometric Operations on GPU"**

Compares six implementations of two core Design Rule Checking operations —
**distance check** and **polygon clipping** — across CPU and GPU, measuring
execution time and speedup under varying topology sizes and densities.

---

## Algorithms

| # | Algorithm | Environment |
|---|-----------|-------------|
| 1 | CPU Naive (O(n²) brute-force) | CPU |
| 2 | CPU Sweep-line | CPU |
| 3 | CPU Parallel (OpenMP) | CPU |
| 4 | CPU Optimized Parallel (OpenMP + sweep) | CPU |
| 5 | GPU Brute-force (CUDA 2D grid) | GPU |
| 6 | GPU Optimized (CUDA + Thrust sort + sweep) | GPU |

---

## Requirements

**Hardware**
- NVIDIA GPU with CUDA Compute Capability ≥ 6.0
- ≥ 2 GB VRAM
- ≥ 8 GB RAM

**Software**
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) ≥ 11.0
- Microsoft Visual Studio (with *Desktop development with C++* workload)
- Windows 10/11 x64 or Linux (Ubuntu 20.04+)

---

## Project Structure

```
project/
├── main.cu                          # Entry point — reads config.json and runs benchmarks
├── config.json                      # All runtime parameters (edit this, not the code)
├── external/
│   └── json.hpp                     # nlohmann/json (header-only, no install needed)
├── topology/
│   └── topology.h                   # Point, GdsPolygon, Layout, Scene — data structures
└── algorithms/
    ├── distance_check/
    │   ├── cpu_naive.h
    │   ├── cpu_sweepline.h
    │   ├── cpu_parallel.h
    │   ├── cpu_optimized_parallel.h
    │   ├── gpu_parallel.cuh
    │   └── gpu_optimized.cuh
    └── polygon_clipping/
        ├── cpu_naive.h
        ├── cpu_sweepline.h
        ├── cpu_parallel.h
        ├── cpu_optimized_parallel.h
        ├── gpu_parallel.cuh
        └── gpu_optimized.cuh
```

---

## Build

Open **x64 Native Tools Command Prompt for VS** (found in the Start menu under Visual Studio),
navigate to the project directory, and run:

```bash
nvcc main.cu -o main.exe -O2 -Xcompiler "/openmp /std:c++17 /EHsc" -arch=sm_86
```

**Pick the right `-arch` flag for your GPU:**

| GPU Series | Flag |
|------------|------|
| RTX 20xx (Turing) | `-arch=sm_75` |
| RTX 30xx (Ampere) | `-arch=sm_86` |
| RTX 40xx (Ada Lovelace) | `-arch=sm_89` |
| Unknown | omit the flag — NVCC will choose automatically |

**Linux:**
```bash
nvcc main.cu -o main -O2 -Xcompiler "-fopenmp -std=c++17" -arch=sm_86
```

---

## Configuration

All benchmark parameters are controlled via `config.json`.
**No code editing required.** Open the file in any text editor and adjust as needed.

### Polygon counts
Any number of entries — a separate benchmark run is performed for each value:
```json
"polygon_counts": [1000, 5000, 10000, 20000]
```

### Densities
Add, remove, or disable individual density scenarios:
```json
"densities": [
  { "name": "Low",    "id": "LOW",    "world_width": 50000, "world_height": 50000, "enabled": true  },
  { "name": "Medium", "id": "MEDIUM", "world_width": 10000, "world_height": 10000, "enabled": true  },
  { "name": "High",   "id": "HIGH",   "world_width":  3000, "world_height":  3000, "enabled": false }
]
```

### Enable / disable individual algorithms
```json
"distance_check": {
  "cpu_naive":        { "enabled": true  },
  "cpu_sweepline":    { "enabled": true  },
  "cpu_parallel":     { "enabled": false },
  "cpu_opt_parallel": { "enabled": true  },
  "gpu_bruteforce":   { "enabled": false },
  "gpu_optimized":    { "enabled": true  }
}
```

### Other parameters

```json
"topology": {
  "seed": 42,             // random generator seed — guarantees reproducibility
  "poly_min_size": 20.0,  // minimum polygon side (um)
  "poly_max_size": 150.0, // maximum polygon side (um)
  "layers": 3             // number of topology layers
},

"check": {
  "min_distance": 20.0    // minimum allowed distance for distance check (um)
},

"algorithms": {
  "runs_per_measurement": 3  // runs per algorithm — median is reported
},

"output": {
  "save_to_file":     true,          // write results to CSV
  "filename":         "results.csv", // output file name
  "print_to_console": true,          // print tables to console after completion
  "format":           "both"         // "table" | "csv" | "both"
}
```

---

## Run

```bash
main.exe          # Windows
./main            # Linux
```

While running, a progress bar is shown in the terminal:

```
  [##############..............] 5/9  MEDIUM n=5000
```

When all measurements are complete, the console is cleared and the final
results are printed as clean formatted tables:

```
================================================================
  Distance Check  |  Medium  |  n = 10000
----------------------------------------------------------------
  Algorithm              Time (ms)      Result      Speedup
----------------------------------------------------------------
  CPU Naive                   1251      7232          base
  CPU Sweep-line                23      7232         54.4x
  CPU Opt.Parallel               8      7232        156.4x
  GPU Brute-force               32      7232         39.1x
  GPU Optimized                  4      7232        312.8x
================================================================
```

If `save_to_file` is enabled, `results.csv` is written to the project
directory simultaneously and can be opened directly in Excel.

---

## CSV Output Format

```
operation, density_id, density_name, n, algorithm, time_ms, result, status, speedup_vs_naive
```

Example:
```
DistanceCheck,MEDIUM,Medium,10000,GPU Optimized,4,7232,OK,312.75
```

---

## Reproducibility

All topology data is generated procedurally using a seeded random number generator.
The same `seed` value always produces the same topology, making all results
fully reproducible across machines given identical hardware.

---

## References

- Gene Amdahl (1967) — *Validity of the single processor approach to achieving large scale computing capabilities*
- He Z. et al. — *X-Check: GPU-Accelerated DRC via Parallel Sweepline*, ICCAD 2022
- He Z. et al. — *OpenDRC: Open-Source DRC with Hierarchical GPU Acceleration*, DAC 2023
- NVIDIA CUDA Toolkit Documentation — https://docs.nvidia.com/cuda/
- KLayout EDA Tool — https://www.klayout.de
- nlohmann/json — https://github.com/nlohmann/json