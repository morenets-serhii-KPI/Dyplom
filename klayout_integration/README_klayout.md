# KLayout Integration

GPU-accelerated DRC pipeline: GDS → violations → GDS.

## Flow

```
input.gds
    │
    ▼  python run.py
    └── drc_results.gds   ← open in KLayout
```

## Requirements

- Python 3.8+  +  `pip install gdstk`
- NVIDIA GPU (Compute Capability ≥ 6.0)
- CUDA Toolkit ≥ 11.0
- x64 Native Tools Command Prompt for VS (build only)

## Build

```
nvcc one_engine.cu -o one_engine.exe -IC.. -IC..\external -O2 -arch=sm_75 -std=c++17 -Xcompiler "/Zc:preprocessor /EHsc"
```

Change `-arch=sm_75` to match your GPU:
`sm_75` RTX 20xx · `sm_86` RTX 30xx · `sm_89` RTX 40xx

## Run

Place your GDS file as `input.gds` and run:

```
python run.py
```

Options:
```
--gds   FILE    input GDS file       (default: input.gds)
--dist  FLOAT   min distance in um   (default: 20.0)
--arch  sm_XX   CUDA arch            (default: sm_75)
--no-build      skip compilation
```

## Output

One file: `drc_results.gds` — open in KLayout.

| Layer | Meaning |
|-------|---------|
| 100/101 | polygon pair that violates distance rule |
| 102 | gap marker between them |
| 200 | critical overlap area |
