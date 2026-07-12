# Roofline Analyst

**CPU roofline profiling for PyTorch modules — C++20 hardware benchmark + Python analysis.**

Roofline Analyst measures CPU hardware limits in C++20, profiles supported PyTorch module invocations in Python, and renders an interactive roofline chart.

It is designed as an ML inference-performance engineering project: every roofline point states what was measured, what was estimated, and what hardware scope applies.

`C++20` · `ARM NEON` · `Python 3.13` · `PyTorch` · `Plotly`

---

## Table of Contents

- [Architecture](#architecture)
- [Measurement Scope](#measurement-scope)
- [Methodology](#methodology)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [C++ Benchmark Parameters](#c-benchmark-parameters)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Repository Structure](#repository-structure)
- [Interview Summary](#interview-summary)

---

## Architecture

```text
C++20 / ARM NEON
  └─ roofline_hw
       ├─ measures CPU copy bandwidth
       ├─ measures single-threaded NEON FP32 FMA throughput
       └─ writes one JSON object to stdout

Python / PyTorch
  └─ roofline_analyst
       ├─ runs roofline_hw as a subprocess
       ├─ validates its JSON contract
       ├─ profiles PyTorch operators
       ├─ captures leaf-module shapes and timings with forward hooks
       ├─ estimates logical bytes moved
       └─ writes an interactive Plotly roofline chart
```

### Why a subprocess instead of pybind11?

The C++ benchmark runs once per analysis session and returns two scalar values:

```json
{
  "memory_bandwidth_gbps": 120.0,
  "cpu_neon_fp32_gflops": 74.0
}
```

A Python extension would add Python ABI coupling, extension-module build rules, headers, linker complexity, and in-process native failure modes — without improving a one-shot call.

**Decision rule:**

| Condition | Approach |
|---|---|
| Low call frequency + simple serialized output | subprocess + JSON |
| High call frequency + rich data crossing boundary | native bindings |

---

## Measurement Scope

| Metric | Scope |
|---|---|
| Memory bandwidth | Single-threaded CPU copy throughput from a large-buffer `memcpy` benchmark |
| Compute throughput | Single-threaded CPU NEON FP32 fused multiply-add throughput |
| Layer bytes moved | Logical read/write estimate from tensors and module parameters |
| Layer FLOPs | Supported module formula and PyTorch profiler estimates |
| Layer timing | CPU forward-hook elapsed time |

> **Out of scope.** This project does not claim to measure Apple GPU, AMX, Neural Engine, multi-core CPU peak, or exact DRAM-controller traffic.

> **Note on Apple's published bandwidth figure.** Apple lists M5 Pro unified memory bandwidth as up to 307 GB/s. That is an aggregate hardware ceiling; the CPU-copy benchmark measures a narrower, machine-local CPU execution path.
> — [Apple M5 Pro specifications](https://support.apple.com/en-us/126319)

---

## Methodology

### Memory bandwidth

Each copy moves:

```text
read source N bytes + write destination N bytes = 2N bytes
bandwidth = iterations × 2 × buffer_bytes / elapsed_seconds
```

The benchmark:

- allocates large heap buffers to avoid cache-resident measurements
- pre-touches pages before timing
- alternates two differently initialized sources
- observes destination contents after every copy to prevent `-O3` from collapsing repeated copies

### CPU FLOPS

The NEON benchmark uses eight independent `float32x4_t` accumulators.

```text
8 accumulators × 4 FP32 lanes × 2 FLOPs per FMA lane = 64 FLOPs per loop iteration
```

Independent accumulators hide FMA dependency latency. One accumulator would measure latency-bound throughput rather than the CPU's FMA issue rate.

### Layer roofline points

```text
arithmetic intensity = FLOPs / estimated logical bytes moved
achieved GFLOPS      = FLOPs / elapsed_nanoseconds
ridge point          = peak_GFLOPS / bandwidth_GBps
```

Points left of the ridge are memory-bound under this model; points right of the ridge are compute-bound.

---

## Requirements

- macOS on ARM64
- Homebrew LLVM clang 17+ (validated locally with LLVM 22)
- CMake 4.2+
- Ninja
- Python 3.13
- [uv](https://docs.astral.sh/uv/)

---

## Getting Started

### 1. Build the C++ benchmark

```bash
cmake -S . -B build-homebrew-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++

cmake --build build-homebrew-release
```

Run the hardware binary directly:

```bash
./build-homebrew-release/roofline_hw
```

Example output:

```json
{
  "memory_bandwidth_gbps": 120.0,
  "cpu_neon_fp32_gflops": 74.0
}
```

> The exact values are machine-, thermal-, compiler-, and workload-specific.

### 2. Set up Python

```bash
uv sync
```

### 3. Generate a roofline chart

```bash
uv run roofline-analyst \
  --binary build-homebrew-release/roofline_hw \
  --model token-mlp \
  --batch-size 2 \
  --sequence-length 16 \
  --output roofline.html
```

Open `roofline.html` in a browser.

The current built-in model is `token-mlp`, with input shape `[batch_size, sequence_length, 128]`. Its two `nn.Linear` invocations receive supported per-layer roofline points. GELU is not plotted as a roofline point because the current implementation has no per-module FLOP formula for it — it is not falsely treated as zero FLOPs.

---

## C++ Benchmark Parameters

```bash
./build-homebrew-release/roofline_hw \
  --buffer-mib 256 \
  --memory-iterations 200 \
  --fma-iterations 500000000
```

Use the defaults for the calibrated benchmark. Smaller settings are useful for CLI smoke tests but should not be interpreted as stable hardware ceilings.

---

## Limitations

- Logical bytes moved are estimates, not direct DRAM hardware-counter readings.
- PyTorch operator FLOP coverage is not universal.
- Forward-hook timing includes framework overhead and is least reliable for tiny layers.
- The CLI currently provides the reproducible built-in `token-mlp` model only.
- CPU roofline limits must not be compared as though they were GPU or AMX limits.

---

## Troubleshooting

### macOS editable-install issues

If Python reports `ModuleNotFoundError: No module named 'roofline_analyst'`, check whether macOS marked the editable-install `.pth` file hidden:

```bash
uv run python -v -c "pass" 2>&1 | grep roofline_analyst.pth
```

If it says `Skipping hidden .pth file`, repair the local virtual environment:

```bash
chflags -R nohidden .venv
xattr -rc .venv
```

Then retry `uv run`.

---

## Repository Structure

```text
src/cpp/                    C++20 hardware benchmark
src/roofline_analyst/       Python package and CLI
docs/                       architecture and validation notes
CMakeLists.txt              C++ build definition
pyproject.toml              Python package metadata
uv.lock                     exact Python dependency resolution
```

---

## Interview Summary

Roofline Analyst empirically measures a machine's CPU copy bandwidth and single-threaded NEON FP32 FMA throughput in C++. Python invokes the binary once through a JSON subprocess contract, profiles a PyTorch model, captures module tensor metadata and timings, estimates logical memory traffic, and plots supported module invocations against the measured roofline.