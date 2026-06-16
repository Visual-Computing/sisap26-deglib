# deglib: C++ Implementation for SISAP 2026 Challenge

This directory contains the C++ implementation of the Dynamic Exploration Graph (DEG) optimized and tailored for the SISAP 2026 Challenge (asks 1 & 2). It includes the core `deglib` header-only library and the task runners inside the `sisap` subdirectory.

## How to use

### Prepare the data

Download and extract the data set files from the main [readme](../readme.md) file.

### Prerequisites

* **C++ Compiler**: A modern C++20 compiler (GCC 10.0+, Clang 11.0+, MSVC 2022+, or AppleClang).
* **CMake**: Version 3.19+

IMPORTANT NOTE: this code is highly optimized using AVX2 instructions for fast distance computation.

### Compile

#### 1. Install Dependencies

Select your operating system and preferred setup method:

##### Windows (Command-line setup via winget)
```powershell
# Install MSVC C++ Compiler (Visual Studio Build Tools)
$ winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --passive"

# Install CMake
$ winget install --id Kitware.CMake
```

##### Windows (Manual setup)
1. Go to the [Visual Studio Download Page](https://visualstudio.microsoft.com/downloads/).
2. Scroll down to the bottom of the page and expand the section **"Tools for Visual Studio"**.
3. Download the installer for **"Build Tools for Visual Studio 2022"**.
4. Run the installer, select the **"Desktop development with C++"** workload (which installs the required C++ Build Tools), and complete the installation.
5. Download and install [CMake for Windows](https://cmake.org/download/).

##### Linux (Ubuntu/Debian)
```bash
$ sudo apt-get update && sudo apt-get install build-essential cmake
```

##### macOS (Command-line setup)
```bash
# Install AppleClang C++ Compiler (Xcode Command Line Tools)
$ xcode-select --install

# Install CMake via Homebrew
$ brew install cmake
```

##### macOS (Manual setup)
1. Open the **Mac App Store**, search for **"Xcode"**, and install it (this will install the AppleClang compiler). Launch Xcode once after installation to accept the license agreement.
2. Download and install [CMake for macOS](https://cmake.org/download/) directly.

#### 2. Configure and Build (via CMake Presets)

Rename `CMakePresets.json.sample` to `CMakePresets.json` and change the `DATA_PATH` cache variable inside of the file to point to the directory where your datasets are located.

Then, compile the project using standard CMake Presets from the root directory:

```bash
# Configure using your environment's preset (e.g. "windows-msvc", "linux-gcc", or "macos-clang")
cmake --preset <Preset-Name>

# Compile the Release target (e.g. "windows-msvc-release", "linux-gcc-release", or "macos-clang-release")
cmake --build --preset <Build-Preset-Name>
```

### Project Structure

This C++ repository is organized as follows:

* **[deglib/include/](deglib/include/)**: The core C++ library of the Dynamic Exploration Graph (DEG). It contains the distance metrics, builders, repository, graph structures, and search algorithms.
* **[sisap/](sisap/)**: Challenge benchmark runner files:
  * **[sisap.cpp](sisap/sisap.cpp)**: The main combined entry point program (`deglib_sisap`). It acts as a router, dispatching execution to either `task1` or `task2` depending on the first CLI argument passed.
  * **[task1.cpp](sisap/task1.cpp)**: Implementation for EVP Benchmark Task 1. Dispatches HDF5-based approximate nearest neighbor (ANN) search benchmarks across 7 different modes (e.g., FP16, EVP bits, asymmetric search, and candidate list rerankings).
  * **[task2.cpp](sisap/task2.cpp)**: Implementation for EVP Benchmark Task 2. Evaluates the `llama-dev` datasets and incorporates advanced features like FLAS pre-sorting, entry-search expansions, and graph optimization sweeps.
  * **[flas/](sisap/flas/)**: Fast Linear Assignment Sorter library used to pre-sort database vectors to optimize graph construction.
  * **[hdf5/](sisap/hdf5/)**: Custom header-only parser to scan, read, and interpret dataset HDF5 files natively.

### Running the Executable

Once compiled, the executable `deglib_sisap` can be run from the build output directory.

#### Basic Usage Syntax

```bash
# General syntax
./deglib_sisap <task> <hdf5_file_path> <mode_name> [options...]
```

* `<task>`: Set to `task1` or `task2`.
* `<hdf5_file_path>`: Path to a valid SISAP HDF5 dataset file (e.g. `benchmark-dev-wikipedia-bge-m3-small.h5` or `llama-dev.h5`).
* `<mode_name>`: The target benchmark mode (e.g. `mode4`, `evp-rerank`).

#### Examples

##### Run Task 1 (EVP Benchmark) locally:
```bash
# On Windows (from the build folder):
.\build\windows-msvc\bin\Release\deglib_sisap.exe task1 data/wikipedia-small.h5 mode4 --threads 8 --max-dist 200

# On Linux:
./build/linux-gcc/bin/Release/deglib_sisap task1 data/wikipedia-small.h5 mode4 --threads 8 --max-dist 200
```

##### Run Task 2 (Llama Dev Benchmark) locally:
```bash
# Run with FLAS pre-sorting and a search limit sweep:
./build/linux-gcc/bin/Release/deglib_sisap task2 data/llama-dev.h5 mode7 --threads 8 --flas --max-dist 5000,6000 --eps-search 0.007
```

#### Key Command Line Options

* `--threads <n>`: Number of CPU worker threads used for parallel construction and search (default: `8` / host allocation).
* `--max-dist <list>`: Exploration search budget (e.g., `200` or a comma-separated list like `100,200,300` for sweeps).
* `--k-top <n>`: Number of nearest neighbors to retrieve per query (default: `15` for task1, `30` for task2).
* `--k-graph <n>`: Degree/edges per vertex in the graph (default: `32`).
* `--no-recall`: Skips loading ground-truth data (requires `--output`).
* `--output <path>`: Path to write retrieved neighbor indices to a binary `.ivecs` file.
* `--flas` (Task 2 only): Enables FLAS pre-sorting of training vectors before graph building.

## Graph modes

The `deglib_sisap` binary implements seven graph modes per task (`mode1`…`mode7`). All modes share the same save-mode contract (writing one result file per operating point holding neighbor IDs and distances), so they are drop-in alternatives.

### Task 1 — EVP variants

| Mode        | Name                           | Description                                  |
|-------------|--------------------------------|----------------------------------------------|
| mode1       | fp16                           | FP16 build + FP16 explore                    |
| mode2       | evp-linear                     | EVP quantization + brute-force linear search |
| mode3       | evp                            | EVP build + EVP explore (no rerank)          |
| **mode4** ⭐ | evp-rerank                     | EVP build + EVP explore + FP16 rerank        |
| mode5       | evp-build-fp16-external-search | EVP build + FP16 external graph search       |
| mode6       | evp-asymmetric                 | EVP build + asymmetric FP16-vs-EVP search    |
| mode7 ⭐     | evp-asymmetric-rerank          | EVP build + asymmetric search + FP16 rerank  |

### Task 2 — L2-lift variants

| Mode        | Name                    | Description                             |
|-------------|-------------------------|-----------------------------------------|
| mode1       | baseline                | FP32 build + FP32 inner-product explore |
| mode2       | fp16-build-fp16-explore | FP16 build + FP16 IP explore            |
| mode3       | baseline-fp16           | FP32 build + FP16 IP explore            |
| mode4       | l2-converted            | FP32 L2(d+1) build + FP32 L2 explore    |
| **mode5** ⭐ | l2-fp16-ip              | FP32 L2(d+1) build + FP16 IP explore    |
| mode6       | l2-fp16-l2              | FP32 L2(d+1) build + FP16 L2 explore    |
| mode7 ⭐     | l2-fp16-d2              | FP32 L2(d+2) build + FP16 L2 explore    |
