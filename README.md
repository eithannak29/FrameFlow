# FrameFlow Video Processing

This project is a video processing application that uses CUDA to perform frame-by-frame processing on a video file. The application reads a video file, processes each frame, and writes the processed frames to an output video file.

<div align="center">
  <img src="lava.gif" alt="FrameFlow">
</div>

## Prerequisites

### Minimum Requirements
- CMake version 3.15 or higher
- CUDA Toolkit version 10.2 or higher (for GPU mode)
- GCC version 8.0 or higher
- Nix (if using Nix on OpenStack)

If you're using Nix on OpenStack, use the provided flake to set up your environment:

```sh
nix develop
```

## 1. Configure the Project

First, you need to configure the project using `cmake`. You can choose between Debug or Release builds:

```sh
export builddir=... # specify a directory (not in AFS)
cmake -S . -B $builddir -DCMAKE_BUILD_TYPE=Debug
```

Alternatively, for a Release build:

```sh
cmake -S . -B $builddir -DCMAKE_BUILD_TYPE=Release
```

## 2. Build the Project

After configuring, build the project:

```sh
make build
```

This will compile the source code and create the necessary binaries in the build directory.

## 3. Run the Project

Run the project with a specified video file using the following command:

```sh
$builddir/stream --mode=[gpu,cpu] <video.mp4> [--output=output.mp4]
```

Replace `<video.mp4>` with the path to your input video and optionally specify an output file.

## Makefile Targets

### Configuring the Project

To configure the project using the Makefile, run:

```sh
make configure
```

This target will use CMake to configure the project in the specified build mode (Debug or Release).

### Building the Project

To build the project:

```sh
make build
```

This target will first run `make configure` and then build the project using CMake.

### Running the Project

To run the project:

```sh
make run input_video=<path/to/video.mp4> out_file=<path/to/output.mp4>
```

If `input_video` or `out_file` is not provided, it will default to `samples/ACET.mp4` and `outputs/output.mp4`, respectively.

### Benchmarking the Project

To benchmark the project:

```sh
make bench input_video=<path/to/video.mp4> out_file_cpu=<path/to/cpu_output.mp4> out_file_gpu=<path/to/gpu_output.mp4>
```

If paths are not provided, default values will be used (`samples/ACET.mp4`, `outputs/cpu.mp4`, and `outputs/gpu.mp4`). This target will benchmark both CPU and GPU modes sequentially.

### Profiling the Project

To profile the project using `nvprof`:

```sh
make profiler input_video=<path/to/video.mp4> out_file=<path/to/output.mp4>
```

This will run the `nvprof` tool to profile the GPU execution of the project. Default values will be used if paths are not provided.

### Cleaning Build Files

To clean all build and output files:

```sh
make clean
```

This target will remove all build files and generated output videos.

## Editing the Code

Edit your CUDA/C++ code in `*/Compute.*` to make modifications to the main processing logic of the project.

## Example Workflow

1. Configure the project:

   ```sh
   make configure
   ```

2. Build the project:

   ```sh
   make build
   ```

3. Run the project:

   ```sh
   make run
   ```

## Notes

- The project defaults to GPU mode, but you can switch to CPU by modifying the `mode` variable in the Makefile or when running the executable.
- Ensure the `output` directory exists or let the Makefile create it for you.

