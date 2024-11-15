# Paths and configurations
builddir := build
outputdir := outputs
outputfile := $(outputdir)/acet_bg_cuda.mp4
outputfile_cpu := $(outputdir)/cpu.mp4
outputfile_gpu := $(outputdir)/gpu.mp4
mode := gpu #cpu #gpu
build_type := Debug
default_video := samples/ACET.mp4

# Target to configure the project with CMake
.PHONY: configure
configure:
	@echo "Configuring the project in $(build_type) mode..."
	cmake -S . -B $(builddir) -DCMAKE_BUILD_TYPE=$(build_type)

# Target to build the project
.PHONY: build
build: configure
	@echo "Building the project..."
	cmake --build $(builddir)

# Target to run the project
.PHONY: run
run: build | $(outputdir)
	@input_file=$(input_video); \
	if [ -z "$$input_file" ]; then \
	    input_file=$(default_video); \
	fi; \
	echo "Running with input video $$input_file..."; \
	$(builddir)/stream --mode=$(mode) $$input_file --output=$(outputfile)

.PHONY: bench
bench: build | $(outputdir)
	@input_file=$(input_video); \
	if [ -z "$$input_file" ]; then \
	    input_file=$(default_video); \
	fi; \
	echo "Starting benchmark for CPU mode..."; \
	$(builddir)/stream --mode=cpu $$input_file --output=$(outputfile_cpu); \
	echo "Starting benchmark for GPU mode..."; \
	$(builddir)/stream --mode=gpu $$input_file --output=$(outputfile_gpu)

# Target to profile the project using nvprof
.PHONY: profiler
profiler: build | $(outputdir)
	@input_file=$(input_video); \
	if [ -z "$$input_file" ]; then \
	    input_file=$(default_video); \
	fi; \
	echo "Profiling with input video $$input_file..."; \
	nvprof ${builddir}/stream --mode=gpu $$input_file --output=$(outputfile_gpu)

.PHONY: profiler_cpp
profiler_cpp: build | $(outputdir)
	@input_file=$(default_video); \
	echo "Profiling the project with gprof using input video $$input_file..."; \
	g++ -pg -o $(builddir)/program main.cpp -O2 && \
	$(builddir)/program $$input_file && \
	gprof $(builddir)/program gmon.out > $(outputfile) && \
	echo "Profiling complete. Report saved to $(outputfile).

# Create the outputs directory if it doesn’t exist
$(outputdir):
	@mkdir -p $(outputdir)

# Target to clean the build files
.PHONY: clean
clean:
	@echo "Cleaning build files..."
	rm -rf $(builddir)/* $(outputdir)/output.mp4
