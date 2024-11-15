# Paths and configurations
builddir := build
outputdir := outputs
outputfile := $(outputdir)/output.mp4
default_output_cpu := $(outputdir)/cpu.mp4
default_output_gpu := $(outputdir)/gpu.mp4
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
	output_file=$(out_file); \
	if [ -z "$$output_file" ]; then \
	    output_file=$(outputfile); \
	fi; \
	echo "Running with input video $$input_file and output file $$output_file..."; \
	$(builddir)/stream --mode=$(mode) $$input_file --output=$$output_file

# Target to benchmark the project
.PHONY: bench
bench: build | $(outputdir)
	@input_file=$(input_video); \
	if [ -z "$$input_file" ]; then \
	    input_file=$(default_video); \
	fi; \
	output_file_cpu=$(out_file_cpu); \
	if [ -z "$$output_file_cpu" ]; then \
	    output_file_cpu=$(default_output_cpu); \
	fi; \
	output_file_gpu=$(out_file_gpu); \
	if [ -z "$$output_file_gpu" ]; then \
	    output_file_gpu=$(default_output_gpu); \
	fi; \
	echo "Starting benchmark for CPU mode..."; \
	$(builddir)/stream --mode=cpu $$input_file --output=$$output_file_cpu; \
	echo "Starting benchmark for GPU mode..."; \
	$(builddir)/stream --mode=gpu $$input_file --output=$$output_file_gpu

# Target to profile the project using nvprof
.PHONY: profiler
profiler: build | $(outputdir)
	@input_file=$(input_video); \
	if [ -z "$$input_file" ]; then \
	    input_file=$(default_video); \
	fi; \
	output_file=$(out_file); \
	if [ -z "$$output_file" ]; then \
	    output_file=$(default_output_gpu); \
	fi; \
	echo "Profiling with input video $$input_file and output file $$output_file..."; \
	nvprof $(builddir)/stream --mode=gpu $$input_file --output=$$output_file

# Create the outputs directory if it doesn’t exist
$(outputdir):
	@mkdir -p $(outputdir)

# Target to clean the build files
.PHONY: clean
clean:
	@echo "Cleaning build files..."
	rm -rf $(builddir)/* $(outputdir)/*.mp4
