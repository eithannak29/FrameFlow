# Paths and configurations
builddir := ~/build
outputdir := outputs
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
	@input_file=$(or $(input_video), $(default_video)); \
	output_file=$(outputdir)/$(or $(output_video), output.mp4); \
	mode=$(or $(mode), gpu); \
	echo "Running with input video $$input_file in mode $$mode..."; \
	$(builddir)/stream --mode=$$mode $$input_file --output=$$output_file

# Target to run benchmark for CPU and GPU
.PHONY: bench
bench: build | $(outputdir)
	@input_file=$(or $(input_video), $(default_video)); \
	base_output_file=$(or $(output_video), output); \
	output_file_cpu=$(outputdir)/$$base_output_file"_cpu.mp4"; \
	output_file_gpu=$(outputdir)/$$base_output_file"_gpu.mp4"; \
	echo "Starting benchmark for CPU mode..."; \
	$(builddir)/stream --mode=cpu $$input_file --output=$$output_file_cpu; \
	echo "Starting benchmark for GPU mode..."; \
	$(builddir)/stream --mode=gpu $$input_file --output=$$output_file_gpu

# Create the outputs directory if it doesn’t exist
$(outputdir):
	@mkdir -p $(outputdir)

# Target to clean the build files
.PHONY: clean
clean:
	@echo "Cleaning build files..."
	rm -rf $(builddir)/* $(outputdir)/*
