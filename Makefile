# Paths and configurations
builddir := ~/build
outputdir := outputs
outputfile := $(outputdir)/acet_bg_cuda.mp4
mode := cpu #cpu #gpu
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

# Create the outputs directory if it doesn’t exist
$(outputdir):
	@mkdir -p $(outputdir)

# Target to clean the build files
.PHONY: clean
clean:
	@echo "Cleaning build files..."
	rm -rf $(builddir)/* $(outputdir)/output.mp4
