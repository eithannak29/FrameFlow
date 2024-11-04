# Paths and configurations
builddir := build
outputdir := outputs
outputfile := $(outputdir)/output.mp4
mode := cpu
build_type := Debug

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
	@if [ -z "$(input_video)" ]; then \
	    echo "Error: please specify the input file with 'make run input_video=path/to/video.mp4'"; \
	    exit 1; \
	fi
	@echo "Running with input video $(input_video)..."
	$(builddir)/stream --mode=$(mode) $(input_video) --output=$(outputfile)

# Create the outputs directory if it doesn’t exist
$(outputdir):
	@mkdir -p $(outputdir)

# Target to clean the build files
.PHONY: clean
clean:
	@echo "Cleaning build files..."
	rm -rf $(builddir)/* $(outputdir)/output.mp4
