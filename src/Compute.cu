#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"

// Single threaded version of the Method
__global__ void mykernel(ImageView<rgb8> in, ImageView<uint8_t> logo)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < in.width && y < in.height)
    {
        rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
        pixel[x].r = 0;

        if (x < logo.width && y < logo.height)
        {
            float alpha = logo.buffer[y * logo.stride + x] / 255.f;
            pixel[x].g = uint8_t(alpha * pixel[x].g + (1 - alpha) * 255);
            pixel[x].b = uint8_t(alpha * pixel[x].b + (1 - alpha) * 255);
        }
    }
}


void compute_cu(ImageView<rgb8> in)
{
    static Image<uint8_t> device_logo;
    static Image<uint8_t> pixel_time_counter;
    static Image<rgb8> device_background;
    static Image<rgb8> device_candidate

    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
    
    // Copy the logo to the device if it is not already there
    if (device_logo.buffer == nullptr)
    {
        device_logo = Image<uint8_t>(logo_width, logo_height, true);
        cudaMemcpy2D(device_logo.buffer, device_logo.stride, logo_data, logo_width, logo_width, logo_height, cudaMemcpyHostToDevice);
    }

    if(device_background.buffer == nullptr)
    {
        device_background = Image<rgb8>(in.width, in.height, true);
        cudaMemcpy2D(device_background.buffer, device_background.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);

        device_candidate = Image<rgb8>(in.width, in.height, true);
        cudaMemset2D(device_candidate.buffer, device_candidate.stride, 0, in.width * sizeof(rgb8), in.height);

        pixel_time_counter = Image<uint8_t>(in.width, in.height, true);
        cudaMemset2D(pixel_time_counter.buffer, pixel_time_counter.stride, 0, in.width, in.height);
    }

    // Copy the input image to the device
    Image<rgb8> device_in(in.width, in.height, true);
    cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);
    
    mykernel<<<grid, block>>>(device_background, device_logo);

    // Copy the result back to the host
    cudaMemcpy2D(in.buffer, in.stride, device_background.buffer, device_background.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost);
}
