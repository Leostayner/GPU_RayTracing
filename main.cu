#include <iostream>
#include "vec3.h"

__global__ void render(vec3 *img, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;
    
    int pixel_index = j*nx + i;
    img[pixel_index] = vec3( float(i)/nx, float(j)/ny, 0.2f);
}

int main() {
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cout << "Rendering Image: " << nx << "x" << ny << std::endl;

    int num_pixels = nx*ny;
    size_t img_size = num_pixels*sizeof(vec3);

    vec3 *img;
    cudaMallocManaged((void **)&img, img_size);

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    render<<<blocks, threads>>>(img, nx, ny);
    cudaDeviceSynchronize();
    
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*img[pixel_index].r());
            int ig = int(255.99*img[pixel_index].g());
            int ib = int(255.99*img[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    cudaFree(img);
}