#include <iostream>

__global__ void render(float *img, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;
    
    int pixel_index = j*nx*3 + i*3;
    img[pixel_index + 0] = float(i) / nx;
    img[pixel_index + 1] = float(j) / ny;
    img[pixel_index + 2] = 0.2;
}

int main() {
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cout << "Rendering Image: " << nx << "x" << ny << std::endl;

    int num_pixels = nx*ny;
    size_t img_size = 3*num_pixels*sizeof(float);

    float *img;
    cudaMallocManaged((void **)&img, img_size);

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    render<<<blocks, threads>>>(img, nx, ny);
    cudaDeviceSynchronize();
    
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*3*nx + i*3;
            float r = img[pixel_index + 0];
            float g = img[pixel_index + 1];
            float b = img[pixel_index + 2];
            
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    cudaFree(img);
}