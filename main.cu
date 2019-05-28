#include <iostream>
#include "vec3.h"
#include "ray.h"
#include "hitable_list.h"
#include "sphere.h"

__global__ void create_world(hitable **d_list, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *d_world    = new hitable_list(d_list,2);
    }
}

__device__ vec3 color(const ray& r, hitable **world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, MAXFLOAT, rec)) {
        return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f*(unit_direction.y() + 1.0f);
        return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

__global__ void render(vec3 *img, int nx, int ny, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, hitable **world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= nx) || (j >= ny)) return;
    
    int pixel_index = j * nx + i;
    float u = float(i) / float(nx);
    float v = float(j) / float(ny);

    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    img[pixel_index] = 255.99 * color(r, world);
}


int main() {
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering Image: " << nx << "x" << ny << std::endl;

    int num_pixels = nx*ny;
    size_t img_size = num_pixels*sizeof(vec3);

    vec3 *img;
    cudaMallocManaged((void **)&img, img_size);

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    hitable **list, **world; 
    cudaMalloc((void **)&list, 2*sizeof(hitable *));
    cudaMalloc((void **)&world, sizeof(hitable *));
    
    create_world<<<1,1>>>(list, world);
    cudaDeviceSynchronize();
    
    vec3 lower_left(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);

    render<<<blocks, threads>>>(img, nx, ny, lower_left, horizontal, vertical, origin, world);
    cudaDeviceSynchronize();
    
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(img[pixel_index].r());
            int ig = int(img[pixel_index].g());
            int ib = int(img[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    cudaFree(img);
}