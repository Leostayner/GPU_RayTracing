#include <iostream>
#include "vec3.h"
#include "ray.h"

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4.0f*a*c;
    return (discriminant > 0.0f);
}

__device__ vec3 color(const ray& r) {
    if (hit_sphere(vec3(0,0,-1), 0.5, r)) return vec3(1,0,0);
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *img, int nx, int ny, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= nx) || (j >= ny)) return;
    
    int pixel_index = j * nx + i;
    float u = float(i) / float(nx);
    float v = float(j) / float(ny);

    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    img[pixel_index] = 255.99 * color(r);
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

    vec3 lower_left(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);

    render<<<blocks, threads>>>(img, nx, ny, lower_left, horizontal, vertical, origin);
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