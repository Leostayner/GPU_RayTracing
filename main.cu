#include <iostream>
#include <curand_kernel.h>
#include <float.h>
#include "vec3.h"
#include "ray.h"
#include "hitable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

//Alterar
__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0.8, 0.3, 0.3)));
        d_list[1] = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vec3(1,0,-1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 1.0));
        d_list[3] = new sphere(vec3(-1,0,-1), 0.5, new metal(vec3(0.8, 0.8, 0.8), 0.3));
        *d_world  = new hitable_list(d_list,4);
        *d_camera = new camera();
    }
}

//Alterar
__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for(int i=0; i < 4; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__device__ vec3 color(const ray& r, hitable **world, curandState *rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0);
}

__global__ void render(vec3 *img, int nx, int ny, int ns, hitable **world, camera **cam) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if((i >= nx) || (j >= ny)) return;
    int pixel_index = j * nx + i;
    
    curandState state;
    curand_init((unsigned long long)clock64() + pixel_index, (unsigned long long)0, 0, &state);
    
    vec3 col(0,0,0);
    for(int s=0; s<ns; s++){
        float u = float(i + curand_uniform(&state)) / float(nx);
        float v = float(j + curand_uniform(&state)) / float(ny);
        ray r = (*cam)->get_ray(u, v);
        col += color(r, world, &state);
    }
    
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    img[pixel_index] = 255.99 * col;
}


int main() {
    int nx = 1200;
    int ny = 600;
    int ns = 100;
    int tx = 8;
    int ty = 8;
    
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    /**********/
    vec3 *img;
    cudaMallocManaged((void **)&img, nx*ny*sizeof(vec3));

    hitable **list, **world; 
    cudaMalloc((void **)&list, 2*sizeof(hitable *));
    cudaMalloc((void **)&world, sizeof(hitable *));

    camera **cam;
    cudaMalloc((void **)&cam, sizeof(camera *));

    
    /**********/
    create_world<<<1,1>>>(list, world, cam);
    cudaDeviceSynchronize();
    
    render<<<blocks, threads>>>(img, nx, ny, ns, world, cam);
    cudaDeviceSynchronize();

    /**********/
    std::cerr << "Rendering Image: " << nx << "x" << ny << std::endl;
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

    /**********/
    cudaDeviceSynchronize();
    free_world<<<1,1>>>(list, world, cam);
    void* freeList[4] = {cam, world, list, img};
    for(int i=0; i<4; i++) cudaFree(freeList[i]);    
    cudaDeviceReset();
}