#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "timer.h"
#include "check.h"
#include "cuda_runtime.h"

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

/*
 * Do not modify this function. A constraint of this exercise is
 * that it remain a host function.
 */

void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
    }
}

/*
 * This function calculates the gravitational impact of all bodies in the system
 * on all others, but does not update their positions.
 */

__global__ void bodyForceKernel(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            // r^2 = x^2 + y^2 + z^2
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            // 1 / r
            float invDist = rsqrtf(distSqr);
            // 1 / r^3
            float invDist3 = invDist * invDist * invDist;

            // F = GMm * r / r^3
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

/*
 * This position integration cannot occur until this round of `bodyForce` has completed.
 * Also, the next round of `bodyForce` cannot begin until the integration is complete.
 */

__global__ void updatePositionKernel(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(const int argc, const char **argv) {
    /*
     * Do not change the value for `nBodies` here. If you would like to modify it,
     * pass values into the command line.
     */

    int nBodies = 2 << 11;
    int salt = 0;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);

#ifdef DEV
    #define GPUID 1
    // in the dev env
    // set gpu id
    cudaSetDevice(GPUID);
#endif

    /*
     * This salt is for assessment reasons. Tampering with it will result in automatic failure.
     */

    if (argc > 2) salt = atoi(argv[2]);

    const float dt = 0.01f; // time step
    const int nIters = 10;  // simulation iterations

    int bytes = nBodies * sizeof(Body);
    float *buf;

#ifdef UNIFIED
    // use unified memory
    cudaMallocManaged((void **)&buf, bytes);
#else
    buf = (float *) malloc(bytes);
    // device mem
    Body *dev_p;
    cudaMalloc((void **)&dev_p, bytes);
#endif
    Body *p = (Body *) buf;

    // num of threads
    size_t block_dim = 32;
    // num of blocks
    size_t grid_dim = (nBodies - 1) / block_dim + 1;

    /*
     * As a constraint of this exercise, `randomizeBodies` must remain a host function.
     */

    // 6 * nBodies of float in total
    randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

#ifndef UNIFIED
    // copy from host to device
    cudaMemcpy(dev_p, p, bytes, cudaMemcpyHostToDevice);
#endif

    double totalTime = 0.0;

    /*
     * This simulation will run for 10 cycles of time, calculating gravitational
     * interaction amongst bodies, and adjusting their positions to reflect.
     */

    /*******************************************************************/
    // Do not modify these 2 lines of code.
    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();
    /*******************************************************************/

        /*
         * You will likely wish to refactor the work being done in `bodyForce`,
         * as well as the work to integrate the positions.
         */

#ifdef UNIFIED
        // compute interbody forces
        bodyForceKernel<<<grid_dim, block_dim>>>(p, dt, nBodies);

        // update positions
        updatePositionKernel<<<grid_dim, block_dim>>>(p, dt, nBodies);

        // if unified memory is used, device level synchronization at the last iter is required
        if (iter == nIters - 1) cudaDeviceSynchronize();
#else
        // compute interbody forces
        bodyForceKernel<<<grid_dim, block_dim>>>(dev_p, dt, nBodies);

        // update positions
        updatePositionKernel<<<grid_dim, block_dim>>>(dev_p, dt, nBodies);

        // copy from device to host at the last iteration
        if (iter == nIters - 1) cudaMemcpy(p, dev_p, bytes, cudaMemcpyDeviceToHost);
#endif

    /*******************************************************************/
        // Do not modify the code in this section.
        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    double avgTime = totalTime / (double) (nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

#ifdef ASSESS
    checkPerformance(buf, billionsOfOpsPerSecond, salt);
#else
    checkAccuracy(buf, nBodies);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
    salt += 1;
#endif
    /*******************************************************************/

    /*
     * Feel free to modify code below.
     */

#ifdef UNIFIED
    cudaFree(buf);
#else
    free(buf);
    cudaFree(dev_p);
#endif
}
