#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

//nvcc -arch=sm_75 -o /tmp/test /tmp/test.cu -lcuda
 

#define CHECK_CU(err) do { \
    CUresult r = (err); \
    if (r != CUDA_SUCCESS) { \
        const char *errStr; \
        cuGetErrorString(r, &errStr); \
        fprintf(stderr, "CUDA Driver error: %s\n", errStr); \
        exit(1); \
    } \
} while(0)

int main() {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction kernel;

    int M = 4, K = 3, N = 2;

    size_t sizeA = M*K*sizeof(float);
    size_t sizeB = K*N*sizeof(float);
    size_t sizeC = M*N*sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    for (int i=0;i<M*K;i++) h_A[i] = float(i+1);
    for (int i=0;i<K*N;i++) h_B[i] = float(i+1);
    for (int i=0;i<M*N;i++) h_C[i] = 0.0f;

    CHECK_CU(cuInit(0));
    CHECK_CU(cuDeviceGet(&dev, 0));
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    CHECK_CU(cuModuleLoad(&mod, "matmul_for_gpu.fatbin"));
    CHECK_CU(cuModuleGetFunction(&kernel, mod, "ll_matmul_gpu"));

    CUdeviceptr d_A, d_B, d_C;
    CHECK_CU(cuMemAlloc(&d_A, sizeA));
    CHECK_CU(cuMemAlloc(&d_B, sizeB));
    CHECK_CU(cuMemAlloc(&d_C, sizeC));

    CHECK_CU(cuMemcpyHtoD(d_A, h_A, sizeA));
    CHECK_CU(cuMemcpyHtoD(d_B, h_B, sizeB));

    int threadsX = 16, threadsY = 16;
    int gridX = (N + threadsX - 1)/threadsX;
    int gridY = (M + threadsY - 1)/threadsY;

    void *args[] = { &d_A, &d_B, &d_C, &M, &N, &K };

    CHECK_CU(cuLaunchKernel(kernel,
        gridX, gridY, 1,
        threadsX, threadsY, 1,
        0, 0,
        args, 0));

    CHECK_CU(cuCtxSynchronize());

    CHECK_CU(cuMemcpyDtoH(h_C, d_C, sizeC));

    printf("result C =\n");
    for (int i=0;i<M;i++) {
        for (int j=0;j<N;j++) {
            printf("%f ", h_C[i*N + j]);
        }
        printf("\n");
    }

    // cleanup !!!!
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
