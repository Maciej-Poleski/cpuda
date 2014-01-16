#include "cuda.h"
#include <cstdio>

int main(){
    // -------- fundamental operations --------

    // initialize driver - before calling any Driver API function
    cuInit(0);
    // get device handler (there can be few devices)
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot acquire device 0\n");
        exit(1);
    }
    // create context
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        printf("cannot create context\n");
        exit(1);
    }
    // load .ptx module with given name
    CUmodule cuModule = (CUmodule)0;
    res = cuModuleLoad(&cuModule, "cuModule.ptx");
    if (res != CUDA_SUCCESS) {
        printf("cannot load module: %d\n", res);
        exit(1);
    }
    // get handler to function with given name from loaded module
    CUfunction cuFun;
    res = cuModuleGetFunction(&cuFun, cuModule, "cuFun");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

    // -------- sample usage --------

    CUdeviceptr dev_tab; // pointer of device memory
    int *tab;
    int n = 32;
    int m = 16;
    int size = n*m*sizeof(int);

    // memory management
    cuMemAlloc(&dev_tab, size); // allocate memory on the device
    cuMemAllocHost((void **)&tab, size); // allocate pinned memory on CPU which would be accessible to the device

    cuMemcpyHtoD(dev_tab, tab, size); // copy memory from host to device (input for device)

    // prepare arguments and launch kernel
    void* args[] = {&dev_tab, &m, &n};
    res = cuLaunchKernel(cuFun,
                         2, 2, 2, 32, 16, 2,  // blocks in grid, threads in block
                         0, 0, args, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }

    // synchronize host with device (wait until kernel will be finished)
    cuCtxSynchronize();

    // memory management
    cuMemcpyDtoH(tab, dev_tab, size); // copy memory from device to host (get result)

    cuMemFree(dev_tab); // free device memory
    cuMemFreeHost(tab); // free host memory

    // destroy context
    cuCtxDestroy(cuContext);
    return 0;
}
