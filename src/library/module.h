#ifndef MODULE_H
#define MODULE_H

#include "dim3.h"

typedef dim3 gridDim_t;
typedef dim3 blockDim_t;
typedef dim3 blockIdx_t;
typedef dim3 threadIdx_t;

class Module;

/**
 * @brief Object of this class gives acces to one kernel from given module.
 * If you want to invoke one selected function (kernel) from some module
 * (.ptx) this is solution you are looking for. You have to provide module
 * and name of kernel. After that you can invoke kernel as many times as you
 * want from as many threads as you want.
 * @note This class is thread-safe.
 * @warning It is your responsibility to ensure that this object WILL NOT
 * outlive module object.
 */
class Function
{
public:
    const Module& module;
    /**
     * @param module Module in which we look for this function
     * @param name Name of kernel function (must be marked with __global__)
     */
    Function(const Module& module, const char *name);

    /**
     * @param args Kernel parameters as provided in cuLaunchKernel
     */
    void run(void* args[],
             blockIdx_t blockIdx,
             threadIdx_t threadIdx);

private:
    void (*const _run)(void* args[],
                       gridDim_t gridDim,
                       blockDim_t blockDim,
                       blockIdx_t blockIdx,
                       threadIdx_t threadIdx);
};

/**
 * @brief Object of this class gives access to one module (.ptx).
 * If you want general access to given module you should use this class.
 * You have to provide module name (witch .ptx extension). Before invoking
 * kernel you have to do appropriate initialization. You have to invoke
 * initializeModule() once before any other function in this class. You have
 * to invoke initializeBlock() before running (@see Function::run()) kernel
 * for the first time in given block. After last thread in given block finished
 * you have to invoke releaseBlock(). After last block was released you have
 * to invoke cleanupModule(). If you want to invoke mare than one kernel from
 * given Module then you have to serialize all invocations. It means that full
 * initialize...cleanup cycle must be performed before subsequent
 * initializeModule() call.
 *
 * @note This class is reentrant
 * @warning It is your responsibility to ensure that one module (.ptx) is handled
 * by at most one Module object.
 */
class Module
{
    friend class Function;
public:
    /**
     * @param name Name of ptx file (including extension ex. myKernl.ptx)
     */
    Module(const char *name);

    ~Module();

    /**
     * Initialize global module state. Must be invoked exactly one time before
     * kernel
     * @param gridDim gridDim as provided to kernel
     */
    void initializeModule(gridDim_t gridDim, blockDim_t blockDim);

    /**
     * Releases any resources associated witch this module. Must be invoked
     * exactly one time after last thread exit.
     */
    void cleanupModule();

    /**
     * Initialize shared block state. Must be invoked exactly one time before
     * start of any thread in given block. Must be invoked after
     * initializeModule
     * @param gridDim gridDim as provided to kernel
     * @param blockIdx blockIdx of block which is about to launch
     */
    void initializeBlock(blockIdx_t blockIdx);

    /**
     * Releases any resources associated with given block. Must be invoked
     * exactly one time after last thread in given block finished but before
     * cleanupModule()
     * @param gridDim gridDim as provided to kernel
     * @param blockIdx blockIdx of block which is about to finish
     */
    void releaseBlock(blockIdx_t blockIdx);

private:
    void* const _moduleHandle;
    void (*const _kernel_global_init)(gridDim_t gridDim);
    void (*const _kernel_global_deinit)();
    void (*const _kernel_block_init)(gridDim_t gridDim, blockIdx_t blockIdx);
    void (*const _kernel_block_deinit)(gridDim_t gridDim, blockIdx_t blockIdx);

    gridDim_t gridDim;
    blockDim_t blockDim;
};


// ------------ temporary implementation ------------

#include <string>
#include <iostream>
#include <stdexcept>

//#include <dlfcn.h>
int RTLD_LAZY = 1, RTLD_NOW = 1<<1, RTLD_GLOBAL = 1<<2, RTLD_LOCAL = 1<<3;
void  *dlopen(const char *, int){};
void  *dlsym(void *, const char *){};
int    dlclose(void *){};
char  *dlerror(void){};

Module::Module(const char *name)
    : _moduleHandle(dlopen(name, RTLD_LOCAL | RTLD_LAZY)),
      _kernel_global_init(reinterpret_cast<decltype(_kernel_global_init)>(dlsym(_moduleHandle, "_kernel_global_init"))),
      _kernel_global_deinit(reinterpret_cast<decltype(_kernel_global_deinit)>(dlsym(_moduleHandle, "_kernel_global_deinit"))),
      _kernel_block_init(reinterpret_cast<decltype(_kernel_block_init)>(dlsym(_moduleHandle, "_kernel_block_init"))),
      _kernel_block_deinit(reinterpret_cast<decltype(_kernel_block_deinit)>(dlsym(_moduleHandle, "_kernel_block_deinit")))
{
    if (!_moduleHandle || !_kernel_global_init || !_kernel_global_deinit ||
            !_kernel_block_init || !_kernel_block_deinit)
        throw std::runtime_error(dlerror());
}

Module::~Module()
{
    if (dlclose(_moduleHandle))
        std::cerr << dlerror() << '\n';
}

void Module::initializeModule(gridDim_t gridDim, blockDim_t blockDim)
{
    this->gridDim = gridDim;
    this->blockDim = blockDim;
    printf("Initialize Module!!!\n");
    //_kernel_global_init(gridDim);
}

void Module::cleanupModule()
{
    _kernel_global_deinit();
}

void Module::initializeBlock(blockIdx_t blockIdx)
{
    _kernel_block_init(this->gridDim, blockIdx);
}

void Module::releaseBlock(blockIdx_t blockIdx)
{
    _kernel_block_deinit(this->gridDim, blockIdx);
}

Function::Function(const Module& module, const char *name)
    : module(module),
      _run(reinterpret_cast<decltype(_run)>(dlsym(module._moduleHandle, (std::string(name) + "_start").c_str())))
{
    if (!_run)
        throw std::runtime_error(dlerror());
}

std::mutex printf_mutex;           // mutex for critical section
void testKernel(void **args, const dim3 &gridDim, const dim3 &blockDim, const dim3 &blockIdx, const dim3 &threadIdx){
    int tID = (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if(tID & 7) // print every eighth
        return;
    int bID = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + blockIdx.x;
    printf_mutex.lock();
    for(int i=0;i<bID;i++)
        printf("    ");
    printf("%4d\n",tID);
    printf_mutex.unlock();
}

void Function::run(void* args[], blockIdx_t blockIdx, threadIdx_t threadIdx)
{
    //_run(args, module.gridDim, module.blockDim, blockIdx, threadIdx);
    testKernel(args, module.gridDim, module.blockDim, blockIdx, threadIdx);
}

#endif // MODULE_H
