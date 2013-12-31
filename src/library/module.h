#ifndef MODULE_H
#define MODULE_H

#include <string>

struct data3d {
    int x, y, z;
};

struct gridDim_t : public data3d {};

struct blockDim_t : public data3d {};

struct blockIdx_t : public data3d {};

struct threadIdx_t : public data3d {};

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
    /**
     * @param module Module in which we look for this function
     * @param name Name of kernel function (must be marked with __global__)
     */
    Function(const Module& module, const std::string& name);

    /**
     * @param args Kernel parameters as provided in cuLaunchKernel
     */
    void run(void* args[],
             gridDim_t gridDim,
             blockDim_t blockDim,
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
    friend Function::Function(const Module& module, const std::string& name);
public:
    /**
     * @param name Name of ptx file (including extension ex. myKernl.ptx)
     */
    Module(const std::string& name);

    ~Module();

    /**
     * Initialize global module state. Must be invoked exactly one time before
     * kernel
     * @param gridDim gridDim as provided to kernel
     */
    void initializeModule(gridDim_t gridDim);

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
    void initializeBlock(gridDim_t gridDim, blockIdx_t blockIdx);

    /**
     * Releases any resources associated with given block. Must be invoked
     * exactly one time after last thread in given block finished but before
     * cleanupModule()
     * @param gridDim gridDim as provided to kernel
     * @param blockIdx blockIdx of block which is about to finish
     */
    void releaseBlock(gridDim_t gridDim, blockIdx_t blockIdx);

private:
    void* const _moduleHandle;
    void (*const _kernel_global_init)(gridDim_t gridDim);
    void (*const _kernel_global_deinit)();
    void (*const _kernel_block_init)(gridDim_t gridDim, blockIdx_t blockIdx);
    void (*const _kernel_block_deinit)(gridDim_t gridDim, blockIdx_t blockIdx);
};


#endif // MODULE_H
