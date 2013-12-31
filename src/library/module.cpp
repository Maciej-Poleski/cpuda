#include "module.h"

#include <iostream>
#include <stdexcept>

#include <dlfcn.h>

Module::Module(const std::string& name)
    : _moduleHandle(dlopen(name.c_str(), RTLD_LOCAL | RTLD_LAZY)),
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

void Module::initializeModule(gridDim_t gridDim)
{
    _kernel_global_init(gridDim);
}

void Module::cleanupModule()
{
    _kernel_global_deinit();
}

void Module::initializeBlock(gridDim_t gridDim, blockIdx_t blockIdx)
{
    _kernel_block_init(gridDim, blockIdx);
}

void Module::releaseBlock(gridDim_t gridDim, blockIdx_t blockIdx)
{
    _kernel_block_deinit(gridDim, blockIdx);
}

Function::Function(const Module& module, const std::string& name)
    : _run(reinterpret_cast<decltype(_run)>(dlsym(module._moduleHandle, (name + "_start").c_str())))
{
    if (!_run)
        throw std::runtime_error(dlerror());
}

void Function::run(void* args[], gridDim_t gridDim, blockDim_t blockDim, blockIdx_t blockIdx, threadIdx_t threadIdx)
{
    _run(args, gridDim, blockDim, blockIdx, threadIdx);
}
