#ifdef __GNUC__
    #if !(__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8))
        #define thread_local __thread
    #endif
#endif

namespace detail
{
struct data3d {
    int x, y, z;
};

struct gridDim_t : public data3d {};

struct blockDim_t : public data3d {};

struct blockIdx_t : public data3d {};

struct threadIdx_t : public data3d {};
}

static thread_local detail::gridDim_t gridDim;
static thread_local detail::blockDim_t blockDim;
static thread_local detail::blockIdx_t blockIdx;
static thread_local detail::threadIdx_t threadIdx;

namespace detail
{
static int grid_flat_size(gridDim_t gridDim)
{
    return gridDim.x * gridDim.y * gridDim.z;
}
static int block_flat_idx(gridDim_t gridDim, blockIdx_t blockIdx)
{
    return gridDim.x * gridDim.y * blockIdx.z +
           gridDim.x * blockIdx.y +
           blockIdx.x;
}
static int block_flat_idx()
{
    return gridDim.x * gridDim.y * blockIdx.z +
           gridDim.x * blockIdx.y +
           blockIdx.x;
}

template<typename T>
struct type_wrapper {
    using type = T;
};
}

#include<cstdio>
#include<stdarg.h>
#include<mutex>
std::mutex printf_mutex; // mutex for synchronized printing

void sync_printf(const char *format, ...){
    va_list arglist;
    va_start(arglist, format);
    printf_mutex.lock();
    vprintf(format, arglist);
    printf_mutex.unlock();
    va_end(arglist);
}

#define printf sync_printf

// Here begins generated code
