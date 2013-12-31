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

// Here begins generated code
