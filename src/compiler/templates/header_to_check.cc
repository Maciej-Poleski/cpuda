#include<cstdio> // CUDA enables printf
namespace detail
{
    struct data3d{
        int x, y, z;
    };
}
static detail::data3d gridDim, blockDim, blockIdx, threadIdx;
void __syncthreads();
#define __device__
#define __global__
#define __shared__
#line 0
// -------------------- KERNEL CODE --------------------
