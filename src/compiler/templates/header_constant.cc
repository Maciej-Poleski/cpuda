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

    typedef data3d gridDim_t;
    typedef data3d blockDim_t;
    typedef data3d blockIdx_t;
    typedef data3d threadIdx_t;
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
        return gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    }
    static int block_flat_idx()
    {
        return gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    }

    template<typename T>
    struct type_wrapper {
        using type = T;
    };
}

#include<mutex>
#include<condition_variable>
namespace detail
{
    int THROAT_SIZE;
    int BLOCK_SIZE;

    class throat_sync{
        int live;
        int in_throat;
        int waiting;

        std::mutex sync_mutex;
        std::condition_variable throat_v;
        std::condition_variable sync_v;

        void enter_throat(std::unique_lock<std::mutex> &lock){
            if(in_throat == THROAT_SIZE){ // apply concurrent limit
                throat_v.wait(lock); // wait to enter the throat
            }
            ++in_throat;
        }

        void leave_throat(){
            --in_throat;
            throat_v.notify_one(); // one thread from previous stop can run
        }

        bool synced(){
            if(waiting == live && live > 0){
                waiting = 0;
                sync_v.notify_all(); // run them further!
                return true;
            }
            return false;
        }

    public:
        /**
         * Initialize throat synchronization, specifying number of threads which will run under the throat.
         */
        throat_sync(int threads_num){
            live = threads_num;
            in_throat = 0;
            waiting = 0;
        }
        /**
         * Start running of the calling thread under the throat.
         */
        void start(){
            std::unique_lock<std::mutex> lock(sync_mutex);
            enter_throat(lock); // simply enter the throat
        }
        /**
         * Synchronize calling thread with all threads still running under the throat.
         */
        void sync(){
            std::unique_lock<std::mutex> lock(sync_mutex);
            ++waiting;
            if(!synced()){      // check whether all threads are in point of synchronization
                leave_throat(); // if not then leave throat and wait for other threads
                sync_v.wait(lock);
                enter_throat(lock); // after wake up enter the throat
            }
            // if yes than synced() notify other threads and this thread can run without leaving throat
        }
        /**
         * Report that calling thread finishes running under the throat.
         */
        void end(){
            std::unique_lock<std::mutex> lock(sync_mutex);
            leave_throat();
            --live;     // decrease number of threads running under the throat
            synced();   // remaining threads could be waiting for this thread so check synchronization
        }
    };

    throat_sync **blocks_synchronization;
}

void __syncthreads(){
    detail::blocks_synchronization[detail::block_flat_idx(gridDim, blockIdx)]->sync();
}

// --- synchronized printing ---
#include<cstdio>
#include<stdarg.h>
std::mutex printf_mutex;
void sync_printf(const char *format, ...){
    printf_mutex.lock();
    va_list arglist;
    va_start(arglist, format);
    vprintf(format, arglist);
    va_end(arglist);
    printf_mutex.unlock();
}
#define printf sync_printf
// -----------------------------

// Here begins generated code
