#include <thread>
#include <mutex>
#include <condition_variable>
using namespace std;

// --- synchronized printing ---
#include<stdarg.h>
mutex printf_mutex;
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

int THREAD_NUM;     // threads number
int THROAT_SIZE;    // maximal concurrent running threads
int SYNC_NUM;       // number of synchronizations

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
    void init(int threads_num){
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

// --------------------------------

throat_sync th;
__thread int num; // thread id (thread local variable)

void sync_threads(){
    printf("%-8s%-8d%-8s%-8s\n", "-", num, "-", "-");
    th.sync();
    printf("%-8s%-8s%-8d%-8s\n", "-", "-", num, "-");
}

void run_thread(int id) {
    num = id;
    th.start();
    printf("%-8d%-8s%-8s%-8s\n", num, "-", "-", "-");
    for(int i=0;i<SYNC_NUM; i++){
        sync_threads();
    }
    printf("%-8s%-8s%-8s%-8d\n", "-", "-", "-", num);
    th.end();
}

// -----------------------------------

#include <stdlib.h>     /* atoi */

/**
 * Test thread throat synchronizing.
 */
int main(int argc, char* argv[]) {
    if(argc != 4){
        printf("specify input arguments: THREAD_NUM THROAT_SIZE SYNC_NUM");
        return 0;
    }
    THREAD_NUM = atoi(argv[1]);
    THROAT_SIZE = atoi(argv[2]);
    SYNC_NUM = atoi(argv[3]);
    printf("GIVEN PARAMETERS:\n");
    printf("%s = %d\n","THREAD_NUM", THREAD_NUM);
    printf("%s = %d\n","THROAT_SIZE", THROAT_SIZE);
    printf("%s = %d\n","SYNC_NUM", SYNC_NUM);
    printf("\n");
    printf("%-8s%-8s%-8s%-8s\n","START","SYNC","SYNCED","END");
    printf("\n");
    th.init(THREAD_NUM);

    thread threads[THREAD_NUM];
    for(int i=0;i<THREAD_NUM;i++){
        threads[i] = thread(run_thread,i);
    }
    for(int i=0;i<THREAD_NUM;i++){
        threads[i].join();
    }
    return 0;
}
