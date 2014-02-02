#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
using namespace std;

// --- synchronized printing ---
#include<stdarg.h>
mutex printf_mutex;
void sync_printf(const char *format, ...){
    va_list arglist;
    va_start(arglist, format);
    printf_mutex.lock();
    vprintf(format, arglist);
    printf_mutex.unlock();
    va_end(arglist);
}
#define printf sync_printf
// -----------------------------

const int t_num = 5; // threads number
__thread int num;

int waiting = 0;
mutex sync_mutex;
condition_variable sync_v;

void sync_threads(){
    printf("-\t#%d\t-\t-\n", num);
    unique_lock<mutex> lock(sync_mutex);
    if(++waiting == t_num){
        waiting = 0;
        sync_v.notify_all();
        printf("----------- SYNCED -----------\n");
    }
    else{
        sync_v.wait(lock);
    }
    lock.unlock();
    printf("-\t-\t#%d\t-\n", num);
}

void call_from_thread(int id) {
    num = id;
    printf("#%d\t-\t-\t-\n", num);
    sync_threads();
    sync_threads();
    printf("-\t-\t-\t#%d\n", num);
}

/**
 * Test thread synchronizing
 */
int main() {
    printf("%-8s%-8s%-8s%-8s\n","START","SYNC","SYNCED","END");
    vector<thread> threads;
    for(int i=0;i<t_num;i++){
        threads.push_back(thread(call_from_thread,i));
    }
    for (auto &t : threads) {
        t.join();
    }
    return 0;
}
