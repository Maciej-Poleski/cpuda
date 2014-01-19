#include <cstdio>
#include <thread>
#include <mutex>
using namespace std;

const int t_num = 7; // threads number
const int r_num = 2; // repeat
mutex mtx;           // mutex for printing

#ifdef __GNUC__
    #if !(__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8))
        #define thread_local __thread
    #endif
#endif

int glob_num;
thread_local int tloc_num;

void print_nums(int tid){
    mtx.lock();
    printf("thread #%2d --> NUM: %4d, LOCAL: %4d\n",tid,glob_num,tloc_num);
    mtx.unlock();
}

void test_local_var(int num) {
    print_nums(num);
    glob_num = num * num  + 1;
    tloc_num = 2 * num + 3;
    print_nums(num);
}

int main() {
    printf("Test thread local storage!\n");

    glob_num = 1;
    tloc_num = 2;

    printf("BEFORE:\n");
    test_local_var(0);

    printf("THREADS:\n");
    thread t1(test_local_var,10);
    thread t2(test_local_var,20);
    thread t3(test_local_var,30);
    t1.join();
    t2.join();
    t3.join();

    printf("AFTER all:\n");
    test_local_var(0);
    return 0;
}
