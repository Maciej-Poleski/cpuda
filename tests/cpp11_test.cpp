#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
using namespace std;

const int t_num = 7; // threads number
const int r_num = 2; // repeat
mutex mtx;           // mutex for critical section

void call_from_thread(int num) {
    for(int i=0;i<r_num;i++){
        mtx.lock();
        printf("#%d --> %d\n",num,i);
        mtx.unlock();
    }
}

int main() {
    printf("Test c++11 threads!\n");
    vector<thread> threads;
    for(int i=0;i<t_num;i++){
        threads.push_back(thread(call_from_thread,i));
    }
    for (auto &t : threads) {
        t.join();
    }
    printf("Hope it all worked!\n");
    return 0;
}
