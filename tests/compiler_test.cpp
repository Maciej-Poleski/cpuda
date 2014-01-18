#include <cstdio>

#ifdef _WIN32
    #define IS_WIN32 "TRUE"
#else
    #define IS_WIN32 "FALSE"
#endif

#ifdef __GNUC__
    #define IS_GNUC "TRUE"
    #if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
        #define VER_48 "TRUE"
    #endif
    #define VER_48 "FALSE"
#else
    #define IS_GNUC "FALSE"
#endif

int main(){
    printf("IS WIN32? -> %s\n",IS_WIN32);
    printf("IS GNUC? -> %s\n",IS_GNUC);
#ifdef __GNUC__
    printf("\tGNUC version: %s\n",__VERSION__);
    printf("\tGNUC version >= 4.8? -> %s\n",VER_48);
#endif
    return 0;
}
