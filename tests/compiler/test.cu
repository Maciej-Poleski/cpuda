extern "C"
{

    __device__ int fun1(int* argIntStar, float argFloat)
    {
        __shared__ int tab[123],temp,* ptr,arr[2];
        ptr=&tab[5];
        tab[5]=argIntStar[threadIdx.x];
        argIntStar[6]=*ptr;
        return 10;
    }

    __global__ void fun2(char argCharArray[],int argInt)
    {
        __shared__ char tab[12],a,b,* c;
        __shared__ signed int*q[1] , w[2],*e;
        argCharArray[threadIdx.x]=5;
    }

    __global__ void fun3()
    {}

    __device__ int*fun4(){ return 0;}

}