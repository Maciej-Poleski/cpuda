#ifndef DIM3_H
#define DIM3_H

namespace cpuda{

typedef unsigned int dim_t;

struct dim3{
    dim_t x, y, z;
};

dim3 doDim3(dim_t x, dim_t y, dim_t z){
    dim3 d;
    d.x = x;
    d.y = y;
    d.z = z;
    return d;
}

dim_t totalSize(const dim3 &dim) { // unsigned long long ?
    return dim.x * dim.y * dim.z;
}

bool checkDimensionsInRange(const dim3 &dim, const dim3 &range){
    return (dim.x <= range.x) && (dim.y <= range.y) && (dim.z <= range.z);
}

bool checkTotalSizeInLimit(const dim3 &dim, const dim_t &limit){
    unsigned long long bound = limit;
    unsigned long long xy = dim.x; xy *= dim.y; // ULL in order to hold product of two UI
    return (xy <= bound) && (xy*dim.z <= bound); // two tests: x*y*z can overlap ULL
}

}

#endif // DIM3_H
