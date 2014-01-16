#ifndef DIM3_H
#define DIM3_H

typedef unsigned int dim_t;

struct dim3{
    dim_t x, y, z;
    dim3():                     x(0), y(0), z(0){}
    dim3(dim_t x, dim_t y, dim_t z):  x(x), y(y), z(z){}
    dim_t totalSize() const { // unsigned long long ?
        return x * y * z;
    }
};

bool checkDimensionsInRange(const dim3 &dim, const dim3 &range){
    return (dim.x <= range.x) && (dim.y <= range.y) && (dim.z <= range.z);
}

bool checkTotalSizeInLimit(const dim3 &dim, const dim_t &limit){
    unsigned long long bound = limit;
    unsigned long long xy = dim.x; xy *= dim.y; // ULL in order to hold product of two UI
    return (xy <= bound) && (xy*dim.z <= bound); // two tests: x*y*z can overlap ULL
}

#endif // DIM3_H
