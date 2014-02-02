/*
 * This file contains CpUDA device configuration. Parameters imitate these from
 * real CUDA but because of running on CPU architecture has different meaning.
 */

#ifndef DEVICE_CONFIG
#define DEVICE_CONFIG

#include "dim3.h"
using namespace cpuda;

/**
 * How many devices imitate on CPU.
 */
const unsigned int NUMBER_OF_DEVICES = 1;

/**
 * Maximal values of grid dimensions.
 */
const dim3 MAX_GRID_DIMENSIONS = doDim3(65535,65535,65535);

/**
 * Maximal values of block dimensions.
 */
const dim3 MAX_BLOCK_DIMENSIONS = doDim3(1024,1024,64);

/**
 * Maximal size of the grid (total number of blocks).
 */
const unsigned int MAX_NUMBER_OF_BLOCKS = 65535;

/**
 * Maximal size of a block (total threads in one block).
 */
const unsigned int MAX_THREADS_PER_BLOCK = 1024;

/**
 * Number of blocks to run simultaneously. This means that the total number of thread
 * objects used per kernel would be NUMBER_OF_MULTIPROCESSORS * block size (so the
 * maximal possible value is NUMBER_OF_MULTIPROCESSORS * MAX_THREADS_PER_BLOCK).
 */
const unsigned int NUMBER_OF_MULTIPROCESSORS = 1;

/**
 * Number of threads running simultaneously inside one block. Each block creates as
 * many thread objects as its size but between synchronization points only WARP_SIZE
 * threads are running. This means that total number of threads concurrently executing
 * kernel code would be NUMBER_OF_MULTIPROCESSORS * WARP_SIZE.
 */
const unsigned int WARP_SIZE = 16;

#endif // DEVICE_CONFIG
