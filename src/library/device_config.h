/*
 * This file contains CpUDA device configuration. Parameters imitate these from
 * real CUDA but because of running on CPU architecture has different meaning.
 */

#ifndef DEVICE_CONFIG
#define DEVICE_CONFIG

#include "dim3.h"

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
 * Minimal number of blocks to run simultaneously. This means that the total number of
 * threads used per kernel would be NUMBER_OF_MULTIPROCESSORS * MAX_THREADS_PER_BLOCK.
 * If block size would be smaller than MAX_THREADS_PER_BLOCK then possibly more blocks
 * would be run simultaneously.
 */
const unsigned int NUMBER_OF_MULTIPROCESSORS = 4;

#endif // DEVICE_CONFIG
