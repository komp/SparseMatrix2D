/*
 *  Based on code found at the Nax.io website:
 *  https://nax.io/2017/05/21/implementing-a-thread-pool-in-c++
 *  It also includes excellent descriptive text about the implementation.
 */

#ifndef LOADER_POOL_H
#define LOADER_POOL_H

#include <thread>
#include <vector>
#include <functional>
#include <condition_variable>
#include <mutex>

#include "LDPC.h"

class LoaderPool
{
public:
    LoaderPool (unsigned int infoLength, unsigned int numBits, unsigned int numParityBits,
                  unsigned int* W_row_rom, unsigned int numRowsW, unsigned int numColsW, unsigned int shiftRegLength,
                  float sigma2, float lc);
    ~LoaderPool();

    LoaderPool(const LoaderPool&) = delete;
    LoaderPool& operator=(const LoaderPool&) = delete;

    void schedule_job(bundleElt *packet_address);

private:
    void worker_main();

    bool                        P_running;
    mutable std::mutex          P_mutex;
    std::vector<std::thread>    P_threads;

    unsigned int                P_infoLeng;
    unsigned int                P_numBits;
    unsigned int                P_numParityBits;
    unsigned int*               P_W_row_rom;
    unsigned int                P_numRowsW;
    unsigned int                P_numColsW;
    unsigned int                P_shiftRegLength;
    float                       P_sigma2;
    float                       P_lc;

    size_t                      P_job_size;
    std::vector<bundleElt*>     P_job_packet_address;
    std::vector<bundleElt*>     P_job_decode_address;

    std::condition_variable     P_cv_worker;
};

#endif
