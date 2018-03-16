/*
 *  Based on code found at the Nax.io website:
 *  https://nax.io/2017/05/21/implementing-a-thread-pool-in-c++
 *  It also includes excellent descriptive text about the implementation.
 */

#ifndef DECODER_POOL_H
#define DECODER_POOL_H

#include <thread>
#include <vector>
#include <functional>
#include <condition_variable>
#include <mutex>

#include "LDPC.h"

class DecoderPool
{
public:
    DecoderPool(H_matrix *hmat, unsigned int maxIterations, size_t decoder_count);
    ~DecoderPool();

    DecoderPool(const DecoderPool&) = delete;
    DecoderPool& operator=(const DecoderPool&) = delete;

    void schedule_job(bundleElt *packet_address, bundleElt *decode_address);

private:
    void worker_main();

    bool                        P_running;
    mutable std::mutex          P_mutex;
    std::vector<std::thread>    P_threads;

    H_matrix*                   P_hmat;
    unsigned int                P_maxIterations;

    size_t                      P_job_size;
    std::vector<bundleElt*>     P_job_packet_address;
    std::vector<bundleElt*>     P_job_decode_address;

    std::condition_variable     P_cv_worker;
};

#endif
