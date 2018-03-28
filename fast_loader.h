/*
 *  Based on code found at the Nax.io website:
 *  https://nax.io/2017/05/21/implementing-a-thread-pool-in-c++
 *  It also includes excellent descriptive text about the implementation.
 */

#ifndef FAST_LOADER_H
#define FAST_LOADER_H

#include <thread>
#include <vector>
#include <functional>
#include <condition_variable>
#include <mutex>

#include "LDPC.h"

class FastLoader
{
public:
    FastLoader (bundleElt* preloads, int nBundles, int numBits);
    ~FastLoader();

    FastLoader(const FastLoader&) = delete;
    FastLoader& operator=(const FastLoader&) = delete;

    void schedule_job(Tpkt *packet);

private:
    void worker_main();

    bool                        P_running;
    mutable std::mutex          P_mutex;
    std::vector<std::thread>    P_threads;

    int                         P_nBundles;
    int                         P_numBits;
    int                         P_currentBundle;
    bundleElt*                  P_preloads;
    size_t                      P_job_size;
    std::vector<Tpkt*>          P_job_packet;

    std::condition_variable     P_cv_worker;
};

#endif
