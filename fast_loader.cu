#include "fast_loader.h"

#include <random>
#include <thread>
#include <vector>
#include <functional>
#include <condition_variable>
#include <mutex>

#include "LDPC.h"

FastLoader::FastLoader (bundleElt* preloads, int nBundles, int numBits)
: P_running(true)
, P_job_size(0u)
, P_nBundles(nBundles)
, P_numBits(numBits)
, P_preloads(preloads)
, P_currentBundle(0)

{
  // Just one thread for this fast loader.
  P_threads.reserve(1);
  P_threads.emplace_back(&FastLoader::worker_main, this);
}

FastLoader::~FastLoader() {
  std::unique_lock<std::mutex> lock(P_mutex);
  P_running = false;
  P_cv_worker.notify_all();
  lock.unlock();
  for (auto& t : P_threads) t.join();
}

void FastLoader::schedule_job(Tpkt *packet) {
  std::lock_guard<std::mutex> lock(P_mutex);

  P_job_packet.push_back(packet);
  P_job_size++;
  P_cv_worker.notify_one();
}

void FastLoader::worker_main() {

  std::unique_lock<std::mutex> lock(P_mutex);

  int index;

  while (P_running) {
    P_cv_worker.wait(lock);
    for (;;) {
      if (P_job_size == 0) break;

      Tpkt *packet = P_job_packet.back();
      P_job_packet.pop_back();
      P_job_size--;

      index = P_currentBundle * P_numBits;
      P_currentBundle++;
      if (P_currentBundle >= P_nBundles) P_currentBundle = 0;

      lock.unlock();
      packet->receivedSigs = &P_preloads[index];
      lock.lock();
      packet->loadStamp = 1.0;
    }
  }
}
