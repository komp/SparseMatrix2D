
#ifndef TPKT_H
#define TPKT_H

enum PktState { LOADING, DECODING };

class Tpkt
{
public:
    Tpkt(bundleElt* in, bundleElt* out);
    ~Tpkt();

    Tpkt(Tpkt &&) = default;
    Tpkt(const Tpkt&) = delete;
    Tpkt& operator=(const Tpkt&) = delete;

    PktState state;
    int loadStamp;
    int decodeStamp;
    bundleElt * receivedSigs;
    bundleElt * decodedSigs;
};

#endif
