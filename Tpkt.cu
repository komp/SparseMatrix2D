#include "bundleElt.h"
#include "Tpkt.h"

Tpkt::Tpkt (bundleElt* in, bundleElt* out)
: loadStamp(0)
, decodeStamp(0)
, receivedSigs(in)
, decodedSigs(out)
    , state(LOADING)
{
}

Tpkt::~Tpkt () {
}
