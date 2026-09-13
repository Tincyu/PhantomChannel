#include "bt_native/version.hpp"

#ifndef BT_NATIVE_VERSION
#define BT_NATIVE_VERSION "0.0.0"
#endif

namespace bt_native {

const char* version()
{
    return BT_NATIVE_VERSION;
}

bool self_test()
{
    return true;
}

}  // namespace bt_native
