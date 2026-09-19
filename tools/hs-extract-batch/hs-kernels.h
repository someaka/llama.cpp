// Internal header for hs-extract-batch: masked-mean kernel entry points.
// The kernel lives in the shared tools/hs-extract-common/masked-mean.h so
// the server's skip_mean pooling and this tool accumulate identically;
// this header keeps the tool's historical names for its own TUs.
#pragma once

#include "masked-mean.h"  // tools/hs-extract-common/

#define compute_masked_mean hs_compute_masked_mean
#define compute_single_range_mean hs_compute_single_range_mean
