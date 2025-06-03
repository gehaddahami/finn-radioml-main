
#define AP_INT_MAX_W 15

#include "bnn-library.h"

// includes for network parameters
#include "activations.hpp"
#include "params.h"

// defines for network parameters
#define NumChannels1 2
#define PE1 1
#define numReps 1

void (hls::stream<ap_uint<14>> &in0_V,
                hls::stream<ap_uint<15>> &out_V
                )
{
#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE axis port=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=1
#pragma HLS RESOURCE variable=threshs.parameters core=ROM_2P_LUTRAM
Thresholding_Batch<1, NumChannels1, PE1, Slice<ap_int<14>>, Slice<ap_int<15>>>
            (in0_V, out_V, threshs, numReps);
}
