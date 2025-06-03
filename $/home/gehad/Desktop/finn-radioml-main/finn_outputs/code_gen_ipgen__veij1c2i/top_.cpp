
#define AP_INT_MAX_W 15

#include "bnn-library.h"

// includes for network parameters
#include "maxpool.h"

// defines for network parameters


void (hls::stream<ap_uint<1*15>> &in0_V,
                hls::stream<ap_uint<1> > &out_V)
{
#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE axis port=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
LabelSelect_Batch<2, 1, 1, ap_int<15>, ap_uint<1> > (in0_V, out_V, 1);
}
