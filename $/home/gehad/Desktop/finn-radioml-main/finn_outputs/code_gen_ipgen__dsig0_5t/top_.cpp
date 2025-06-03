
#define AP_INT_MAX_W 2

#include "bnn-library.h"

// includes for network parameters
#include "maxpool.h"

// defines for network parameters
#define ImgDim 128
 #define PoolDim 2

                #define NumChannels 4
 #define PE 1
 #define OutputSize 64
                
 #define numReps 1

void (hls::stream<ap_uint<2> > &in0_V, hls::stream<ap_uint<2> > &out_V)
{
#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE axis port=out_V
#pragma HLS INTERFACE ap_ctrl_none port=return
StreamingMaxPool_Precision_1d<ImgDim, PoolDim, NumChannels, PE,
                     OutputSize, ap_uint<2>, 0>(in0_V, out_V);
}
