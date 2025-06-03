clear; clc;

%%% Data Inspector for Receiver%%%
f_system = 64e6;


%%% Transmitter %%%
% Create message data
mbuf_data = fi(double('Hi!'), 0, 8, 0);
%mbuf_data = fi(double('Hello World!'), 0, 8, 0);

% Get the message data length
length_mbuf_data = fi(numel(mbuf_data), 0, 8, 0);

% Prepend the message data length to the message
mbuf = [length_mbuf_data, mbuf_data];

% Prepare the barker code for frame synchronisation
barker_13 = fi([0 0 0 0 0 1 1 0 0 1 0 1 0], 0, 1, 0);
bbuf = reshape([barker_13; barker_13; barker_13], [1, length(barker_13)*3]);

% 8-bit Barker and zero padding
bbuf_padded = [zeros(1, 1) bbuf];
tx_data = zeros(1, numel(bbuf_padded)/8);
for i = 1:numel(bbuf_padded)/8
    tx_data(i) = bitconcat(bbuf_padded((i-1)*8 + 8),...
                           bbuf_padded((i-1)*8 + 7),...
                           bbuf_padded((i-1)*8 + 6),...
                           bbuf_padded((i-1)*8 + 5),...
                           bbuf_padded((i-1)*8 + 4),...
                           bbuf_padded((i-1)*8 + 3),...
                           bbuf_padded((i-1)*8 + 2),...
                           bbuf_padded((i-1)*8 + 1));
end

% Append the message data to the Barker code
tx_data = [tx_data mbuf];


%%% Receiver %%%
% Prepare barker code for FIR
fir_barker_13 = fi([1 1 1 1 1 -1 -1 1 1 -1 1 -1 1], 1, 2, 0);
fir_bbuf = reshape([fir_barker_13; fir_barker_13; fir_barker_13], [1, length(fir_barker_13)*3]);

%%% FINN Adapter decimation filters %%%
% Set sample rates
fsample = 1024e6;

% Design the RF-ADC Filters
fir0 = dsp.FIRHalfbandDecimator(...
    'Specification','Coefficients',...
    'Numerator', [-6, 0, 54, 0, -254, 0, 1230, 2048, 1230, 0, -254, 0, 54, 0, -6]/4096, ...
    'SampleRate', fsample);

fir1 = dsp.FIRHalfbandDecimator(...
    'Specification','Coefficients',...
    'Numerator', [-12, 0, 84, 0, -337, 0, 1008, 0, -2693, 0, 10142, 16384, 10142, 0, -2693, 0, 1008, 0, -337, 0, 84, 0, -12]/32768, ...
    'SampleRate', fsample);

fir2 = dsp.FIRHalfbandDecimator(...
    'Specification','Coefficients',...
    'Numerator', [5, 0, -17, 0, 44, 0, -96, 0, 187, 0, -335, 0, 565, 0, -906, 0, 1401, 0, -2112, 0, 3145, 0, -4723, 0, 7415, 0, -13331, 0, 41526, 65536, 41526, 0, -13331, 0, 7415, 0, -4723, 0, 3145, 0, -2112, 0, 1401, 0, -906, 0, 565, 0, -335, 0, 187, 0, -96, 0, 44, 0, -17, 0, 5]/131072, ...
    'SampleRate', fsample);

%fvtool(fir2)
%fvtool(fir2, fir1, fir0, dsp.FilterCascade(fir0, fir1, fir2))
%fvtool(fir2, dsp.FilterCascade(fir2,fir2), dsp.FilterCascade(fir2,fir2,fir2), dsp.FilterCascade(fir2,fir2,fir2,fir2))
