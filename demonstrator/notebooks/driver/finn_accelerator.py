import numpy as np
import os
import time
from pynq import DefaultIP
from pynq import allocate

from .finn.core.datatype import DataType
from .finn.util.data_packing import packed_bytearray_to_finnpy

class FINN_accelerator():
    def __init__(self, adapter, odma):
        
        super().__init__()

        self.adapter = adapter
        self.odma = odma
        
        self.obuf_packed_device = None
        #self.batch_size = 1

        # dictionary describing the I/O of the FINN-generated accelerator
        self._io_shape_dict = {
        "odt" : [DataType['UINT8']],
        "oshape_normal" : [(1, 1)],
        "oshape_folded" : [(1, 1, 1)],
        "oshape_packed" : [(1, 1, 1)],
        }

        # configure FINN adapter via AXI-Lite
        self.adapter.decimation = 0
        self.adapter.iq_swap = 0

    def odt(self, ind=0):
        return self._io_shape_dict["odt"][ind]

    def oshape_normal(self, ind=0):
        ret = list(self._io_shape_dict["oshape_normal"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    def oshape_folded(self, ind=0):
        ret = list(self._io_shape_dict["oshape_folded"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    def oshape_packed(self, ind=0):
        ret = list(self._io_shape_dict["oshape_packed"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        # free the old buffers by setting to None
        # (reference counting should care of it)
        if self.obuf_packed_device is not None:
            self.obuf_packed_device = None
        cacheable = True # for zynq-iodma platform
        self.obuf_packed_device = []
        self.obuf_packed = []
    
        new_packed_obuf = allocate(
            shape=self.oshape_packed(), dtype=np.uint8, cacheable=cacheable
        )
        self.obuf_packed_device.append(new_packed_obuf)
        self.obuf_packed.append(np.empty_like(new_packed_obuf))

    def unpack_output(self, obuf_packed, ind=0):
        """Unpacks the packed output buffer from accelerator.
        Gets packed output and returns output data in folded shape."""
        obuf_folded = packed_bytearray_to_finnpy(
            obuf_packed,
            self.odt(ind),
            self.oshape_folded(ind),
            reverse_endian=True,
            reverse_inner=True,
            fast_mode=True,
        )
        return obuf_folded

    def unfold_output(self, obuf_folded, ind=0):
        """Unfolds output data to normal shape.
        Gets folded output data and returns output data in normal shape."""
        obuf_normal = obuf_folded.reshape(self.oshape_normal(ind))
        return obuf_normal

    def copy_output_data_from_device(self, data, ind=0):
        """Copies PYNQ output buffer from device."""
        self.obuf_packed_device[ind].invalidate()
        np.copyto(data, self.obuf_packed_device[ind])

    def initiate_read(self, asynch=False, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.batch_size, "Specified batch_size is too large."

        assert self.odma.read(0x00) & 0x4 != 0, "Output DMA is not idle"
        # manually launch IODMAs since signatures are missing
        self.odma.write(0x10, self.obuf_packed_device[0].device_address)
        self.odma.write(0x1C, batch_size)
        self.odma.write(0x00, 1)
        # blocking behavior depends on asynch parameter
        if asynch is False:
            self.wait_until_finished()

    def wait_until_finished(self):
        "Block until all output DMAs have finished writing."
        # check if output IODMA is finished via register reads
        status = self.odma.read(0x00)
        while status & 0x2 == 0:
            status = self.odma.read(0x00)
        
    def read_output(self):
        self.initiate_read()
        self.copy_output_data_from_device(self.obuf_packed[0])
        obuf_folded = self.unpack_output(self.obuf_packed[0])
        obuf_normal = self.unfold_output(obuf_folded)
        return obuf_normal

class FINNadapterCore(DefaultIP):
    """Driver for FINNadapter's core logic IP
    Exposes all the configuration registers by name via data-driven properties
    """
    def __init__(self, description):
        super().__init__(description=description)
        
    bindto = ['User_Company:SysGen:FINNadapter:1.0']
    
# LUT of property addresses for our data-driven properties
_FINNadapter_props = [("decimation", 0),
                          ("iq_swap", 4)]
    
# Function to return a MMIO Getter and Setter based on a relative address
def _create_mmio_property(addr):
    def _get(self):
        return self.read(addr)
        
    def _set(self, value):
        self.write(addr, value)
            
    return property(_get, _set)
    
# Generate getters and setters based on _Transmitter_props
for (name, addr) in _FINNadapter_props:
    setattr(FINNadapterCore, name, _create_mmio_property(addr))
