# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.backends.base import ComputeKernel

class OpenMPOversetBridge(OpenMPKernelProvider):
    
    # get a kernel to write/read data at specific location?
    def read_ram(self):
        pass

    def write_ram(self):
        pass
    
    # get a kernel to return the address?
    def addrptr(self):
        pass
    
