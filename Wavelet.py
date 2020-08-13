# identical to pywt.wavelet class

import numpy as numpy

class Wavelet:
    def __init__(self, name):
        filename = 'Wavelets/' + name + '.txt'
        filter_bank = np.loadtxt(filename)

        if (len(filter_bank) != 4):
            e = "ERROR: Wavelet expects four filter coefficients"
            raise ValueError(e)
        else:
            self.dec_lo = np.asarray(filter_bank[0], dtype=np.float64)
            self.dec_hi = np.asarray(filter_bank[1], dtype=np.float64)
            self.rec_lo = np.asarray(filter_bank[2], dtype=np.float64)
            self.rec_hi = np.asarray(filter_bank[3], dtype=np.float64)

        if (self.dec_lo.ndim != 1) or (self.dec_hi.ndim != 1) or (self.rec_lo != 1) or (self.rec_hi.ndim != 1):
            e = "ERROR: filters must be one dimensional"
            raise ValueError(e)
