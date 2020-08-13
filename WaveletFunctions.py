import numpy as np
import scipy.fftpack as fft
from scipy import signal
import math
import Wavelet

class WaveletFunctions:
    """
    Class containing wavelet specific functions
    Based on pywavelet and AviaNZ
    A simpler implementation used for denoising only

    Therefore only contains waveletDenoise and associated helpers
    """

    def __init__(self, data, wavelet, maxLevel, samplerate):
        if data is None:
            e = "ERROR: data must be provided"
            raise ValueError(e)
        if wavelet is None:
            e = "ERROR: wavelet must be provided"
            raise ValueError(e)

        self.data = data
        self.maxLevel = maxLevel
        self.tree = None
        self.treefs = samplerate

        self.wavelet = Wavelet.Wavelet(name=wavelet)

    def ShannonEntropy(self, s):
        entropy = s[np.nonzero(s)]**2 * np.log(s[np.nonzero(s)]**2)
        return np.sum(entropy)

    def BestLevel(self, maxLevel=None):
        """
        Compute best level for WP decomposition using the shannon entropy using iterative methods
        """

        if maxLevel is None:
            maxLevel = self.maxLevel

        allnodes = range(2 ** (maxlevel + 1) - 1)

        previouslevelmaxE = self.ShannonEntropy(self.data)
        self.WaveletPacket(allnodes, 'symmetric', aaWP=False, antialiasFilter=True)

        level = 1

        currentlevelmaxE = np.max([self.ShannonEntropy(self.tree[n]) for n in range(1,3)])
        
        while currentlevelmaxE < previouslevelmaxE and level < maxlevel:
            previouslevelmaxE = currentlevelmaxE
            level += 1
            currentlevelmaxE = np.max([self.ShannonEntropy(self.tree[n]) for n in range(2**level-1, 2**(level+1)-1)])
        
        return level
    
    def reconstructWP(self, node, antialias=False, antialiasFilter=False):
        """
        Reconstruct the signal from a wavelet packet decomposition
        Makes use of wavelet and data from current instance

        Returns: reconstructed signal
        """

        wv = self.wavelet
        data = self.tree[node]
        
        lvl = math.floor(math.log2(node + 1))
        # position of node in its level
        nodepos = node - (2**lvl - 1)
        # gray coded as wp not in natural order
        # nodepos = self.graycode(nodepos)
        # number of nodes
        numnodes = 2**(lvl+1)

        # reconstruction wavlets and lengths
        wv_rec_hi = wv.rec_hi
        wv_rec_lo = wv.rec_lo

        wv_hi_len = len(wv_rec_hi)
        wv_lo_len = len(wv_rec_lo)

        # perform convolutions to get signal and upsample
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype='float64')

        if node % 2 == 0:
            data = np.convolve(data, wv_rec_hi, 'same')
        else:
            data = np.convolve(data, wv_rec_lo, 'same')

        data = data[wv_hi_len//2-1 : -(wv_lo_len//2-1)]

        return data
        

    def WaveletPacket(self, nodes, mode='symmetric', antialias=False, antialiasFilter=True):
        """
        Reimplementation of pwwt.WaveletPacket
        """
        
        if (len(nodes) == 0) or not isinstance(nodes[0], int):
            print("ERROR: must provide a list of node IDs")
            return
        
        # find max level
        maxlevel = math.floor(math.log2(max(nodes)+1))
        if (maxlevel > 10):
            print("ERROR: got level above 10, probably nodes badly specified")
            return

        # determine nodes that need to be produced
        nodes = list(nodes)
        for child in nodes:
            parent = (child - 1) // 2
            if parent not in nodes and (parent >= 0):
                nodes.append(parent)
        nodes.sort()

        # wavelet object
        wavelet = self.wavelet
        flen = max(len(wavelet.dec_lo), len(wavelet.dec_hi), len(wavelet.rec_lo), len(wavelet.rec_hi))

        # tree to hold downsampled coefficients
        self.tree = [self.data]
        if mode != 'symmetric':
            print("ERROR: only symmetric implemented")
            return

        # anti-aliasing filter
        if antialiasFilter:    
            low = 0.5
            hb, ha = signal.butter(20, low, btype='highpass')
            lb,la = signal.butter(20, low, btype='lowpass')

        # loop over possible parent nodes and construnct WPD
        for node in range(2**maxlevel-1):
            childa = node*2 + 1
            childd = node*2 + 2
            # if not relevant, set children to empty to keep tree structure
            if childa not in nodes and childd not in nodes:
                self.tree.append(np.array([]))
                self.tree.append(np.array([]))
                continue

            # retrieve parent from Jth level
            data = self.tree[node]
            # downsample all non-root nodes
            if node != 0:
                data = data[0::2]

            # symmetric mode
            data = np.concatenate((data[0:flen:-1], data, data[-flen:]))

            ll = len(data)

            # make A_(j+1) and D_(j+1)
            # create the A child
            if childa in nodes:
                nexta = np.convolve(data, wavelet.dec_lo, 'same')[1:-1]
                # antialias A
                if antialias:
                    if antialiasFilter:
                        # bandpass filtering method
                        nexta = signal.lfilter(lb, la, nexta)
                    else:
                        # take fft and set coefficients to 0
                        ft = fft.fft(nexta)
                        ft[ll//4 : 3*ll//4] = 0
                        nexta = np.real(fft.ifft(ft))         
               # store A
                self.tree.append(nexta)
            else:
                self.tree.append(np.array([]))

            # create the D child
            if childd in nodes:
                nextd = np.convolve(data, wavelet.dec_hi, 'same')[1:-1]
                # antialias A
                if antialias:
                    if antialiasFilter:
                        # bandpass filtering method
                        nextd = signal.lfilter(hb, ha, nextd)
                    else:
                        # take fft and set coefficients to 0
                        ft = fft.fft(nextd)
                        ft[ll//4 : 3*ll//4] = 0
                        nextd = np.real(fft.ifft(ft))
               # store A
                self.tree.append(nextd)
            else:
                self.tree.append(np.array([]))

    def BestTree(self, wp, threshold, costfn='threshold'):
        """
        Compute best wavelet tree using the given cost function
        Assigns a score to each node which is used to identify leaves to keep

        Returns: list of new tree leaves
        """
        nnodes = 2 ** (wp.maxLevel + 1) - 1
        cost = np.zeors(nnodes)
        count = 0

        for level in range(wp.maxLevel + 1):
            for n in wp.get_level(level, 'natural'):
                d = np.abs(n.data)
                cost[count] = np.sum(d > threshold)
                count += 1

        # compute best tree from these costs
        flags = 2 * np.ones(nnodes)
        flags[2 ** wp.maxLevel - 1:] = 1
        inds = np.arange(2 ** wp.maxLevel - 1)
        inds = inds[-1::-1]

        for i in inds:
            # get children
            children = (i + 1) * 2 + np.arange(2) - 1
            c = cost[children[0]] + cost[children[1]]
            if c < cost[i]:
                cost[i] = c
                flags[i] = 2
            else:
                flags[i] = flags[children[0]] + 2
                flags[children] = -flags[children]

        newleaves = np.where(flags > 2)[0]

        # function to make list of children, recursively
        def getchildren(n):
            level = int(np.floor(np.log2(n+1)))
            if level < wp.maxlevel:
                tbd.append((n + 1) * 2 - 1)
                tbd.append((n + 1) * 2)
                getchildren((n + 1) * 2 - 1)
                getchildren((n + 1) * 2)

        tbd = []
        for i in newleaves:
            getchildren(i)

        tbd = np.unique(tbd)

        # delete other nodes
        listnodes = np.arange(2 ** (wp.maxlevel + 1) - 1)
        listnodes = np.delete(listnodes, tbd)
        notleaves = np.intersect1d(newleaves, tbd)
        for i in notleaves:
            newleaves = np.delete(newleaves, np.where(newleaves == i))

        listleaves = np.intersect1d(np.arange(2 ** (wp.maxlevel) - 1, 2 ** (wp.maxlevel + 1) -1), listnodes)
        listleaves = np.unique(np.concatenate((listleaves, newleaves)))

        return listleaves

    def waveletDenoise(self, thresholdType='soft', threshold=4.5, maxLevel=5, bandpass=False, costfn='threshold', aarec=False, aaWP=False, thrfun='c'):
        """
        Wavelet Denoising
        Constructs a wavelet packet tree to specified depth, finds the best tree.
        Thresholds the coefficients then reconstructs the wavelet the data from the resulting tree.

        Data and wavelet taken from classes self

        Returns: Reconstructed signal
        """

        if maxLevel == 0:
            self.maxLevel = self.BestLevel()
            print("Best Level is %d".format(self.maxLevel))
        else:
            self.maxLevel = maxLevel

        self.thresholdMultiplier = threshold

        # create wavelet decomposition
        allnodes = range(2 ** (self.maxLevel + 1) - 1)
        wp = self.WaveletPacket(allnodes, 'symmetric', aaWP, antialiasFilter=True)

        # get the threshold
        det1 = self.tree[2]
        # magic number here, why(?)
        sigma = np.median(np.abs(det1)) / 0.6745
        threshold = self.thresholdMultiplier * sigma

        # NOTE: Node order in best tree not the same
        bestleaves = self.BestTree(wp, threshold)
        print("Keeping leaves: ", bestleaves)

        # thresholding
        exit_code = self.ThresholdNodes(self, self.tree, bestleaves, thresholdType)

        if exit_code != 0:
            print("ERROR: error thresholding nodes")
            return
        
        data = self.tree[0]
        reconstructed = np.zeros(len(data))
        for i in bestleaves:
            tmp = self.reconstructWP(i, aaRec, True)[0:len(data)]
            reconstructed = reconstructed + tmp

        return newWP
