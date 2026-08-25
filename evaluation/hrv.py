# """

# @author: Anonymous 
# """

# import numpy as np
# from pyampd.ampd import find_peaks

# def ibi_calculate(signal, fps):
#     """
#     Calculate Inter-Beat Intervals (IBI) from a physiological signal.

#     Parameters:
#     -----------
#     signal : array_like
#         Input physiological signal (e.g., PPG waveform, ECG signal)
#     fps : float
#         Sampling frequency (frames per second) of the signal
        
#     Returns:
#     --------
#     ibi : numpy.ndarray
#         Array of inter-beat intervals in milliseconds
        
#     """
    
#     # Detect peak locations in the signal
#     pk_lc = find_peaks(signal)
    
#     # Calculate differences between consecutive peak locations
#     ibi = np.diff(pk_lc)
    
#     # Convert sample intervals to milliseconds
#     ibi = ibi * 1000 / fps
#     return ibi
    


# class HRV:
#     def __init__(self, signal, fps):
        
#         ibi = ibi_calculate(signal, fps)

#         if ibi[-1] > 10:  # ibi should be in ms
#             self.RR = ibi
#             self.time = np.array([np.sum(ibi[:i])
#                                  for i in range(1, len(ibi)+1)])/1000
#         else:
#             self.RR = ibi * 1000
#             self.time = np.array([np.sum(ibi[:i])
#                                  for i in range(1, len(ibi)+1)])

#         self.SD = np.diff(self.RR)

#     def sdnn(self):
#         return np.std(self.RR)

#     def rmssd(self):
#         rmssd = np.sqrt(np.mean(self.SD**2))
#         return rmssd
    


"""

@author: Anonymous 
"""

import numpy as np
from pyampd.ampd import find_peaks

def ibi_calculate(signal, fps):
    """
    Calculate Inter-Beat Intervals (IBI) from a physiological signal.

    Parameters:
    -----------
    signal : array_like
        Input physiological signal (e.g., PPG waveform, ECG signal)
    fps : float
        Sampling frequency (frames per second) of the signal
        
    Returns:
    --------
    ibi : numpy.ndarray
        Array of inter-beat intervals in milliseconds
        
    """
    
    # Detect peak locations in the signal
    pk_lc = find_peaks(signal)
    
    # Calculate differences between consecutive peak locations
    ibi = np.diff(pk_lc)
    
    # Convert sample intervals to milliseconds
    ibi = ibi * 1000 / fps
    return ibi
    


class HRV:
    def __init__(self, signal, fps):
        
        ibi = ibi_calculate(signal, fps)

        if ibi[-1] > 10:  # ibi should be in ms
            self.RR = ibi
            self.time = np.array([np.sum(ibi[:i])
                                 for i in range(1, len(ibi)+1)])/1000
        else:
            self.RR = ibi * 1000
            self.time = np.array([np.sum(ibi[:i])
                                 for i in range(1, len(ibi)+1)])

        self.SD = np.diff(self.RR)

    def sdnn(self):
        return np.std(self.RR)

    def rmssd(self):
        rmssd = np.sqrt(np.mean(self.SD**2))
        return rmssd
    
    