#Using 2 RPCs (train, decode) that gets called
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA
from collections import deque
from enum import Enum
import numpy as np
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.utils.buffers import DataBuffer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics
from physiolabxr.rpc.decorator import rpc, async_rpc
import time
import random

class CVEPdummyScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    def init(self):
        return
    def loop(self):
        return                #adding the detected choice to the list of detected choices
    def cleanup(self):
        return

    @async_rpc
    def trainingModel(self) -> int:
        time.sleep(1)
        return 1

    @async_rpc
    def addSeqData(self, sequenceNum: int, Duration: float):  # Data is going to come in sequencially seq1 -> seq2 -> seq3 repeat
        time.sleep(Duration)
        print(sequenceNum)

    @async_rpc
    def decodeChoice(self) -> int:
        return random.randint(1,3)


