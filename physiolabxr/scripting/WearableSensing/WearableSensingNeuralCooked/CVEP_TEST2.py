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


#This is a test file to test out different approaches to the game
#We will be using a database to store data for training
class NeuralCooked(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    def init(self):
        self.mSequence = []
        self.frequency = 300
        self.freq_bands = [(8, 60), (12, 60), (30, 60)]
        self.data = DataBuffer()
        self.training_EEG_label = np.array([])
        self.cca_models = None
        self.decoded_choices = []

    def loop(self):
        #Grabbing p3, p4, o1 and o1 data and putting it into another data buffer
        EEG_Data = {
            'stream_name': 'EEG Data',
            'frames': self.inputs['DSI-24'][0][14:18, :],
            'timestamps': self.inputs['DSI-24'][1]
        }
        self.data.update_buffer(EEG_Data)
        #Removes the data if its longer than 20 seconds
        if self.data.get_data('EEG Data').shape[1] > 60000:
            self.data.clear_stream_up_to_index(self, stream_name= 'EEG Data', cut_to_index= len(self.data.get_data('EEG Data').shape[1]))
        #if training is done(can be seen if cca_models is not none) then we can start playing
        if self.cca_models != None:
            # Get the EEG Data and split it into band channels
            band_data = self.apply_filter_banks(self.data.get_data('EEG Data'))
            # Apply shifting window CCA to each band channel
            self.correlation_coefficients = self.apply_shifting_window_cca(band_data)
            # Look at correlation coefficient averages for each band channel
            # Look at who has the highest coefficient average
            highest_correlation, detected_choice = self.evaluate_correlation_coefficients(self.correlation_coefficients)
            self.decoded_choices.append[detected_choice]

    #Training
    def train_cca(self, m_sequences):
        self.mSequence = m_sequences
        # Split the data via band channels using filter banks
        band_data = self.apply_filter_banks(self.data.get_data('EEG Data'))
        # Train a shifting CCA for each band channel
        EEG_split =[]
        #Need to split the training data into 3 different sequences
        #Training has to be around 200s or 3 minutes and 20 seconds can be increased if needed
        EEG_split[0] = f'0:20000'
        EEG_split[1] = f'20000:40000'
        EEG_split[2] = f'40000:60000'
        cca_model = {}
        for seq in range(3):
            for band in self.freq_bands:
                band_key = f'band_{band[0]}_{band[1]}'
                cca_model[band_key] = self.shifting_cca(band_data[band_key][EEG_split[seq]],m_sequences[seq])
            self.cca_models[seq] = cca_model

    def apply_filter_banks(self, data):
        band_data = {}
        for band in self.freq_bands:
            band_key = f'band_{band[0]}_{band[1]}'
            band_data[band_key] = self.bandpass_filter(data, band[0], band[1])
        return band_data

    def bandpass_filter(data, lowcut, highcut, fs, order=8):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data= filtfilt(b, a, data)
        return filtered_data

    def shifting_cca(self, band_data, m_sequence_number):
        # Implement CCA training for shifting window
        window_size = 300  # Define window size (e.g., 1 second for a 300 Hz signal)
        step_size = 150  # Define step size (e.g., 0.5 seconds for a 300 Hz signal)
        num_windows = (band_data.shape[1] - window_size) // step_size + 1
        cca_model = []
        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            window_data = band_data[:, start:end]
            cca = CCA(n_components=1)
            cca.fit(window_data.T, self.mSequence[m_sequence_number])
            cca_model.append(cca)
        return cca_model

    #Playing
    def apply_shifting_window_cca(self, band_data):
        correlation_coefficients = {1: [], 2: [], 3: []}
        for band in self.freq_bands:
            band_key = f'band_{band[0]}_{band[1]}'
            for seq in correlation_coefficients.keys():
                if self.cca_models[seq] and band_key in self.cca_models[seq]:
                    correlation_coefficients[seq].append(
                        self.calculate_correlations(band_data[band_key], self.cca_models[seq][band_key]))
        return correlation_coefficients

    def evaluate_correlation_coefficients(self, correlation_coefficients):
        avg_correlations = {
            'mSequence1': np.mean(correlation_coefficients[0]),
            'mSequence2': np.mean(correlation_coefficients[1]),
            'mSequence3': np.mean(correlation_coefficients[2])
        }

        # Sort the sequences by their average correlation in descending order
        sorted_correlations = sorted(avg_correlations.items(), key=lambda item: item[1], reverse=True)

        # Get the highest and second-highest correlations
        highest_sequence, highest_corr = sorted_correlations[0]
        second_highest_sequence, second_highest_corr = sorted_correlations[1]

        # Check if the highest correlation is at least 0.15 higher than the second highest
        if highest_corr >= second_highest_corr + 0.15:
            return highest_corr, highest_sequence
        else:
            return None, None
    def cleanup(self):
        return

    @rpc
    def decode(self) -> int:
        choices = self.decoded_choices
        user_choice = max(set(choices), key=choices.count)
        return user_choice


    @async_rpc
    def training(self, input0: int, input1: int):
        """
        Args:
            input0: int - 1 for choice 1, 2 for choice 2, 3 for choice 3
        Returns: Generates correlation coefficients for EEG data x m-sequence
        """
        self.mSequence = input1
        # Train the CCA
        self.train_cca(input0)

        return

