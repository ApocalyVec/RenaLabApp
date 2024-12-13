from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA
import numpy as np
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.utils.buffers import DataBuffer
from physiolabxr.rpc.decorator import rpc, async_rpc

class NeuralCooked(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    def init(self):
        self.freq_bands = [(8, 60), (12, 60), (30, 60)]         #defining frequency bands for filter bank
        self.mSequence = [                                      #creating a list of m_sequences
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],  #mSequence1
            [1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1],  #mSequence2
            [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]   #mSequence3
        ]
        self.frequency = 300                                    #default frequency of DSI-24
        self.data = DataBuffer()                                #generating a data buffer for EEG data
        self.cca_models = []                                    #creating a list to store all of the CCA models
        self.decoded_choices = []                               #creating a list to store all of the decoded choices

    def loop(self):
        EEG_Data = {                                            #creating a dictionary for EEG data
            'stream_name': 'EEG Data',                          #defining the stream name
            'frames': self.inputs['DSI-24'][0][14:18, :],       #choosing the correct channels
            'timestamps': self.inputs['DSI-24'][1]              #defining the timestamps
        }
        self.data.update_buffer(EEG_Data)                       #updating the data buffer with EEG data
        if self.data.get_data('EEG Data').shape[1] > 60000:     #if the data is longer than 200 seconds then cut off beginning of data so that it is to 200 seconds
            self.data.clear_stream_up_to_index(self, stream_name='EEG Data', cut_to_index=len(self.data.get_data('EEG Data').shape[1])-60000)

        band_data = self.apply_filter_banks(self.data.get_data('EEG Data'))             #applying filter banks to EEG data
        self.correlation_coefficients = self.apply_shifting_window_cca(band_data)       #getting the correlation coefficients by applying shifting window CCA
        highest_correlation, detected_choice = self.evaluate_correlation_coefficients(  #evaluating the correlation coefficients to get the highest correlation and the detected choice
            self.correlation_coefficients)
        self.decoded_choices.append(detected_choice)                                    #adding the detected choice to the list of detected choices

    def cleanup(self):
        self.freq_bands = [(8, 60), (12, 60), (30, 60)]
        self.mSequence = []
        self.frequency = 300
        self.data = DataBuffer()
        self.cca_models = []
        self.decoded_choices = []

    ## Basic Tools
    def apply_filter_banks(self, data):
        band_data = {}
        for band in self.freq_bands:
            band_key = f'band_{band[0]}_{band[1]}'
            band_data[band_key] = self.bandpass_filter(data, band[0], band[1], self.frequency)
        return band_data

    def bandpass_filter(self, data, lowcut, highcut, fs, order=8):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    ## Signal Generation and CCA
    def generate_reference_signal(self, m_sequence, length):
        # Repeat the m-sequence to match the length of the EEG data
        repetitions = length // len(m_sequence) + 1
        reference_signal = np.tile(m_sequence, repetitions)[:length]
        return reference_signal

    def apply_shifting_window_cca(self, band_data):
        correlation_coefficients = {1: [], 2: [], 3: []}
        for band in self.freq_bands:
            band_key = f'band_{band[0]}_{band[1]}'
            for seq_index in range(3):
                reference_signal = self.generate_reference_signal(self.mSequence[seq_index], band_data[band_key].shape[1])
                cca = CCA(n_components=1)
                cca.fit(band_data[band_key].T, reference_signal)
                transformed_data = cca.transform(band_data[band_key].T)
                corr = np.corrcoef(transformed_data[:, 0], reference_signal)[0, 1]
                correlation_coefficients[seq_index + 1].append(corr)
        return correlation_coefficients

    def evaluate_correlation_coefficients(self, correlation_coefficients):
        avg_correlations = {
            'mSequence1': np.mean(correlation_coefficients[1]),
            'mSequence2': np.mean(correlation_coefficients[2]),
            'mSequence3': np.mean(correlation_coefficients[3])
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
            return highest_corr, -1

    @rpc
    def decode(self) -> int:
        choices = self.decoded_choices
        user_choice = max(set(choices), key=choices.count)
        self.decoded_choices = []
        return user_choice
