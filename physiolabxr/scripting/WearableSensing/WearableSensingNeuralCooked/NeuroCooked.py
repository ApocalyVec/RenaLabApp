from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA
import numpy as np
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.utils.buffers import DataBuffer
from physiolabxr.rpc.decorator import rpc, async_rpc

class NeuroCooked(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    def init(self):
        self.freq_bands = [(8, 60), (12, 60), (30, 60)]         #defining frequency bands for filter bank
        self.frequency = 300                                    #default frequency of DSI-24
        self.data = DataBuffer()                                #generating a data buffer for EEG data
        self.templates = {}
        self.ccaModel = {}                                  #creating a list to store all of the CCA models
        self.ccaResults= {}
        self.decoded_choices = []                               #creating a list to store all of the decoded choices
        self.mSequence = [
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],  #mSequence1
            [1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1],  #mSequence2
            [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]   #mSequence3
        ]
        self.sequence_length = len(self.mSequence[0])
        self.mSequenceSignal =  {
            'segment1': self.generateMSignal(0),
            'segment2': self.generateMSignal(1),
            'segment3': self.generateMSignal(2)
        }
        self.seq1_data = np.array([[]])
        self.seq2_data = np.array([[]])
        self.seq3_data = np.array([[]])


    def loop(self):
        if self.inputs:
            EEG_Data = {                                            #creating a dictionary for EEG data
                'stream_name': 'EEG Data',                          #defining the stream name
                'frames': self.inputs['DSI24'][0][14:18, :].astype(float),       #choosing the correct channels
                'timestamps': self.inputs['DSI24'][1].astype(float)           #defining the timestamps
            }
            Sequence_Data = {}                                      #creating a dictionary for sequence data
            self.data.update_buffer(EEG_Data)                       #updating the data buffer with EEG data
            if self.data.get_data('EEG Data').shape[1] > 60000:     #if the data is longer than 200 seconds then cut off beginning of data so that it is to 200 seconds
                self.data.clear_stream_up_to_index(stream_name= 'EEG Data', cut_to_index= self.data.get_data('EEG Data').shape[1]-60000)
            if len(self.ccaModel) == 3:                           #if training is complete (i.e there are 3 CCA models) then we can start decoding everything asyncronously
                self.decode_choice()                                  #adding

    def cleanup(self):
        self.freq_bands = [(8, 60), (12, 60), (30, 60)]
        self.mSequence = []
        self.frequency = 300
        self.data = DataBuffer()
        self.cca_models = []
        self.decoded_choices = []

        return

    #Data Manipulation
    def bandpassFilter(self, data, low, high, order = 9):
        """
        Takes in data and applies a bandpass filter to it
        :param data: EEG data to be bandpassed
        :param low: Low pass
        :param high: High pass
        :param fs: sampling frequency
        :param order: Default is 8
        :return: A bandpassed version of the data
        """
        nq = 0.5 * self.frequency
        lowpass = low / nq
        highpass = high / nq
        b, a = butter(order, [ lowpass, highpass], bype = 'band')
        BPdata = filtfilt(b,a,data)
        return BPdata

    def appplyFilterBank(self, data):
        """
        Returns data in 3 different arrays of different frequncy ranges
        :param data: list of data to be filtered after data has been segmented
        :return: a dictionary that contains 3 arrays for each frequency band where the keys are the found in self.freq_bands
        """
        band = {}           #Dictionary created to fill in
        for i in range(3):
            filtered_segments = []  # List to hold filtered segments for the current frequency band
            for segment in data:
                # Apply bandpass filter to each segment and append the result to the list
                filtered_segment = self.bandpassFilter(segment, self.freq_bands[i][0], self.freq_bands[i][1],
                                                       self.frequency)
                filtered_segments.append(filtered_segment)

            # Average the filtered segments across the first axis
            band[self.freq_bands[i]] = np.mean(filtered_segments, axis=0)
        return band
    def adjust_segments(self, segments, segment_length):
        adjusted_segments = []
        for segment in segments:
            # If the segment is shorter than the desired length, pad it with zeros
            if segment.shape[1] < segment_length:
                padding = np.zeros((segment.shape[0], segment_length - segment.shape[1]))
                adjusted_segment = np.hstack((segment, padding))  # Pad with zeros
            else:
                # If the segment is longer, trim it to the desired length
                adjusted_segment = segment[:, :segment_length]

            adjusted_segments.append(adjusted_segment)

        return adjusted_segments
    def createTemplates(self):
        """
        Creates templates that are EEG arrays that are filtered
        :param data: EEG data for segmenting into templates
        :return: dictionary of dictionary of filtered EEG arrays Keys: segment number -> keys frequency band
        """
        seq1_segments = np.array_split(self.seq1_data, self.seq1_data.shape[1] // self.sequence_length, axis=1)
        seq2_segments = np.array_split(self.seq2_data, self.seq2_data.shape[1] // self.sequence_length, axis=1)
        seq3_segments = np.array_split(self.seq3_data, self.seq3_data.shape[1] // self.sequence_length, axis=1)

        seq1_segments = self.adjust_segments(seq1_segments, self.sequence_length)
        seq2_segments = self.adjust_segments(seq2_segments, self.sequence_length)
        seq3_segments = self.adjust_segments(seq3_segments, self.sequence_length)


        self.templates['segment1'] = self.applyFilterBank(seq1_segments)
        self.templates['segment3'] = self.applyFilterBank(seq2_segments)
        self.templates['segment3'] = self.applyFilterBank(seq3_segments)


    def generateMSignal(self, seqNum):

        # Step 1: Calculate the total number of samples needed
        total_samples = self.sequence_length * 0.033 * 300

        # Step 2: Calculate the number of samples per m-sequence element
        samples_per_bit = total_samples // len(self.mSequence[seqNum])

        # Step 3: Create the binary signal by repeating each bit
        signal = np.repeat(self.mSequence[seqNum], samples_per_bit)

        # Step 4: If the signal is longer than required, truncate it
        if len(signal) > total_samples:
            signal = signal[:total_samples]
        return signal
    def train_cca(self):
        """
        Training the CCA model
        By generated spatial filters and templates for each target m-sequence
        """
        self.createTemplates()

        for segment in self.templates.keys():
            for freqBand in self.templates[segment].keys():
                cca = CCA(n_components = 1)
                cca.fit(self.templates[segment][freqBand].T,self.mSequenceSignal[segment])
                self.ccaModel[segment][freqBand] = cca

    @async_rpc
    def add_seq_data(self, sequenceNum: int,
                     duration: float):  # Data is going to come in sequencially seq1 -> seq2 -> seq3 repeat
        eegData = self.data.get_data('EEG Data')[:, int(-duration * 300):]  # 4 by (samples)

        if sequenceNum == 1:
            if self.seq1_data.size == 0:
                self.seq1_data = eegData
            else:
                self.seq1_data = np.concatenate((self.seq1_data, eegData), axis=1)
        elif sequenceNum == 2:
            if self.seq2_data.size == 0:
                self.seq2_data = eegData
            else:
                self.seq2_data = np.concatenate((self.seq2_data, eegData), axis=1)
        elif sequenceNum == 3:
            if self.seq3_data.size == 0:
                self.seq3_data = eegData
            else:
                self.seq3_data = np.concatenate((self.seq3_data, eegData), axis=1)

                @async_rpc
                def training(self) -> int:
                    """
                    Args:
                        input0: int - 1 for choice 1, 2 for choice 2, 3 for choice 3
                    Returns: Generates correlation coefficients for EEG data x m-sequence
                    """
                    # Train the CCA
                    self.train_cca()  # start training the CCA
                    return 1
    @async_rpc
    def decode(self) -> int:
        # Get the choices decoded so far
        choices = self.decoded_choices

        # Determine the most common choice
        user_choice = max(set(choices), key=choices.count)

        # Clear the decoded choices list for the next round
        self.decoded_choices = []
        self.data.clear_buffer_data('EEG Data')

        # Return the most common detected choice
        return user_choice
    def decode_choice(self):
        self.correlation_coefficients = self.apply_shifting_window_cca(
            self.data.get_data('EEG Data'))  # getting the correlation coefficients by applying shifting window CCA
        highest_correlation, detected_choice = self.evaluate_correlation_coefficients(
            self.correlation_coefficients)  # evaluating the correlation coefficients to get the highest correlation and the detected choice
        self.decoded_choices.append[detected_choice]

    def apply_shifting_window_cca(self, data):
        """
        Applies shifting window CCA to the filtered band data.
        """
        window_size = self.sequence_length * 0.033 * 300  # For example, 1 second window for 300 Hz sampling rate
        step_size = window_size/2  # For example, 0.5 second step size for 300 Hz sampling rate

        segments = []
        for start in range(0, len(data) - window_size + 1, step_size):
            segment = data[start:start+window_size]
            segments.append(segment)

        #Filter the data
        filtered_data = {}
        for band in self.freq_bands:
            filtered_data[band] = self.appplyFilterBank(segments)
        correlation = {}
        avg_correlation = {}
        #Transform the data with CCA
        for segment in self.templates.keys():
            for freqBand in self.templates[segment].keys():
                cca = self.ccaModel[segment][freqBand]
                self.ccaResults[segment][freqBand] = cca.transform(filtered_data[freqBand])

                correlation[segment][freqBand] = np.corrcoef(self.ccaResults[segment][freqBand], self.templates[segment][freqBand])
            avg_correlation[segment] = np.mean(correlation[segment].values())
        return avg_correlation

    def evaluate_correlation_coefficients(self, correlation_coefficients):
        # Sort the sequences by their average correlation in descending order
        sorted_correlations = sorted(correlation_coefficients.items(), key=lambda item: item[1], reverse=True)

        # Get the highest and second-highest correlations
        highest_sequence, highest_corr = sorted_correlations[0]
        second_highest_sequence, second_highest_corr = sorted_correlations[1]

        # Check if the highest correlation is at least 0.15 higher than the second highest
        if highest_corr >= second_highest_corr + 0.15:
            return highest_corr, highest_sequence
        else:
            return highest_corr, -1