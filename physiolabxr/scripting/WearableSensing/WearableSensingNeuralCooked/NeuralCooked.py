from scipy.signal import butter, filtfilt, iirnotch
from sklearn.cross_decomposition import CCA
import numpy as np
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.utils.buffers import DataBuffer
from physiolabxr.rpc.decorator import async_rpc
import time
class NeuralCooked(RenaScript):

    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    def init(self):
        self.freqBands = [(8, 60), (13, 60), (30, 60)]          #defining frequency bands for filter bank
        self.frequency = 300                                    #default frequency of DSI-24
        self.refreshRate  = 0.0166                              #Duration of  flicker
        self.data = DataBuffer()                                #generating a data buffer for EEG data
        self.templates = {}
        self.ccaModel = {}                                      #creating a list to store all of the CCA models
        self.ccaResults= {}
        self.decodedChoices = []                                #creating a list to store all of the decoded choices
        self.mSequence = [
            [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],  #mSequence1
            [1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],  #mSequence2
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0]   #mSequence3
        ]
        self.sequenceLength = len(self.mSequence[0])
        self.segmentLength = int(np.ceil(self.sequenceLength*self.frequency * self.refreshRate))
        self.mSequenceSignal = {
            'sequence1': self.generateMSignal(0),
            'sequence2': self.generateMSignal(1),
            'sequence3': self.generateMSignal(2)
        }
        self.seq1Data = np.array([[[ ]]])
        self.seq2Data = np.array([[[ ]]])
        self.seq3Data = np.array([[[ ]]])
        self.gameState = 0


    def loop(self):
        if self.inputs:
            unfilteredData = self.inputs['DSI24'][0][14:18, :].astype(float)
            filteredData1 = self.bandpassFilter(unfilteredData, 2, 120)
            filteredData2 = self.notchFilter(filteredData1, 60)


            EEG_Data = {                                                        #creating a dictionary for EEG data
                'stream_name': 'EEG Data',                                      #defining the stream name
                'frames': filteredData2,                                        #choosing the correct channels
                'timestamps': self.inputs['DSI24'][1].astype(float)             #defining the timestamps
            }
            self.data.update_buffer(EEG_Data)                                   #updating the data buffer with EEG data
            Data = self.data.get_data('EEG Data')
            if Data.shape[1] > 400:                                            #if the data is longer than 12 seconds then cut off beginning of data so that it is to 200 seconds
                self.data.clear_stream_up_to_index(stream_name='EEG Data', cut_to_index=self.data.get_data('EEG Data').shape[1]-400)
            if self.data.get_data('EEG Data').shape[1] > self.segmentLength:
                if self.gameState == 1:
                    self.decodeChoice()



    def cleanup(self):
        self.freqBands = [(8, 60), (12, 60), (30, 60)]
        self.mSequence = []
        self.frequency = 300
        self.data = DataBuffer()
        self.ccaModels = []
        self.decodedChoices = []
        self.gameState = 0

        return

    #Data Manipulation
    def bandpassFilter(self, data, lowcutoff, highcutoff):
        """
        Function that takes in data and applies a bandpass filter to it
        :param data: EEG data to be band passed
        :param lowcutoff: Low pass
        :param highcutoff: High pass
        :return: A band passed version of the data
        """
        nq = 0.5 * self.frequency
        order = 3
        lowcutOffNorm = lowcutoff / nq
        highcutOffNorm = highcutoff / nq
        b, a = butter(order, [ lowcutOffNorm, highcutOffNorm], btype = 'band')
        BPdata = filtfilt(b,a,data)
        return BPdata
    def notchFilter(self, data, notchfreq, qualityFactor = 30):
        """
        Function that takes in data and applies a notch filter to it
        :param data: signal to be filtered
        :param notchfreq: frequency to notch
        :param qualityFactor: quality factor of the notch filter that determines the width of the notch
        :return: filtered array
        """
        # Design the notch filter
        b, a = iirnotch(notchfreq, qualityFactor, self.frequency)
        # Apply the filter to the data
        filteredData = filtfilt(b, a, data)
        return filteredData
    def applyFilterBank(self, data):
        """
        Function that takes segmented data filters it with a bandpass filter, and averages the signal out over all the
        segments to return a dictionary that contains 3 arrays for each frequency band
        :param data: list of data to be filtered after data has been segmented
        :return: a dictionary that contains 3 arrays for each frequency band where the keys are the found in self.freqBands
        """
        band = {}           #Dictionary created to fill in
        for i in range(3):
            filteredSegments = np.empty((4,self.segmentLength,0))  # List to hold filtered segments for the current frequency band
            # for segment in data:
            #     # Apply bandpass filter to each segment and append the result to the list
            #     filteredSegment = self.bandpassFilter(segment, self.freqBands[i][0], self.freqBands[i][1])
            #     filteredSegments = np.concatenate((filteredSegments, filteredSegment[:, :, np.newaxis]), axis =2)
            for segmentNum in range(data.shape[2]):
                # Apply bandpass filter to each segment and append the result to the list
                filteredSegment = self.bandpassFilter(data[:, :, segmentNum], self.freqBands[i][0], self.freqBands[i][1])
                filteredSegments = np.concatenate((filteredSegments, filteredSegment[:, :, np.newaxis]), axis =2)

            band[self.freqBands[i]]= np.mean(filteredSegments, axis=2)
            # Average the filtered segments across the first axis
            # averageSegments = np.mean(filteredSegments, axis=2)
            # Downsampling
            # band[self.freqBands[i]] = np.mean(averageSegments.reshape(-1, 5), axis=1)
        return band
    def adjustSegments(self, segments):
        """
        Function that ensures that all segments created by the splitting of data to be the same shape as the
        expected segment length
        :param segments: EEG data that has been segmented
        :return: Returns the segments but truncated to be the same size as self.segmentLength
        """
        adjustedSegments = []
        for segment in segments:
            # If the segment is shorter than the desired length, pad it with zeros
            if segment.shape[1] < self.segmentLength:
                padding = np.ones((segment.shape[0], self.segmentLength - segment.shape[1]))
                adjustedSegment = np.hstack((segment, padding))  # Pad with zeros
            else:
                # If the segment is longer, trim it to the desired length
                adjustedSegment = segment[:, :self.segmentLength]

            adjustedSegments.append(adjustedSegment)

        return adjustedSegments
    def createTemplates(self):
        """
        Function that generates templates of the training EEG data by segmenting the data based on expected length of
        samples within one iteration of the m-sequence playing in Unity and averaging the data out.
        :param: EEG data for segmenting into templates
        :return: dictionary of segment numbers that contain a dictionary of frequency banded EEG data that has been
        averaged out for each segment: segment number -> keys frequency band
        """
        filteredData1= np.zeros(self.seq1Data.shape)
        filteredData2= np.zeros(self.seq2Data.shape)
        filteredData3= np.zeros(self.seq3Data.shape)
        for segment in range(self.seq1Data.shape[2]):
            filteredData1[:,:,segment] = self.bandpassFilter(self.notchFilter(self.seq1Data[:, :, segment], 60), 2, 120)
            filteredData2[:,:,segment] = self.bandpassFilter(self.notchFilter(self.seq2Data[:, :, segment], 60), 2, 120)
            filteredData3[:,:,segment] = self.bandpassFilter(self.notchFilter(self.seq3Data[:, :, segment], 60), 2, 120)

        self.templates['sequence1'] = self.applyFilterBank(filteredData1) #size_segment_length
        self.templates['sequence2'] = self.applyFilterBank(filteredData2)
        self.templates['sequence3'] = self.applyFilterBank(filteredData3)
    def generateMSignal(self, seqNum):
        """
        Function to generate a signal template for the m-sequence for CCA.fit. It functions by repeating the sequence
        element by a certain number of times based on the sampling rate of the EEG and the refresh rate of the monitor.
        :param seqNum: The number associated with which m-sequence the function will generate a signal for
        :return: Returns an 8 x segmentLength array where each row is the m-sequence stretched to fit the required
        segment length necessary for analysis via CCA
        """
        samplesPerBit = self.segmentLength // len(self.mSequence[seqNum])
        signal = np.repeat(self.mSequence[seqNum], samplesPerBit )

        if len(signal) > self.segmentLength:
            signal = signal[:, :self.segmentLength]


        elif len(signal) < self.segmentLength:
            padding = int(np.ceil((self.segmentLength - len(signal))/2)+1)
            signal = np.pad(signal, pad_width=padding, mode='constant', constant_values=0)
            signal = signal[:self.segmentLength]
        mSignal = np.expand_dims(np.tile(signal, (4, 1)),axis=2)
        filteredMsignal = self.applyFilterBank(mSignal)
        return filteredMsignal

    def trainCCA(self):
        """
        Training the CCA model
        By generated spatial filters and templates for each target m-sequence
        """
        self.createTemplates()
        self.ccaModel = {k: {sub_k: None for sub_k in v.keys()} for k, v in self.templates.items() if isinstance(v, dict)}
        for sequence in self.templates.keys():
            for freqBand in self.templates[sequence].keys():
                cca = CCA(n_components=1)
                cca.fit(self.templates[sequence][freqBand].T, self.mSequenceSignal[sequence][freqBand].T)
                self.ccaModel[sequence][freqBand] = cca
                test1,test2 = cca.transform(self.templates[sequence][freqBand].T, self.mSequenceSignal[sequence][freqBand].T)
                print(np.abs(np.corrcoef(test1.T,test2.T)[0, 1]))

        print('training done')
        self.gameState = 1

    @async_rpc
    def add_seq_data(self, sequenceNum: int, duration: float):
        """
        RPC function that gets called by Unity to grab the EEG data from *duration* samples ago and adds it to the
        respective m-sequence data storage
        :param sequenceNum: The number that represents the m-sequence that was played for *duration* time
        :param duration: How long the respective m-sequence was played for
        :return: Does not return anything to Unity it is an asynchronous process
        """

        eegData = np.expand_dims(self.data.get_data('EEG Data')[:, -self.segmentLength:], axis=2) # 4 by (samples)
        print(self.segmentLength,duration*300)
        if sequenceNum == 1:
            if self.seq1Data.size == 0:
                self.seq1Data = eegData
            else:
                self.seq1Data = np.concatenate((self.seq1Data, eegData), axis=2)
        elif sequenceNum == 2:
            if self.seq2Data.size == 0:
                self.seq2Data = eegData
            else:
                self.seq2Data = np.concatenate((self.seq2Data, eegData), axis=2)
        elif sequenceNum == 3:
            if self.seq3Data.size == 0:
                self.seq3Data = eegData
            else:
                self.seq3Data = np.concatenate((self.seq3Data, eegData), axis=2)

    @async_rpc
    def training(self) -> int:
        """
        RPC function that gets called by Unity to begin training off of the saved data from function: add_seq_data
        :return: Returns 1 to Unity once it is done training
        """
        # Train the CCA
        self.trainCCA()  # start training the CCA
        self.data.clear_buffer_data() #Clear the buffer after training
        return 1
    @async_rpc
    def decode(self) -> int:
        """
        RPC function that gets called by Unity to receive what the player is looking at by sending Unity the most
        frequently occurring m-sequence using the mode of the decoded choices
        :return: Sends Unity the most common choice to Unity, sends 0 if there is no common choice
        """

        userChoice = 0
        if len(self.decodedChoices) > 5:
            choices = [x for x in self.decodedChoices if x is not None]
            counts = {choice: choices.count(choice) for choice in set(choices)}
            max_count = max(counts.values())
            modes = [choice for choice, count in counts.items() if count == max_count]
            userChoice = modes[0] if len(modes) == 1 else 0
            self.decodedChoices = []
            self.data.clear_buffer_data()
        return userChoice

    def decodeChoice(self):
        """
        A looping function that decodes the correlation coefficients of the data and adds the highest correlation
        m-sequence (what m-sequence the user is likely looking at) to a list called decodedChoices
        :return: Updates decodedChoice with what has been decoded
        """
        data = np.expand_dims(self.data.get_data('EEG Data')[:, -self.segmentLength:],axis=2)
        self.correlationCoefficients = self.applyCCA(data)  # getting the correlation coefficients by applying CCA
        highestCorrelation, detectedChoice = self.evaluateCorrelationCoefficients(self.correlationCoefficients)  # evaluating the correlation coefficients to get the highest correlation and the detected choice
        if detectedChoice != None:
            self.decodedChoices.append(detectedChoice)

    def applyCCA(self, data):
        """
        Function for applying CCA
        :param data: Data that needs to be classified
        :return: A dictionary that contains the correlation of the EEG data with the templates
        """

        #Filter the data
        filteredData = self.applyFilterBank(data)
        self.ccaResults = {k: {sub_k: None for sub_k in v.keys()} for k, v in self.templates.items() if isinstance(v, dict)} #temporarily assigns the dictionary so that we can use the keys

        correlation = {k: {sub_k: None for sub_k in v.keys()} for k, v in self.templates.items() if isinstance(v, dict)}
        avgCorrelation = {}
        #Transform the data with CCA
        for sequence in self.templates.keys():
            for freqBand in self.templates[sequence].keys():
                cca = self.ccaModel[sequence][freqBand]
                self.ccaResults[sequence][freqBand], refMSeq = cca.transform(filteredData[freqBand].T, self.mSequenceSignal[sequence][freqBand].T)
                correlation[sequence][freqBand] = np.abs(np.corrcoef(self.ccaResults[sequence][freqBand].T, refMSeq.T)[0, 1])
                print(sequence, freqBand, correlation[sequence][freqBand])
            avgCorrelation[sequence] = np.mean(list(correlation[sequence].values()))
        return avgCorrelation

    def evaluateCorrelationCoefficients(self, correlationCoefficients):
        """
        Function for determining the highest correlation coefficient and whether the coefficient is greater than the
         second-highest correlation by a certain threshold.
        :param correlationCoefficients:  Correlation coefficients between the EEG data and the templates
        :return: The highest correlation as well as the corresponding m-sequence number if the correlation surpasses the
        threshold
        """
        # Sort the sequences by their average correlation in descending order
        sortedCorrelations = sorted(correlationCoefficients.items(), key=lambda item: item[1], reverse=True)
        # Get the highest and second-highest correlations
        highestSequence, highestCorr = sortedCorrelations[0]
        secondHighestSequence, secondHighestCorr = sortedCorrelations[1]
        # Check if the highest correlation is at least 0.15 higher than the second highest
        if highestCorr >= secondHighestCorr + 0.01:
            return highestCorr, int(highestSequence[-1])
        else:
            return highestCorr, None
