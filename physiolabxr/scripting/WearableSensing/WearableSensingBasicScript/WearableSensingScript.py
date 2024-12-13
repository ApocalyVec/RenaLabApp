from pylsl import local_clock
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.third_party.WearableSensing.DSI_py3 import *
import numpy as np
import sys
from physiolabxr.utils.buffers import DataBuffer

#Creating a data buffer with the DataBuffer class
data_buffer = DataBuffer()

is_first_time = True
time_offset = 0  # time offset for the first packet to the local_clock()
@MessageCallback
def ExampleMessageCallback( msg, lvl=0 ):
    if lvl <= 3:  # ignore messages at debugging levels higher than 3
 	    print( "DSI Message (level %d): %s" % ( lvl, IfStringThenNormalString( msg ) ) )
    return 1
@SampleCallback
def ExampleSampleCallback_Signals(headsetPtr, packetTime, userData):
    #This is the function that will be called every time a new packet is received
    global data_buffer
    global is_first_time
    global time_offset

    #Grab the headset by using a pointer
    h = Headset(headsetPtr)
    #Get the signal from each channel and format it so that it can be created into an array
    new_data = np.array(['%+08.2f' % (ch.GetSignal()) for ch in h.Channels()])
    #Reshapes the array into a 24x1 array so that it can be inputted into the data_buffer
    new_data = new_data.reshape(24,1)
    #Rearrange new_data to fit with desired output format
    new_data = new_data[[9, 10, 3, 2, 4, 17, 18, 7, 1, 5, 11, 22, 12, 21, 8, 0, 6, 13, 14, 20, 23, 19, 16, 15], :]
    #Get the time of the packet as a temporary solution to timestamps
    if is_first_time:
        time_offset = local_clock() - float(packetTime)
        is_first_time = False

    t = [float(packetTime) + time_offset]
    if new_data.shape[1] != len(t):
        print('Data and timestamp mismatch')
        print(new_data.shape)
        print(new_data.shape)
        print(len(t))

    #Create a dictionary with the stream name, data, and timestamps
    new_data_dict = {
        'stream_name': 'DSI-24',
        'frames': new_data,
        'timestamps': t
    }
    #Update the data buffer with the new data
    data_buffer.update_buffer(new_data_dict)
class DSI24(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    def init(self, arg = ''):
        #Create a headset object
        self.headset = Headset()
        #Set the message callback to ExampleMessageCallback
        self.headset.SetMessageCallback(ExampleMessageCallback)
        #Retrieves the command line arguments
        args = getattr(sys, 'argv', [''])
        #Set the default port to the first command line argument based on the parameter provided by user
        default_port = self.params['COM Port']
        #Connect the headset
        self.headset.Connect(default_port)
        #Start the data acquisition based on the parameter provided by user
        if arg.lower().startswith('imp'):
            #Currently not used
            self.headset.SetSampleCallback(ExampleSampleCallback_Impedances, 0)
            self.headset.StartImpedanceDriver()
        else:
            #Set the sample callback to ExampleSampleCallback_Signals
            self.headset.SetSampleCallback(ExampleSampleCallback_Signals, 0)
            if len(arg.strip()): self.headset.SetDefaultReference(arg, True)
        #Start the data acquisition
        self.headset.StartBackgroundAcquisition()


    def loop(self):
        #Called every loop based on the user's chosen frequency
        global data_buffer
        #If the data buffer has data, then set the output to the data buffer
        if len(data_buffer.keys()) > 0:
            self.set_output(stream_name = 'DSI-24', data = data_buffer.get_data('DSI-24'), timestamp = data_buffer.get_timestamps('DSI-24'))
            #Clear the data buffer
            data_buffer.clear_stream_buffer_data('DSI-24')

    def cleanup(self):
        #Called when the script is stopped
        global data_buffer
        global is_first_time
        global time_offset
        #Stop the data acquisition
        self.headset.StopBackgroundAcquisition()
        #Disconnect the headset
        time_offset = 0
        is_first_time = True
        self.headset.Disconnect()
        data_buffer.clear_buffer()
