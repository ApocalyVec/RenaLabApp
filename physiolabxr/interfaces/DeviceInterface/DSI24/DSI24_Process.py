import math
from multiprocessing import Process

import zmq

from physiolabxr.third_party.WearableSensing.DSI_py3 import *
import numpy as np
from pylsl import local_clock
is_first_time = True
time_offset = 0
dsi24_data_socket = None


@MessageCallback
def example_message_callback( msg, lvl=0 ):
    global dsi24_data_socket
    if lvl <= 3:  # ignore messages at debugging levels higher than 3
        msg_str = IfStringThenNormalString(msg)
        print( "DSI Message (level %d): %s" % ( lvl, msg_str ) )
        new_data_dict = {
            't': 'i',  # 'd' for data, 'i' for info, 'e' for error
            'message': msg_str
        }
        # Send data via ZMQ socket to the main process
        try:
            dsi24_data_socket.send_json(new_data_dict)
        except zmq.error.ZMQError:
            print("Socket already closed.")
    return 1

@SampleCallback
def example_sample_callback_signals(headsetPtr, packetTime, userData):
    global is_first_time
    global time_offset
    global dsi24_data_socket

    # This function is called when a new packet is received
    h = Headset(headsetPtr)
    new_data = np.array(['%+08.2f' % (ch.GetSignal()) for ch in h.Channels()])
    new_data = new_data.reshape(24, 1)
    new_data = new_data[[9, 10, 3, 2, 4, 17, 18, 7, 1, 5, 11, 22, 12, 21, 8, 0, 6, 13, 14, 20, 23, 19, 16, 15], :]

    # Calculate the time offset on the first packet
    if is_first_time:
        time_offset = local_clock() - float(packetTime)
        is_first_time = False

    t = [float(packetTime) + time_offset]
    if new_data.shape[1] != len(t):
        print('Data and timestamp mismatch')
        print(new_data.shape)
        print(len(t))
    batteryLevel = list(map(int, IfStringThenNormalString(h.GetBatteryLevelString()).strip("[]").split(",")))
    # Create a dictionary with the stream name, data, and timestamps
    # need to convert the new_data to list to make it json serializable
    new_data_dict = {
        't': 'd', # 'd' for data, 'i' for info, 'e' for error
        'frame': new_data.tolist(),
        'timestamp': t,
        'battery': batteryLevel,
        'impedance': 0
    }

    # Send data via ZMQ socket to the main process
    try:
        dsi24_data_socket.send_json(new_data_dict)
    except zmq.error.ZMQError:
        print("Socket already closed.")

@SampleCallback
def example_sample_callback_impedances(headsetPtr, packetTime, userData):
    global is_first_time
    global time_offset
    global dsi24_data_socket

    # Create the headset object
    h = Headset(headsetPtr)

    # Collect impedance values from all referential EEG channels (excluding factory reference)
    impedance_data = np.array(
        ['%+08.2f' % (src.GetImpedanceEEG()) for src in h.Sources() if src.IsReferentialEEG() and not src.IsFactoryReference()])
    impedance_data = impedance_data.reshape(len(impedance_data), 1)
    impedance_data = impedance_data[[10,11,5,4,6,16,17,9,3,7,12,19,13,18,0,8,14,15,1,2], :]
    empty_rows = np.empty((3,impedance_data.shape[1]), dtype=object)
    impedance_data = np.concatenate((impedance_data, empty_rows))

    # Add the common-mode function (CMF) impedance value at the end
    cmf_impedance = np.array([h.GetImpedanceCMF()])
    impedance_data = np.vstack([impedance_data, cmf_impedance])

    # Calculate the time offset on the first packet
    if is_first_time:
        time_offset = local_clock() - float(packetTime)
        is_first_time = False

    # Create timestamp
    t = [float(packetTime) + time_offset]

    # Ensure that data and timestamps are aligned
    if impedance_data.shape[1] != len(t):
        print('Data and timestamp mismatch')
        print(impedance_data.shape)
        print(len(t))
    batteryLevel = list(map(int, IfStringThenNormalString(h.GetBatteryLevelString()).strip("[]").split(",")))
    # Convert impedance data to a dictionary for streaming
    impedance_data_dict = {
        't': 'd',  # 'd' for data, 'i' for info, 'e' for error
        'frame': impedance_data.tolist(),  # Convert impedance data to a list for JSON serialization
        'timestamp': t,
        'battery': batteryLevel,
        'impedance': 1
    }

    # Send the impedance data via ZMQ socket to the main process
    try:
        dsi24_data_socket.send_json(impedance_data_dict)
    except zmq.error.ZMQError:
        print("Socket already closed.")


def DSI24_process(terminate_event, network_port, com_port, impedance,args=''):
    """Process to connect to the DSI-24 device and send data to the main process

    Args:
        network_port (int): The port number to send data to the main process
        com_port (str): The COM port to connect to the DSI-24 device
        impedance (str): The mode of the headset (default: None), NOT IMPLEMENTED
    """
    global dsi24_data_socket
    global is_first_time
    global time_offset
    global batteryLevel

    context = zmq.Context()
    dsi24_data_socket = context.socket(zmq.PUSH)
    dsi24_data_socket.connect(f"tcp://localhost:{network_port}")

    headset = Headset()
    headset.SetMessageCallback(example_message_callback)
    try:
        headset.Connect(com_port)
    except Exception as e:
        dsi24_data_socket.send_json({
                          't': 'e',
                          'message': f"Error connecting to DSI-24 device: {e}. You might want to restart the device/computer and try again."})
        headset.Disconnect()
        return

    if impedance == 1:

        headset.SetSampleCallback(example_sample_callback_impedances, 0)
        headset.StartImpedanceDriver()
    else:
        # Set the sample callback to ExampleSampleCallback_Signals
        headset.SetSampleCallback(example_sample_callback_signals, 0)
        if len(args.strip()): headset.SetDefaultReference(args, True)
        print("EEG mode")
        # Start the data acquisition
    print("starting background acquisition")
    headset.StartBackgroundAcquisition()
    while not terminate_event.is_set():
        headset.Idle(1)
    print("DSI24 Process received termination event")
    is_first_time = True
    time_offset = 0
    # Stop the data acquisition and reset state
    headset.StopDataAcquisition()
    headset.Disconnect()
    dsi24_data_socket.close()
    context.term()
    print("DSI24 Process Stopped")