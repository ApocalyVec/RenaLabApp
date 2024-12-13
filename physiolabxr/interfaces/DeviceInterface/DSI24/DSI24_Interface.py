from multiprocessing import Process, Event

import numpy as np
import zmq

from physiolabxr.interfaces.DeviceInterface.DSI24.DSI24_Process import DSI24_process
from physiolabxr.third_party.WearableSensing.DSI_py3 import *
from physiolabxr.interfaces.DeviceInterface.DeviceInterface import DeviceInterface


def run_dsi24_headset_process(port, com_port, impedance):
    terminate_event = Event()
    headset_process = Process(target=DSI24_process, args=(terminate_event, port, com_port, impedance))
    headset_process.start()
    return headset_process, terminate_event

class DSI24_Interface(DeviceInterface):
    def __init__(self,
                 _device_name='DSI24',
                 _device_type='eeg',
                 _device_nominal_sampling_rate=300):
        super(DSI24_Interface, self).__init__(_device_name=_device_name,
                                              _device_type=_device_type,
                                              device_nominal_sampling_rate=_device_nominal_sampling_rate,
                                              is_supports_device_availability=False)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind("tcp://*:0")  # Bind to port 0 for an available random port
        self.port = self.socket.getsockopt(zmq.LAST_ENDPOINT).decode("utf-8").split(":")[-1]
        self.data_process = None
        self.terminate_event = None
        self.battery_level = None
        self.impedanceValues = []

    def start_stream(self, bluetooth_port, impedance):
        self.data_process, self.terminate_event = run_dsi24_headset_process(self.port, bluetooth_port, impedance)

    def process_frames(self):
        frames, timestamps, messages = [], [], []
        while True:  # get all available data
            try:
                data = self.socket.recv_json(flags=zmq.NOBLOCK)
                if data['t'] == 'i':
                    messages.append(data['message'])
                elif data['t'] == 'e':
                    raise DSIException(data['message']) # this will cause stop_stream to be called
                elif data['t'] == 'd':
                    self.battery_level = data['battery']
                    if data['impedance'] == 0:
                        frames.append(data['frame'])
                        timestamps.append(data['timestamp'])
                    else:
                        self.impedanceValues = data['frame']
                        frames.append(data['frame'])
                        timestamps.append(data['timestamp'])


            except zmq.error.Again:
                break

        if len(frames) > 0:
            return np.array(frames).transpose(2, 1, 0)[0], np.array(timestamps)[:, 0], messages
        else:
            return frames, timestamps, messages

    def stop_stream(self):
        self.terminate_event.set()
        self.data_process.join()
        self.data_process = None
        # empty the socket buffer, so that the next time we start the stream, we don't get old data
        while True:  # do this after the process has been terminated
            try:
                self.socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.error.Again:
                break

    def is_stream_available(self):
        return self.device_available

    def get_sampling_rate(self):
        return self.device_nominal_sampling_rate

    def __del__(self):
        self.socket.close()
        self.context.term()
