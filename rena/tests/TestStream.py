"""Example program to demonstrate how to send a multi-channel time series to
LSL."""
import random
import sys
import getopt
import string

import time
from collections import deque
from random import random as rand

import numpy as np
import zmq
from pylsl import StreamInfo, StreamOutlet, local_clock


def LSLTestStream(stream_name, n_channels=81):
    letters = string.digits

    srate = 2048
    print('Test stream name is ' + stream_name)
    type = 'EEG'
    help_string = 'SendData.py -s <sampling_rate> -n <stream_name> -t <stream_type>'
    info = StreamInfo(stream_name, type, n_channels, srate, 'float32', 'someuuid1234')

    # next make an outlet
    outlet = StreamOutlet(info)

    print("now sending data...")
    start_time = local_clock()
    sent_samples = 0
    while True:
        elapsed_time = local_clock() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples
        for sample_ix in range(required_samples):
            # make a new random n_channels sample; this is converted into a
            # pylsl.vectorf (the data type that is expected by push_sample)
            mysample = [rand() for _ in range(n_channels)]
            # now send it
            outlet.push_sample(mysample)
        sent_samples += required_samples
        # now send it and wait for a bit before trying again.
        time.sleep(1e-3)

def ZMQTestStream(stream_name, port, n_channels=81):
    topic = stream_name
    srate = 30

    c_channels = 3
    width = 800
    height = 800
    n_channels = c_channels * width * height

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:%s" % port)

    # next make an outlet
    print("now sending data...")
    send_times = deque(maxlen=srate * 10)
    start_time = time.time()
    sent_samples = 0
    while True:
        elapsed_time = time.time() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples
        if required_samples > 0:
            samples = np.random.rand(required_samples * n_channels).reshape((required_samples, -1))
            samples = (samples * 255).astype(np.uint8)
            for sample_ix in range(required_samples):
                mysample = samples[sample_ix]
                socket.send_multipart([bytes(topic, "utf-8"), np.array(local_clock()), mysample])
                send_times.append(time.time())
            sent_samples += required_samples
        # now send it and wait for a bit before trying again.
        time.sleep(0.01)
        if len(send_times) > 0:
            fps = len(send_times) / (np.max(send_times) - np.min(send_times))
            print("Send FPS is {0}".format(fps), end='\r')