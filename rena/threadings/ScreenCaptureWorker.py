import time

import cv2
import numpy as np
import pyautogui
import pyqtgraph as pg
from PyQt5.QtCore import QObject, pyqtSignal
from pylsl import local_clock

from rena.presets.Presets import VideoDeviceChannelOrder
from rena.threadings.workers import RenaWorker
from rena.utils.image_utils import process_image


class ScreenCaptureWorker(QObject, RenaWorker):
    tick_signal = pyqtSignal()  # note that the screen capture follows visualization refresh rate
    change_pixmap_signal = pyqtSignal(tuple)

    def __init__(self, screen_label, video_scale: float, channel_order: VideoDeviceChannelOrder):
        super().__init__()
        self.tick_signal.connect(self.process_on_tick)
        self.screen_label = screen_label
        self.is_streaming = True

        self.video_scale = video_scale
        self.channel_order = channel_order

    def stop_stream(self):
        self.is_streaming = False

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            pull_data_start_time = time.perf_counter()
            img = pyautogui.screenshot()
            frame = np.array(img)
            frame = frame.astype(np.uint8)
            frame = process_image(frame, self.channel_order, self.video_scale)
            frame = np.flip(frame, axis=0)
            self.pull_data_times.append(time.perf_counter() - pull_data_start_time)
            self.change_pixmap_signal.emit((self.screen_label, frame, local_clock()))  # uses lsl local clock for syncing