import numpy as np
import os
import pytest
import threading
import time

from multiprocessing import Process
from random import random as rand
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QMessageBox
from rena.config import stream_availability_wait_time
from rena.tests.test_utils import get_random_test_stream_names, update_test_cwd, app_fixture, ContextBot
from rena.tests.TestStream import CSVTestStream
from rena.utils.data_utils import CsvStoreLoad, RNStream
from pytestqt.qtbot import QtBot
from rena.utils.ui_utils import CustomDialog

@pytest.fixture
def app_main_window(qtbot):
    app, test_renalabapp_main_window = app_fixture(qtbot)
    yield test_renalabapp_main_window
    app.quit()

@pytest.fixture
def context_bot(app_main_window, qtbot):
    test_context = ContextBot(app=app_main_window, qtbot=qtbot)

    yield test_context
    test_context.clean_up()


def test_csv_store_load(app_main_window, qtbot) -> None:
    num_stream_to_test = 3
    recording_time_second = 4
    srate = 2048
    stream_availability_timeout = 2 * stream_availability_wait_time * 1e3
    n_channels = 81
    record_path = 'E:\\Data\\RenaRecording'

    test_stream_names = []
    test_stream_processes = []
    test_stream_samples = []
    ts_names = get_random_test_stream_names(num_stream_to_test)
    # ts_names = ['TestStream1', 'TestStream2']
    samples = {}

    for ts_name in ts_names:
        test_stream_names.append(ts_name)
        sample = np.random.random((n_channels, 10 * recording_time_second * srate))
        samples[ts_name] = np.array(sample)
        p = Process(target=CSVTestStream, args=(ts_name, sample), kwargs={'n_channels':n_channels, 'srate':srate})
        test_stream_processes.append(p)
        test_stream_samples.append(sample)
        p.start()
        app_main_window.create_preset(ts_name, 'float', None, 'LSL', num_channels=81)  # add a default preset

    for ts_name in test_stream_names:
        app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
        qtbot.mouseClick(app_main_window.addStreamWidget.stream_name_combo_box, QtCore.Qt.LeftButton)  # click the add widget combo box
        qtbot.keyPress(app_main_window.addStreamWidget.stream_name_combo_box, 'a', modifier=Qt.ControlModifier)
        qtbot.keyClicks(app_main_window.addStreamWidget.stream_name_combo_box, ts_name)
        qtbot.mouseClick(app_main_window.addStreamWidget.add_btn, QtCore.Qt.LeftButton) # click the add widget combo box

    qtbot.mouseClick(app_main_window.addStreamWidget.stream_name_combo_box, QtCore.Qt.LeftButton)  # click the add widget combo box
    qtbot.keyPress(app_main_window.addStreamWidget.stream_name_combo_box, 'a', modifier=Qt.ControlModifier)
    qtbot.keyClicks(app_main_window.addStreamWidget.stream_name_combo_box, 'monitor 0')
    qtbot.mouseClick(app_main_window.addStreamWidget.add_btn, QtCore.Qt.LeftButton)  # click the add widget combo box

    app_main_window.settings_widget.set_recording_file_location(record_path)
    app_main_window.settings_widget.saveFormatComboBox.setCurrentIndex(3) # set recording file format as .csv
    app_main_window.settings_widget.saveFormatComboBox.activated.emit(app_main_window.settings_widget.saveFormatComboBox.currentIndex())
    print('set format to csv')

    def stream_is_available():
        for ts_name in test_stream_names:
            assert app_main_window.stream_widgets[ts_name].is_stream_available
    def stream_is_unavailable():
        for ts_name in test_stream_names:
            assert not app_main_window.stream_widgets[ts_name].is_stream_available

    qtbot.waitUntil(stream_is_available, timeout=stream_availability_timeout)
    # time.sleep(0.5)
    for ts_name in test_stream_names:
        qtbot.mouseClick(app_main_window.stream_widgets[ts_name].StartStopStreamBtn, QtCore.Qt.LeftButton)

    # app_main_window.ui.tabWidget.setCurrentWidget(
    #     app_main_window.ui.tabWidget.findChild(QWidget, 'recording_tab'))  # switch to the recoding widget
    qtbot.mouseClick(app_main_window.recording_tab.StartStopRecordingBtn, QtCore.Qt.LeftButton)  # start the recording

    qtbot.wait(int(recording_time_second * 1e3))

    # test if the data are being received
    # for ts_name in test_stream_names:
    #     assert app_main_window.stream_widgets[ts_name].viz_data_buffer.has_data()
    #
    # def handle_custom_dialog_ok(patience=0):
    #     w = QtWidgets.QApplication.activeWindow()
    #     if patience == 0:
    #         if isinstance(w, CustomDialog):
    #             yes_button = w.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
    #             qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=1000)  # delay 1 second for the data to come in
    #     else:
    #         time_started = time.time()
    #         while not isinstance(w, CustomDialog):
    #             time_waited = time.time() - time_started
    #             if time_waited > patience:
    #                 raise TimeoutError
    #             time.sleep(0.5)
    #         yes_button = w.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
    #         qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=1000)
    #
    # t = threading.Timer(1, handle_custom_dialog_ok)
    # t.start()
    #
    # t = threading.Timer(1, handle_custom_dialog_ok)
    # t.start()
    print("Stopping recording")
    qtbot.mouseClick(app_main_window.recording_tab.StartStopRecordingBtn, QtCore.Qt.LeftButton)  # stop the recording
    print("recording stopped")
    def conversion_complete():
        assert app_main_window.recording_tab.conversion_dialog.is_conversion_complete

    qtbot.waitUntil(conversion_complete, timeout=stream_availability_timeout)  # wait until the lsl processes are closed

    # t.join()  # wait until the dialog is closed
    qtbot.mouseClick(app_main_window.stop_all_btn, QtCore.Qt.LeftButton)  # stop all the streams, so we don't need to handle stream lost
    #
    print("Waiting for test stream processes to close")
    [p.kill() for p in test_stream_processes]
    qtbot.waitUntil(stream_is_unavailable, timeout=stream_availability_timeout)  # wait until the lsl processes are closed

    # reload recorded file
    saved_file_path = app_main_window.recording_tab.save_path.replace('.dats', '')
    load_csv = CsvStoreLoad()
    csv_data = load_csv.reload_csv(saved_file_path)

    def compare_column_vec(vec1, vec2, persentage):
        percentage_diff = np.abs((vec1 - vec2) / vec2) * 100
        result = np.all(percentage_diff < persentage)
        return result

    def compare(sent_sample_array, loaded_array, persentage=1):
        check = []
        sent_sample_array_trans = sent_sample_array.T
        loaded_array_trans = loaded_array.T
        for row in sent_sample_array_trans:
            check.append(compare_column_vec(row, loaded_array_trans[0], persentage))
        if np.any(check):
            start = np.where(check)[0]
            if len(start) > 1:
                raise Exception(f"Multiple start point detected")
            else:
                start_idx = int(start)
                for i in range(loaded_array.shape[1]):
                    if not compare_column_vec(sent_sample_array_trans[start_idx + i], loaded_array_trans[i], persentage):
                        return False
            return True
        else:
            return False



    for ts_name in ts_names:
        is_passing = compare(samples[ts_name], csv_data[ts_name][0], persentage=1)
        assert is_passing

    assert compare(samples['monitor 0'], csv_data[ts_name][0], persentage=1)
