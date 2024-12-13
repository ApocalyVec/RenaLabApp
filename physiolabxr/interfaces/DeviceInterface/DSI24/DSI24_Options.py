from physiolabxr.ui.BaseDeviceOptions import BaseDeviceOptions
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QProgressBar
from physiolabxr.configs.configs import AppConfigs
class DSI24_Options(BaseDeviceOptions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Impedance = 0
        self.check_impedance_chkbx.clicked.connect(self.check_impedance_chkbx_clicked)
        self.device_interface.battery_level
        self.batteryBar1.setValue(0)  # Set the value of the first progress bar
        self.batteryBar2.setValue(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.batteryUpdate)
        self.timer.start(5000)
        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(self.updateImpedance)
        self.timer2.start(1000)
        self.impedanceDictionary = {
            0: self.FP1,
            1: self.FP2,
            2: self.Fz,
            3: self.F3,
            4: self.F4,
            5: self.F7,
            6: self.F8,
            7: self.Cz,
            8: self.C3,
            9: self.C4,
            10: self.T3,
            11: self.T4,
            12: self.T5,
            13: self.T6,
            14: self.P3,
            15: self.P4,
            16: self.O1,
            17: self.O2,
            18: self.A1,
            19: self.A2,
        }
    def check_impedance_chkbx_clicked(self):
        # This method will be called when the checkbox is clicked
        if self.check_impedance_chkbx.isChecked():
            print("Check Impedance is checked.")
            self.Impedance = 1

            # Add logic to handle the case when the checkbox is checked
        else:
            print("Check Impedance is unchecked.")
            self.Impedance = 0
            
    def updateImpedance(self):
        if self.Impedance == 1:
            if self.device_interface.impedanceValues:
                impedanceValues = self.device_interface.impedanceValues
                for i in range(20):
                    if float(impedanceValues[i][0]) < 1:
                        color = 'green'
                    elif float(impedanceValues[i][0]) < 10 and float(impedanceValues[i][0]) > 1:
                        color = 'yellow'
                    else:
                        color = 'red'
                    self.impedanceDictionary[i].setStyleSheet(f"""background-color: {color}; color: black""")

        
    def batteryUpdate(self):
        if self.device_interface.battery_level != None:
            self.batteryBar1.setValue(self.device_interface.battery_level[0])
            self.batteryBar2.setValue(self.device_interface.battery_level[1])
    def start_stream_args(self):
        return {
            'bluetooth_port': self.device_port_lineedit.text(),
            'impedance': self.Impedance
                }
