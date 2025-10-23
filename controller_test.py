# A testing scrpt for EventDataController class
# Author: Chengyi Ma

import os

from event_data_controller import EventDataController

data_dir = "C:\\Users\\cxm3593\\Academic\\Workspace\\Data\\Calib"
# data_path = os.path.join(data_dir, "calib_recording.raw")
data_path = os.path.join(data_dir, "calib_recording.hdf5")

controller = EventDataController(data_path, (1280, 720))

print(f"Data info: {controller.data_info}")