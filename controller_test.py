# A testing scrpt for EventDataController class
# Author: Chengyi Ma

import os

from event_data_controller import EventDataController

data_dir = "C:\\Users\\cxm3593\\Academic\\Workspace\\Data\\Calib"
# data_path = os.path.join(data_dir, "calib_recording.raw")
data_path = os.path.join(data_dir, "calib_recording.hdf5")

controller = EventDataController(data_path)
start_time = controller.data_info['start_time_us']
end_time = start_time + (1 * 1e6)  # 5 seconds later

# 3D Visualization
# controller.visualization_3D(start_time=start_time, end_time=end_time)

# Video Visualization
controller.visualization_video()