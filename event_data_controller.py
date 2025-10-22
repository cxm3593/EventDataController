# EventDataController.py
# A class to manage event data. Designed for general usage.
# Author: Chengyi Ma


from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
import cv2
import numpy as np
import os
import plotly.graph_objects as go
import pandas as pd
import h5py
import time, threading, queue

class EventDataController:
    def __init__(
            self, 
            data_path: str,
            load_full_data: bool = False,
            events_iterator_mode = 'delta_t',
            events_iterator_delta_t = 10000,
            events_iterator_n_events = 100000
        ):
        """
        Initialize the EventDataController with a path to event data file.
        
        Args:
            data_path: Path to the event data file (.raw or .hdf5 format)
            
        Raises:
            ValueError: If the data path is invalid or file is not readable
        """
        self.data_path = data_path
        if not self.__validate_path():
            raise ValueError(f"Invalid data path: {data_path}")
        
        # Save iterator config and create streaming iterator
        self.events_iterator_mode = events_iterator_mode
        self.events_iterator_delta_t = events_iterator_delta_t
        self.events_iterator_n_events = events_iterator_n_events
        self.iterator = EventsIterator(
            self.data_path, 
            mode=self.events_iterator_mode, 
            delta_t=self.events_iterator_delta_t, 
            n_events=self.events_iterator_n_events
        )
        self.data_info = self.__get_data_info()
        print("EventDataController initialized successfully.")
        print("Data information:", self.data_info)

        self.data = None
        if load_full_data == True:
            ## identify the data format and load accordingly
            if self.data_path.endswith('.raw'):
                self.data = self.__load_data_raw()
            elif self.data_path.endswith('.hdf5'):
                self.data = self.__load_data_hdf5()
            else:
                raise ValueError(f"Unsupported data format: {self.data_path}")

    def __validate_path(self) -> bool:
        """
        Validate that the data path exists and is a readable file.
        
        Returns:
            bool: True if path is valid and readable, False otherwise
        """
        return os.path.isfile(self.data_path) and os.access(self.data_path, os.R_OK)
    
    def __get_data_info(self) -> dict:
        """
        Extract metadata from the event data file.
        Returns:
            dict: A dictionary containing metadata information
        """
        info = {}

        # Grab a separate iterator to avoid consuming the main one
        local_iterator = EventsIterator(self.data_path)
        
        # Get spatial resolution
        height, width = local_iterator.get_size()
        info['resolution'] = (height, width)
        
        # Iterate through events to get count, start/end times
        total_events = 0
        start_time = None
        end_time = None

        for evs in local_iterator:
            if len(evs) > 0:
                # Get start time from first batch
                if start_time is None:
                    start_time = evs['t'][0]
                # Update end time with last event in current batch
                end_time = evs['t'][-1]
                # Count events
                total_events += len(evs)
        
        info['total_events'] = total_events
        info['start_time_us'] = int(start_time) if start_time is not None else None
        info['end_time_us'] = int(end_time) if end_time is not None else None
        info['duration_us'] = int(end_time - start_time) if (start_time is not None and end_time is not None) else None
        
        return info
    
    def __load_data_raw(self) -> pd.DataFrame:
        """
        Load event data from a raw file
        """
        evs_list: list[np.ndarray] = []
        for evs in self.iterator:
            evs_list.append(evs)

        evs_array = np.concatenate(evs_list)

        data = pd.DataFrame(evs_array, columns=['x', 'y', 'p', 't'])

        return data
        

    def __load_data_hdf5(self) -> pd.DataFrame:
        """
        Load event data from an HDF5 file
        """
        file = h5py.File(self.data_path, 'r')
        events = file['CD']['events'][:]
        data = pd.DataFrame(events, columns=['x', 'y', 'p', 't'])
        
        return data
    
    def visualization_3D(self, start_time:float=None, end_time:float=None):
        """
        Visualize events as an interactive 3D point cloud using Plotly.
        
        Args:
            start_time: Start time in microseconds (default: None, uses beginning of recording)
            end_time: End time in microseconds (default: None, uses end of recording)
        """
        # Use full duration if times not specified
        if start_time is None:
            start_time = self.data_info['start_time_us']
        if end_time is None:
            end_time = self.data_info['end_time_us']
        
        # Stream events from iterator and collect within the time window
        x_on, y_on, t_on = [], [], []
        x_off, y_off, t_off = [], [], []
        for evs in self.iterator:
            if len(evs) > 0:
                mask = (evs['t'] >= start_time) & (evs['t'] <= end_time)
                filtered = evs[mask]
                if len(filtered) > 0:
                    on_m = filtered['p'] == 1
                    off_m = filtered['p'] == 0
                    x_on.extend(filtered['x'][on_m])
                    y_on.extend(filtered['y'][on_m])
                    t_on.extend(filtered['t'][on_m])
                    x_off.extend(filtered['x'][off_m])
                    y_off.extend(filtered['y'][off_m])
                    t_off.extend(filtered['t'][off_m])

        # Reset iterator for future use
        self.iterator = EventsIterator(
            self.data_path,
            mode=self.events_iterator_mode,
            delta_t=self.events_iterator_delta_t,
            n_events=self.events_iterator_n_events
        )
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add ON events (blue)
        if len(x_on) > 0:
            fig.add_trace(go.Scatter3d(
                x=x_on, y=y_on, z=t_on,
                mode='markers',
                marker=dict(size=2, color='blue', opacity=0.6),
                name='ON events'
            ))
        
        # Add OFF events (red)
        if len(x_off) > 0:
            fig.add_trace(go.Scatter3d(
                x=x_off, y=y_off, z=t_off,
                mode='markers',
                marker=dict(size=2, color='red', opacity=0.6),
                name='OFF events'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'3D Event Visualization ({start_time} - {end_time} μs)',
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)',
                zaxis_title='Time (μs)'
            ),
            showlegend=True
        )
        
        fig.show()

    def visualization_video(self, start_time=None, end_time=None, fps=30, accumulation_time_us=10_000, playback_speed=1.0):
        if start_time is None: start_time = self.data_info['start_time_us']
        if end_time   is None: end_time   = self.data_info['end_time_us']
        h, w = self.data_info['resolution']

        window = "Event Video"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        # Use callback like in the official docs
        periodic_gen = PeriodicFrameGenerationAlgorithm(
            w, h, accumulation_time_us=accumulation_time_us, fps=fps
        )

        def periodic_cb(ts_us, frame):
            cv2.putText(frame, f"Timestamp: {ts_us} us", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window, frame)

        periodic_gen.set_output_callback(periodic_cb)

        # Stream from iterator
        for evs in self.iterator:
            if len(evs) > 0:
                mask = (evs['t'] >= start_time) & (evs['t'] <= end_time)
                filtered = evs[mask]
                if len(filtered) > 0:
                    periodic_gen.process_events(filtered)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if evs['t'][-1] > end_time:
                    break

        cv2.destroyAllWindows()
        

    def generate_binary_frames(
            self, 
            saving_dir:str,
            accumulation_time_us:int=10000, 
            fps:int=30, 
            start_time:float=None,
            end_time:float=None
        ):
        """
        Generate binary event frames and save them to the specified directory.
        """
