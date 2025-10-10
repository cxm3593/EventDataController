# EventDataController.py
# A class to manage event data. Designed for general usage.
# Author: Chengyi Ma


from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
import cv2
import numpy as np
import os
import plotly.graph_objects as go

class EventDataController:
    def __init__(self, data_path: str):
        """
        Initialize the EventDataController with a path to event data file.
        
        Args:
            data_path: Path to the event data file (.raw or .dat format)
            
        Raises:
            ValueError: If the data path is invalid or file is not readable
        """
        self.data_path = data_path
        if not self.__validate_path():
            raise ValueError(f"Invalid data path: {data_path}")
        
        # Load the event data
        self.iterator = EventsIterator(self.data_path)
        self.data_info = self.__get_data_info()
        print("EventDataController initialized successfully.")
        print("Data information:", self.data_info)

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
        
        # Get spatial resolution
        height, width = self.iterator.get_size()
        info['resolution'] = (height, width)
        
        # Iterate through events to get count, start/end times
        total_events = 0
        start_time = None
        end_time = None
        
        for evs in self.iterator:
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
        
        # Reset iterator for future use
        self.iterator = EventsIterator(self.data_path)
        
        return info
    
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
        
        # Collect events within the time window
        x_on, y_on, t_on = [], [], []
        x_off, y_off, t_off = [], [], []
        
        for evs in self.iterator:
            if len(evs) > 0:
                # Filter events within time window
                mask = (evs['t'] >= start_time) & (evs['t'] <= end_time)
                filtered_evs = evs[mask]
                
                if len(filtered_evs) > 0:
                    # Separate ON and OFF events
                    on_mask = filtered_evs['p'] == 1
                    off_mask = filtered_evs['p'] == 0
                    
                    # ON events (blue)
                    x_on.extend(filtered_evs['x'][on_mask])
                    y_on.extend(filtered_evs['y'][on_mask])
                    t_on.extend(filtered_evs['t'][on_mask])
                    
                    # OFF events (red)
                    x_off.extend(filtered_evs['x'][off_mask])
                    y_off.extend(filtered_evs['y'][off_mask])
                    t_off.extend(filtered_evs['t'][off_mask])
        
        # Reset iterator
        self.iterator = EventsIterator(self.data_path)
        
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

    def visualization_video(self, start_time:float=None, end_time:float=None, fps:int=30, accumulation_time_us:int=10000):
        """
        Display accumulated event frames as a video using PeriodicFrameGenerationAlgorithm.
        
        Args:
            start_time: Start time in microseconds (default: None, uses beginning of recording)
            end_time: End time in microseconds (default: None, uses end of recording)
            fps: Frames per second for video display (default: 30)
            accumulation_time_us: Accumulation time in microseconds (default: 10000 = 10ms)
        """
        # Use full duration if times not specified
        if start_time is None:
            start_time = self.data_info['start_time_us']
        if end_time is None:
            end_time = self.data_info['end_time_us']
        
        # Get resolution
        height, width = self.data_info['resolution']
        
        # Create window for display
        window_name = "Event Video"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Define the callback function for frame display
        def periodic_cb(ts, frame):
            cv2.putText(frame, f"Timestamp: {ts} us", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)
        
        # Instantiate the frame generator with callback
        periodic_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=accumulation_time_us, fps=fps)
        periodic_gen.set_output_callback(periodic_cb)
        
        # Process events
        for evs in self.iterator:
            if len(evs) > 0:
                # Filter events within time window
                mask = (evs['t'] >= start_time) & (evs['t'] <= end_time)
                filtered_evs = evs[mask]
                
                if len(filtered_evs) > 0:
                    # Feed events to the frame generator
                    periodic_gen.process_events(filtered_evs)
                    
                    # Check for exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Stop if we've passed the end time
                if evs['t'][-1] > end_time:
                    break
        
        # Cleanup
        cv2.destroyAllWindows()
        
        # Reset iterator for future use
        self.iterator = EventsIterator(self.data_path)