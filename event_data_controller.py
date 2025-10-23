# EventDataController.py
# A class to manage event data. Designed for general usage.
# Author: Chengyi Ma


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
            data_resolution: tuple,
            load_full_data: bool = False,
            events_iterator_mode = 'delta_t',
            events_iterator_delta_t = 10000,
            events_iterator_n_events = 100000
        ):
        """
        Initialize the EventDataController with a path to event data file.
        
        Args:
            data_path (str): Path to the event data file (HDF5 format)
            data_resolution (tuple): Resolution of the event data (width, height)
            load_full_data (bool): Whether to load the full dataset into memory
            events_iterator_mode (str): Mode for event iteration ('delta_t' or 'n_events')
            events_iterator_delta_t (int): Time window for delta_t mode (in microseconds)
            events_iterator_n_events (int): Number of events for n_events mode
            
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

        # load hdf5 handle
        self.data_hdf5 = h5py.File(self.data_path, 'r')
        self.events_handle = self.data_hdf5['CD']['events'] # By default
        self.data_resolution = data_resolution
        self.data_info = self.__compute_data_info()
        print("EventDataController initialized successfully.")

        self.data = None
        if load_full_data == True:
            ## identify the data format and load accordingly
            if self.data_path.endswith('.hdf5'):
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
    

    def __compute_data_info(self) -> pd.DataFrame:
        """
        Compute basic information about the event data
        """
        data_info = {}
        data_info['num_events'] = self.events_handle.shape[0]
        data_info['start_time_us'] = self.events_handle['t'].min()
        data_info['end_time_us'] = self.events_handle['t'].max()
        data_info['duration_us'] = data_info['end_time_us'] - data_info['start_time_us']

        data_info['num_indexes'] = self.data_hdf5['CD']['indexes'].shape[0]

        data_info['ext_triggers'] = self.data_hdf5['EXT_TRIGGER']['events'].shape[0]

        return data_info

        

    def __load_data_hdf5(self) -> pd.DataFrame:
        """
        Load event data from an HDF5 file
        Returns:
            pd.DataFrame: DataFrame containing event data with columns ['x', 'y', 'p', 't']
        """
        file = h5py.File(self.data_path, 'r')
        events = file['CD']['events'][:]
        data = pd.DataFrame(events, columns=['x', 'y', 'p', 't'])
        
        return data
    
    def generate_frames(
            self, 
            path:str, 
            start_time_us:int, 
            end_time_us:int,
            accumulation_mode:str = 'fixed_time',
            accumulation_time_us:int = 10000,
            accumulation_n_events:int = 5000
        ):
        '''
        Generate frames from event data
        Args:
            path: Path to save the generated frames
            start_time_us: Start time for frame generation (in microseconds)
            end_time_us: End time for frame generation (in microseconds)
            accumulation_mode: Mode of accumulation ('fixed_time' or 'n_events')
            accumulation_time_us: Time window for accumulation (in microseconds)
            accumulation_n_events: Number of events for accumulation

        Returns:
            None
        '''

        # Validate output path
        if not os.path.exists(path):
            os.makedirs(path)
        elif not os.path.isdir(path):
            raise ValueError(f"Output path is not a directory: {path}")
        
        # Filter events within the specified time range
        events = self.events_handle
        events_in_range = events[(events['t'] >= start_time_us) & (events['t'] <= end_time_us)]
        current_time = start_time_us
        frame_idx = 0

        # Iteratively select events and then generate frames
        while current_time < end_time_us:

            if accumulation_mode == 'fixed_time':
                frame_start_time = current_time
                frame_end_time = current_time + accumulation_time_us

                frame_events = events_in_range[(events_in_range['t'] >= frame_start_time) & (events_in_range['t'] < frame_end_time)]
            
            elif accumulation_mode == 'n_events':
                raise NotImplementedError("Accumulation by number of events is not implemented yet.")
            
            # Create frame from frame_events

            # update current time and frame index
            current_time += accumulation_time_us
            frame_idx += 1
