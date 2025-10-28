# Event Data Controller

The project is built as a library for loading and processing data from event sensors. It is supposed to provide a simplified and lightweight interface for tasks like generating event frames, visualizations, and converting to other formats. 

**Currently only supports HDF5 data files from Prophesee's Metavision SDK**

## Description

The project is intended to be used as a utility toolbox to interact with event data. It will take a HDF5 data file handle and interact with it like a pandas Dataframe in a lazy style. 

## Getting Started

### Dependencies

Major Python packages: OpenCV Python, Plotly, Pandas, H5py, tqdm.

Metavision SDK is not required, but I am planning to make this controller interact with it in future.

### Installing

* clone the code from this github repository
* install with 
```
pip install -e .
```


## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Chengyi Ma (cxm3593@rit.edu)

## Version History


* 0.1.0
    * Project Setup

## License

