# Preparing Data

This package is designed to work with data from a variety of sources, including NWB files, MATLAB files, and HDF5 files. To streamline the process of feeding data into our models, we have developed a set of classes and utilities to help with the process of preparing the data.

The following tutorial will guide you through the process of preparing your data, and 
provides multiple examples for different types of data.


## :obj:`kirby.data.IrregularTimeSeries`

This class is used to encapsulate irregularly sampled data, such as spike times (that don't have regular sampling frequency).
The rationale is to encode the sparse timestamps and associated data (amplitude, unit-id etc.) in a dense format that the model directly ingests.

```python

IrregularTimeserSeries(
  timestamps = timestamps, # shape: (T, ), required
  values = values, # shape: (T, ), required
  waveforms = waveforms, # free to add more key-value pairs as long as its first dimension is T
)

```
The `timestamps` and `values` are required attributes of the `IrregularTimeSeries` object.
One can add their own key-value pairs to the `IrregularTimeSeries` object as long as the values have first dimension `T`, where `T` is the length of the timestamps.

This object is generally used to encode spiking activity and behavior data. Can be extended to other types of data like Calcium traces data etc.

Here are some examples of how `IrregularTimeSeries` is used in the project:

```python


spikes = IrregularTimeSeries(
	timestamps = timestamps, #(T,)
	names = names, #(T,)
	waveforms = waveforms, #(T,)
)

go_cue_time = IrregularTimeSeries(
	timestamps = timestamps_1, #(T,)
)

reward = IrregularTimeSeries(
	timestamps = timestamps_2, #(T,)
	reward_value = reward_value, #(T,)
)

video_stimuli = IrregularTimeSeries(
timestamps = timestamps_3, #(T,)
frames = frames, #(T, 3, H, W)
)

calcium = IrregularTimeSeries(
	timestamps = timestamps, #(T,)
	names = names, #(T,)
	values = values,
)

```

## {func}`kirby.data.data.Interval`

This class is simply a wrapper around two `torch.Tensor` objects, `start` and `end`, that represent the start and end times of every "trial" in a session. Since trials can have other metadata that may be important for downstream processing, the `Interval` class supports arbitrary key-value pairs.

```python

Interval(
  start= start, # required, shape: (num_trials, )
  end = end, # required, shape: (num_trials, )
  key1 = value1 # free to add more key-value pairs
)

```

Here are some examples of how `Interval` is used in the project:

```python
trials = Interval(
	start = start,
	end = end,
)

drifting_gratings = Interval(
	start = start,
	end = end,
	orientation = orientation,
	temp_freq = temp_freq,
	spatial_freq = spatial_freq,
)

image_stimuli = Interval(
	start = start,
	end = end,
	images = images,
)
```

## {func}`kirby.data.data.Data`

This class is used to encapsulate all the data for a session into a single object, much like a dictionary.
```python
data = Data(
  spikes = spikes, #IrregularTimeseries
  trials = trials, #Interval
  units = units, #Data
  subject_id = subject_id, #String
)
```

But it can also be used to represent other data like unit-metadata, session-metadata, etc.

```python

units_metadata = Data(
  names = names, # shape: (num_units, )
  area_name = area_name, # shape: (num_units, )
)
```

There are other classes, data structures, and utilities used to encode the various metadata for a session, subject, etc. These are defined in the [project-kirby/kirby/taxonomy/taxonomy.py](../../../kirby/taxonomy/taxonomy.py) file.

# Prepare data script

### Using argparse to parse command line arguments

In general, we will need to specify two directories: one that contains the raw data and one for saving the processed data. We use the `argparse` library to parse these command line arguments.

```python
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="./raw")
parser.add_argument("--output_dir", type=str, default="./processed")

args = parser.parse_args()
```

### Instantiating the DatasetBuilder

To assist with the process of preparing the data, we have developed a {class}`kirby.data.DatasetBuilder` class. First, we instantiate the `DatasetBuilder` object by providing 
various metadata about the experiment.

````{tab} template
```python
db = DatasetBuilder(
    raw_folder_path=args.input_dir,
    processed_folder_path=args.output_dir,
    experiment_name="firstAuthor_lastAuthor_firstWord_year",
    origin_version="origin_version",
    derived_version="1.0.0",
    source="https://...",
    description="...",
    )
```
````

````{tab} perich_miller_population_2018
```python
db = DatasetBuilder(
    raw_folder_path=args.input_dir,
    processed_folder_path=args.output_dir,
    experiment_name="perich_miller_population_2018",
    origin_version="dandi/000688/draft",
    derived_version="1.0.0",
    source="https://dandiarchive.org/dandiset/000688",
    description="This dataset contains electrophysiology and behavioral data from...",
    )
```
````


## Load data from files

````{tab} .nwb
```python
from pynwb import NWBHDF5IO

io = NWBHDF5IO(file_path, "r")
nwbfile = io.read()
... # do stuff with nwbfile
io.close()
```
````

````{tab} .mat
```python
from scipy.io import loadmat

mat = loadmat(file_path)
```
````



````{tab} .h5
```python
import h5py

h5file = h5py.File(file_path, "r")
... # do stuff with h5file
h5file.close()
```
````

## Sessions


Usually, we want to process multiple raw data files corresponding to different sessions.
The helper `find_files_by_extension` in {func}`kirby.utils.dir_utils` will return a list of files with the given extension in the given directory.
For each session, the `db.new_session()` returns a `SessionContextManager` used to register the various information about the data.

## Register subject of experiment

Create a `SubjectDescription` object with the subject's information.
  ```python
  subject = SubjectDescription(
    id=...,
    species=...,
    sex=...
  )
  ```
```python
# this dataset is from dandi, which has structured subject metadata, so we
# can use the helper function extract_subject_from_nwb
subject = extract_subject_from_nwb(nwbfile)
session.register_subject(subject)
```

Register the subject using `session.register_subject(subject)`

## Register Session metadata
  - Create a `session_id` that uniquely identifies the session.
  - Register session with context manager by providing session metadata.
  ```python
  session.register_session(
    id=session_id,
    recording_date=...,
    task=...,
    fields={...}
  )
  ```
## Extracting the data
Define the following methods to extract the data:
- `extract_units()` : Extracts all unit metadata and packages it into a `Data` object of the form:
    ```python
    Data(
      unit_name=["unit_1", "unit_2", ...], # required
      unit_index=[0, 1, ...], # required
      ... # other unit metadata like recording tech etc.
    )
    ```
- `extract_spikes()` : Extracts all spike data and returns an `IrregularTimeSeries` object of the form:
    ```python
    IrregularTimeSeries(
      timestamps=np.array([...]), # required
      unit_index=np.array([0, 1, ...]), # required
      ... # other spike data like waveforms etc.
    )
    ```
- `extract_behavior()` : Extracts all behavior data and returns an `IrregularTimeSeries` object of the form:
    ```python
    IrregularTimeSeries(
      timestamps=np.array([...]), # required
      behavior=np.array([...]) # required, other behavior data like position coordinates, velocity, etc.
    )
    ```
- `extract_trials()` : Extracts details about `(start, end)` for each trial in a session and returns it into an `Interval` object of the form:
    ```python
    Interval(
      start=np.array([...]), # required
      end=np.array([...]), # required
      ... # other trial data like trial type, etc.
    )
    ```


Create one single instance of class `Data` that contains all the info extracted so far for the session:
  ```python
  data = Data(
    units=extract_units(),
    spikes=extract_spikes(),
    behavior=extract_behavior(),
    trials=extract_trials(),
    start=session_start_time,
    end=session_end_time
  )
  ```



<!-- The first code block in a tab content will "join" with the tabs, making things f
fairly clean for language snippets and OS-based command suggestions.

````{tab} Python
```python
print("Hello World!")
```

It's pretty simple!
````

````{tab} C++
```cpp
#include <iostream>

int main() {
  std::cout << "Hello World!" << std::endl;
}
```


More code, but it works too!
````

````{tab} Text
```none
Hello World!
```


Why not.
```` -->