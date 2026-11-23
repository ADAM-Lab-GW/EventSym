# EventSym: A Hierarchical Event-Based Symbol Dataset

EventSym is a hierarchical event-based vision dataset designed to support research on symbol recognition and spatial scalability. Using RGB reconstructions of the scaled down, we demonstrate the scalable nature of symbols, which makes them optimal for edge devices and small networks. We also evaluate the impact of reduction of size on continual learning, however that is not our novel contribution (to know more about the CL experiments, please visit https://github.com/VadymV/events_lifelong_learning.git as we simply use their pipeline on our data, more details can be made available upon request ). 

The dataset contains standardized symbols captured using an event-based camera, including categories such as traffic signs, hazard labels, universal symbols, and wayfinding signs. These symbols are particularly suitable for event-based vision research because they are typically high-contrast, edge-rich, and semantically meaningful, making them useful for studying recognition under spatial compression and hardware-constrained settings.

The dataset is organized hierarchically, with each sample belonging to a main symbol category and a corresponding subclass. In addition to the original-resolution event streams, EventSym includes spatially scaled-down versions generated using biologically inspired foveation. These scaled versions make the dataset suitable for benchmarking how event-based representations behave under reduced spatial resolution, and for evaluating trade-offs between accuracy, memory footprint, and computational efficiency.

This repository provides the code and instructions needed to reproduce the main dataset processing and experimental pipeline:

1. Recording event-based symbol data
2. Scaling down event streams
3. Reconstructing RGB frames from events

---

## Requirements and Prerequisites

Before running the repository, ensure that the following requirements are satisfied.

### Python Version

This repository requires:

```text
Python version 3.11.5 
```

Using versions outher than this may result in dependency or compatibility issues.

It is strongly recommended to create a dedicated virtual environment before installing dependencies.

### Create a Virtual Environment

Using `venv`:

```bash
python -m venv eventsym_env
```

Activate the environment:

Linux/macOS:

```bash
source eventsym_env/bin/activate
```

Windows:

```bash
eventsym_env\Scripts\activate
```

### Install Dependencies

Install all required packages using:

```bash
pip install -r requirements_py311.txt
```

We also need to install  GTK bundle for windows or linux or mac

For windows:
Download the GTK runtime (which includes Cairo) from:
 👉 https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases


Install it. It usually goes to:
C:\Program Files\GTK3-Runtime Win64\


Add the bin folder to your system's PATH:
C:\Program Files\GTK3-Runtime Win64\bin

Restart your terminal or IDE and re-run your script.

If added to path but restarting terminal does not help, restart PC

### Verify Python Version

To verify the installed Python version:

```bash
python --version
```

### Optional GPU Support

Some reconstruction and continual learning experiments may optionally use GPU acceleration through CUDA-enabled PyTorch installations. Ensure that the installed PyTorch version is compatible with the local CUDA version if GPU execution is required.


### Download DV GUI

Visit https://docs.inivation.com/software/dv/gui/index.html and follow the installation instructions to install DV-GUI, which comes in handy to ensure if the camera is properly focusing or not.
---

## Repository Overview

```text
<EventSym>/
├── <recordings>/                 # Scripts/tools for collecting event-based recordings
├── <input_files>                 # Folder containing images to be displayed on screen
├── <outputs>/                    # Generated outputs, logs, reconstructions, checkpoints
└── README.md
```

---

## Dataset Description

EventSym consists of event streams recorded from standardized visual symbols. Each recording represents asynchronous events of the form:

```text
t, x, y, p
```

where:

* `t` is the event timestamp,
* `x` and `y` are spatial pixel coordinates,
* `p` is the event polarity.

The original recordings are captured at the native resolution of the event camera. Scaled versions are produced by applying foveation to the event coordinates while preserving the temporal structure of the stream.

A typical dataset structure is:

```text
EventSym/
├── original_resolution/
│   ├── <main_class>/
│   │   ├── <subclass>/
│   │   │   ├── 1_events.csv
│   │   │   ├── 2_events.csv
│   │   │   └── ...
├── scaled_x2/
├── scaled_x5/
├── scaled_x7/
└── scaled_x9/
```

Each scaled folder follows the same hierarchical class/subclass structure as the original-resolution data.

---

## 1. Recording Event-Based Symbol Data

EventSym recordings are collected using an event-based camera. Each visual symbol is displayed or presented to the camera, and the resulting stream of asynchronous events is saved in AEDAT4 format or converted into CSV format for downstream processing.

### Recording Workflow

1. Download the symbol image from the internet or collect the images from a reliable source.
2. Position the event camera so that the symbol is clearly visible, preferably using the DV software.
3. Record the event stream using the recording script.


## 0. Downloading images
1. Ensure that you are downloading the correct images, given a subclass
2. Organize the data hierarchically. All scripts in this repository expect a main class -> subclass -> sample structure.
3. For the following examples, it is assumed that the root folder containing images is input_files

## 1. Recording in AEDAT4 files
### Example Command

```bash
cd recordings
python synchronized_display_record.py --base_path input_files --output_dir aedat4_recordings --display_time 2.0
```
where 
--base_path = root folder containing images to be displayed
--output_dir = output folder containing aedat4 recordings. Will be created inside current directory if it doesn't exist.
--display_time = time the image will be displayed on the monitor. 

Simultaneously, a mechanism to move the camera continuously should be started. We used a camera glider which is contollable through a phone app.

## 2. Converting AEDAT4 files to CSVs and Frame Extraction

For easier processing, `.aedat4` files recorded from a DAVIS/DVS event camera is converted into CSV files. If frame data is available in the recording, it also extracts and saves the frames as `.png` images.

### What the Script Does

For each `.aedat4` file, the script:

1. Opens the recording using `dv-processing`.
2. Checks whether a frame stream is available.
3. Saves available frames as `.png` images using their timestamps as filenames.
4. Checks whether an event stream is available.
5. Extracts all event batches and combines them into a single table.
6. Normalizes timestamps so that each recording starts from `t = 0`.
7. Removes the first second of events to avoid unstable/noisy startup activity.
8. Renames event columns:

   * `timestamp` → `t`
   * `polarity` → `p`
9. Saves the processed events as a CSV file.

### Input

The input should be a directory containing one or more `.aedat4` files. The script searches recursively, so files inside subfolders will also be processed.

Example input structure:

```text
aedat4_recordings/
├── hazard_symbols/
│   ├── flammable/
│   │   ├── 1.aedat4
│   │   └── 2.aedat4
│   └── toxic/
│       └── 1.aedat4
```

### Output

For each `.aedat4` file, the script creates an output folder and saves:

```text
original_resolution/
├── hazard_symbols/
│   ├── flammable/
│   │   ├── 1/
│   │   │   ├── frames/
│   │   │   │   ├── 123456.png
│   │   │   │   └── 123789.png
│   │   │   └── 1_events.csv
```

The event CSV contains columns such as:

```text
t,x,y,p
```

where:

* `t` is the normalized timestamp in microseconds
* `x` is the event x-coordinate
* `y` is the event y-coordinate
* `p` is the event polarity

### Usage

Run the script from the command line:

```bash
python process_aedat4.py -i aedat4_recordings -o csvs/original_resolution
```
where 
-i = input folder where aedat4 files are stored
-o= output folder containing new csv files. Will be created inside current directory if it doesn't exist.

### Notes

* The default camera resolution is set as `346 × 260`, matching the DAVIS346 sensor.
* The first `1,000,000` microseconds, or 1 second, of events are removed from each file.
* Frame extraction only runs if the `.aedat4` file contains a frame stream.
* Event extraction only runs if the `.aedat4` file contains an event stream.

---

## 3. Scaling Down Event Streams

Spatial downscaling is applied directly to event coordinates. The goal is to reduce spatial resolution while preserving the temporal event structure. This repository supports scaled versions of the dataset, such as `x2`, `x5`, `x7`, and `x9`.

The scaling process can be implemented using foveation-inspired spatial remapping, where the central region is preserved with higher fidelity while peripheral regions are compressed more aggressively. This simulates a retina-like representation and allows the dataset to be used for studying the trade-off between spatial compression and recognition accuracy.

### Input

expected and example input structure:

```text
original_resolution/
├── hazard_symbols/
│   ├── flammable/
│   │   ├── 1_events.csv
│   │   └── 2_events.csv
│   └── toxic/
├── traffic_symbols/
```

### Output

```text
output_files/
├── hazard_symbols/
│   ├── flammable/
│   │   └── factor_x5/
│   │       ├── 1_events_foveated_events.csv
│   │       └── 2_events_foveated_events.csv
```
### How the Foveation Works

The algorithm creates a non-linear coordinate mapping:

1. Central pixels remain mostly unchanged
2. Peripheral pixels are compressed
3. Spatial density decreases with distance from the center

The transformation is controlled using:
| Parameter | Description                       |
| --------- | --------------------------------- |
| `radius`  | Size of high-detail foveal region |
| `alpha`   | Peripheral compression strength   |

### Supported Scaling Factors:
| Factor | Output Resolution | Radius | Alpha |
| ------ | ----------------- | ------ | ----- |
| 1      | 345×259           | 70     | 5.0   |
| 2      | 173×130           | 40     | 2.0   |
| 5      | 69×52             | 2      | 1.1   |
| 6      | 58×43             | 2      | 1.1   |
| 7      | 49×37             | 1.8    | 1.1   |
| 8      | 43×32             | 1.6    | 1.1   |
| 9      | 38×29             | 1.4    | 1.1   |
| 10     | 35×26             | 1      | 1.0   |

** these values for radius and alpha were fixed through trial and error and analyzing the resultant RGB reconstructions of the scaled versions.


### Example Command

```bash
python3 scale_events.py --input_dir csvs/original_resolution --output_dir csvs/scaled -f 2

```
where 
--input_dir = input folder containing csv files of original resolution
--output_dir = folder to save processed outputs
--factor = Scaling factor (1, 2, 5–10)  
---

## 3. RGB Reconstruction from Events

RGB or grayscale reconstructions are generated from event streams to visually inspect the quality of the recorded and scaled event data. These reconstructions are useful for validating whether the semantic content of each symbol remains recognizable after downscaling.

The pipeline:

1. Reads event streams from CSV files
2. Converts events into voxel grids
3. Runs inference using a pretrained E2VID model
4. Generates reconstructed image frames
5. Saves outputs into organized directories

The implementation supports recursive dataset processing and resumable execution using log files.

Based on: E2VID (https://github.com/uzh-rpg/rpg_e2vid.git)

### Input

Example input dataset:

```text
test/
├── hazard_symbols/
│   ├── flammable/
│   │   ├── 1_events.csv
│   │   └── 2_events.csv
│   └── toxic/
└── traffic_symbols/
```

The script recursively searches for all `.csv` files.

---

### Output:

Example generated output:

```text
test/
├── hazard_symbols/
│   ├── flammable/
│   │   ├── 1_events/
│   │   │   ├── events/
│   │   │   ├── frames_events/
│   │   │   └── sample1_events.txt
│   │   │   └── frame_*.png
```

Each CSV file gets its own reconstruction subfolder.

---


### Model Weights

Download or place pretrained E2VID weights in:

```text
pretrained/
```

Example:

```text
pretrained/E2VID_lightweight.pth.tar
```

---


### Command Arguments

| Argument                      | Description                                              |
| ----------------------------- | -------------------------------------------------------- |
| `-c`                          | Path to pretrained E2VID model                           |
| `-i`                          | Input directory containing CSV event files               |
| `-o`                          | Output directory                                         |
| `-N`                          | Number of events per reconstruction window (default= 15000)               |
| `-T`                          | Window duration in milliseconds                          |
| `--fixed_duration`            | Use duration-based windows instead of fixed event counts |
| `--auto_hdr`                  | Enable automatic HDR normalization                       |
| `--show_events`               | Display event visualization during processing            |
| `--compute_voxel_grid_on_cpu` | Build voxel grids on CPU                                 |

---


Example:

```bash
 python run_reconstruction_from_events.py -c pretrained\E2VID_lightweight.pth.tar -i csvs/scaled -o reconstructions --auto_hdr --show_events       
```

The same reconstruction process can be repeated for each scaled version of the dataset.

---


## 4. Removing the 86 Hz Component (Optional)

Some recordings contain a weak 86 Hz modulation of the event rate (Appendix A.3 of the paper). It does not affect the reported results, and the released recordings are unmodified. `thinning/thin_86hz.py` removes it in the same way as in the paper, for a single file, a subclass, a category or the whole dataset, at any scale. See [thinning/README.md](thinning/README.md) for usage.
