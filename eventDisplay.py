import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
import torch
import dv_processing as dv
import os
from os.path import join

from utils.event_readers import FixedSizeEventReader
from utils.inference_utils import events_to_voxel_grid
from utils.timers import Timer
from run_reconstruction_from_events import save_events_to_file
import imageio.v2 as imageio
import stat
 

def handle_error(func, path, exc_info):
    # Change the permission of the file/folder to allow deletion
    os.chmod(path, stat.S_IWRITE)
    func(path)

def create_gif(image_folder, output_gif, fps=0.5):
        """
        Creates a GIF from images in the specified folder.

        :param image_folder: Path to the folder containing images.
        :param output_gif: Path for the output GIF file.
        :param duration: Duration per frame in seconds.
        """
        images = []
        for file_name in sorted(os.listdir(image_folder)):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Adjust for needed formats
                file_path = os.path.join(image_folder, file_name)
                print(file_path)
                images.append(imageio.imread(file_path))

        if images:
            imageio.mimsave(output_gif, images, fps=fps)
            print(f"GIF saved at {output_gif}")
        else:
            print("No images found to create GIF.")

def make_event_preview(events, mode='red-blue', num_bins_to_show=-1):
    # events: [1 x C x H x W] event tensor
    # mode: 'red-blue' or 'grayscale'
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    assert(mode in ['red-blue', 'grayscale'])

    if num_bins_to_show < 0:
        sum_events = torch.sum(events[0, :, :, :], dim=0).detach().cpu().numpy()
    else:
        sum_events = torch.sum(events[0, -num_bins_to_show:, :, :], dim=0).detach().cpu().numpy()

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.full((sum_events.shape[0], sum_events.shape[1], 3), 255, dtype=np.uint8)

        pos_mask = sum_events > 0
        neg_mask = sum_events < 0

        # Set positive events to blue
        event_preview[pos_mask] = [0, 0, 255]

        # Set negative events to red
        event_preview[neg_mask] = [255, 0, 0]

    else:
        # Grayscale mode
        m, M = -10.0, 10.0
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)).astype(np.uint8), 0, 255)

    return event_preview



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating event previews from event files')
    parser.add_argument('-i', '--input_dir', required=True, type=str, help='path to directory containing input CSV files')

    args = parser.parse_args()

    folder_name = "events"

    # Create the folder in the current working directory
    output_path = Path.cwd() / folder_name

    if os.path.exists(output_path):
        shutil.rmtree(folder_name, onerror=handle_error)

    output_path.mkdir(parents=True, exist_ok=True)

    output_path_gifs = Path.cwd() / "gifs"
    output_path_gifs.mkdir(parents=True, exist_ok=True)
    

    input_dir = Path.cwd()/ args.input_dir

    print(f"Processing AEDAT4 files in directory: {input_dir}")
    for file in input_dir.rglob("*.aedat4"):
        print(f"Processing file: {file.relative_to(input_dir)}")

        # Get the relative path from the root directory (excluding 'files/')
        relative_path = file.relative_to(input_dir)
        
        # Extract the directory names and the base file name (without extension)
        dir_structure = "_".join(relative_path.parts[:-1])  # all parts except the last file part
        base_filename = relative_path.stem  # get the file name without extension
        
        # Build the output CSV filename
        output_csv = Path.cwd() / f"events.csv"

        # Open the recording file
        recording = dv.io.MonoCameraRecording(str(file))

        #Save events with and without color if available
        if recording.isEventStreamAvailable():
            events_packets = []
            while True:
                events = recording.getNextEventBatch()
                if events is None:
                    break
                events_df = pd.DataFrame(events.numpy())
                events_packets.append(events_df)

            events_pandas = pd.concat(events_packets)
            
            N= events_pandas.shape[0]

            # Save normal events (no color)
            events_pandas['timestamp'] = events_pandas['timestamp'] -events_pandas['timestamp'].min()
            events_pandas = events_pandas[events_pandas['timestamp'] >= 1000000]

            events_pandas.rename(columns={'timestamp': 't', 'polarity': 'p'}, inplace=True)
            events_pandas.to_csv(output_csv, index=False)

            output_txt = Path.cwd() / "events.txt"
            save_events_to_file(output_csv, output_txt)

            start_index = 0
            event_window_iterator = FixedSizeEventReader(output_txt, num_events=15000, start_index=start_index)

            with Timer('Processing entire dataset'):
                for event_window in event_window_iterator:
                    last_timestamp = event_window[-1, 0]

                    with Timer('Building event tensor'):
                        
                        event_tensor = events_to_voxel_grid(event_window, num_bins=5, width=346, height=260)
                        event_tensor = torch.from_numpy(event_tensor)
                        

                        num_events_in_window = event_window.shape[0]
                        with torch.no_grad():

                            #with CudaTimer('Reconstruction'):
                            with Timer('Reconstruction'):

                                with Timer('NumPy (CPU) -> Tensor (GPU)'):
                                    events = event_tensor.unsqueeze(dim=0)
                                    events = events.to("cpu")

                                with Timer('Normalization'):
                                    nonzero_ev = (events != 0)
                                    num_nonzeros = nonzero_ev.sum()
                                    if num_nonzeros > 0:
                                        # compute mean and stddev of the **nonzero** elements of the event tensor
                                        # we do not use PyTorch's default mean() and std() functions since it's faster
                                        # to compute it by hand than applying those funcs to a masked array
                                        mean = events.sum() / num_nonzeros
                                        stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
                                        mask = nonzero_ev.float()
                                        events = mask * (events - mean) / stddev

                                event_preview = make_event_preview(events, mode='red-blue', num_bins_to_show=-1)
                                start_index += num_events_in_window
                                cv2.imwrite(join(output_path,'events _{:010d}.png'.format(start_index)), event_preview)
            output_gif = Path.cwd() /"gifs"/ f"{dir_structure}_{base_filename}.gif"
            
            create_gif(output_path, output_gif, fps=30)
