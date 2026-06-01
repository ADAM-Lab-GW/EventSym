import torch
from utils.loading_utils import load_model, get_device
import argparse
import pandas as pd
from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
import os
import os



def save_events_to_file(csv_file, output_file):
    # Open the AEDAT4 file
    print("check2", csv_file)
    df = pd.read_csv(csv_file)

    with open(output_file, 'w') as out:
            # Write the header line
            Width = df['x'].max()+1
            Height = df['y'].max()+1
            
             # Ensure Width is even
            if Width % 2 != 0:
                Width += 1

            # Ensure Height is even
            if Height % 2 != 0:
                Height += 1
            out.write(str(Width)+" "+str(Height)+"\n")
            
            # Iterate through packets if needed or directly through events
            for _, row in df.iterrows():
                polarity_flag = "False" if row["p"] == 0 else "True"
                out.write(f"{row['t']} {row['x']} {row['y']} {polarity_flag}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', required=True, type=str, help='path to model weights')
    parser.add_argument('-i', '--input_dir', required=True, type=str, help='path to directory containing input CSV files')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='path to save output folders')
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=False)
    parser.add_argument('-N', '--window_size', default=15000, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--window_duration', default=33.33, type=float,
                        help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float)
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=True)

    set_inference_options(parser)
    args = parser.parse_args()

    input_dir_abs = os.path.abspath(args.input_dir)
    output_dir_abs = os.path.abspath(args.output_dir)

    csv_files = []
    for root, _, files in os.walk(input_dir_abs):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    

    log_file_path = os.path.join(output_dir_abs, "processed_files.log")
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            processed_files = set(line.strip() for line in log_file)
    else:
        processed_files = set()

    for csv_file_path in sorted(csv_files):
        relative_path = os.path.relpath(csv_file_path, input_dir_abs)
        if relative_path in processed_files:
            print(f"Skipping already processed file: {relative_path}")
            continue

        stripped_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
        output_subfolder = os.path.join(output_dir_abs, os.path.dirname(relative_path), stripped_filename)
        os.makedirs(output_subfolder, exist_ok=True)

        print("Processing:", stripped_filename)
        output_file_path = os.path.join(output_subfolder, f"{stripped_filename}.txt")
        save_events_to_file(csv_file_path, output_file_path)

        path_to_events = output_file_path
        header = pd.read_csv(path_to_events, delim_whitespace=True, header=None, names=['width', 'height'],
                             dtype={'width': int, 'height': int}, nrows=1)
        width, height = header.values[0]
        args.width = width
        args.height = height

        print('Sensor size: {} x {}'.format(width, height))
        with open(path_to_events, 'r') as file:
            total_number_of_events = sum(1 for _ in file) - 1
        print('total_number_of_events:', total_number_of_events)

        model = load_model(args.path_to_model)
        device = get_device(args.use_gpu)
        model = model.to(device)
        model.eval()

        args.output_dir = output_subfolder
        args.dataset_name = stripped_filename

        reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

        N = args.window_size
        if not args.fixed_duration:
            if N is None:
                N = int(width * height * args.num_events_per_pixel)
                print(f'Will use {N} events per tensor (estimated).')
            else:
                print(f'Will use {N} events per tensor (user-specified).')

        start_index = args.skipevents + args.suboffset

        if args.fixed_duration:
            event_window_iterator = FixedDurationEventReader(path_to_events, duration_ms=args.window_duration, start_index=start_index)
        else:
            event_window_iterator = FixedSizeEventReader(path_to_events, num_events=N, start_index=start_index)

        with Timer('Processing entire dataset'):
            for event_window in event_window_iterator:
                last_timestamp = event_window[-1, 0]

                with Timer('Building event tensor'):
                    if args.compute_voxel_grid_on_cpu:
                        event_tensor = events_to_voxel_grid(event_window, num_bins=model.num_bins, width=width, height=height)
                        event_tensor = torch.from_numpy(event_tensor)
                    else:
                        event_tensor = events_to_voxel_grid_pytorch(event_window, num_bins=model.num_bins, width=width, height=height, device=device)

                num_events_in_window = event_window.shape[0]
                reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)
                start_index += num_events_in_window
        with open(log_file_path, "a") as log_file:
            log_file.write(relative_path + "\n")



