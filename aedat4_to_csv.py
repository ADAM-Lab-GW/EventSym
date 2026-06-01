import cv2
import dv_processing as dv
import argparse
import pathlib
import pandas as pd

WIDTH = 346
HEIGHT = 260

def process_aedat4_file(input_file, output_base_dir):
    """
    Process a single AEDAT4 file and save outputs in the specified output directory.
    """
    file = pathlib.Path(input_file)
    file_stem = file.stem

    # Create output directories
    output_dir = output_base_dir / file_stem
    frames_path = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_path.mkdir(parents=True, exist_ok=True)

    # Open the recording file
    recording = dv.io.MonoCameraRecording(str(input_file))

    # Save frames if available
    if recording.isFrameStreamAvailable():
        print(f"Saving frames for {file_stem}...")
        count_frames = 0
        while True:
            frame = recording.getNextFrame()
            
            if frame is None:
                break           
            
            count_frames += 1
            cv2.imwrite(str(frames_path / (str(frame.timestamp) + '.png')), frame.image)

        print("Resolution:", WIDTH, "x", HEIGHT)
        print(f"Saved {count_frames} frames for {file_stem}.")

    #Save events with and without color if available
    if recording.isEventStreamAvailable():
        normal_events_path = output_dir / f"{file_stem}_events.csv"
        print(f"Saving events for {file_stem}...")

        events_packets = []
        while True:
            events = recording.getNextEventBatch()
            if events is None:
                break
            events_df = pd.DataFrame(events.numpy())
            events_packets.append(events_df)

        events_pandas = pd.concat(events_packets)
        # Save normal events (no color)
        events_pandas['timestamp'] = events_pandas['timestamp'] -events_pandas['timestamp'].min()

            # Step 2: remove first second (1,000,000 microseconds)
        events_pandas = events_pandas[events_pandas['timestamp'] >= 1_000_000]

        events_pandas.rename(columns={'timestamp': 't', 'polarity': 'p'}, inplace=True)
        events_pandas.to_csv(normal_events_path, index=False)
   

def process_aedat4_directory(input_dir, output_base_dir):
    """
    Recursively process all AEDAT4 files, skipping files already processed.
    """
    input_dir = pathlib.Path(input_dir).resolve()
    output_base_dir = pathlib.Path(output_base_dir).resolve()

    for file in input_dir.rglob("*.aedat4"):
        relative_subpath = file.relative_to(input_dir).parent
        file_stem = file.stem
        output_dir = output_base_dir / relative_subpath
        output_csv = output_dir / f"{file_stem}_events.csv"

        if output_csv.exists():
            print(f"Skipping already processed file: {file.relative_to(input_dir)}")
            continue

        print(f"Processing file: {file.relative_to(input_dir)}")
        process_aedat4_file(file, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process AEDAT4 files in a directory and save outputs.')
    parser.add_argument('-i', '--input_dir',
                        dest='input_dir',
                        type=str,
                        required=True,
                        help='Path to the input directory containing AEDAT4 files')
    parser.add_argument('-o', '--output_dir',
                        dest='output_dir',
                        type=str,
                        required=True,
                        help='Path to the output directory where results will be saved')

    args = parser.parse_args()

    # Process the entire directory
    process_aedat4_directory(args.input_dir, args.output_dir)


