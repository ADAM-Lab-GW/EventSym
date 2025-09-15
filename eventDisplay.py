import argparse
from datetime import timedelta
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

import dv_processing as dv

from os.path import join

from utils.event_readers import FixedSizeEventReader
from utils.inference_utils import events_to_voxel_grid
from utils.timers import Timer
from run_reconstruction_from_events import save_events_to_file
import imageio.v2 as imageio



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating event previews from event files')
    parser.add_argument('-i', '--input_dir', required=True, type=str, help='path to directory containing input CSV files')

    GIF_FRAME_INTERVAL_MS = 100     # how often slicer calls (matches your 15 ms)
    SAVE_EVERY_NTH_FRAME = 1        # use >1 to thin frames & shrink GIF
    MAX_FRAMES = None               # e.g., 2000 to cap size; None = no cap
    args = parser.parse_args()


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
        OUTPUT_GIF = Path.cwd() /"gifs"/ f"{dir_structure}_{base_filename}.gif"
        # -------------------------------

        # Open recording (your path)
        capture = dv.io.MonoCameraRecording(str(file))

        if not capture.isEventStreamAvailable():
            raise RuntimeError("Input camera does not provide an event stream.")

        visualizer = dv.visualization.EventVisualizer(capture.getEventResolution())
        visualizer.setBackgroundColor(dv.visualization.colors.white())
        visualizer.setPositiveColor(dv.visualization.colors.iniBlue())
        visualizer.setNegativeColor(dv.visualization.colors.darkGrey())


        slicer = dv.EventStreamSlicer()

        # storage for GIF frames (RGB uint8)
        gif_frames = []
        _frame_count = 0

        def slicing_callback(events: dv.EventStore):
            global _frame_count, gif_frames

            # Generate BGR frame for OpenCV
            frame_bgr = visualizer.generateImage(events)

            # Thinning / cap to keep GIF reasonable
            if (SAVE_EVERY_NTH_FRAME <= 1 or _frame_count % SAVE_EVERY_NTH_FRAME == 0) and \
            (MAX_FRAMES is None or len(gif_frames) < MAX_FRAMES):
                # Convert BGR -> RGB for imageio
                frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
                # Ensure contiguous uint8 array
                frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
                gif_frames.append(frame_rgb)

            _frame_count += 1

        # Register callback roughly every 15 ms (≈ 66 FPS)
        slicer.doEveryTimeInterval(timedelta(milliseconds=GIF_FRAME_INTERVAL_MS), slicing_callback)

        def write_gif_safe(path, frames, interval_ms):
            if not frames:
                print("No frames captured; GIF not written.")
                return
            # duration per frame in seconds
            duration = interval_ms / 1000.0
            # Write GIF (loop=0 means loop forever)
            imageio.mimsave(path, frames, duration=duration, loop=0)
            print(f"GIF saved: {path} ({len(frames)} frames, ~{int(1/duration) if duration>0 else 'N/A'} FPS)")

        try:
            while capture.isRunning():
                events = capture.getNextEventBatch()
                if events is not None:
                    slicer.accept(events)

                # Allow manual stop with 'q'
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Always attempt to write the GIF, even on Ctrl+C
            write_gif_safe(OUTPUT_GIF, gif_frames, GIF_FRAME_INTERVAL_MS)
            

                
                    
