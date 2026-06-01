import numpy as np
import pandas as pd
import time
import argparse
import os
import math

def generate_mappings(source_width, source_height, output_width, output_height, p0_radius, alpha=1.4):
    source_center_x, source_center_y = source_width // 2, source_height // 2
    output_center_x, output_center_y = output_width // 2, output_height // 2

    mappingX = np.zeros((output_height, output_width), dtype=np.int32)
    mappingY = np.zeros((output_height, output_width), dtype=np.int32)

    for r in range(output_height):
        for c in range(output_width):
            dx = c - output_center_x
            dy = r - output_center_y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance < p0_radius:
                map_x = source_center_x + dx
                map_y = source_center_y + dy
            else:
                angle = math.atan2(dy, dx)
                scaled_dist = p0_radius + (distance - p0_radius) * (alpha ** ((distance - p0_radius) / p0_radius))
                map_x = int(round(source_center_x + scaled_dist * np.cos(angle)))
                map_y = int(round(source_center_y + scaled_dist * np.sin(angle)))

                # Fix: clip to valid range
                map_x = np.clip(map_x, 0, source_width - 1)
                map_y = np.clip(map_y, 0, source_height - 1)


            mappingX[r, c] = map_x
            mappingY[r, c] = map_y

    maxShiftX = (source_width // 2) - (output_width // 2) - 1
    maxShiftY = (source_height // 2) - (output_height // 2) - 1

    return mappingX.flatten(), mappingY.flatten(), maxShiftX, maxShiftY


def getFov(mappingX, mappingY, source_width, source_height, output_width, output_height, maxShiftX, maxShiftY):
    minShiftX = -maxShiftX
    minShiftY = -maxShiftY
    maxValX = source_width - 1
    maxValY = source_height - 1
    zdim = output_width * output_height

    shift_range_x = int(maxShiftX - minShiftX + 1)
    shift_range_y = int(maxShiftY - minShiftY + 1)

    cache = np.zeros((shift_range_y, shift_range_x, 3, zdim), dtype='uint32')
    arr_len = np.zeros((shift_range_y, shift_range_x), dtype='uint32')
    ind_len = np.zeros((shift_range_y, shift_range_x), dtype='uint32')

    for x in range(int(minShiftX), int(maxShiftX) + 1):
        for y in range(int(minShiftY), int(maxShiftY) + 1):
            mX = np.copy(mappingX) + x
            mY = np.copy(mappingY) + y
            ab = np.where((mX >= 0) & (mY >= 0) & (mX <= maxValX) & (mY <= maxValY))
            mX_valid = mX[ab]
            mY_valid = mY[ab]
            cache_idx_x = int(x - minShiftX)
            cache_idx_y = int(y - minShiftY)
            ind_len[cache_idx_y][cache_idx_x] = len(ab[0])
            arr_len[cache_idx_y][cache_idx_x] = len(mX_valid)
            cache[cache_idx_y][cache_idx_x][0][:arr_len[cache_idx_y][cache_idx_x]] = mX_valid
            cache[cache_idx_y][cache_idx_x][1][:arr_len[cache_idx_y][cache_idx_x]] = mY_valid
            cache[cache_idx_y][cache_idx_x][2][:ind_len[cache_idx_y][cache_idx_x]] = ab[0]

    def process(im, fovX=0, fovY=0):
        fovX_clamped = min(max(fovX, minShiftX), maxShiftX)
        fovY_clamped = min(max(fovY, minShiftY), maxShiftY)
        cache_idx_x = int(fovX_clamped - minShiftX)
        cache_idx_y = int(fovY_clamped - minShiftY)
        channels = im.shape[-1] if len(im.shape) > 2 else 1
        fov = np.zeros((zdim, channels), dtype='int32')
        fov -= 1
        mX = cache[cache_idx_y][cache_idx_x][0][:arr_len[cache_idx_y][cache_idx_x]]
        mY = cache[cache_idx_y][cache_idx_x][1][:arr_len[cache_idx_y][cache_idx_x]]
        ab = cache[cache_idx_y][cache_idx_x][2][:ind_len[cache_idx_y][cache_idx_x]]
        fov[ab, :] = im[mY, mX, :]
        return fov.reshape((output_height, output_width, channels))

    return process

def foveate_events(file_path, output_width, output_height, p0_radius, alpha, output_dir):
    output_name = os.path.splitext(os.path.basename(file_path))[0]
    output_csv_file = os.path.join(output_dir, f'{output_name}_foveated_events.csv')

    EVENTS_PER_FRAME = 15000
    SRC_WIDTH = 346
    SRC_HEIGHT = 260

    print(f"Loading event data from '{file_path}'...")
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")

    mappingX, mappingY, maxShiftX, maxShiftY = generate_mappings(
        SRC_WIDTH, SRC_HEIGHT, output_width, output_height, p0_radius, alpha
    )

    # Build a reverse map: from (SRC x, SRC y) to (new x, new y)
    reverse_map = dict()
    for out_y in range(output_height):
        for out_x in range(output_width):
            src_x = mappingX[out_y * output_width + out_x]
            src_y = mappingY[out_y * output_width + out_x]
            reverse_map[(src_x, src_y)] = (out_x, out_y)

    total_events = len(df)
    num_frames = int(np.ceil(total_events / EVENTS_PER_FRAME))

    all_foveated_events = []
    print(f"\nStarting per-event foveation..'")
    start_time = time.time()

    for i in range(num_frames):
        start_idx = i * EVENTS_PER_FRAME
        end_idx = min((i + 1) * EVENTS_PER_FRAME, total_events)
        frame_events = df.iloc[start_idx:end_idx]

        # Blank image just for visualization
        frame_vis = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        for _, row in frame_events.iterrows():
            x, y = int(row['x']), int(row['y'])
            timestamp = row['t']
            polarity = row['p']

            if (x, y) in reverse_map:
                new_x, new_y = reverse_map[(x, y)]

                all_foveated_events.append({
                    't': timestamp,
                    'x': new_x,
                    'y': new_y,
                    'p': polarity
                })

        if (i + 1) % 10 == 0 or (i + 1) == num_frames:
            print(f"  Processed frame {i + 1}/{num_frames}")

    foveated_df = pd.DataFrame(all_foveated_events)
    foveated_df.to_csv(output_csv_file, index=False)
    end_time = time.time()

    print("\n--------------------")
    print("Processing complete!")
    print(f"CSV saved as '{output_csv_file}'")
    print(f"Total time: {end_time - start_time:.2f} seconds")


def process_directory(input_dir, output_dir, factor):
    foveation_params = {
        1: {'width': 345, 'height': 259, 'radius': 70, 'alpha': 5.0},
        2: {'width': 173, 'height': 130, 'radius': 40, 'alpha': 2},
        5: {'width': 69, 'height': 52, 'radius': 2, 'alpha': 1.1},
        6: {'width': 58, 'height': 43, 'radius': 2, 'alpha': 1.1},
        7: {'width': 49, 'height': 37, 'radius': 1.8, 'alpha': 1.1},
        8: {'width': 43, 'height': 32, 'radius': 1.6, 'alpha': 1.1},
        9: {'width': 38, 'height': 29, 'radius': 1.4, 'alpha': 1.1},
        10: {'width': 35, 'height': 26, 'radius': 1, 'alpha': 1.0}
    }

    factor_folder = f"factor_x{factor}"
    log_file_name = f"processed_files_{factor_folder}.log"
    log_file_path = os.path.join(output_dir, log_file_name)

    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            processed_files = set(line.strip() for line in log_file)
    else:
        processed_files = set()


    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_events.csv"):
                input_file_path = os.path.join(root, file)
                relative_input_path = os.path.relpath(input_file_path, input_dir)

                if relative_input_path in processed_files:
                    print(f"Skipping already processed file: {relative_input_path}")
                    continue

                relative_path = os.path.relpath(root, input_dir)
                factor_folder = f"factor_x{factor}"
                output_subdir = os.path.join(output_dir, relative_path, factor_folder)

                os.makedirs(output_subdir, exist_ok=True)



                if factor not in foveation_params:
                    print(ValueError("Invalid scale factor. Options are: 1, 2, 5–10"))
                    return
                params = foveation_params[factor]
                print(f"\nFoveating {input_file_path} with parameters: {params}")
                foveate_events(
                    input_file_path,
                    params['width'],
                    params['height'],
                    params['radius'],
                    params['alpha'],
                    output_subdir
                )


                with open(log_file_path, "a") as log_file:
                    log_file.write(relative_input_path + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run event scaling pipeline for a directory')
    parser.add_argument('--input_dir', required=True, help='Path to input folder')
    parser.add_argument('--output_dir', required=True, help='Path to save output files')
    parser.add_argument('-f', '--factor', type=int, required=True, help='Scale factor. Options: 1, 2, 5–10')

    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.factor)