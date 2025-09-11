import os
import cv2
import numpy as np
import argparse
from datetime import datetime
import time
import signal
import json
from screeninfo import get_monitors
import dv_processing as dv
from PIL import Image
import cairosvg
import io

capture = dv.io.CameraCapture()
running = True

def load_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".svg":
        try:
            png_data = cairosvg.svg2png(url=image_path)
            img_pil = Image.open(io.BytesIO(png_data)).convert("RGBA")
        except Exception as e:
            print(f"SVG load failed for {image_path}: {e}")
            return None
    else:
        try:
            img_pil = Image.open(image_path)
            if img_pil.mode == 'P':
                img_pil = img_pil.convert('RGBA')
            elif img_pil.mode != 'RGBA':
                img_pil = img_pil.convert('RGBA')
        except Exception as e:
            print(f"Failed to open image: {image_path}. Error: {e}")
            return None

    # Use light gray background for better visibility
    if img_pil.mode == 'RGBA':
        background = Image.new("RGBA", img_pil.size, (200, 200, 200, 255))
        img_pil = Image.alpha_composite(background, img_pil)

    img_np = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def display_image_centered(image_path, x_offset, y_offset, screen_w, screen_h, display_time):
    img = load_image(image_path)
    if img is None:
        print(f"Warning: Failed to load image: {image_path}. Skipping.")
        return

    img_h, img_w = img.shape[:2]
    target_area = 0.5 * screen_w * screen_h
    scale_factor = (target_area / (img_w * img_h)) ** 0.5
    scale_factor = min(scale_factor, 1.0) if img_w > screen_w or img_h > screen_h else min(scale_factor, 2.0)

    # Resize while respecting canvas bounds
    new_w = int(img_w * scale_factor)
    new_h = int(img_h * scale_factor)

    # Ensure resized image fits in canvas
    if new_w > screen_w or new_h > screen_h:
        scale_w = screen_w / img_w
        scale_h = screen_h / img_h
        scale_factor = min(scale_w, scale_h, scale_factor)
        new_w = int(img_w * scale_factor)
        new_h = int(img_h * scale_factor)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255
    y_start = max((screen_h - new_h) // 2, 0)
    x_start = max((screen_w - new_w) // 2, 0)

    # Final bounds check before insertion
    if y_start + new_h > screen_h or x_start + new_w > screen_w:
        print(f"Warning: Resized image {image_path} still doesn't fit screen ({screen_w}x{screen_h}). Skipping.")
        return

    canvas[y_start:y_start+new_h, x_start:x_start+new_w] = resized_img

    window_name = "ImageDisplay"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, x_offset, y_offset)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow(window_name, canvas)
    cv2.waitKey(int(display_time * 1000))



def handle_interrupt(signum, frame):
    global running
    running = False
    print("\nInterrupted. Will stop after current image and recording.")

signal.signal(signal.SIGINT, handle_interrupt)

def get_secondary_screen_geometry():
    monitors = get_monitors()
    if len(monitors) < 2:
        print("Only one screen detected. Using primary screen.")
        m = monitors[0]
    else:
        print("Secondary screen detected. Using it.")
        m = monitors[1]
    return m.x, m.y, m.width, m.height



def record_event_data(duration_sec, output_file):
    writer = dv.io.MonoCameraWriter(output_file, capture)
    start_time = time.time()
    end_time = start_time + duration_sec

    while time.time() < end_time and running and capture.isRunning():
        events = capture.getNextEventBatch()
        frame = capture.getNextFrame()
        if events is not None:
            writer.writeEvents(events, streamName='events')
        if frame is not None:
            writer.writeFrame(frame, streamName='frames')

    print(f"Saved: {output_file}")

def load_log(log_path):
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return json.load(f)
    return {}

def update_log(log_path, log_data, category, subcategory, filename):
    log_data.setdefault(category, {})
    log_data[category].setdefault(subcategory, {"count": 0, "files": []})

    if filename not in log_data[category][subcategory]["files"]:
        log_data[category][subcategory]["files"].append(filename)
        log_data[category][subcategory]["count"] += 1

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)

def display_and_record(base_path, output_dir, display_time):
    x_offset, y_offset, screen_w, screen_h = get_secondary_screen_geometry()
    log_path = os.path.join(output_dir, "record_log.json")
    os.makedirs(output_dir, exist_ok=True)
    log_data = load_log(log_path)

    try:
        for root, dirs, files in os.walk(base_path):
            for file in sorted(files):
                if not running:
                    break
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.svg', '.tiff', '.webp', '.tif')):
                    image_path = os.path.join(root, file)
                    rel_path = os.path.relpath(image_path, base_path)
                    parts = rel_path.split(os.sep)
                    if len(parts) < 3:
                        continue  # skip files not in category/subcategory

                    category, subcategory = parts[0], parts[1]
                    if file in log_data.get(category, {}).get(subcategory, {}).get("files", []):
                        print(f"Skipping already recorded file: {file}")
                        continue

                    print(f"Displaying: {image_path}")
                    display_image_centered(image_path, x_offset, y_offset, screen_w, screen_h, display_time)

                    print("Recording events...")
                    # Build mirrored output directory
                    output_subdir = os.path.join(output_dir, *parts[:-1])
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, os.path.splitext(file)[0] + ".aedat4")
                    record_event_data(display_time, output_file)

                    update_log(log_path, log_data, category, subcategory, file)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Finished all images or interrupted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display images and record synchronized AEDAT4 files.")
    parser.add_argument("--base_path", type=str, required=True, help="Root folder: Symbols/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for recordings.")
    parser.add_argument("--display_time", type=float, default=2.0, help="Time to show each image (seconds)")
    args = parser.parse_args()


    display_and_record(args.base_path, args.output_dir, args.display_time)
    # python synchronized_display_record.py --base_path Symbols --output_dir Recordings --display_time 4.0


