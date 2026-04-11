import os
import time
from datetime import datetime
from picamera2 import Picamera2



# --- Configuration ---
PICS_PER_SET = 2

# 1. Initialize the camera
picam2 = Picamera2()

try:
    print("Configuring camera module for maximum quality...")
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()

    # 2. Create ONE main folder on the Desktop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"captures_{timestamp}"
    desktop_path = os.path.expanduser(f"~/Desktop/{folder_name}")

    if not os.path.exists(desktop_path):
        os.makedirs(desktop_path)
        print(f"Created directory: {desktop_path}")

    # 3. Warm up sensor
    print("Warming up sensor...")
    time.sleep(2)

    total_photos_taken = 0  # This ensures file names don't reset and overwrite
    set_counter = 1

    # 4. Interactive Loop for Multiple Sets in the SAME folder
    while True:
        print("-" * 40)
        user_input = input(f"Press [ENTER] to capture Set {set_counter} (or type 'q' to quit): ").strip().lower()
        
        if user_input == 'q':
            print("Exiting capture sequence.")
            break

        print(f"\nStarting capture of {PICS_PER_SET} images...")
        
        for i in range(1, PICS_PER_SET + 1):
            total_photos_taken += 1
            
            # Generate a continuous filename (photo_001, photo_002... photo_021...)
            file_path = os.path.join(desktop_path, f"photo_{total_photos_taken:03d}.jpg")
            
            print(f"  Capturing image {i}/{PICS_PER_SET} (File: photo_{total_photos_taken:03d}.jpg)...")
            picam2.capture_file(file_path)
            
        print(f"Finished Set {set_counter}! Total photos currently in folder: {total_photos_taken}")
        set_counter += 1

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # 5. Clean up
    picam2.stop()
    # Check if total_photos_taken exists in case it failed before initialization
    final_count = total_photos_taken if 'total_photos_taken' in locals() else 0
    print(f"\nCamera safely closed. All {final_count} photos are saved in:\n{desktop_path}")