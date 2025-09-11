# CODE 3.0.21 - AI Metal Classifier GUI with Gated Automation
# Description: Displays live sensor data and camera feed.
#              - On startup, waits for a LOW->HIGH signal on GPIO 23 to classify.
#              - While GPIO 23 is LOW, it continuously auto-calibrates.
#              - After classifying, it enters a PAUSED state, ignoring new triggers.
#              - Clicking 'Classify Another' RE-ARMS the system for the next trigger.
# Version: 3.0.24 - MODIFIED: Updated with final scaler parameters from the new training script.
#                  -           TFLite models (Visual, Magnetism, Resistivity).
#                  - MODIFIED: Implemented a weighted-average system to combine model outputs.
#                  - REMOVED:  Removed joblib dependency; scaler parameters are now hardcoded.
# FIXED:       Potential mismatch between sensor data processing and scaler expectation.
# DEBUG:       Enhanced prints in capture_and_classify, preprocess_input, run_inference, postprocess_output.

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
from tkinter import messagebox
import cv2 # OpenCV for camera access
from PIL import Image, ImageTk, ImageDraw, ImageFont # Added ImageDraw, ImageFont
import time
import os
import statistics
from collections import deque
import numpy as np
import math
import warnings # To potentially suppress warnings later if needed
import traceback # For more detailed error logging

# --- AI Imports ---
try:
    # Preferred import for modern TFLite runtime
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback to full TensorFlow if tflite_runtime is not available
    print("tflite_runtime not found. Falling back to full TensorFlow library.")
    from tensorflow.lite.python.interpreter import Interpreter

# --- Hardware Imports (with MOCK classes for PC testing) ---
try:
    import RPi.GPIO as GPIO
    import board
    import busio
    import adafruit_ldc1101
    from adafruit_ms4988 import S49E
    # Check if we are on a Pi by checking a known device file
    if not os.path.exists('/sys/bus/platform/devices/soc'):
        raise ImportError("Not on a Raspberry Pi, forcing fallback to MOCK hardware.")
    ON_PI = True
except (ImportError, NotImplementedError):
    ON_PI = False
    print("\n" + "="*40)
    print("WARNING: RPi.GPIO or other hardware library not found.")
    print("Application will run in MOCK mode on this PC.")
    print("Sensor values will be simulated.")
    print("="*40 + "\n")

    # --- MOCK Hardware Classes for PC development ---
    class MockI2C:
        def __init__(self): pass
        def writeto(self, address, buffer, *, start=0, end=None): pass
        def readfrom_into(self, address, buffer, *, start=0, end=None): pass
        def writeto_then_readfrom(self, address, buffer_out, buffer_in, *, out_start=0, out_end=None, in_start=0, in_end=None): pass

    class MockLDC1101:
        def __init__(self, i2c_bus, address=0x2a):
            self._rp_data = 60000
            self._last_update = time.time()
        @property
        def rp_data(self):
            # Simulate slight fluctuation
            if time.time() - self._last_update > 0.1:
                self._rp_data += np.random.randint(-50, 51)
                self._last_update = time.time()
            return self._rp_data

    class MockS49E:
        def __init__(self, i2c, address=0x4a): self.magnetic_flux = 0.0

    class MockGPIO:
        BCM = 1; IN = 1; PUD_DOWN = 1; RISING = 1; FALLING = 1; LOW = 0; HIGH = 1
        def setmode(self, mode): pass
        def setup(self, pin, direction, pull_up_down=None): pass
        def input(self, pin): return self.LOW # Default to not triggered
        def add_event_detect(self, pin, edge, callback, bouncetime): pass
        def cleanup(self): pass
        def setwarnings(self, val): pass

    GPIO = MockGPIO()


# =================================================================================
# --- GLOBAL CONFIGURATION & STATE VARIABLES ---
# =================================================================================

# --- Application State ---
APP_STATE = "INITIALIZING" # "INITIALIZING", "ARMED", "CLASSIFYING", "PAUSED", "CALIBRATING"
trigger_event_detected = False # Global flag for GPIO trigger event

# --- UI & Display ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
FONT_LARGE = ("Helvetica", 20, "bold")
FONT_MEDIUM = ("Helvetica", 14)
FONT_SMALL = ("Helvetica", 10)
UPDATE_INTERVAL_MS = 50 # UI update frequency (milliseconds)

# --- Hardware & Sensor ---
I2C_BUS = None # To be initialized
LDC_SENSOR = None
HALL_SENSOR = None
VIDEO_CAPTURE = None
TRIGGER_PIN = 23 # BCM pin number for the trigger input
BASELINE_SAMPLES = 50 # Number of samples to take for baseline calibration

# --- Data Buffers ---
ldc_deque = deque(maxlen=BASELINE_SAMPLES)
mag_deque = deque(maxlen=BASELINE_SAMPLES)
ldc_baseline = 0
mag_baseline = 0

# --- AI Model & Preprocessing Configuration ---
VISUAL_MODEL_PATH = 'visual_model.tflite'
MAGNETISM_MODEL_PATH = 'magnetism_model.tflite'
RESISTIVITY_MODEL_PATH = 'resistivity_model.tflite'

# ### MODIFIED: Hardcoded Scaler Parameters with your trained values ###
# These values MUST match the ones used during the training of the respective models.
MAGNETISM_SCALER_MEAN = -0.01960512
MAGNETISM_SCALER_STD = 3.21212715
RESISTIVITY_SCALER_MEAN = 60780.67241379
RESISTIVITY_SCALER_STD = 1543.13269432

# --- Hierarchical Ensemble Configuration ---
# CLASS_LABELS = ['Aluminum', 'Copper', 'Others', 'Steel'] # OLD ORDER
CLASS_LABELS = ['Steel', 'Aluminum', 'Copper', 'Others'] # NEW ORDER from training script

# Model weights: How much we trust each model's prediction in the final vote.
MODEL_WEIGHTS = {
    'visual': 0.0,
    'magnetism': 1.0, # Highest trust for steel detection
    'resistivity': 0.0
}

# --- AI Model Interpreter References ---
interpreters = {} # Dictionary to hold loaded TFLite interpreters
input_details = {}
output_details = {}


# =================================================================================
# --- CORE APPLICATION LOGIC ---
# =================================================================================

def initialize_hardware():
    """Initializes I2C, sensors, GPIO, and the camera."""
    global I2C_BUS, LDC_SENSOR, HALL_SENSOR, VIDEO_CAPTURE
    print("--- Initializing Hardware ---")

    if ON_PI:
        print("Running on Raspberry Pi. Initializing real hardware.")
        # Initialize I2C bus
        I2C_BUS = busio.I2C(board.SCL, board.SDA)

        # Initialize LDC1101 Sensor
        print("Initializing LDC1101 sensor...")
        LDC_SENSOR = adafruit_ldc1101.LDC1101(I2C_BUS)
        print("LDC1101 sensor initialized.")

        # Initialize S49E Hall Effect Sensor
        print("Initializing S49E Hall sensor...")
        HALL_SENSOR = S49E(I2C_BUS)
        print("S49E sensor initialized.")

        # Setup GPIO
        print(f"Setting up GPIO on pin {TRIGGER_PIN}...")
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIGGER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        # Add event detection for the trigger
        GPIO.add_event_detect(TRIGGER_PIN, GPIO.RISING, callback=gpio_trigger_callback, bouncetime=200)
        print("GPIO setup complete.")

    else:
        print("Running in MOCK mode. Initializing mock hardware.")
        I2C_BUS = MockI2C()
        LDC_SENSOR = MockLDC1101(I2C_BUS)
        HALL_SENSOR = MockS49E(I2C_BUS)
        # Mock GPIO is already instantiated, no setup needed.

    # Initialize Camera (for both Pi and PC)
    print("Initializing camera...")
    VIDEO_CAPTURE = cv2.VideoCapture(0)
    if not VIDEO_CAPTURE.isOpened():
        print("FATAL: Cannot open camera.")
        messagebox.showerror("Camera Error", "Could not open the camera. Please check if it's connected and not in use.")
        return False
    VIDEO_CAPTURE.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    VIDEO_CAPTURE.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    print("Camera initialized successfully.")
    return True

def initialize_ai():
    """Loads all TFLite models into memory."""
    global interpreters, input_details, output_details
    print("\n--- Initializing AI Models ---")
    model_paths = {
        'visual': VISUAL_MODEL_PATH,
        'magnetism': MAGNETISM_MODEL_PATH,
        'resistivity': RESISTIVITY_MODEL_PATH
    }

    all_models_loaded = True
    for model_name, path in model_paths.items():
        try:
            print(f"Loading {model_name} model from: {path}")
            interpreters[model_name] = Interpreter(model_path=path)
            interpreters[model_name].allocate_tensors()
            input_details[model_name] = interpreters[model_name].get_input_details()
            output_details[model_name] = interpreters[model_name].get_output_details()
            print(f"-> {model_name} model loaded successfully.")
        except Exception as e:
            print(f"FATAL: Failed to load {model_name} model from '{path}'. Error: {e}")
            all_models_loaded = False

    return all_models_loaded


def gpio_trigger_callback(channel):
    """
    Interrupt handler for the GPIO trigger pin.
    Sets a global flag to be handled by the main GUI loop.
    This is safer than modifying GUI elements directly from a thread.
    """
    global trigger_event_detected
    # Only set the flag if the system is armed to avoid multiple triggers
    if APP_STATE == "ARMED":
        print(f"GPIO TRIGGER detected on channel {channel}! Setting flag.")
        trigger_event_detected = True


def read_sensors():
    """Reads the latest values from the LDC and Hall sensors."""
    try:
        ldc_val = LDC_SENSOR.rp_data if LDC_SENSOR else 0
        # S49E gives flux in mT (millitesla), which is what our training data used
        mag_val = HALL_SENSOR.magnetic_flux if HALL_SENSOR else 0
        return {"ldc_rp": ldc_val, "magnetism": mag_val}
    except Exception as e:
        # print(f"Warning: Could not read sensor data: {e}") # Can be noisy
        return {"ldc_rp": 0, "magnetism": 0}


def update_baselines(sensor_values):
    """Updates the moving averages for sensor baselines."""
    global ldc_baseline, mag_baseline
    ldc_deque.append(sensor_values["ldc_rp"])
    mag_deque.append(sensor_values["magnetism"])

    if len(ldc_deque) >= BASELINE_SAMPLES:
        ldc_baseline = statistics.mean(ldc_deque)
        mag_baseline = statistics.mean(mag_deque)


def capture_and_classify():
    """
    Core function triggered by the GPIO event. Captures data, runs inference,
    and updates the UI with the result.
    """
    global APP_STATE
    set_app_state("CLASSIFYING")
    print("\n" + "="*20 + " CAPTURE & CLASSIFY " + "="*20)

    # 1. Capture Data
    print("Step 1: Capturing sensor data and image...")
    sensor_readings = read_sensors()
    ret, frame = VIDEO_CAPTURE.read()
    if not ret:
        print("Error: Failed to capture frame for classification.")
        messagebox.showerror("Capture Error", "Could not capture an image from the camera for classification.")
        set_app_state("PAUSED") # Go to paused state on failure
        return

    # 2. Preprocess all inputs
    print("Step 2: Preprocessing inputs for all models...")
    try:
        visual_input = preprocess_input(frame, 'visual')
        magnetism_input = preprocess_input(sensor_readings, 'magnetism')
        resistivity_input = preprocess_input(sensor_readings, 'resistivity')
    except Exception as e:
        print(f"FATAL during preprocessing: {e}"); traceback.print_exc()
        messagebox.showerror("Preprocessing Error", f"Failed to prepare data for AI models: {e}")
        set_app_state("PAUSED"); return

    # 3. Run Inference on all models
    print("Step 3: Running inference on all three models...")
    try:
        visual_preds = run_inference(visual_input, 'visual')
        magnetism_preds = run_inference(magnetism_input, 'magnetism')
        resistivity_preds = run_inference(resistivity_input, 'resistivity')
    except Exception as e:
        print(f"FATAL during inference: {e}"); traceback.print_exc()
        messagebox.showerror("Inference Error", f"An error occurred while running the AI models: {e}")
        set_app_state("PAUSED"); return

    # 4. Post-process and Combine Results (Ensemble)
    print("Step 4: Combining model predictions...")
    final_result, all_predictions = postprocess_output({
        'visual': visual_preds,
        'magnetism': magnetism_preds,
        'resistivity': resistivity_preds
    })

    # 5. Update UI
    print("Step 5: Displaying final result.")
    # Show the captured image with the result overlay
    display_final_result(frame, final_result, all_predictions)
    # Update the result label
    result_var.set(f"Result: {final_result}")
    set_app_state("PAUSED") # Move to paused state after classification
    print("="*20 + " CLASSIFICATION COMPLETE " + "="*20 + "\n")


def preprocess_input(data, model_type):
    """
    Preprocesses raw data (image or sensor values) for a specific model.
    - For sensors: Converts magnetism to µT and applies standardization.
    - For visual: Resizes, normalizes, and adds a batch dimension.
    """
    # print(f"DEBUG: Preprocessing for '{model_type}' model...") # Can be noisy
    if model_type == 'magnetism':
        # 1. Convert from millitesla (mT) to microtesla (µT)
        magnetism_uT = data['magnetism'] * 1000.0
        # 2. Standardize using the hardcoded scaler for the magnetism model
        scaled_value = (magnetism_uT - MAGNETISM_SCALER_MEAN) / MAGNETISM_SCALER_STD
        # print(f"  DEBUG Mag: Raw(mT)={data['magnetism']:.4f} -> µT={magnetism_uT:.4f} -> Scaled={scaled_value:.4f}")
        return np.array([[scaled_value]], dtype=np.float32)

    elif model_type == 'resistivity':
        # 1. Standardize using the hardcoded scaler for the resistivity model
        scaled_value = (data['ldc_rp'] - RESISTIVITY_SCALER_MEAN) / RESISTIVITY_SCALER_STD
        # print(f"  DEBUG Res: Raw={data['ldc_rp']} -> Scaled={scaled_value:.4f}")
        return np.array([[scaled_value]], dtype=np.float32)

    elif model_type == 'visual':
        # 1. Resize the image to the model's expected input size (e.g., 224x224)
        input_shape = input_details['visual'][0]['shape']
        target_height, target_width = input_shape[1], input_shape[2]
        img_resized = cv2.resize(data, (target_width, target_height))
        # 2. Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # 3. Normalize pixel values to [0, 1]
        img_normalized = img_rgb / 255.0
        # 4. Add a batch dimension and ensure correct data type
        return np.expand_dims(img_normalized, axis=0).astype(np.float32)

    else:
        raise ValueError(f"Unknown model type for preprocessing: {model_type}")


def run_inference(input_data, model_type):
    """Runs the specified TFLite model and returns the raw output."""
    # print(f"DEBUG: Running inference for '{model_type}'...") # Can be noisy
    interpreter = interpreters[model_type]
    input_idx = input_details[model_type][0]['index']
    output_idx = output_details[model_type][0]['index']

    interpreter.set_tensor(input_idx, input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_idx)
    # print(f"  DEBUG {model_type} raw output: {output}")
    return output[0] # Remove batch dimension from output


def postprocess_output(predictions):
    """
    Combines predictions from all models using a weighted average.
    Also implements a hard rule for steel based on magnetism.
    """
    num_classes = len(CLASS_LABELS)
    weighted_scores = np.zeros(num_classes)
    all_preds_info = {}

    print("--- Post-processing and Ensemble Logic ---")

    # Hard Rule Check: Is the magnetism model overwhelmingly confident about 'Steel'?
    # The 'Steel' class is at index 0 in our CLASS_LABELS
    steel_index = CLASS_LABELS.index('Steel')
    magnetism_steel_confidence = predictions['magnetism'][steel_index]

    if magnetism_steel_confidence > 0.95:
        print("HARD RULE APPLIED: Magnetism model is >95% confident of 'Steel'.")
        print("Overriding other models. Final result is 'Steel'.")
        # To explain the result, we can still show the other preds
        for model_name, preds in predictions.items():
            best_idx = np.argmax(preds)
            all_preds_info[model_name] = (CLASS_LABELS[best_idx], preds[best_idx])
        return CLASS_LABELS[steel_index], all_preds_info

    # If hard rule is not met, proceed with weighted averaging
    print("Weighted averaging predictions...")
    for model_name, preds in predictions.items():
        weight = MODEL_WEIGHTS[model_name]
        weighted_scores += preds * weight

        # Store individual model top predictions for display
        best_idx = np.argmax(preds)
        confidence = preds[best_idx]
        all_preds_info[model_name] = (CLASS_LABELS[best_idx], confidence)
        print(f"  -> {model_name.capitalize()} pred: {CLASS_LABELS[best_idx]} (Conf: {confidence:.2f}, Weight: {weight})")


    # Final decision is the class with the highest weighted score
    final_prediction_index = np.argmax(weighted_scores)
    final_class = CLASS_LABELS[final_prediction_index]
    print(f"Final Weighted Scores: {[f'{s:.3f}' for s in weighted_scores]}")
    print(f"--> Final Ensemble Result: {final_class}")
    return final_class, all_preds_info


def set_app_state(new_state):
    """Updates the global application state and the UI status label."""
    global APP_STATE
    if new_state != APP_STATE:
        APP_STATE = new_state
        status_var.set(f"Status: {APP_STATE}")
        print(f"Application state changed to: {APP_STATE}")

        # Configure button states based on the new app state
        if APP_STATE == "PAUSED":
            classify_button.config(state=tk.NORMAL)
        else:
            classify_button.config(state=tk.DISABLED)


def rearm_system():
    """Resets the state to 'ARMED' to wait for a new trigger."""
    global trigger_event_detected
    print("\nSystem RE-ARMED. Waiting for next trigger...")
    trigger_event_detected = False
    result_var.set("Result: --")
    set_app_state("ARMED")


def main_loop():
    """The main GUI update loop."""
    global trigger_event_detected

    # --- State Machine Logic ---
    if trigger_event_detected and APP_STATE == "ARMED":
        trigger_event_detected = False # Consume the event
        capture_and_classify()

    elif APP_STATE in ["ARMED", "CALIBRATING"]:
        # When waiting for a trigger, continuously calibrate baselines
        current_sensors = read_sensors()
        update_baselines(current_sensors)
        # Check if calibration is complete
        if len(ldc_deque) == BASELINE_SAMPLES and APP_STATE == "CALIBRATING":
            set_app_state("ARMED") # Move to armed once baseline is established

    # --- UI Updates ---
    # Read sensor data for display
    sensor_values = read_sensors()
    ldc_val = sensor_values.get("ldc_rp", 0)
    mag_val = sensor_values.get("magnetism", 0)

    # Update sensor data labels
    ldc_var.set(f"LDC RP: {ldc_val}")
    mag_var.set(f"Magnetism: {mag_val:.4f} mT") # Display in mT

    # Update LDC delta label
    if ldc_baseline > 0:
        delta_rp = ldc_val - ldc_baseline
        ldc_delta_var.set(f"Delta RP: {delta_rp:.0f}")
    else:
        ldc_delta_var.set("Delta RP: Calibrating...")

    # Update video feed (unless a result is being shown)
    if APP_STATE != "PAUSED":
        ret, frame = VIDEO_CAPTURE.read()
        if ret:
            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            # Convert to a format Tkinter can use
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

    # Schedule the next run of the main_loop
    window.after(UPDATE_INTERVAL_MS, main_loop)


def display_final_result(frame, final_result, all_predictions):
    """Overlays the final result and individual model predictions on the image."""
    # Flip the frame horizontally to match the live mirror view
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        # Use a more common font, handle case where it might be missing
        title_font = ImageFont.truetype("arial.ttf", 32)
        detail_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        print("Arial font not found, using default.")
        title_font = ImageFont.load_default()
        detail_font = ImageFont.load_default()


    # --- Draw Final Result Box ---
    # Position for the main result box
    box_x0, box_y0 = 10, 10
    box_x1, box_y1 = VIDEO_WIDTH - 10, 70
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill="black", outline="yellow", width=2)
    # Text for the final result
    text = f"FINAL RESULT: {final_result}"
    text_bbox = draw.textbbox((0, 0), text, font=title_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (VIDEO_WIDTH - text_width) / 2
    text_y = box_y0 + (box_y1 - box_y0 - text_height) / 2
    draw.text((text_x, text_y), text, font=title_font, fill="yellow")


    # --- Draw Individual Predictions Box ---
    detail_y_start = 80
    line_height = 22
    # Background box for details
    draw.rectangle([10, detail_y_start, 250, detail_y_start + (line_height * 4)], fill=(0,0,0,128))

    draw.text((15, detail_y_start + 5), "--- Model Predictions ---", font=detail_font, fill="white")
    y_pos = detail_y_start + line_height
    for model_name, (pred, conf) in all_predictions.items():
        text = f"{model_name.capitalize()}: {pred} ({conf:.1%})"
        draw.text((15, y_pos), text, font=detail_font, fill="cyan")
        y_pos += line_height

    # Convert back to Tkinter format and display
    imgtk = ImageTk.PhotoImage(image=pil_img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)


def cleanup_resources():
    """Releases hardware resources gracefully."""
    print("\n--- Cleaning Up Resources ---")
    if ON_PI:
        print("Cleaning up GPIO...")
        GPIO.cleanup()
        print("GPIO cleaned up.")
    if VIDEO_CAPTURE and VIDEO_CAPTURE.isOpened():
        print("Releasing video capture...")
        VIDEO_CAPTURE.release()
        print("Video capture released.")


# =================================================================================
# --- GUI SETUP ---
# =================================================================================

def setup_gui(root):
    """Creates and arranges all the Tkinter widgets."""
    global status_var, ldc_var, mag_var, ldc_delta_var, result_var, video_label, classify_button

    # --- Configure Styles ---
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TFrame", background="#2c3e50")
    style.configure("TLabel", background="#2c3e50", foreground="#ecf0f1", font=FONT_MEDIUM)
    style.configure("Title.TLabel", font=FONT_LARGE, foreground="#ffffff")
    style.configure("Status.TLabel", font=FONT_MEDIUM, foreground="#e74c3c")
    style.configure("Result.TLabel", font=FONT_LARGE, foreground="#2ecc71")
    style.configure("TButton", font=FONT_MEDIUM, background="#3498db", foreground="#ffffff")
    style.map("TButton", background=[('active', '#2980b9')])


    # --- StringVars for dynamic labels ---
    status_var = tk.StringVar(value="Status: INITIALIZING")
    ldc_var = tk.StringVar(value="LDC RP: --")
    mag_var = tk.StringVar(value="Magnetism: --")
    ldc_delta_var = tk.StringVar(value="Delta RP: --")
    result_var = tk.StringVar(value="Result: --")

    # --- Main Frame ---
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(expand=True, fill=tk.BOTH)

    # --- Title ---
    title_label = ttk.Label(main_frame, text="AI Metal Classification System", style="Title.TLabel")
    title_label.pack(pady=(0, 10))

    # --- Content Frame (Video + Data) ---
    content_frame = ttk.Frame(main_frame)
    content_frame.pack(expand=True, fill=tk.BOTH, pady=5)

    # --- Video Label ---
    video_label = ttk.Label(content_frame)
    video_label.pack(side=tk.LEFT, padx=(0, 10), expand=True, fill=tk.BOTH)
    # Create a black placeholder image
    placeholder = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), 'black')
    imgtk = ImageTk.PhotoImage(image=placeholder)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)


    # --- Data & Controls Frame ---
    data_frame = ttk.Frame(content_frame, padding="10")
    data_frame.pack(side=tk.RIGHT, fill=tk.Y)

    # Status Label
    status_label = ttk.Label(data_frame, textvariable=status_var, style="Status.TLabel", font=("Helvetica", 16, "bold"))
    status_label.pack(pady=10, anchor='w')

    # Sensor Labels
    ttk.Label(data_frame, textvariable=ldc_var).pack(pady=5, anchor='w')
    ttk.Label(data_frame, textvariable=ldc_delta_var).pack(pady=5, anchor='w')
    ttk.Label(data_frame, textvariable=mag_var).pack(pady=5, anchor='w')

    # Separator
    ttk.Separator(data_frame, orient='horizontal').pack(fill='x', pady=20)

    # Result Label
    result_label = ttk.Label(data_frame, textvariable=result_var, style="Result.TLabel")
    result_label.pack(pady=10, anchor='w')

    # 'Classify Another' Button
    classify_button = ttk.Button(data_frame, text="Classify Another / Re-Arm", command=rearm_system)
    classify_button.pack(pady=20, fill='x')
    classify_button.config(state=tk.DISABLED) # Initially disabled

    return root


# =================================================================================
# --- APPLICATION ENTRY POINT ---
# =================================================================================

if __name__ == "__main__":
    # --- Setup Main Window ---
    window = tk.Tk()
    window.title("AI Metal Classifier v3.0.24")
    window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    window.configure(background="#2c3e50")

    # --- Run Application ---
    print("\n" + "="*30 + "\nStarting AI Metal Classifier (Ensemble) \n" + "="*30)
    hw_init_attempted = False
    try:
        window = setup_gui(window)
        # We need the GUI to exist before we can show error popups
        if initialize_hardware() and initialize_ai():
            hw_init_attempted = True
            set_app_state("CALIBRATING") # Start in calibration mode
            main_loop() # Start the main application loop
            window.mainloop() # Start the Tkinter event loop
        else:
            print("\nApplication cannot start due to initialization failure.")
            # Don't destroy window immediately, let user see error
            # messagebox.showerror("Init Error", "Could not initialize hardware or AI models. Please check console.")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting application.")
    except Exception as e:
        print("\n" + "="*30 + f"\nFATAL ERROR in main execution: {e}\n" + "="*30)
        traceback.print_exc()
        if 'window' in globals() and window and window.winfo_exists():
            try:
                messagebox.showerror("Fatal Application Error", f"Unrecoverable error:\n\n{e}\n\nPlease check console.")
            except Exception:
                pass
    finally:
        # Graceful shutdown
        cleanup_resources()
        if 'window' in globals() and window and window.winfo_exists():
            print("Destroying Tkinter window...")
            window.destroy()
        print("\nApplication finished.\n" + "="*30)
