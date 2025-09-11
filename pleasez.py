# CODE 3.0.25 - AI Metal Classifier GUI with Enhanced Calibration Accuracy
# Description: Displays live sensor data and camera feed with improved calibration system.
# - ENHANCED: Continuous calibration during pause for better accuracy
# - ENHANCED: Fresh calibration before each classification
# - ENHANCED: Drift detection and compensation
# Version: 3.0.25 - MODIFIED: Enhanced calibration system for improved accuracy

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import time
import os
import statistics
from collections import deque
import numpy as np
import math
import warnings
import traceback

# --- AI Imports ---
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        print("ERROR: TensorFlow Lite Runtime is not installed.")
        exit()

# --- I2C/ADS1115 Imports ---
I2C_ENABLED = False
try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    I2C_ENABLED = True
    print("I2C/ADS1115 libraries imported successfully.")
except ImportError:
    print("Warning: I2C/ADS1115 libraries not found. Magnetism readings will be disabled.")

# --- SPI/LDC1101 & RPi.GPIO Imports ---
SPI_ENABLED = False
RPi_GPIO_AVAILABLE = False
try:
    import spidev
    SPI_ENABLED = True
    print("SPI library (spidev) imported successfully.")
except ImportError:
    print("Warning: SPI library (spidev) not found. LDC readings will be disabled.")

try:
    import RPi.GPIO as GPIO
    RPi_GPIO_AVAILABLE = True
    print("RPi.GPIO library imported successfully.")
except ImportError:
    print("Warning: RPi.GPIO library not found.")

# --- Configuration Constants ---
NUM_SAMPLES_PER_UPDATE = 3
NUM_SAMPLES_CALIBRATION = 15
GUI_UPDATE_INTERVAL_MS = 100
CAMERA_UPDATE_INTERVAL_MS = 50
LDC_DISPLAY_BUFFER_SIZE = 5
CAMERA_INDEX = 0
DISPLAY_IMG_WIDTH = 640
DISPLAY_IMG_HEIGHT = 480
RESULT_IMG_DISPLAY_WIDTH = 280

CALIBRATION_INTERVAL_PAUSED = 2.0  # Calibrate every 2 seconds during pause
DRIFT_THRESHOLD_VOLTAGE = 0.001    # Voltage drift threshold (1mV)
DRIFT_THRESHOLD_RP = 50           # RP drift threshold
MAX_CALIBRATION_AGE = 10.0        # Max age before forced recalibration (seconds)

try:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_PATH = os.getcwd()

# --- GPIO Configuration ---
SORTING_GPIO_ENABLED = False
SORTING_DATA_PIN_LSB = 16
SORTING_DATA_PIN_MID = 6
SORTING_DATA_READY_PIN = 26
CONTROL_PIN = 23
CONTROL_PIN_SETUP_OK = False
CONTROL_CHECK_INTERVAL_MS = 50

# --- Model Configuration ---
MODEL_VISUAL_FILENAME = "visual_model.tflite"
MODEL_MAGNETISM_FILENAME = "magnetism_model.tflite"
MODEL_RESISTIVITY_FILENAME = "resistivity_model.tflite"
LABELS_FILENAME = "material_labels.txt"

MODEL_VISUAL_PATH = os.path.join(BASE_PATH, MODEL_VISUAL_FILENAME)
MODEL_MAGNETISM_PATH = os.path.join(BASE_PATH, MODEL_MAGNETISM_FILENAME)
MODEL_RESISTIVITY_PATH = os.path.join(BASE_PATH, MODEL_RESISTIVITY_FILENAME)
LABELS_PATH = os.path.join(BASE_PATH, LABELS_FILENAME)

MODEL_WEIGHTS = {
    'visual': 0.0,
    'magnetism': 1.0,
    'resistivity': 0.0
}

SCALER_PARAMS = {
    'magnetism': {
        'mean': [0.00048415711947626843],
        'scale': [0.0007762457818081904]
    },
    'resistivity': {
        'mean': [61000.82880523732],
        'scale': [1362.7716526399417]
    }
}

TESTING_FOLDER_NAME = "testing"
AI_IMG_WIDTH = 224
AI_IMG_HEIGHT = 224
HALL_ADC_CHANNEL = ADS.P0 if I2C_ENABLED else None
SENSITIVITY_V_PER_TESLA = 0.0002
SENSITIVITY_V_PER_MILLITESLA = SENSITIVITY_V_PER_TESLA * 1000

# --- LDC Configuration ---
SPI_BUS = 0
SPI_DEVICE = 0
SPI_SPEED = 500000
SPI_MODE = 0b00
CS_PIN = 8
LDC_CHIP_ID = 0xD4

# LDC Registers
START_CONFIG_REG, RP_SET_REG, TC1_REG, TC2_REG, DIG_CONFIG_REG, ALT_CONFIG_REG, \
D_CONF_REG, INTB_MODE_REG, RP_DATA_MSB_REG, RP_DATA_LSB_REG, CHIP_ID_REG = \
0x0B, 0x01, 0x02, 0x03, 0x04, 0x05, 0x0C, 0x0A, 0x22, 0x21, 0x3F

ACTIVE_CONVERSION_MODE, SLEEP_MODE = 0x00, 0x01

# === Global Objects/State ===
camera = None
i2c = None
ads = None
hall_sensor = None
spi = None
ldc_initialized = False

# AI Models
interpreter_visual = None
interpreter_magnetism = None
interpreter_resistivity = None
input_details_visual = None
output_details_visual = None
input_details_magnetism = None
output_details_magnetism = None
input_details_resistivity = None
output_details_resistivity = None
loaded_labels = []

RP_DISPLAY_BUFFER = deque(maxlen=LDC_DISPLAY_BUFFER_SIZE)
g_last_live_magnetism_mT = 0.0

# GUI Globals
window = None
main_frame = None
live_view_frame = None
results_view_frame = None
label_font, readout_font, button_font, title_font, result_title_font, result_value_font, pred_font = (None,) * 7
lv_camera_label, lv_magnetism_label, lv_ldc_label, lv_save_checkbox = (None,) * 4
rv_image_label, rv_prediction_label, rv_confidence_label, rv_magnetism_label, rv_ldc_label, rv_classify_another_button = (None,) * 6
placeholder_img_tk = None
save_output_var = None

g_accepting_triggers = True
g_previous_control_state = None
g_last_calibration_time = 0
g_last_paused_calibration_time = 0  # Track calibration during pause
g_calibration_baseline_age = 0      # Track how old current calibration is
IDLE_VOLTAGE = 0.0
IDLE_RP_VALUE = 0
g_baseline_voltage_history = deque(maxlen=5)  # Track baseline drift
g_baseline_rp_history = deque(maxlen=5)       # Track baseline drift


def initialize_hardware():
    global camera, i2c, ads, hall_sensor, spi, ldc_initialized, CS_PIN
    global SORTING_GPIO_ENABLED, RPi_GPIO_AVAILABLE
    global CONTROL_PIN_SETUP_OK, CONTROL_PIN
    
    print("\n--- Initializing Hardware ---")
    
    # Camera Initialization
    print(f"Attempting to open camera at index {CAMERA_INDEX}...")
    try:
        camera = cv2.VideoCapture(CAMERA_INDEX)
        time.sleep(0.5)
        if not camera or not camera.isOpened():
            raise ValueError(f"Could not open camera at index {CAMERA_INDEX}.")
        print(f"Camera {CAMERA_INDEX} opened successfully.")
    except Exception as e:
        print(f"ERROR: Failed to open camera {CAMERA_INDEX}: {e}")
        camera = None

    # I2C/ADS1115 Initialization
    if I2C_ENABLED:
        print("Initializing I2C and ADS1115...")
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            ads = ADS.ADS1115(i2c)
            if HALL_ADC_CHANNEL is not None:
                hall_sensor = AnalogIn(ads, HALL_ADC_CHANNEL)
                print(f"ADS1115 initialized. Hall sensor assigned to channel {HALL_ADC_CHANNEL}.")
            else:
                print("Warning: HALL_ADC_CHANNEL not defined, cannot create Hall sensor input.")
                hall_sensor = None
        except Exception as e:
            print(f"ERROR: Initializing I2C/ADS1115 failed: {e}")
            i2c = ads = hall_sensor = None
    else:
        print("Skipping I2C/ADS1115 setup (libraries not found or disabled).")

    # GPIO setup
    gpio_bcm_mode_set = False
    if RPi_GPIO_AVAILABLE:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            gpio_bcm_mode_set = True
            print("GPIO BCM mode set successfully.")
        except Exception as e:
            print(f"ERROR: GPIO.setmode(GPIO.BCM) failed: {e}")
    else:
        print("RPi.GPIO library not available. Skipping all GPIO-dependent setups.")

    # SPI/LDC1101 Initialization
    if SPI_ENABLED and gpio_bcm_mode_set:
        print("Initializing SPI and LDC1101...")
        try:
            GPIO.setup(CS_PIN, GPIO.OUT, initial=GPIO.HIGH)
            print(f"LDC CS Pin {CS_PIN} set as OUTPUT HIGH.")
            spi = spidev.SpiDev()
            spi.open(SPI_BUS, SPI_DEVICE)
            spi.max_speed_hz = SPI_SPEED
            spi.mode = SPI_MODE
            print(f"SPI initialized for LDC (Bus={SPI_BUS}, Device={SPI_DEVICE}).")
            if initialize_ldc1101():
                enable_ldc_rpmode()
                print("LDC1101 initialized and RP+L mode enabled.")
            else:
                print("ERROR: LDC1101 Low-level Initialization Failed.")
                ldc_initialized = False
        except Exception as e:
            print(f"ERROR: An error occurred during SPI/LDC initialization: {e}")
            if spi:
                spi.close()
            spi = None
            ldc_initialized = False
    else:
        print("Skipping SPI/LDC1101 setup.")

    # Sorting GPIO Pin Initialization
    if gpio_bcm_mode_set:
        print("Attempting to initialize GPIO pins for Sorting Mechanism...")
        try:
            GPIO.setup(SORTING_DATA_PIN_LSB, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(SORTING_DATA_PIN_MID, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(SORTING_DATA_READY_PIN, GPIO.OUT, initial=GPIO.LOW)
            SORTING_GPIO_ENABLED = True
            print(f"Sorting GPIO pins set. Sorting is ENABLED.")
        except Exception as e:
            print(f"ERROR: Failed to set up sorting GPIO pins: {e}. Sorting is DISABLED.")
            SORTING_GPIO_ENABLED = False
    else:
        print("Skipping Sorting GPIO setup. Sorting is DISABLED.")
        SORTING_GPIO_ENABLED = False

    # Automation Control Pin Initialization
    if gpio_bcm_mode_set:
        print(f"Attempting to initialize GPIO pin {CONTROL_PIN} for Automation Control...")
        try:
            GPIO.setup(CONTROL_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            CONTROL_PIN_SETUP_OK = True
            print(f"Automation Control Pin {CONTROL_PIN} set as INPUT with PULL-DOWN.")
        except Exception as e:
            print(f"ERROR: Failed to set up Automation Control Pin {CONTROL_PIN}: {e}")
            CONTROL_PIN_SETUP_OK = False
    else:
        print(f"Skipping Automation Control Pin {CONTROL_PIN} setup.")
        CONTROL_PIN_SETUP_OK = False

    # Create Testing Folder
    try:
        testing_path = os.path.join(BASE_PATH, TESTING_FOLDER_NAME)
        os.makedirs(testing_path, exist_ok=True)
        print(f"Ensured testing folder exists: {testing_path}")
    except Exception as e:
        print(f"ERROR: Could not create testing folder: {e}")

    print("--- Hardware Initialization Complete ---")

def initialize_ai():
    global loaded_labels
    global interpreter_visual, input_details_visual, output_details_visual
    global interpreter_magnetism, input_details_magnetism, output_details_magnetism
    global interpreter_resistivity, input_details_resistivity, output_details_resistivity
    
    print("\n--- Initializing AI Components (Hierarchical) ---")
    
    # Load Labels
    print(f"Loading labels from: {LABELS_PATH}")
    try:
        with open(LABELS_PATH, 'r') as f:
            loaded_labels = [line.strip() for line in f.readlines()]
        if not loaded_labels:
            raise ValueError("Labels file is empty.")
        print(f"Loaded {len(loaded_labels)} labels: {loaded_labels}")
    except Exception as e:
        print(f"FATAL ERROR: Reading labels file '{LABELS_FILENAME}' failed: {e}")
        return False

    def load_model(model_name, model_path):
        print(f"\n--- Loading {model_name} Model ---")
        try:
            interpreter = Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            if not input_details or not output_details:
                print(f"ERROR: Failed to get input/output details for {model_name}.")
                return None, None, None
            if output_details[0]['shape'][-1] != len(loaded_labels):
                print(f"ERROR: {model_name} output size ({output_details[0]['shape'][-1]}) does not match label count ({len(loaded_labels)}).")
                return None, None, None
            print(f"{model_name} model loaded successfully.")
            return interpreter, input_details, output_details
        except Exception as e:
            print(f"ERROR: Failed to load TFLite model '{os.path.basename(model_path)}': {e}")
            traceback.print_exc()
            return None, None, None

    # Load all three models
    interpreter_visual, input_details_visual, output_details_visual = load_model("Visual", MODEL_VISUAL_PATH)
    interpreter_magnetism, input_details_magnetism, output_details_magnetism = load_model("Magnetism", MODEL_MAGNETISM_PATH)
    interpreter_resistivity, input_details_resistivity, output_details_resistivity = load_model("Resistivity", MODEL_RESISTIVITY_PATH)

    all_models_loaded = all([interpreter_visual, interpreter_magnetism, interpreter_resistivity])
    if not all_models_loaded:
        print("\n--- AI Initialization Failed: One or more models could not be loaded. ---")
        return False
    else:
        if not math.isclose(sum(MODEL_WEIGHTS.values()), 1.0):
            print(f"WARNING: Model weights sum to {sum(MODEL_WEIGHTS.values())}, not 1.0.")
        print("\n--- All AI Models Initialized Successfully ---")
        return True


def calibrate_sensors(is_manual_call=False, force_calibration=False):
    global IDLE_VOLTAGE, IDLE_RP_VALUE, g_calibration_baseline_age
    global hall_sensor, ldc_initialized
    global g_baseline_voltage_history, g_baseline_rp_history
    
    if is_manual_call:
        print("\n" + "="*10 + " Manual Sensor Calibration Triggered " + "="*10)
    
    hall_avail, ldc_avail = hall_sensor is not None, ldc_initialized
    if not hall_avail and not ldc_avail:
        if is_manual_call:
            print("Warning: Calibration - No sensors available.")
        return

    # Check if calibration is needed
    current_time = time.time()
    calibration_age = current_time - g_calibration_baseline_age
    
    if not force_calibration and not is_manual_call and calibration_age < MAX_CALIBRATION_AGE:
        # Skip calibration if baseline is still fresh
        return

    old_voltage = IDLE_VOLTAGE
    old_rp = IDLE_RP_VALUE

    if hall_avail:
        avg_v = get_averaged_hall_voltage(num_samples=NUM_SAMPLES_CALIBRATION)
        if avg_v is not None:
            IDLE_VOLTAGE = avg_v
            g_baseline_voltage_history.append(avg_v)
        else:
            IDLE_VOLTAGE = 0.0

    if ldc_avail:
        avg_rp = get_averaged_rp_data(num_samples=NUM_SAMPLES_CALIBRATION)
        if avg_rp is not None:
            IDLE_RP_VALUE = int(round(avg_rp))
            g_baseline_rp_history.append(avg_rp)
        else:
            IDLE_RP_VALUE = 0

    g_calibration_baseline_age = current_time

    # Detect significant drift
    voltage_drift = abs(IDLE_VOLTAGE - old_voltage) if old_voltage != 0 else 0
    rp_drift = abs(IDLE_RP_VALUE - old_rp) if old_rp != 0 else 0
    
    drift_detected = (voltage_drift > DRIFT_THRESHOLD_VOLTAGE or 
                     rp_drift > DRIFT_THRESHOLD_RP)
    
    if is_manual_call or drift_detected:
        drift_msg = f" (Drift: V={voltage_drift:.4f}, RP={rp_drift})" if drift_detected else ""
        print(f"Calibration Results: Hall Idle={IDLE_VOLTAGE:.4f}V, LDC Idle={IDLE_RP_VALUE}{drift_msg}")

def get_averaged_hall_voltage(num_samples=NUM_SAMPLES_PER_UPDATE):
    if not hall_sensor:
        return None
    readings = []
    for _ in range(num_samples):
        try:
            readings.append(hall_sensor.voltage)
        except Exception as e:
            print(f"Warning: Error reading Hall sensor: {e}")
            return None
    if readings:
        return statistics.mean(readings)
    else:
        return None

def get_averaged_rp_data(num_samples=NUM_SAMPLES_PER_UPDATE):
    if not ldc_initialized:
        return None
    readings = []
    for _ in range(num_samples):
        rp_value = get_ldc_rpdata()
        if rp_value is not None:
            readings.append(rp_value)
    if readings:
        return statistics.mean(readings)
    else:
        return None


def manage_automation_flow():
    global window, g_previous_control_state, g_last_calibration_time, g_accepting_triggers
    global g_last_paused_calibration_time
    global CONTROL_PIN, CONTROL_PIN_SETUP_OK, RPi_GPIO_AVAILABLE
    
    if not window or not window.winfo_exists():
        return
    
    if not CONTROL_PIN_SETUP_OK or not RPi_GPIO_AVAILABLE:
        if window.winfo_exists():
            window.after(CONTROL_CHECK_INTERVAL_MS, manage_automation_flow)
        return

    try:
        current_state = GPIO.input(CONTROL_PIN)
        current_time = time.time()
        
        if g_previous_control_state is None:
            g_previous_control_state = current_state

        # RISING EDGE (LOW -> HIGH): Trigger classification if system is armed
        if g_accepting_triggers and current_state == GPIO.HIGH and g_previous_control_state == GPIO.LOW:
            print(f"AUTOMATION: Armed and rising edge detected. Scheduling classification...")
            window.after(2000, capture_and_classify)
        
        # STATE IS LOW: Perform periodic calibration
        elif current_state == GPIO.LOW:
            if (current_time - g_last_calibration_time) >= 0.5:
                calibrate_sensors(is_manual_call=False)
                g_last_calibration_time = current_time
        
        elif not g_accepting_triggers:  # System is paused
            if (current_time - g_last_paused_calibration_time) >= CALIBRATION_INTERVAL_PAUSED:
                print("Performing background calibration during pause...")
                calibrate_sensors(is_manual_call=False)
                g_last_paused_calibration_time = current_time

        g_previous_control_state = current_state
        
    except Exception as e:
        print(f"ERROR: Could not read Automation Control Pin {CONTROL_PIN}: {e}")
    
    if window.winfo_exists():
        window.after(CONTROL_CHECK_INTERVAL_MS, manage_automation_flow)

def capture_and_classify():
    global window, camera, IDLE_VOLTAGE, IDLE_RP_VALUE
    global rv_image_label, rv_prediction_label, rv_confidence_label, rv_magnetism_label, rv_ldc_label
    global save_output_var, g_accepting_triggers
    global interpreter_visual, interpreter_magnetism, interpreter_resistivity
    
    # Disarm the trigger
    g_accepting_triggers = False
    print("\n" + "="*10 + " Automatic Classification Triggered (System Paused) " + "="*10)
    
    print("Performing fresh calibration for maximum accuracy...")
    calibrate_sensors(is_manual_call=True, force_calibration=True)
    
    # Pre-flight Checks
    if not all([interpreter_visual, interpreter_magnetism, interpreter_resistivity]):
        messagebox.showerror("Error", "One or more AI Models are not initialized. Cannot classify.")
        print("Classification aborted: AI not ready.")
        show_live_view()
        return
    
    if not camera or not camera.isOpened():
        messagebox.showerror("Error", "Camera is not available. Cannot capture image.")
        print("Classification aborted: Camera not ready.")
        show_live_view()
        return
    
    window.update_idletasks()
    
    # Capture Image
    ret, frame = camera.read()
    if not ret or frame is None:
        messagebox.showerror("Capture Error", "Failed to capture image from camera.")
        print("ERROR: Failed to read frame from camera.")
        show_live_view()
        return
    
    try:
        img_captured_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        messagebox.showerror("Image Error", f"Failed to process captured image: {e}")
        print(f"ERROR: Failed converting captured frame to PIL Image: {e}")
        show_live_view()
        return
    
    # Capture Sensor Data
    print(f"Capturing fresh sensor values for classification...")
    
    # Magnetism Reading
    current_mag_mT = None
    avg_v_capture = get_averaged_hall_voltage(num_samples=NUM_SAMPLES_CALIBRATION)
    if avg_v_capture is not None and abs(SENSITIVITY_V_PER_MILLITESLA) > 1e-9:
        current_mag_mT = (avg_v_capture - IDLE_VOLTAGE) / SENSITIVITY_V_PER_MILLITESLA
    else:
        print("ERROR: Hall sensor read failed during capture.")
    
    # LDC Reading
    current_rp_raw = None
    current_rp_delta = None
    resistivity_features = None
    avg_rp_val = get_averaged_rp_data(num_samples=NUM_SAMPLES_CALIBRATION)
    if avg_rp_val is not None:
        current_rp_raw = avg_rp_val
        current_rp_delta = current_rp_raw - IDLE_RP_VALUE
        resistivity_features = [current_rp_raw, current_rp_delta]
    else:
        print("ERROR: LDC read failed during capture.")
    
    # Preprocess Data for Each Model
    print("\n--- Preprocessing all inputs ---")
    visual_input = preprocess_visual_input(img_captured_pil, input_details_visual)
    magnetism_input = preprocess_numerical_input(current_mag_mT, 'magnetism', input_details_magnetism)
    resistivity_input = preprocess_numerical_input(resistivity_features, 'resistivity', input_details_resistivity)
    
    # Run Inference on Each Model
    print("\n--- Running inference on all models ---")
    output_visual = run_single_inference(interpreter_visual, input_details_visual, visual_input)
    output_magnetism = run_single_inference(interpreter_magnetism, input_details_magnetism, magnetism_input)
    output_resistivity = run_single_inference(interpreter_resistivity, input_details_resistivity, resistivity_input)
    
    # Combine Results with Hierarchical Logic
    model_outputs = {
        'visual': output_visual,
        'magnetism': output_magnetism,
        'resistivity': output_resistivity
    }
    predicted_label, confidence = postprocess_hierarchical_output(model_outputs)
    
    print(f"\n--- HIERARCHICAL RESULT: Prediction='{predicted_label}', Confidence={confidence:.1%} ---")
    
    # Handle Saving and Sorting
    mag_display_text = ""
    if current_mag_mT is not None:
        if abs(current_mag_mT) < 0.1:
            mag_display_text = f"{current_mag_mT * 1000:+.1f}µT"
        else:
            mag_display_text = f"{current_mag_mT:+.2f}mT"
    else:
        mag_display_text = "ReadErr"
    
    ldc_display_text = f"{int(round(current_rp_raw))}" if current_rp_raw is not None else "ReadErr"
    if current_rp_delta is not None:
        ldc_display_text += f" (Δ{current_rp_delta:+,})"
    
    if save_output_var and save_output_var.get() == 1:
        save_result_screenshot(img_captured_pil, predicted_label, confidence, mag_display_text, ldc_display_text)
    
    send_sorting_signal(predicted_label)
    
    # Update Results Display
    if rv_image_label:
        try:
            w, h_img = img_captured_pil.size
            aspect = h_img/w if w>0 else 0.75
            display_h = int(RESULT_IMG_DISPLAY_WIDTH * aspect) if aspect > 0 else int(RESULT_IMG_DISPLAY_WIDTH * 0.75)
            img_disp = img_captured_pil.resize((RESULT_IMG_DISPLAY_WIDTH, max(1, display_h)), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_disp)
            rv_image_label.img_tk = img_tk
            rv_image_label.config(image=img_tk, text="")
        except Exception as e:
            print(f"ERROR: Results image update: {e}")
            if placeholder_img_tk:
                rv_image_label.config(image=placeholder_img_tk, text="ImgErr")
                rv_image_label.img_tk = placeholder_img_tk
            else:
                rv_image_label.config(image='', text="ImgErr")
                rv_image_label.img_tk = None
    
    if rv_prediction_label:
        rv_prediction_label.config(text=f"{predicted_label}")
    if rv_confidence_label:
        rv_confidence_label.config(text=f"{confidence:.1%}")
    if rv_magnetism_label:
        rv_magnetism_label.config(text=mag_display_text)
    if rv_ldc_label:
        rv_ldc_label.config(text=ldc_display_text)
    
    show_results_view()
    print("="*10 + " Capture & Classify Complete " + "="*10 + "\n")

def calibrate_and_show_live_view():
    global g_accepting_triggers, g_last_paused_calibration_time
    print("\n--- 'Classify Another' clicked: Re-arming GPIO trigger ---")
    g_accepting_triggers = True
    g_last_paused_calibration_time = time.time()  # Reset paused calibration timer
    calibrate_sensors(is_manual_call=True, force_calibration=True)
    show_live_view()
