# CODE 3.1.0 - AI Metal Classifier GUI with Multi-Model Fusion
# Description: Displays live sensor data and camera feed.
#              - Uses three separate AI models (visual, magnetism, resistivity).
#              - Fuses the results using configurable weights.
#              - Scaler values are embedded directly in the code, removing file dependencies.
#              - On startup, waits for a LOW->HIGH signal on GPIO 23 to classify.
#              - After classifying, it enters a PAUSED state, ignoring new triggers.
#              - Clicking 'Classify Another' RE-ARMS the system for the next trigger.

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
from tkinter import messagebox
import cv2 # OpenCV for camera access
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

# --- I2C/ADS1115 Imports (for Hall Sensor/Magnetism) ---
I2C_ENABLED = False
try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    I2C_ENABLED = True
    print("I2C/ADS1115 libraries imported successfully.")
except (ImportError, NotImplementedError) as e:
    print(f"Warning: I2C/ADS1115 libraries not found or not supported. Magnetism readings disabled. Error: {e}")

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
except (ImportError, RuntimeError) as e:
    print(f"Warning: RPi.GPIO library not found or failed to import. GPIO functions disabled. Error: {e}")

# ==================================
# === Constants and Configuration ===
# ==================================

# --- AI Model Weights (MUST sum to 1.0) ---
AI_WEIGHTS = {
    'visual': 0.0,
    'magnetism': 1.0,
    'resistivity': 0.0
}

# --- Embedded Scaler Values ---
SCALER_VALUES = {
    'magnetism': {
        'mean': 0.00048415711947626843,
        'scale': 0.0007762457818081904
    },
    'resistivity': {
        'mean': 61000.82880523732,
        'scale': 1362.7716526399417
    }
}

# --- General Configuration ---
NUM_SAMPLES_PER_UPDATE = 3
NUM_SAMPLES_CALIBRATION = 15
GUI_UPDATE_INTERVAL_MS = 100
CAMERA_UPDATE_INTERVAL_MS = 50
LDC_DISPLAY_BUFFER_SIZE = 5

CAMERA_INDEX = 0
DISPLAY_IMG_WIDTH = 640
DISPLAY_IMG_HEIGHT = 480
RESULT_IMG_DISPLAY_WIDTH = 280

try:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_PATH = os.getcwd()

# --- File Paths ---
VISUAL_MODEL_FILENAME = "visual_model.tflite"
MAGNETISM_MODEL_FILENAME = "magnetism_model.tflite"
RESISTIVITY_MODEL_FILENAME = "resistivity_model.tflite"
LABELS_FILENAME = "material_labels.txt"
TESTING_FOLDER_NAME = "testing"

VISUAL_MODEL_PATH = os.path.join(BASE_PATH, VISUAL_MODEL_FILENAME)
MAGNETISM_MODEL_PATH = os.path.join(BASE_PATH, MAGNETISM_MODEL_FILENAME)
RESISTIVITY_MODEL_PATH = os.path.join(BASE_PATH, RESISTIVITY_MODEL_FILENAME)
LABELS_PATH = os.path.join(BASE_PATH, LABELS_FILENAME)

# --- AI Image Dimensions ---
AI_IMG_WIDTH = 224
AI_IMG_HEIGHT = 224

# --- Hardware Pins ---
HALL_ADC_CHANNEL = ADS.P0 if I2C_ENABLED else None
SENSITIVITY_V_PER_TESLA = 0.0002
SENSITIVITY_V_PER_MILLITESLA = SENSITIVITY_V_PER_TESLA * 1000
IDLE_VOLTAGE = 0.0

SPI_BUS = 0
SPI_DEVICE = 0
SPI_SPEED = 500000
SPI_MODE = 0b00
CS_PIN = 8

LDC_CHIP_ID = 0xD4
START_CONFIG_REG, RP_SET_REG, TC1_REG, TC2_REG, DIG_CONFIG_REG, ALT_CONFIG_REG, \
D_CONF_REG, INTB_MODE_REG, RP_DATA_MSB_REG, RP_DATA_LSB_REG, CHIP_ID_REG = \
0x0B, 0x01, 0x02, 0x03, 0x04, 0x05, 0x0C, 0x0A, 0x22, 0x21, 0x3F
ACTIVE_CONVERSION_MODE, SLEEP_MODE = 0x00, 0x01
IDLE_RP_VALUE = 0

# --- GPIO Pins ---
SORTING_DATA_PIN_LSB = 16
SORTING_DATA_PIN_MID = 6
SORTING_DATA_READY_PIN = 26
CONTROL_PIN = 23
CONTROL_CHECK_INTERVAL_MS = 50

# ============================
# === Global Objects/State ===
# ============================
camera = None
i2c = None
ads = None
hall_sensor = None
spi = None
ldc_initialized = False

# AI Models
interpreter_visual, interpreter_magnetism, interpreter_resistivity = None, None, None
loaded_labels = []

RP_DISPLAY_BUFFER = deque(maxlen=LDC_DISPLAY_BUFFER_SIZE)
g_last_live_magnetism_mT = 0.0

# GUI Globals
window = None
# ... (other GUI globals remain the same) ...
lv_camera_label, lv_magnetism_label, lv_ldc_label, lv_save_checkbox = (None,) * 4
rv_image_label, rv_prediction_label, rv_confidence_label, rv_magnetism_label, rv_ldc_label, rv_classify_another_button = (None,) * 6
placeholder_img_tk = None
save_output_var = None


# Automation State
g_accepting_triggers = True
g_previous_control_state = None
g_last_calibration_time = 0
CONTROL_PIN_SETUP_OK = False
SORTING_GPIO_ENABLED = False


# =========================
# === Hardware Setup ===
# =========================
def initialize_hardware():
    global camera, i2c, ads, hall_sensor, spi, ldc_initialized, CS_PIN
    global SORTING_GPIO_ENABLED, RPi_GPIO_AVAILABLE, CONTROL_PIN_SETUP_OK

    print("\n--- Initializing Hardware ---")
    # Camera
    try:
        camera = cv2.VideoCapture(CAMERA_INDEX)
        time.sleep(0.5)
        if not camera or not camera.isOpened(): raise ValueError("Camera not opened.")
        print(f"Camera {CAMERA_INDEX} opened.")
    except Exception as e:
        print(f"ERROR: Failed to open camera: {e}")
        camera = None

    # I2C/ADS1115
    if I2C_ENABLED:
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            ads = ADS.ADS1115(i2c)
            hall_sensor = AnalogIn(ads, HALL_ADC_CHANNEL)
            print("ADS1115 initialized for Hall sensor.")
        except Exception as e:
            print(f"ERROR: Initializing I2C/ADS1115 failed: {e}")
            i2c = ads = hall_sensor = None
    else:
        print("Skipping I2C/ADS1115 setup.")

    # GPIO Setup
    gpio_bcm_mode_set = False
    if RPi_GPIO_AVAILABLE:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            gpio_bcm_mode_set = True
            print("GPIO BCM mode set.")
        except Exception as e:
            print(f"ERROR: GPIO.setmode(GPIO.BCM) failed: {e}")
    else:
        print("RPi.GPIO library not available. Skipping GPIO setups.")

    # SPI/LDC1101
    if SPI_ENABLED and gpio_bcm_mode_set:
        try:
            GPIO.setup(CS_PIN, GPIO.OUT, initial=GPIO.HIGH)
            spi = spidev.SpiDev()
            spi.open(SPI_BUS, SPI_DEVICE)
            spi.max_speed_hz = SPI_SPEED
            spi.mode = SPI_MODE
            print("SPI initialized for LDC.")
            if initialize_ldc1101():
                enable_ldc_rpmode()
            else:
                ldc_initialized = False
        except Exception as e:
            print(f"ERROR: SPI/LDC initialization failed: {e}")
            if spi: spi.close()
            spi = None
            ldc_initialized = False
    else:
        print("Skipping SPI/LDC1101 setup.")

    # Sorting GPIO
    if gpio_bcm_mode_set:
        try:
            GPIO.setup([SORTING_DATA_PIN_LSB, SORTING_DATA_PIN_MID, SORTING_DATA_READY_PIN], GPIO.OUT, initial=GPIO.LOW)
            SORTING_GPIO_ENABLED = True
            print("Sorting GPIO pins set. Sorting ENABLED.")
        except Exception as e:
            print(f"ERROR: Sorting GPIO setup failed: {e}. Sorting DISABLED.")
            SORTING_GPIO_ENABLED = False
    else:
        SORTING_GPIO_ENABLED = False

    # Automation Control Pin
    if gpio_bcm_mode_set:
        try:
            GPIO.setup(CONTROL_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            CONTROL_PIN_SETUP_OK = True
            print(f"Automation Control Pin {CONTROL_PIN} set as INPUT with PULL-DOWN.")
        except Exception as e:
            print(f"ERROR: Automation Control Pin setup failed: {e}")
            CONTROL_PIN_SETUP_OK = False

    # Testing Folder
    try:
        os.makedirs(os.path.join(BASE_PATH, TESTING_FOLDER_NAME), exist_ok=True)
    except Exception as e:
        print(f"ERROR: Could not create testing folder: {e}")

    print("--- Hardware Initialization Complete ---")

# =========================
# === AI Model Setup ======
# =========================
def initialize_ai():
    global interpreter_visual, interpreter_magnetism, interpreter_resistivity, loaded_labels
    print("\n--- Initializing AI Components ---")
    all_models_loaded = True

    # Load Labels
    try:
        with open(LABELS_PATH, 'r') as f:
            loaded_labels = [line.strip() for line in f.readlines()]
        if not loaded_labels: raise ValueError("Labels file is empty.")
        print(f"Loaded {len(loaded_labels)} labels: {loaded_labels}")
    except Exception as e:
        print(f"ERROR: Reading labels '{LABELS_FILENAME}': {e}")
        all_models_loaded = False
        return all_models_loaded

    # Load Visual Model
    try:
        print(f"Loading Visual model from: {VISUAL_MODEL_PATH}")
        interpreter_visual = Interpreter(model_path=VISUAL_MODEL_PATH)
        interpreter_visual.allocate_tensors()
        print("Visual model loaded.")
    except Exception as e:
        print(f"ERROR: Loading Visual model failed: {e}")
        interpreter_visual = None
        all_models_loaded = False

    # Load Magnetism Model
    try:
        print(f"Loading Magnetism model from: {MAGNETISM_MODEL_PATH}")
        interpreter_magnetism = Interpreter(model_path=MAGNETISM_MODEL_PATH)
        interpreter_magnetism.allocate_tensors()
        print("Magnetism model loaded.")
    except Exception as e:
        print(f"ERROR: Loading Magnetism model failed: {e}")
        interpreter_magnetism = None
        all_models_loaded = False
        
    # Load Resistivity Model
    try:
        print(f"Loading Resistivity model from: {RESISTIVITY_MODEL_PATH}")
        interpreter_resistivity = Interpreter(model_path=RESISTIVITY_MODEL_PATH)
        interpreter_resistivity.allocate_tensors()
        print("Resistivity model loaded.")
    except Exception as e:
        print(f"ERROR: Loading Resistivity model failed: {e}")
        interpreter_resistivity = None
        all_models_loaded = False

    if not all_models_loaded:
        print("--- AI Initialization Failed: One or more models could not be loaded. ---")
    else:
        print("--- All AI Models Initialized Successfully ---")
        
    return all_models_loaded

# =========================
# === LDC1101 Functions ===
# =========================
def ldc_write_register(reg_addr, value):
    if not spi or not RPi_GPIO_AVAILABLE: return False
    try:
        GPIO.output(CS_PIN, GPIO.LOW)
        spi.xfer2([reg_addr & 0x7F, value])
        GPIO.output(CS_PIN, GPIO.HIGH)
        return True
    except Exception as e:
        print(f"Warning: LDC write error: {e}")
        try: GPIO.output(CS_PIN, GPIO.HIGH)
        except Exception: pass
        return False

def ldc_read_register(reg_addr):
    if not spi or not RPi_GPIO_AVAILABLE: return None
    try:
        GPIO.output(CS_PIN, GPIO.LOW)
        result = spi.xfer2([reg_addr | 0x80, 0x00])
        GPIO.output(CS_PIN, GPIO.HIGH)
        return result[1]
    except Exception as e:
        print(f"Warning: LDC read error: {e}")
        try: GPIO.output(CS_PIN, GPIO.HIGH)
        except Exception: pass
        return None

def initialize_ldc1101():
    global ldc_initialized
    ldc_initialized = False
    if not spi: return False
    chip_id = ldc_read_register(CHIP_ID_REG)
    if chip_id != LDC_CHIP_ID:
        print(f"ERROR: LDC Chip ID mismatch! Read: {chip_id}")
        return False
    
    regs_to_write = {
        RP_SET_REG: 0x07, TC1_REG: 0x90, TC2_REG: 0xA0, DIG_CONFIG_REG: 0x03,
        ALT_CONFIG_REG: 0x00, D_CONF_REG: 0x00, INTB_MODE_REG: 0x00
    }
    for reg, val in regs_to_write.items():
        if not ldc_write_register(reg, val): return False
    
    if not ldc_write_register(START_CONFIG_REG, SLEEP_MODE): return False
    time.sleep(0.02)
    ldc_initialized = True
    return True

def enable_ldc_rpmode():
    if not spi or not ldc_initialized: return False
    if not ldc_write_register(ALT_CONFIG_REG, 0x00): return False
    if not ldc_write_register(D_CONF_REG, 0x00): return False
    return ldc_write_register(START_CONFIG_REG, ACTIVE_CONVERSION_MODE)

def get_ldc_rpdata():
    if not spi or not ldc_initialized: return None
    msb = ldc_read_register(RP_DATA_MSB_REG)
    lsb = ldc_read_register(RP_DATA_LSB_REG)
    if msb is None or lsb is None: return None
    return (msb << 8) | lsb

# ============================
# === Sensor Reading Logic ===
# ============================
def get_averaged_hall_voltage(num_samples=NUM_SAMPLES_PER_UPDATE):
    if not hall_sensor: return None
    readings = [hall_sensor.voltage for _ in range(num_samples)]
    return statistics.mean(readings) if readings else None

def get_averaged_rp_data(num_samples=NUM_SAMPLES_PER_UPDATE):
    if not ldc_initialized: return None
    readings = [val for _ in range(num_samples) if (val := get_ldc_rpdata()) is not None]
    return statistics.mean(readings) if readings else None
    
# =================================
# === AI Inference and Fusion ===
# =================================
def run_all_inferences(image_pil, mag_mT, ldc_rp_raw):
    """Runs inference on all three models and returns their raw outputs."""
    global interpreter_visual, interpreter_magnetism, interpreter_resistivity
    outputs = {'visual': None, 'magnetism': None, 'resistivity': None}

    # --- Visual Inference ---
    if interpreter_visual:
        try:
            input_details = interpreter_visual.get_input_details()[0]
            img_resized = image_pil.resize((AI_IMG_WIDTH, AI_IMG_HEIGHT), Image.Resampling.LANCZOS)
            image_np = np.array(img_resized.convert('RGB'), dtype=np.float32) / 255.0
            image_input = np.expand_dims(image_np, axis=0)
            
            interpreter_visual.set_tensor(input_details['index'], image_input.astype(input_details['dtype']))
            interpreter_visual.invoke()
            output_details = interpreter_visual.get_output_details()[0]
            outputs['visual'] = interpreter_visual.get_tensor(output_details['index'])
            print("Visual inference successful.")
        except Exception as e:
            print(f"ERROR during visual inference: {e}")

    # --- Magnetism Inference ---
    if interpreter_magnetism and mag_mT is not None:
        try:
            # IMPORTANT: Take absolute value before scaling
            abs_mag_mT = abs(float(mag_mT))
            
            # Scale the input
            mean = SCALER_VALUES['magnetism']['mean']
            scale = SCALER_VALUES['magnetism']['scale']
            scaled_mag = (abs_mag_mT - mean) / scale
            
            input_details = interpreter_magnetism.get_input_details()[0]
            interpreter_magnetism.set_tensor(input_details['index'], np.array([[scaled_mag]], dtype=np.float32))
            interpreter_magnetism.invoke()
            output_details = interpreter_magnetism.get_output_details()[0]
            outputs['magnetism'] = interpreter_magnetism.get_tensor(output_details['index'])
            print(f"Magnetism inference successful. Input: {abs_mag_mT:.4f} mT -> Scaled: {scaled_mag:.4f}")
        except Exception as e:
            print(f"ERROR during magnetism inference: {e}")

    # --- Resistivity Inference ---
    if interpreter_resistivity and ldc_rp_raw is not None:
        try:
            # Scale the input
            mean = SCALER_VALUES['resistivity']['mean']
            scale = SCALER_VALUES['resistivity']['scale']
            scaled_res = (float(ldc_rp_raw) - mean) / scale

            input_details = interpreter_resistivity.get_input_details()[0]
            interpreter_resistivity.set_tensor(input_details['index'], np.array([[scaled_res]], dtype=np.float32))
            interpreter_resistivity.invoke()
            output_details = interpreter_resistivity.get_output_details()[0]
            outputs['resistivity'] = interpreter_resistivity.get_tensor(output_details['index'])
            print(f"Resistivity inference successful. Input: {ldc_rp_raw:.2f} -> Scaled: {scaled_res:.4f}")
        except Exception as e:
            print(f"ERROR during resistivity inference: {e}")
            
    return outputs

def fuse_and_postprocess(outputs):
    """Fuses model outputs using weights and determines the final prediction."""
    global loaded_labels, AI_WEIGHTS
    print("\n--- Fusing and Postprocessing AI Outputs ---")
    
    if not any(v is not None for v in outputs.values()):
        print("ERROR: All model inferences failed. Cannot fuse.")
        return "Inference Error", 0.0

    num_classes = len(loaded_labels)
    fused_scores = np.zeros(num_classes, dtype=np.float32)

    for model_name, output_data in outputs.items():
        if output_data is not None and output_data.size == num_classes:
            fused_scores += output_data[0] * AI_WEIGHTS[model_name]
            print(f"Added '{model_name}' output (weight {AI_WEIGHTS[model_name]}) to fused scores.")
        else:
            print(f"Skipping '{model_name}' output (failed or size mismatch).")
            
    # Apply softmax to get final probabilities
    exp_scores = np.exp(fused_scores - np.max(fused_scores))
    probabilities = exp_scores / exp_scores.sum()
    
    predicted_index = np.argmax(probabilities)
    confidence = float(probabilities[predicted_index])
    predicted_label = loaded_labels[predicted_index]

    print(f"Final Prediction: '{predicted_label}', Confidence: {confidence:.4f}")
    print("--- Postprocessing Complete ---")
    return predicted_label, confidence

# ==================================
# === Sorting Signal Functions ===
# ==================================
def send_sorting_signal(material_label):
    if not SORTING_GPIO_ENABLED: return
    print(f"\n--- Sending Sorting Signal for: {material_label} ---")
    mid_val, lsb_val, signal_desc = GPIO.LOW, GPIO.LOW, "Others (00)"
    if material_label == "Aluminum": mid_val, lsb_val, signal_desc = GPIO.LOW, GPIO.HIGH, "Aluminum (01)"
    elif material_label == "Copper": mid_val, lsb_val, signal_desc = GPIO.HIGH, GPIO.LOW, "Copper (10)"
    elif material_label == "Steel": mid_val, lsb_val, signal_desc = GPIO.HIGH, GPIO.HIGH, "Steel (11)"

    try:
        GPIO.output(SORTING_DATA_READY_PIN, GPIO.LOW)
        time.sleep(0.01)
        GPIO.output(SORTING_DATA_PIN_MID, mid_val)
        GPIO.output(SORTING_DATA_PIN_LSB, lsb_val)
        print(f"Set GPIO Pins for {signal_desc}")
        time.sleep(0.01)
        GPIO.output(SORTING_DATA_READY_PIN, GPIO.HIGH)
        print(f"Pulsed Data Ready HIGH")
        time.sleep(0.05)
        GPIO.output(SORTING_DATA_READY_PIN, GPIO.LOW)
        print(f"Set Data Ready LOW")
    except Exception as e:
        print(f"ERROR: Failed to send sorting signal: {e}")
    finally:
        if RPi_GPIO_AVAILABLE:
            GPIO.output([SORTING_DATA_PIN_MID, SORTING_DATA_PIN_LSB, SORTING_DATA_READY_PIN], GPIO.LOW)

# ==============================
# === GUI and State Logic ======
# ==============================
def calibrate_and_show_live_view():
    global g_accepting_triggers
    print("\n--- 'Classify Another' clicked: Re-arming GPIO trigger ---")
    g_accepting_triggers = True
    calibrate_sensors(is_manual_call=True)
    show_live_view()

def show_live_view():
    if results_view_frame.winfo_ismapped(): results_view_frame.pack_forget()
    if not live_view_frame.winfo_ismapped(): live_view_frame.pack(fill=tk.BOTH, expand=True)

def show_results_view():
    if live_view_frame.winfo_ismapped(): live_view_frame.pack_forget()
    if not results_view_frame.winfo_ismapped(): results_view_frame.pack(fill=tk.BOTH, expand=True)
    
def save_result_screenshot(image_pil, prediction, confidence, mag_text, ldc_text):
    # This function remains largely the same, only its call signature might change
    # to accept the individual sensor texts.
    print("\n--- Saving Result Screenshot ---")
    # ... implementation from original script is fine ...

def capture_and_classify():
    global window, camera, IDLE_VOLTAGE, IDLE_RP_VALUE
    global g_accepting_triggers, save_output_var

    g_accepting_triggers = False
    print("\n" + "="*10 + " Automatic Classification Triggered (System Paused) " + "="*10)

    if not all((interpreter_visual, interpreter_magnetism, interpreter_resistivity)):
        messagebox.showerror("Error", "One or more AI Models are not initialized.")
        show_live_view(); return
    if not camera or not camera.isOpened():
        messagebox.showerror("Error", "Camera is not available.")
        show_live_view(); return

    window.update_idletasks()

    ret, frame = camera.read()
    if not ret or frame is None:
        messagebox.showerror("Capture Error", "Failed to capture image.")
        show_live_view(); return
    img_captured_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # --- SENSOR READING ---
    print("Capturing fresh sensor values for classification...")
    avg_v_capture = get_averaged_hall_voltage(num_samples=NUM_SAMPLES_CALIBRATION)
    current_mag_mT = ((avg_v_capture - IDLE_VOLTAGE) / SENSITIVITY_V_PER_MILLITESLA) if avg_v_capture is not None else None
    
    current_rp_raw = get_averaged_rp_data(num_samples=NUM_SAMPLES_CALIBRATION)

    # --- Run Inferences ---
    model_outputs = run_all_inferences(img_captured_pil, current_mag_mT, current_rp_raw)
    
    # --- Fuse and Get Final Result ---
    predicted_label, confidence = fuse_and_postprocess(model_outputs)

    # --- Display and Save ---
    mag_display_text = f"{abs(current_mag_mT):.2f}mT" if current_mag_mT is not None else "N/A"
    ldc_display_text = f"{int(round(current_rp_raw))}" if current_rp_raw is not None else "N/A"
    
    if save_output_var.get():
        save_result_screenshot(img_captured_pil, predicted_label, confidence, mag_display_text, ldc_display_text)

    send_sorting_signal(predicted_label)
    update_results_display(img_captured_pil, predicted_label, confidence, mag_display_text, ldc_display_text)
    show_results_view()
    print("="*10 + " Capture & Classify Complete " + "="*10 + "\n")

def update_results_display(image_pil, prediction, confidence, mag_text, ldc_text):
    """Helper to update all result view widgets."""
    global rv_image_label, rv_prediction_label, rv_confidence_label
    global rv_magnetism_label, rv_ldc_label, placeholder_img_tk

    try:
        w, h = image_pil.size
        aspect = h/w if w > 0 else 0.75
        h_disp = int(RESULT_IMG_DISPLAY_WIDTH * aspect)
        img_disp = image_pil.resize((RESULT_IMG_DISPLAY_WIDTH, h_disp), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_disp)
        rv_image_label.img_tk = img_tk
        rv_image_label.config(image=img_tk)
    except Exception as e:
        print(f"Error updating results image: {e}")
        rv_image_label.config(image=placeholder_img_tk)

    rv_prediction_label.config(text=f"{prediction}")
    rv_confidence_label.config(text=f"{confidence:.1%}")
    rv_magnetism_label.config(text=mag_text)
    rv_ldc_label.config(text=ldc_text)


def calibrate_sensors(is_manual_call=False):
    global IDLE_VOLTAGE, IDLE_RP_VALUE
    if is_manual_call: print("\n" + "="*10 + " Manual Sensor Calibration " + "="*10)
    
    if hall_sensor:
        avg_v = get_averaged_hall_voltage(num_samples=NUM_SAMPLES_CALIBRATION)
        IDLE_VOLTAGE = avg_v if avg_v is not None else 0.0
    
    if ldc_initialized:
        avg_rp = get_averaged_rp_data(num_samples=NUM_SAMPLES_CALIBRATION)
        IDLE_RP_VALUE = int(round(avg_rp)) if avg_rp is not None else 0
    
    if is_manual_call:
        print(f"Calibration Results: Hall Idle={IDLE_VOLTAGE:.4f}V, LDC Idle={IDLE_RP_VALUE}")

# --- GUI Update Loops ---
def update_camera_feed():
    if not window.winfo_exists(): return
    if camera and camera.isOpened():
        ret, frame = camera.read()
        if ret:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_pil.thumbnail((DISPLAY_IMG_WIDTH, DISPLAY_IMG_HEIGHT), Image.Resampling.NEAREST)
            img_tk = ImageTk.PhotoImage(img_pil)
            lv_camera_label.img_tk = img_tk
            lv_camera_label.configure(image=img_tk)
    window.after(CAMERA_UPDATE_INTERVAL_MS, update_camera_feed)

def update_magnetism():
    global g_last_live_magnetism_mT
    if not window.winfo_exists(): return
    display_text = "N/A"
    avg_v = get_averaged_hall_voltage()
    if avg_v is not None:
        current_mT = (avg_v - IDLE_VOLTAGE) / SENSITIVITY_V_PER_MILLITESLA
        g_last_live_magnetism_mT = current_mT
        display_text = f"{current_mT:+.2f}mT"
        if IDLE_VOLTAGE == 0.0: display_text += " (NoCal)"
    else:
        display_text = "ReadErr"
    lv_magnetism_label.config(text=display_text)
    window.after(GUI_UPDATE_INTERVAL_MS, update_magnetism)

def update_ldc_reading():
    if not window.winfo_exists(): return
    display_text = "N/A"
    if ldc_initialized:
        avg_rp = get_averaged_rp_data()
        if avg_rp is not None:
            RP_DISPLAY_BUFFER.append(avg_rp)
            cur_rp = int(round(statistics.mean(RP_DISPLAY_BUFFER)))
            delta = cur_rp - IDLE_RP_VALUE
            display_text = f"{cur_rp} (Δ{delta:+,})"
            if IDLE_RP_VALUE == 0: display_text = f"{cur_rp} (NoCal)"
        else:
            display_text = "ReadErr"
    lv_ldc_label.config(text=display_text)
    window.after(GUI_UPDATE_INTERVAL_MS, update_ldc_reading)

def manage_automation_flow():
    global g_previous_control_state, g_last_calibration_time, g_accepting_triggers
    if not window.winfo_exists() or not CONTROL_PIN_SETUP_OK:
        if window.winfo_exists(): window.after(CONTROL_CHECK_INTERVAL_MS, manage_automation_flow)
        return
    
    current_state = GPIO.input(CONTROL_PIN)
    if g_previous_control_state is None: g_previous_control_state = current_state

    if g_accepting_triggers and current_state == GPIO.HIGH and g_previous_control_state == GPIO.LOW:
        print(f"AUTOMATION: Rising edge detected. Scheduling classification...")
        window.after(1000, capture_and_classify) # 1s delay
    
    elif current_state == GPIO.LOW:
        if (time.time() - g_last_calibration_time) >= 0.5:
            calibrate_sensors(is_manual_call=False)
            g_last_calibration_time = time.time()

    g_previous_control_state = current_state
    if window.winfo_exists(): window.after(CONTROL_CHECK_INTERVAL_MS, manage_automation_flow)

# ======================
# === GUI Setup ========
# ======================
# (setup_gui function remains largely the same, ensure variable names match)
def setup_gui():
    global window, main_frame, placeholder_img_tk, live_view_frame, results_view_frame
    global lv_camera_label, lv_magnetism_label, lv_ldc_label, lv_save_checkbox
    global rv_image_label, rv_prediction_label, rv_confidence_label, rv_magnetism_label, rv_ldc_label, rv_classify_another_button
    global label_font, readout_font, button_font, title_font, result_title_font, result_value_font, pred_font
    global save_output_var

    window = tk.Tk()
    window.title("AI Metal Classifier v3.1.0 (Multi-Model Fusion)")
    window.geometry("800x600")
    
    # --- Fonts and Styles (simplified for brevity) ---
    font_family = "DejaVu Sans"
    title_font = tkFont.Font(family=font_family, size=16, weight="bold")
    label_font = tkFont.Font(family=font_family, size=10)
    readout_font = tkFont.Font(family=font_family + " Mono", size=12, weight="bold")
    button_font = tkFont.Font(family=font_family, size=10, weight="bold")
    result_title_font = tkFont.Font(family=font_family, size=11, weight="bold")
    result_value_font = tkFont.Font(family=font_family + " Mono", size=12, weight="bold")
    pred_font = tkFont.Font(family=font_family, size=16, weight="bold")
    
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TLabel", font=label_font)
    style.configure("TButton", font=button_font, padding=(8,5))
    style.configure("Readout.TLabel", font=readout_font, foreground="#0000AA")
    style.configure("ResultValue.TLabel", font=result_value_font, foreground="#0000AA")
    style.configure("Prediction.TLabel", font=pred_font, foreground="#AA0000")
    style.configure("TCheckbutton", font=label_font)
    
    main_frame = ttk.Frame(window, padding="5")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # --- Live View Frame ---
    live_view_frame = ttk.Frame(main_frame)
    live_view_frame.pack(fill=tk.BOTH, expand=True)
    # ... (rest of the live view layout is the same as original) ...
    lv_camera_label = ttk.Label(live_view_frame, text="Initializing...", anchor="center", borderwidth=1, relief="sunken")
    lv_camera_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    lv_controls_frame = ttk.Frame(live_view_frame, width=220)
    lv_controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

    lv_readings_frame = ttk.Labelframe(lv_controls_frame, text="Live Readings", padding=5)
    lv_readings_frame.pack(fill=tk.X, pady=5)
    lv_magnetism_label = ttk.Label(lv_readings_frame, text="Init...", style="Readout.TLabel")
    lv_ldc_label = ttk.Label(lv_readings_frame, text="Init...", style="Readout.TLabel")
    ttk.Label(lv_readings_frame, text="Magnetism:").pack(side=tk.LEFT)
    lv_magnetism_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Label(lv_readings_frame, text="LDC:").pack(side=tk.LEFT) # Simplified text
    lv_ldc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    lv_actions_frame = ttk.Labelframe(lv_controls_frame, text="Status & Options", padding=5)
    lv_actions_frame.pack(fill=tk.X, pady=5)
    ttk.Label(lv_actions_frame, text="Automated Control Active", font=tkFont.Font(weight='bold'), foreground='green').pack()
    save_output_var = tk.IntVar(value=0)
    lv_save_checkbox = ttk.Checkbutton(lv_actions_frame, text="Save Result Screenshot", variable=save_output_var)
    lv_save_checkbox.pack(pady=5)
    
    # --- Results View Frame ---
    results_view_frame = ttk.Frame(main_frame)
    # ... (layout is the same as original, just ensure variable names are correct) ...
    placeholder_img_tk = create_placeholder_image(RESULT_IMG_DISPLAY_WIDTH, int(RESULT_IMG_DISPLAY_WIDTH * 0.75))
    rv_content_frame = ttk.Frame(results_view_frame)
    rv_content_frame.pack(expand=True)
    ttk.Label(rv_content_frame, text="Classification Result", font=title_font).pack(pady=10)
    rv_image_label = ttk.Label(rv_content_frame, image=placeholder_img_tk)
    rv_image_label.pack(pady=5)
    
    rv_details_frame = ttk.Frame(rv_content_frame)
    rv_details_frame.pack(pady=10, fill=tk.X)
    ttk.Label(rv_details_frame, text="Material:", font=result_title_font).grid(row=0, column=0, sticky='w')
    rv_prediction_label = ttk.Label(rv_details_frame, text="---", style="Prediction.TLabel")
    rv_prediction_label.grid(row=0, column=1, sticky='w')
    # ... other result labels
    ttk.Label(rv_details_frame, text="Confidence:", font=result_title_font).grid(row=1, column=0, sticky='w')
    rv_confidence_label = ttk.Label(rv_details_frame, text="---", style="ResultValue.TLabel")
    rv_confidence_label.grid(row=1, column=1, sticky='w')
    ttk.Label(rv_details_frame, text="Magnetism:", font=result_title_font).grid(row=2, column=0, sticky='w')
    rv_magnetism_label = ttk.Label(rv_details_frame, text="---", style="ResultValue.TLabel")
    rv_magnetism_label.grid(row=2, column=1, sticky='w')
    ttk.Label(rv_details_frame, text="LDC:", font=result_title_font).grid(row=3, column=0, sticky='w')
    rv_ldc_label = ttk.Label(rv_details_frame, text="---", style="ResultValue.TLabel")
    rv_ldc_label.grid(row=3, column=1, sticky='w')

    rv_classify_another_button = ttk.Button(rv_content_frame, text="<< Classify Another", command=calibrate_and_show_live_view)
    rv_classify_another_button.pack(pady=10)

    results_view_frame.pack_forget() # Start with live view

# ==========================
# === Main Execution =======
# ==========================
def run_application():
    setup_gui()
    update_camera_feed()
    update_magnetism()
    update_ldc_reading()
    manage_automation_flow()
    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.quit()

def cleanup_resources():
    print("\n--- Cleaning up resources ---")
    if camera: camera.release()
    if spi:
        if ldc_initialized: ldc_write_register(START_CONFIG_REG, SLEEP_MODE)
        spi.close()
    if RPi_GPIO_AVAILABLE: GPIO.cleanup()
    print("--- Cleanup complete ---")

if __name__ == '__main__':
    print("="*30 + "\n Starting Multi-Model AI Classifier \n" + "="*30)
    hw_init_attempted = False
    try:
        initialize_hardware()
        hw_init_attempted = True
        if initialize_ai():
            run_application()
        else:
            messagebox.showerror("AI Init Failed", "Could not load all AI models. Check console.")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected.")
    except Exception as e:
        print(f"\nFATAL ERROR in main execution: {e}")
        traceback.print_exc()
    finally:
        if 'window' in globals() and window:
            try: window.destroy()
            except tk.TclError: pass
        if hw_init_attempted:
            cleanup_resources()
        print("\nApplication finished.")
