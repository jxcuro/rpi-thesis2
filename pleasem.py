# CODE 3.0.22 - Multi-AI Metal Classifier GUI with Gated Automation
# Description: Displays live sensor data and camera feed.
#              - Uses THREE separate AI models: visual, magnetism, and resistivity
#              - Each AI has configurable weights for ensemble prediction

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
    # Preferred import for dedicated TFLite runtime package
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        # Fallback for full TensorFlow package (less common on Pi for inference)
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        print("ERROR: TensorFlow Lite Runtime is not installed.")
        print("Please install it (e.g., 'pip install tflite-runtime' or follow official Pi instructions)")
        exit()


# --- I2C/ADS1115 Imports (for Hall Sensor/Magnetism) ---
I2C_ENABLED = False # Default to False, set True if libraries import successfully
try:
    import board      # Adafruit Blinka library for hardware pins
    import busio      # For I2C communication
    import adafruit_ads1x15.ads1115 as ADS # ADS1115 library
    from adafruit_ads1x15.analog_in import AnalogIn # Helper for reading analog pins
    I2C_ENABLED = True
    print("I2C/ADS1115 libraries imported successfully.")
except ImportError:
    print("Warning: I2C/ADS1115 libraries (board, busio, adafruit-circuitpython-ads1x15) not found.")
    print("Ensure Adafruit Blinka is installed and configured for your Pi.")
    print("Magnetism readings will be disabled.")
except NotImplementedError:
    # This can happen if Blinka doesn't detect the board/platform correctly
    print("Warning: I2C not supported on this platform according to Blinka. Magnetism readings disabled.")
except Exception as e:
    print(f"Warning: Error importing I2C/ADS1115 libraries: {e}. Magnetism readings disabled.")

# --- SPI/LDC1101 & RPi.GPIO Imports ---
SPI_ENABLED = False # For spidev library itself
RPi_GPIO_AVAILABLE = False # For RPi.GPIO library itself
try:
    import spidev   # For SPI communication
    SPI_ENABLED = True
    print("SPI library (spidev) imported successfully.")
except ImportError:
    print("Warning: SPI library (spidev) not found. LDC readings will be disabled.")

try:
    import RPi.GPIO as GPIO # For controlling Chip Select pin and sorting pins
    RPi_GPIO_AVAILABLE = True
    print("RPi.GPIO library imported successfully (needed for LDC CS, Sorting, and Automation Control).")
except ImportError:
    print("Warning: RPi.GPIO library not found. LDC CS control, Sorting, and Automation Control will be disabled.")
except RuntimeError:
    print("Warning: RPi.GPIO library likely requires root privileges (sudo). LDC CS, Sorting, and Automation Control may fail.")
except Exception as e:
    print(f"Warning: Error importing RPi.GPIO library: {e}. LDC CS control, Sorting, and Automation Control disabled.")


# --- Sorting GPIO Configuration ---
SORTING_GPIO_ENABLED = False # Default to False, set True if RPi.GPIO is available and setup succeeds
SORTING_DATA_PIN_LSB = 16 # BCM Pin for LSB of sorting data
SORTING_DATA_PIN_MID = 6  # BCM Pin for MID/MSB of sorting data (2-bit signal)
SORTING_DATA_READY_PIN = 26 # BCM Pin to signal data is ready for sorter

# --- Automation GPIO Configuration ---
CONTROL_PIN = 23 # BCM Pin (Physical Pin 16) for automation control
CONTROL_PIN_SETUP_OK = False # Tracks if the control pin was set up
CONTROL_CHECK_INTERVAL_MS = 50 # How often to check the control pin (in milliseconds)


# ==================================
# === Constants and Configuration ===
# ==================================
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
    print(f"Warning: __file__ not defined, using current working directory as base path: {BASE_PATH}")

MODEL_FILENAME = "material_classifier_model.tflite"
LABELS_FILENAME = "material_labels.txt"
SCALER_FILENAME = "numerical_scaler.joblib"
MODEL_PATH = os.path.join(BASE_PATH, MODEL_FILENAME)
LABELS_PATH = os.path.join(BASE_PATH, LABELS_FILENAME)
SCALER_PATH = os.path.join(BASE_PATH, SCALER_FILENAME)
TESTING_FOLDER_NAME = "testing" # Folder to save screenshots

AI_IMG_WIDTH = 224
AI_IMG_HEIGHT = 224

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

# =========================
# === AI Model Configuration ===
# =========================

# Model file paths - update these to point to your three model files
VISUAL_MODEL_PATH = "visual_model.tftlite"
MAGNETISM_MODEL_PATH = "magnetism_model.tftlite" 
RESISTIVITY_MODEL_PATH = "resistivity_model.tftlite"

# AI Model Weights - EASILY EDITABLE
AI_WEIGHTS = {
    'visual': 0.0,      # Weight for visual AI model
    'magnetism': 1.0,   # Weight for magnetism AI model  
    'resistivity': 0.0  # Weight for resistivity AI model
}

# Ensure weights sum to 1.0
total_weight = sum(AI_WEIGHTS.values())
if abs(total_weight - 1.0) > 1e-6:
    print(f"WARNING: AI weights sum to {total_weight}, normalizing to 1.0")
    for key in AI_WEIGHTS:
        AI_WEIGHTS[key] /= total_weight

# Magnetism Model Scaler (embedded)
MAGNETISM_SCALER = {
    'mean': 0.00048415711947626843,
    'scale': 0.0007762457818081904
}

# Resistivity Model Scaler (embedded) 
RESISTIVITY_SCALER = {
    'mean': 61000.82880523732,
    'scale': 1362.7716526399417
}

# ============================
# === Global Objects/State ===
# ============================
camera = None
i2c = None
ads = None
hall_sensor = None
spi = None
ldc_initialized = False

interpreter = None
input_details = None
output_details = None
loaded_labels = []
numerical_scaler = None

RP_DISPLAY_BUFFER = deque(maxlen=LDC_DISPLAY_BUFFER_SIZE)
g_last_live_magnetism_mT = 0.0

# --- GUI Globals ---
window = None
main_frame = None
live_view_frame = None
results_view_frame = None
label_font, readout_font, button_font, title_font, result_title_font, result_value_font, pred_font = (None,) * 7
lv_camera_label, lv_magnetism_label, lv_ldc_label, lv_save_checkbox = (None,) * 4 
rv_image_label, rv_prediction_label, rv_confidence_label, rv_magnetism_label, rv_ldc_label, rv_classify_another_button = (None,) * 6
placeholder_img_tk = None
save_output_var = None

# --- NEW: State for Gated GPIO Automation ---
g_accepting_triggers = True     # Controls if the system will respond to a GPIO signal
g_previous_control_state = None # Tracks GPIO 23 state to detect rising edges
g_last_calibration_time = 0     # Timestamp of the last auto-calibration

# =========================
# === Hardware Setup ===
# =========================
def initialize_hardware():
    global camera, i2c, ads, hall_sensor, spi, ldc_initialized, CS_PIN
    global SORTING_GPIO_ENABLED, RPi_GPIO_AVAILABLE
    global CONTROL_PIN_SETUP_OK, CONTROL_PIN

    print("\n--- Initializing Hardware ---")

    # --- Camera Initialization ---
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

    # --- I2C/ADS1115 Initialization ---
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

    # --- GPIO general setup (BCM mode) ---
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

    # --- SPI/LDC1101 Initialization ---
    if SPI_ENABLED and gpio_bcm_mode_set:
        print("Initializing SPI and LDC1101 (CS pin setup)...")
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
            if spi: spi.close()
            spi = None
            ldc_initialized = False
    elif SPI_ENABLED and not gpio_bcm_mode_set: # spidev present, but GPIO failed
        print("Skipping LDC1101 setup because RPi.GPIO (needed for CS pin) failed BCM mode setup.")
        spi = None
        ldc_initialized = False
    else: # SPI_ENABLED is False or RPi.GPIO not available/BCM failed
        print("Skipping SPI/LDC1101 setup.")


    # --- Sorting GPIO Pin Initialization ---
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
        print("Skipping Sorting GPIO setup (RPi.GPIO not available or BCM mode failed). Sorting is DISABLED.")
        SORTING_GPIO_ENABLED = False


    # --- Automation Control Pin Initialization ---
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
        print(f"Skipping Automation Control Pin {CONTROL_PIN} setup (RPi.GPIO not available or BCM mode failed).")
        CONTROL_PIN_SETUP_OK = False

    # --- Create Testing Folder ---
    try:
        testing_path = os.path.join(BASE_PATH, TESTING_FOLDER_NAME)
        os.makedirs(testing_path, exist_ok=True)
        print(f"Ensured testing folder exists: {testing_path}")
    except Exception as e:
        print(f"ERROR: Could not create testing folder: {e}")


    print("--- Hardware Initialization Complete ---")

# =========================
# === AI Model Setup ======
# =========================
# Global AI variables for three models
visual_interpreter = None
magnetism_interpreter = None  
resistivity_interpreter = None
visual_input_details = None
magnetism_input_details = None
resistivity_input_details = None
visual_output_details = None
magnetism_output_details = None
resistivity_output_details = None

def initialize_ai():
    global visual_interpreter, magnetism_interpreter, resistivity_interpreter
    global visual_input_details, magnetism_input_details, resistivity_input_details
    global visual_output_details, magnetism_output_details, resistivity_output_details
    global loaded_labels
    
    print("=== Initializing Multi-AI System ===")
    
    # Initialize visual model
    visual_ready = False
    if os.path.exists(VISUAL_MODEL_PATH):
        try:
            print(f"Loading Visual AI model from: {VISUAL_MODEL_PATH}")
            visual_interpreter = Interpreter(model_path=VISUAL_MODEL_PATH)
            visual_interpreter.allocate_tensors()
            visual_input_details = visual_interpreter.get_input_details()
            visual_output_details = visual_interpreter.get_output_details()
            visual_ready = True
            print("Visual AI model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Loading Visual AI model: {e}")
            traceback.print_exc()
    else:
        print(f"ERROR: Visual model file not found: {VISUAL_MODEL_PATH}")
    
    # Initialize magnetism model
    magnetism_ready = False
    if os.path.exists(MAGNETISM_MODEL_PATH):
        try:
            print(f"Loading Magnetism AI model from: {MAGNETISM_MODEL_PATH}")
            magnetism_interpreter = Interpreter(model_path=MAGNETISM_MODEL_PATH)
            magnetism_interpreter.allocate_tensors()
            magnetism_input_details = magnetism_interpreter.get_input_details()
            magnetism_output_details = magnetism_interpreter.get_output_details()
            magnetism_ready = True
            print("Magnetism AI model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Loading Magnetism AI model: {e}")
            traceback.print_exc()
    else:
        print(f"ERROR: Magnetism model file not found: {MAGNETISM_MODEL_PATH}")
    
    # Initialize resistivity model
    resistivity_ready = False
    if os.path.exists(RESISTIVITY_MODEL_PATH):
        try:
            print(f"Loading Resistivity AI model from: {RESISTIVITY_MODEL_PATH}")
            resistivity_interpreter = Interpreter(model_path=RESISTIVITY_MODEL_PATH)
            resistivity_interpreter.allocate_tensors()
            resistivity_input_details = resistivity_interpreter.get_input_details()
            resistivity_output_details = resistivity_interpreter.get_output_details()
            resistivity_ready = True
            print("Resistivity AI model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Loading Resistivity AI model: {e}")
            traceback.print_exc()
    else:
        print(f"ERROR: Resistivity model file not found: {RESISTIVITY_MODEL_PATH}")
    
    # Check if at least one model is ready
    ai_ready = visual_ready or magnetism_ready or resistivity_ready
    
    if not ai_ready:
        print("--- Multi-AI Initialization Failed - No models loaded ---")
        return False
    else:
        print(f"--- Multi-AI Initialization Complete ---")
        print(f"Visual AI: {'Ready' if visual_ready else 'Failed'}")
        print(f"Magnetism AI: {'Ready' if magnetism_ready else 'Failed'}")
        print(f"Resistivity AI: {'Ready' if resistivity_ready else 'Failed'}")
        return True

def scale_magnetism_value(raw_value):
    """Scale magnetism value using embedded scaler parameters"""
    if raw_value is None:
        return 0.0
    # Convert to absolute value as requested
    abs_value = abs(float(raw_value))
    return (abs_value - MAGNETISM_SCALER['mean']) / MAGNETISM_SCALER['scale']

def scale_resistivity_value(raw_value):
    """Scale resistivity value using embedded scaler parameters"""
    if raw_value is None:
        return 0.0
    return (float(raw_value) - RESISTIVITY_SCALER['mean']) / RESISTIVITY_SCALER['scale']

# =========================
# === LDC1101 Functions ===
# =========================
def ldc_write_register(reg_addr, value):
    if not spi or not RPi_GPIO_AVAILABLE: return False
    success = False
    try:
        GPIO.output(CS_PIN, GPIO.LOW)
        spi.xfer2([reg_addr & 0x7F, value])
        GPIO.output(CS_PIN, GPIO.HIGH)
        success = True
    except Exception as e:
        print(f"Warning: LDC write error (Reg 0x{reg_addr:02X}): {e}")
        try:
            if RPi_GPIO_AVAILABLE: GPIO.output(CS_PIN, GPIO.HIGH)
        except Exception: pass
    return success

def ldc_read_register(reg_addr):
    if not spi or not RPi_GPIO_AVAILABLE: return None
    read_value = None
    try:
        GPIO.output(CS_PIN, GPIO.LOW)
        result = spi.xfer2([reg_addr | 0x80, 0x00])
        GPIO.output(CS_PIN, GPIO.HIGH)
        read_value = result[1]
    except Exception as e:
        print(f"Warning: LDC read error (Reg 0x{reg_addr:02X}): {e}")
        try:
            if RPi_GPIO_AVAILABLE: GPIO.output(CS_PIN, GPIO.HIGH)
        except Exception: pass
    return read_value

def initialize_ldc1101():
    global ldc_initialized
    ldc_initialized = False
    if not spi: print("Cannot initialize LDC1101: SPI not available."); return False
    print("Initializing LDC1101...")
    try:
        chip_id = ldc_read_register(CHIP_ID_REG)
        if chip_id is None: print("ERROR: Failed to read LDC Chip ID."); return False
        if chip_id != LDC_CHIP_ID: print(f"ERROR: LDC Chip ID mismatch! (Read:0x{chip_id:02X})"); return False
        print(f"LDC1101 Chip ID verified (0x{chip_id:02X}).")
        regs_to_write = { RP_SET_REG: 0x07, TC1_REG: 0x90, TC2_REG: 0xA0, DIG_CONFIG_REG: 0x03,
                          ALT_CONFIG_REG: 0x00, D_CONF_REG: 0x00, INTB_MODE_REG: 0x00 }
        for reg, val in regs_to_write.items():
            if not ldc_write_register(reg, val): print(f"ERROR: LDC write reg 0x{reg:02X} failed."); return False
        if not ldc_write_register(START_CONFIG_REG, SLEEP_MODE): return False
        time.sleep(0.02)
        print("LDC1101 Configuration successful.")
        ldc_initialized = True
        return True
    except Exception as e: print(f"ERROR: Exception during LDC1101 Initialization: {e}"); ldc_initialized = False; return False

def enable_ldc_powermode(mode):
    if not spi or not ldc_initialized: return False
    if ldc_write_register(START_CONFIG_REG, mode): time.sleep(0.01); return True
    else: print(f"Warning: Failed to set LDC power mode register."); return False

def enable_ldc_rpmode():
    if not spi or not ldc_initialized: print("Warning: Cannot enable LDC RP mode (SPI/LDC not ready)."); return False
    print("Enabling LDC RP+L Mode...")
    try:
        if not ldc_write_register(ALT_CONFIG_REG, 0x00): return False
        if not ldc_write_register(D_CONF_REG, 0x00): return False
        if enable_ldc_powermode(ACTIVE_CONVERSION_MODE): print("LDC RP+L Mode Enabled and Active."); return True
        else: print("Failed to set LDC to Active mode for RP+L."); return False
    except Exception as e: print(f"Warning: Failed to enable LDC RP mode: {e}"); return False

def get_ldc_rpdata():
    if not spi or not ldc_initialized: return None
    try:
        msb = ldc_read_register(RP_DATA_MSB_REG)
        lsb = ldc_read_register(RP_DATA_LSB_REG)
        if msb is None or lsb is None: return None
        return (msb << 8) | lsb
    except Exception as e: print(f"Warning: Exception while reading LDC RP data: {e}"); return None

# ============================
# === Sensor Reading (Avg) ===
# ============================
def get_averaged_hall_voltage(num_samples=NUM_SAMPLES_PER_UPDATE):
    if not hall_sensor: return None
    readings = []
    for _ in range(num_samples):
        try: readings.append(hall_sensor.voltage)
        except Exception as e: print(f"Warning: Error reading Hall sensor: {e}. Aborting average."); return None
    if readings: return statistics.mean(readings)
    else: return None

def get_averaged_rp_data(num_samples=NUM_SAMPLES_PER_UPDATE):
    if not ldc_initialized: return None
    readings = []
    for _ in range(num_samples):
        rp_value = get_ldc_rpdata()
        if rp_value is not None: readings.append(rp_value)
    if readings: return statistics.mean(readings)
    else: return None

# ==========================
# === AI Processing ========
# ==========================
def preprocess_visual_input(image_pil):
    """Preprocess image for visual AI model"""
    global visual_input_details
    print("--- Preprocessing Visual Input ---")
    
    if visual_interpreter is None or visual_input_details is None:
        print("ERROR: Visual AI Model not initialized.")
        return None
        
    try:
        img_resized = image_pil.resize((AI_IMG_WIDTH, AI_IMG_HEIGHT), Image.Resampling.LANCZOS)
        image_np = np.array(img_resized.convert('RGB'), dtype=np.float32)
        image_np /= 255.0
        image_input = np.expand_dims(image_np, axis=0)
        
        # Convert to expected dtype
        input_dtype = visual_input_details[0]['dtype']
        if input_dtype == np.uint8:
            image_input = (image_input * 255.0).astype(np.uint8)
        else:
            image_input = image_input.astype(input_dtype)
            
        print(f"Visual input preprocessed. Shape: {image_input.shape}")
        return image_input
    except Exception as e:
        print(f"ERROR: Visual preprocessing failed: {e}")
        return None

def preprocess_magnetism_input(mag_mT):
    """Preprocess magnetism data for magnetism AI model"""
    global magnetism_input_details
    print("--- Preprocessing Magnetism Input ---")
    
    if magnetism_interpreter is None or magnetism_input_details is None:
        print("ERROR: Magnetism AI Model not initialized.")
        return None
        
    try:
        # Scale the magnetism value (with absolute value conversion)
        scaled_mag = scale_magnetism_value(mag_mT)
        magnetism_input = np.array([[scaled_mag]], dtype=magnetism_input_details[0]['dtype'])
        print(f"Magnetism input preprocessed. Raw: {mag_mT}, Scaled: {scaled_mag}")
        return magnetism_input
    except Exception as e:
        print(f"ERROR: Magnetism preprocessing failed: {e}")
        return None

def preprocess_resistivity_input(ldc_rp_raw):
    """Preprocess resistivity data for resistivity AI model"""
    global resistivity_input_details
    print("--- Preprocessing Resistivity Input ---")
    
    if resistivity_interpreter is None or resistivity_input_details is None:
        print("ERROR: Resistivity AI Model not initialized.")
        return None
        
    try:
        # Scale the resistivity value
        scaled_rp = scale_resistivity_value(ldc_rp_raw)
        resistivity_input = np.array([[scaled_rp]], dtype=resistivity_input_details[0]['dtype'])
        print(f"Resistivity input preprocessed. Raw: {ldc_rp_raw}, Scaled: {scaled_rp}")
        return resistivity_input
    except Exception as e:
        print(f"ERROR: Resistivity preprocessing failed: {e}")
        return None

def run_visual_inference(visual_input):
    """Run inference on visual AI model"""
    global visual_interpreter, visual_output_details
    print("--- Running Visual AI Inference ---")
    
    if visual_interpreter is None or visual_input is None:
        print("ERROR: Visual interpreter/input not ready.")
        return None
        
    try:
        visual_interpreter.set_tensor(visual_input_details[0]['index'], visual_input)
        visual_interpreter.invoke()
        output_data = visual_interpreter.get_tensor(visual_output_details[0]['index'])
        print(f"Visual inference complete. Output shape: {output_data.shape}")
        return output_data
    except Exception as e:
        print(f"ERROR: Visual inference failed: {e}")
        return None

def run_magnetism_inference(magnetism_input):
    """Run inference on magnetism AI model"""
    global magnetism_interpreter, magnetism_output_details
    print("--- Running Magnetism AI Inference ---")
    
    if magnetism_interpreter is None or magnetism_input is None:
        print("ERROR: Magnetism interpreter/input not ready.")
        return None
        
    try:
        magnetism_interpreter.set_tensor(magnetism_input_details[0]['index'], magnetism_input)
        magnetism_interpreter.invoke()
        output_data = magnetism_interpreter.get_tensor(magnetism_output_details[0]['index'])
        print(f"Magnetism inference complete. Output shape: {output_data.shape}")
        return output_data
    except Exception as e:
        print(f"ERROR: Magnetism inference failed: {e}")
        return None

def run_resistivity_inference(resistivity_input):
    """Run inference on resistivity AI model"""
    global resistivity_interpreter, resistivity_output_details
    print("--- Running Resistivity AI Inference ---")
    
    if resistivity_interpreter is None or resistivity_input is None:
        print("ERROR: Resistivity interpreter/input not ready.")
        return None
        
    try:
        resistivity_interpreter.set_tensor(resistivity_input_details[0]['index'], resistivity_input)
        resistivity_interpreter.invoke()
        output_data = resistivity_interpreter.get_tensor(resistivity_output_details[0]['index'])
        print(f"Resistivity inference complete. Output shape: {output_data.shape}")
        return output_data
    except Exception as e:
        print(f"ERROR: Resistivity inference failed: {e}")
        return None

def combine_predictions(visual_output, magnetism_output, resistivity_output):
    """Combine predictions from all three AI models using weighted ensemble"""
    global loaded_labels, AI_WEIGHTS
    print("--- Combining Multi-AI Predictions ---")
    
    if not loaded_labels:
        print("ERROR: No labels loaded for prediction combination.")
        return "Error", 0.0
    
    # Initialize combined probabilities
    num_classes = len(loaded_labels)
    combined_probs = np.zeros(num_classes)
    total_weight_used = 0.0
    
    # Process visual predictions
    if visual_output is not None and visual_interpreter is not None:
        try:
            visual_probs = visual_output[0] if len(visual_output.shape) == 2 else visual_output
            if len(visual_probs) == num_classes:
                combined_probs += visual_probs * AI_WEIGHTS['visual']
                total_weight_used += AI_WEIGHTS['visual']
                print(f"Visual AI contribution: weight={AI_WEIGHTS['visual']}")
            else:
                print(f"WARNING: Visual output size mismatch: {len(visual_probs)} vs {num_classes}")
        except Exception as e:
            print(f"ERROR processing visual predictions: {e}")
    
    # Process magnetism predictions  
    if magnetism_output is not None and magnetism_interpreter is not None:
        try:
            magnetism_probs = magnetism_output[0] if len(magnetism_output.shape) == 2 else magnetism_output
            if len(magnetism_probs) == num_classes:
                combined_probs += magnetism_probs * AI_WEIGHTS['magnetism']
                total_weight_used += AI_WEIGHTS['magnetism']
                print(f"Magnetism AI contribution: weight={AI_WEIGHTS['magnetism']}")
            else:
                print(f"WARNING: Magnetism output size mismatch: {len(magnetism_probs)} vs {num_classes}")
        except Exception as e:
            print(f"ERROR processing magnetism predictions: {e}")
    
    # Process resistivity predictions
    if resistivity_output is not None and resistivity_interpreter is not None:
        try:
            resistivity_probs = resistivity_output[0] if len(resistivity_output.shape) == 2 else resistivity_output
            if len(resistivity_probs) == num_classes:
                combined_probs += resistivity_probs * AI_WEIGHTS['resistivity']
                total_weight_used += AI_WEIGHTS['resistivity']
                print(f"Resistivity AI contribution: weight={AI_WEIGHTS['resistivity']}")
            else:
                print(f"WARNING: Resistivity output size mismatch: {len(resistivity_probs)} vs {num_classes}")
        except Exception as e:
            print(f"ERROR processing resistivity predictions: {e}")
    
    # Normalize by total weight used (in case some models failed)
    if total_weight_used > 0:
        combined_probs /= total_weight_used
        print(f"Total weight used: {total_weight_used}")
    else:
        print("ERROR: No valid predictions from any AI model.")
        return "No AI", 0.0
    
    # Get final prediction
    predicted_index = np.argmax(combined_probs)
    confidence = float(combined_probs[predicted_index])
    predicted_label = loaded_labels[predicted_index]
    
    print(f"Final Ensemble Prediction: '{predicted_label}', Confidence: {confidence:.4f}")
    print("--- Multi-AI Prediction Complete ---")
    
    return predicted_label, confidence


def capture_and_classify():
    global window, camera, IDLE_VOLTAGE, IDLE_RP_VALUE
    global visual_interpreter, magnetism_interpreter, resistivity_interpreter
    global rv_image_label, rv_prediction_label, rv_confidence_label, rv_magnetism_label, rv_ldc_label
    global save_output_var
    global g_accepting_triggers

    # --- MODIFICATION: Disarm the trigger as soon as we start processing ---
    g_accepting_triggers = False
    print("\n" + "="*10 + " Multi-AI Classification Triggered (System Paused) " + "="*10)
    
    # Check if at least one AI model is ready
    if not (visual_interpreter or magnetism_interpreter or resistivity_interpreter):
        messagebox.showerror("Error", "No AI Models are initialized. Cannot classify.")
        print("Classification aborted: No AI models ready.")
        show_live_view()
        return
        
    if not camera or not camera.isOpened():
        messagebox.showerror("Error", "Camera is not available. Cannot capture image.")
        print("Classification aborted: Camera not ready.")
        show_live_view()
        return

    window.update_idletasks()

    # Capture image
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

    # --- SENSOR READING SECTION ---
    print(f"Capturing fresh sensor values for multi-AI classification...")
    mag_display_text, sensor_warning = "N/A", False
    current_mag_mT = None
    
    # Get magnetism reading
    avg_v_capture = get_averaged_hall_voltage(num_samples=NUM_SAMPLES_CALIBRATION)
    if avg_v_capture is not None and abs(SENSITIVITY_V_PER_MILLITESLA) > 1e-9:
        try:
            current_mag_mT = (avg_v_capture - IDLE_VOLTAGE) / SENSITIVITY_V_PER_MILLITESLA
            
            # Format display text
            if abs(current_mag_mT) < 0.1:
                mag_display_text = f"{current_mag_mT * 1000:+.1f}µT"
            else:
                mag_display_text = f"{current_mag_mT:+.2f}mT"
            
            if IDLE_VOLTAGE == 0.0:
                mag_display_text += " (NoCal)"
        except Exception as e_calc: 
            mag_display_text = "CalcErr"
            print(f"Warn: Magnetism calculation for capture failed: {e_calc}")
            sensor_warning = True
    else:
        mag_display_text = "ReadErr"
        print("ERROR: Hall sensor read failed during capture.")
        sensor_warning = True

    # Get resistivity reading
    avg_rp_val = get_averaged_rp_data(num_samples=NUM_SAMPLES_CALIBRATION)
    current_rp_raw, ldc_display_text = None, "N/A"
    if avg_rp_val is not None:
        current_rp_raw = avg_rp_val
        current_rp_int = int(round(avg_rp_val))
        delta_rp_display = current_rp_int - IDLE_RP_VALUE
        ldc_display_text = f"{current_rp_int}"
        if IDLE_RP_VALUE != 0:
            ldc_display_text += f" (Δ{delta_rp_display:+,})"
        else:
            ldc_display_text += " (NoCal)"
    else:
        ldc_display_text = "ReadErr"
        print("ERROR: LDC read fail.")
        sensor_warning = True
        
    if sensor_warning:
        print("WARNING: Sensor issues may affect classification.")

    # --- MULTI-AI INFERENCE SECTION ---
    print("=== Starting Multi-AI Inference ===")
    
    # Preprocess inputs for each AI model
    visual_input = preprocess_visual_input(img_captured_pil) if visual_interpreter else None
    magnetism_input = preprocess_magnetism_input(current_mag_mT) if magnetism_interpreter else None
    resistivity_input = preprocess_resistivity_input(current_rp_raw) if resistivity_interpreter else None
    
    # Run inference on each available model
    visual_output = run_visual_inference(visual_input) if visual_input is not None else None
    magnetism_output = run_magnetism_inference(magnetism_input) if magnetism_input is not None else None
    resistivity_output = run_resistivity_inference(resistivity_input) if resistivity_input is not None else None
    
    # Combine predictions from all models
    predicted_label, confidence = combine_predictions(visual_output, magnetism_output, resistivity_output)
    
    print(f"--- Multi-AI Classification Result: Prediction='{predicted_label}', Confidence={confidence:.1%} ---")

    # Save result if requested
    if save_output_var and save_output_var.get() == 1:
        save_result_screenshot(img_captured_pil, predicted_label, confidence, mag_display_text, ldc_display_text)

    # Send sorting signal
    send_sorting_signal(predicted_label)

    # Update Results Display
    if rv_image_label:
        try:
            w, h_img = img_captured_pil.size; aspect = h_img/w if w>0 else 0.75
            display_h = int(RESULT_IMG_DISPLAY_WIDTH * aspect) if aspect > 0 else int(RESULT_IMG_DISPLAY_WIDTH * 0.75)
            img_disp = img_captured_pil.resize((RESULT_IMG_DISPLAY_WIDTH, max(1, display_h)), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_disp); rv_image_label.img_tk = img_tk
            rv_image_label.config(image=img_tk, text="")
        except Exception as e:
            print(f"ERROR: Results image update: {e}")
            if placeholder_img_tk: rv_image_label.config(image=placeholder_img_tk, text="ImgErr"); rv_image_label.img_tk = placeholder_img_tk
            else: rv_image_label.config(image='', text="ImgErr"); rv_image_label.img_tk = None
    if rv_prediction_label: rv_prediction_label.config(text=f"{predicted_label}")
    if rv_confidence_label: rv_confidence_label.config(text=f"{confidence:.1%}")
    if rv_magnetism_label: rv_magnetism_label.config(text=mag_display_text)
    if rv_ldc_label: rv_ldc_label.config(text=ldc_display_text)

    show_results_view()
    print("="*10 + " Capture & Classify Complete " + "="*10 + "\n")

def calibrate_sensors(is_manual_call=False):
    global IDLE_VOLTAGE, IDLE_RP_VALUE
    global hall_sensor, ldc_initialized

    if is_manual_call:
        print("\n" + "="*10 + " Manual Sensor Calibration Triggered " + "="*10)

    hall_avail, ldc_avail = hall_sensor is not None, ldc_initialized
    if not hall_avail and not ldc_avail:
        if is_manual_call: print("Warning: Calibration - No sensors available.")
        return

    if hall_avail:
        avg_v = get_averaged_hall_voltage(num_samples=NUM_SAMPLES_CALIBRATION)
        if avg_v is not None: IDLE_VOLTAGE = avg_v
        else: IDLE_VOLTAGE = 0.0
    
    if ldc_avail:
        avg_rp = get_averaged_rp_data(num_samples=NUM_SAMPLES_CALIBRATION)
        if avg_rp is not None: IDLE_RP_VALUE = int(round(avg_rp))
        else: IDLE_RP_VALUE = 0
    
    if is_manual_call:
        print(f"Calibration Results: Hall Idle={IDLE_VOLTAGE:.4f}V, LDC Idle={IDLE_RP_VALUE}")

def update_camera_feed():
    global lv_camera_label, window, camera
    if not window or not window.winfo_exists(): return
    img_tk = None
    if camera and camera.isOpened():
        ret, frame = camera.read()
        if ret and frame is not None:
            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((DISPLAY_IMG_WIDTH, DISPLAY_IMG_HEIGHT), Image.Resampling.NEAREST)
                img_tk = ImageTk.PhotoImage(img_pil)
            except Exception: pass
    if lv_camera_label:
        if img_tk: lv_camera_label.img_tk = img_tk; lv_camera_label.configure(image=img_tk, text="")
        else:
            if not hasattr(lv_camera_label, 'no_cam_img'):
                lv_camera_label.no_cam_img = create_placeholder_image(DISPLAY_IMG_WIDTH // 2, DISPLAY_IMG_HEIGHT // 2, '#BDBDBD', "No Feed")
            if lv_camera_label.no_cam_img and str(lv_camera_label.cget("image")) != str(lv_camera_label.no_cam_img):
                 lv_camera_label.configure(image=lv_camera_label.no_cam_img, text=""); lv_camera_label.img_tk = lv_camera_label.no_cam_img
            elif not lv_camera_label.no_cam_img and lv_camera_label.cget("text") != "Camera Failed":
                 lv_camera_label.configure(image='', text="Camera Failed"); lv_camera_label.img_tk = None
    if window and window.winfo_exists(): window.after(CAMERA_UPDATE_INTERVAL_MS, update_camera_feed)

def update_magnetism():
    global lv_magnetism_label, window, IDLE_VOLTAGE, hall_sensor
    global g_last_live_magnetism_mT

    if not window or not window.winfo_exists(): return
    display_text = "N/A"
    if hall_sensor:
        avg_v = get_averaged_hall_voltage(num_samples=NUM_SAMPLES_PER_UPDATE)
        if avg_v is not None:
            try:
                if abs(SENSITIVITY_V_PER_MILLITESLA) < 1e-9:
                    raise ZeroDivisionError("Sensitivity is zero")
                # Calculate magnetism directly from the averaged voltage
                current_mT = (avg_v - IDLE_VOLTAGE) / SENSITIVITY_V_PER_MILLITESLA
                
                # Store this raw value for other parts of the app if needed (though capture now does its own reading)
                g_last_live_magnetism_mT = current_mT

                # Format the display text based on the magnitude
                if abs(current_mT) < 0.1:
                    display_text = f"{current_mT * 1000:+.1f}µT"
                else:
                    display_text = f"{current_mT:+.2f}mT"
                
                # Add a note if the sensor hasn't been calibrated
                if IDLE_VOLTAGE == 0.0:
                    display_text += " (NoCal)"
            except Exception:
                display_text = "CalcErr"
                g_last_live_magnetism_mT = 0.0 # Reset on error
        else:
            display_text = "ReadErr"
            g_last_live_magnetism_mT = 0.0 # Reset on error

    if lv_magnetism_label and lv_magnetism_label.cget("text") != display_text:
        lv_magnetism_label.config(text=display_text)
        
    if window and window.winfo_exists():
        window.after(GUI_UPDATE_INTERVAL_MS, update_magnetism)

def update_ldc_reading():
    global lv_ldc_label, window, RP_DISPLAY_BUFFER, IDLE_RP_VALUE, ldc_initialized
    if not window or not window.winfo_exists(): return
    display_text = "N/A"
    if ldc_initialized:
        avg_rp = get_averaged_rp_data(num_samples=NUM_SAMPLES_PER_UPDATE)
        if avg_rp is not None:
            RP_DISPLAY_BUFFER.append(avg_rp)
            if RP_DISPLAY_BUFFER:
                cur_rp = int(round(statistics.mean(RP_DISPLAY_BUFFER)))
                delta = cur_rp - IDLE_RP_VALUE
                display_text = f"{cur_rp}"
                if IDLE_RP_VALUE != 0: display_text += f"(Δ{delta:+,})"
                else: display_text += "(NoCal)"
            else: display_text = "Buffer..."
        else: display_text = "ReadErr"
    if lv_ldc_label and lv_ldc_label.cget("text") != display_text: lv_ldc_label.config(text=display_text)
    if window and window.winfo_exists(): window.after(GUI_UPDATE_INTERVAL_MS, update_ldc_reading)

# --- MODIFIED: Automation loop now checks the g_accepting_triggers flag ---
def manage_automation_flow():
    """
    Checks the GPIO pin to manage calibration and classification.
    - If pin is LOW: Calibrates every 0.5 seconds.
    - On LOW->HIGH transition: Triggers classification ONLY if the system is armed.
    """
    global window, g_previous_control_state, g_last_calibration_time, g_accepting_triggers
    global CONTROL_PIN, CONTROL_PIN_SETUP_OK, RPi_GPIO_AVAILABLE

    if not window or not window.winfo_exists(): return
    if not CONTROL_PIN_SETUP_OK or not RPi_GPIO_AVAILABLE:
        if window.winfo_exists():
            window.after(CONTROL_CHECK_INTERVAL_MS, manage_automation_flow)
        return
    
    try:
        current_state = GPIO.input(CONTROL_PIN)
        
        if g_previous_control_state is None:
            g_previous_control_state = current_state

        # RISING EDGE (LOW -> HIGH): Trigger classification if system is armed
        if g_accepting_triggers and current_state == GPIO.HIGH and g_previous_control_state == GPIO.LOW:
            print(f"AUTOMATION: Armed and rising edge detected. Scheduling classification...")
            window.after(2000, capture_and_classify) # 2s delay
        
        # STATE IS LOW: Perform periodic calibration
        elif current_state == GPIO.LOW:
            current_time = time.time()
            if (current_time - g_last_calibration_time) >= 0.5:
                calibrate_sensors(is_manual_call=False)
                g_last_calibration_time = current_time

        g_previous_control_state = current_state

    except Exception as e:
        print(f"ERROR: Could not read Automation Control Pin {CONTROL_PIN}: {e}")

    if window.winfo_exists():
        window.after(CONTROL_CHECK_INTERVAL_MS, manage_automation_flow)


# ======================
# === GUI Setup ========
# ======================
def setup_gui():
    global window, main_frame, placeholder_img_tk, live_view_frame, results_view_frame
    global lv_camera_label, lv_magnetism_label, lv_ldc_label, lv_save_checkbox 
    global rv_image_label, rv_prediction_label, rv_confidence_label, rv_magnetism_label, rv_ldc_label, rv_classify_another_button
    global label_font, readout_font, button_font, title_font, result_title_font, result_value_font, pred_font
    global save_output_var 

    print("Setting up GUI...")
    window = tk.Tk()
    window.title("AI Metal Classifier v3.0.22 (RPi - Gated Automation)")
    window.geometry("800x600")
    style = ttk.Style()
    available_themes = style.theme_names(); style.theme_use('clam' if 'clam' in available_themes else 'default')
    try:
        font_family = "DejaVu Sans"
        title_font = tkFont.Font(family=font_family, size=16, weight="bold"); label_font = tkFont.Font(family=font_family, size=10)
        readout_font = tkFont.Font(family=font_family+" Mono", size=12, weight="bold"); button_font = tkFont.Font(family=font_family, size=10, weight="bold")
        result_title_font = tkFont.Font(family=font_family, size=11, weight="bold"); result_value_font = tkFont.Font(family=font_family+" Mono", size=12, weight="bold")
        pred_font = tkFont.Font(family=font_family, size=16, weight="bold")
    except tk.TclError:
        print("Warning: DejaVu fonts not found, using Tkinter default fonts.")
        title_font=tkFont.nametofont("TkHeadingFont"); label_font=tkFont.nametofont("TkTextFont"); readout_font=tkFont.nametofont("TkFixedFont")
        button_font=tkFont.nametofont("TkDefaultFont"); result_title_font=tkFont.nametofont("TkDefaultFont"); result_value_font=tkFont.nametofont("TkFixedFont")
        pred_font=tkFont.nametofont("TkHeadingFont")

    style.configure("TLabel", font=label_font, padding=2); style.configure("TButton", font=button_font, padding=(8,5))
    style.configure("Readout.TLabel", font=readout_font, foreground="#0000AA"); style.configure("ResultValue.TLabel", font=result_value_font, foreground="#0000AA")
    style.configure("Prediction.TLabel", font=pred_font, foreground="#AA0000")
    style.configure("TCheckbutton", font=label_font)

    main_frame = ttk.Frame(window, padding="5 5 5 5"); main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    main_frame.rowconfigure(0, weight=1); main_frame.columnconfigure(0, weight=1)

    live_view_frame = ttk.Frame(main_frame, padding="5 5 5 5"); live_view_frame.columnconfigure(0, weight=3); live_view_frame.columnconfigure(1, weight=1); live_view_frame.rowconfigure(0, weight=1)
    lv_camera_label = ttk.Label(live_view_frame, text="Initializing Camera...", anchor="center", borderwidth=1, relief="sunken", background="#CCCCCC")
    lv_camera_label.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="nsew")
    lv_controls_frame = ttk.Frame(live_view_frame); lv_controls_frame.grid(row=0, column=1, sticky="nsew", padx=(5,0)); lv_controls_frame.columnconfigure(0, weight=1)
    lv_readings_frame = ttk.Labelframe(lv_controls_frame, text=" Live Readings ", padding="8 4 8 4"); lv_readings_frame.grid(row=0, column=0, sticky="new", pady=(0, 10)); lv_readings_frame.columnconfigure(1, weight=1)
    ttk.Label(lv_readings_frame, text="Magnetism:").grid(row=0, column=0, sticky="w", padx=(0, 8)); lv_magnetism_label = ttk.Label(lv_readings_frame, text="Init...", style="Readout.TLabel"); lv_magnetism_label.grid(row=0, column=1, sticky="ew")
    ttk.Label(lv_readings_frame, text="LDC (Delta):").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(2,0)); lv_ldc_label = ttk.Label(lv_readings_frame, text="Init...", style="Readout.TLabel"); lv_ldc_label.grid(row=1, column=1, sticky="ew", pady=(2,0))
    
    lv_actions_frame = ttk.Labelframe(lv_controls_frame, text=" Status & Options ", padding="8 4 8 8")
    lv_actions_frame.grid(row=1, column=0, sticky="new", pady=(0,10)); lv_actions_frame.columnconfigure(0, weight=1)
    
    status_label_font = tkFont.Font(family="DejaVu Sans", size=10, weight="bold")
    ttk.Label(
        lv_actions_frame, 
        text="Automated Control Active", 
        font=status_label_font, 
        foreground="#006400" # Dark Green
    ).grid(row=0, column=0, sticky="ew", pady=(4,8))
    
    save_output_var = tk.IntVar(value=0)
    lv_save_checkbox = ttk.Checkbutton(lv_actions_frame, text="Save Result Screenshot", variable=save_output_var)
    lv_save_checkbox.grid(row=2, column=0, sticky="w", pady=(6,4), padx=(5,0))

    results_view_frame = ttk.Frame(main_frame, padding="10 10 10 10"); results_view_frame.rowconfigure(0, weight=1); results_view_frame.rowconfigure(1, weight=0); results_view_frame.rowconfigure(2, weight=1)
    results_view_frame.columnconfigure(0, weight=1); results_view_frame.columnconfigure(1, weight=0); results_view_frame.columnconfigure(2, weight=1)
    rv_content_frame = ttk.Frame(results_view_frame); rv_content_frame.grid(row=1, column=1, sticky="")
    ttk.Label(rv_content_frame, text="Classification Result", font=title_font).grid(row=0, column=0, columnspan=2, pady=(5, 15))
    placeholder_h = int(RESULT_IMG_DISPLAY_WIDTH * 0.75); placeholder_img_tk = create_placeholder_image(RESULT_IMG_DISPLAY_WIDTH, placeholder_h)
    rv_image_label = ttk.Label(rv_content_frame, anchor="center", borderwidth=1, relief="sunken")
    if placeholder_img_tk: rv_image_label.config(image=placeholder_img_tk); rv_image_label.img_tk = placeholder_img_tk
    else: rv_image_label.config(text="Image Area", width=30, height=15)
    rv_image_label.grid(row=1, column=0, columnspan=2, pady=(0, 15))
    rv_details_frame = ttk.Frame(rv_content_frame); rv_details_frame.grid(row=2, column=0, columnspan=2, pady=(0,15)); rv_details_frame.columnconfigure(1, weight=1); res_row = 0
    ttk.Label(rv_details_frame, text="Material:", font=result_title_font).grid(row=res_row, column=0, sticky="w", padx=(0,5)); rv_prediction_label = ttk.Label(rv_details_frame, text="---", style="Prediction.TLabel"); rv_prediction_label.grid(row=res_row, column=1, sticky="ew", padx=5); res_row += 1
    ttk.Label(rv_details_frame, text="Confidence:", font=result_title_font).grid(row=res_row, column=0, sticky="w", padx=(0,5), pady=(3,0)); rv_confidence_label = ttk.Label(rv_details_frame, text="---", style="ResultValue.TLabel"); rv_confidence_label.grid(row=res_row, column=1, sticky="ew", padx=5, pady=(3,0)); res_row += 1
    ttk.Separator(rv_details_frame, orient='horizontal').grid(row=res_row, column=0, columnspan=2, sticky='ew', pady=8); res_row += 1
    ttk.Label(rv_details_frame, text="Sensor Values Used:", font=result_title_font).grid(row=res_row, column=0, columnspan=2, sticky="w", pady=(0,3)); res_row += 1
    ttk.Label(rv_details_frame, text=" Magnetism:", font=result_title_font).grid(row=res_row, column=0, sticky="w", padx=(5,5)); rv_magnetism_label = ttk.Label(rv_details_frame, text="---", style="ResultValue.TLabel"); rv_magnetism_label.grid(row=res_row, column=1, sticky="ew", padx=5); res_row += 1
    ttk.Label(rv_details_frame, text=" LDC Reading:", font=result_title_font).grid(row=res_row, column=0, sticky="w", padx=(5,5)); rv_ldc_label = ttk.Label(rv_details_frame, text="---", style="ResultValue.TLabel"); rv_ldc_label.grid(row=res_row, column=1, sticky="ew", padx=5); res_row += 1
    rv_classify_another_button = ttk.Button(rv_content_frame, text="<< Classify Another", command=calibrate_and_show_live_view); rv_classify_another_button.grid(row=3, column=0, columnspan=2, pady=(15, 5))

    clear_results_display()
    show_live_view()
    print("GUI setup complete.")

# ==========================
# === Main Execution =======
# ==========================
def run_application():
    global window, lv_camera_label, lv_magnetism_label, lv_ldc_label
    global visual_interpreter, magnetism_interpreter, resistivity_interpreter, camera, hall_sensor, ldc_initialized

    print("Setting up GUI...")
    try: setup_gui()
    except Exception as e:
        print(f"FATAL ERROR: Failed to set up GUI: {e}"); traceback.print_exc()
        try: root_err = tk.Tk(); root_err.withdraw(); messagebox.showerror("GUI Setup Error", f"GUI Init Failed:\n{e}\n\nConsole for details."); root_err.destroy()
        except Exception: pass
        return

    # Initial state of GUI elements
    if not camera and lv_camera_label: lv_camera_label.configure(text="Camera Failed", image='')
    if not hall_sensor and lv_magnetism_label: lv_magnetism_label.config(text="N/A")
    if not ldc_initialized and lv_ldc_label: lv_ldc_label.config(text="N/A")

    print("Starting GUI update loops...")
    update_camera_feed()
    update_magnetism()
    update_ldc_reading()
    manage_automation_flow() 

    print("Starting Tkinter main loop... (Press Ctrl+C in console to exit)")
    try:
        window.protocol("WM_DELETE_WINDOW", on_closing)
        window.mainloop()
    except Exception as e: print(f"ERROR: Exception in Tkinter main loop: {e}")
    print("Tkinter main loop finished.")

def on_closing():
    global window
    print("Window close requested by user.")
    if messagebox.askokcancel("Quit", "Do you want to quit the AI Metal Classifier application?"):
        print("Proceeding with shutdown...")
        if window: window.quit()
    else: print("Shutdown cancelled by user.")

# ==========================
# === Cleanup Resources ====
# ==========================
def cleanup_resources():
    print("\n--- Cleaning up resources ---")
    global camera, spi, ldc_initialized, CS_PIN, RPi_GPIO_AVAILABLE, SORTING_GPIO_ENABLED
    if camera and camera.isOpened():
        try: print("Releasing camera..."); camera.release(); print("Camera released.")
        except Exception as e: print(f"Warning: Error releasing camera: {e}")
    if spi:
        try:
            if ldc_initialized and RPi_GPIO_AVAILABLE and CS_PIN is not None:
                print("Putting LDC1101 to sleep...")
                try:
                    if ldc_write_register(START_CONFIG_REG, SLEEP_MODE): print("LDC sleep command sent.")
                    else: print("Note: Failed to send LDC sleep command.")
                except Exception as ldc_e: print(f"Note: Error sending LDC sleep cmd: {ldc_e}")
        finally:
            try: print("Closing LDC SPI..."); spi.close(); print("LDC SPI closed.")
            except Exception as e: print(f"Warning: Error closing LDC SPI: {e}")
    if RPi_GPIO_AVAILABLE:
        try:
            current_gpio_mode = GPIO.getmode()
            if current_gpio_mode is not None:
                print(f"Cleaning up GPIO (mode: {current_gpio_mode})..."); GPIO.cleanup(); print("GPIO cleaned up.")
            else: print("Note: GPIO mode not set/already cleaned, skipping cleanup.")
        except RuntimeError as e: print(f"Note: GPIO cleanup runtime error: {e}")
        except Exception as e: print(f"Warning: Error during GPIO cleanup: {e}")
    else: print("Note: RPi.GPIO not available, skipping cleanup.")
    print("--- Cleanup complete ---")

# ==========================
# === Main Entry Point =====
# ==========================
if __name__ == '__main__':
    print("="*30 + "\n Starting AI Metal Classifier (RPi Gated Automation) \n" + "="*30)
    hw_init_attempted = False
    try:
        initialize_hardware(); hw_init_attempted = True
        initialize_ai()
        run_application()
    except KeyboardInterrupt: print("\nKeyboard interrupt detected. Exiting application.")
    except Exception as e:
        print("\n" + "="*30 + f"\nFATAL ERROR in main execution: {e}\n" + "="*30); traceback.print_exc()
        if 'window' in globals() and window and window.winfo_exists():
            try: messagebox.showerror("Fatal Application Error", f"Unrecoverable error:\n\n{e}\n\nPlease check console.")
            except Exception: pass
    finally:
        if 'window' in globals() and window:
            try:
                if window.winfo_exists(): print("Ensuring Tkinter window is destroyed..."); window.destroy(); print("Tkinter window destroyed.")
            except tk.TclError: print("Note: Tkinter window already destroyed.")
            except Exception as e: print(f"Warning: Error destroying Tkinter window: {e}")
        if hw_init_attempted: cleanup_resources()
        else: print("Skipping resource cleanup as hardware init not fully attempted.")
        print("\nApplication finished.\n" + "="*30)
