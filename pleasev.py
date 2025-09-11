# CODE 3.0.21 - AI Metal Classifier GUI with Gated Automation
# Description: Displays live sensor data and camera feed.
#              - On startup, waits for a LOW->HIGH signal on GPIO 23 to classify.
#              - While GPIO 23 is LOW, it continuously auto-calibrates.
#              - After classifying, it enters a PAUSED state, ignoring new triggers.
#              - Clicking 'Classify Another' RE-ARMS the system for the next trigger.
# Version: 3.0.28 - FIXED: Magnetism reading instability by implementing "smart"
#                  -           auto-calibration. The system now checks if a magnetic
#                  -           field is present before recalibrating the idle voltage,
#                  -           preventing the baseline from being incorrectly reset.
# Version: 3.0.27 - MODIFIED: LDC display is now "rawer", similar to magnetism-test.py.
#                  -           Removed the temporal smoothing buffer (RP_DISPLAY_BUFFER) to make
#                  -           the live reading more responsive. A small amount of smoothing is
#                  -           retained by taking the median of a few fast samples per update.
# Version: 3.0.26 - IMPROVED: Averaging logic for sensor readings now uses the median
#                  -           instead of the mean. This provides more robust noise
#                  -           rejection against outlier data spikes.
# Version: 3.0.25 - MODIFIED: Magnetism display logic updated to match 'magnetism-test.py'.
#                  -           The display now switches from milliTesla (mT) to microTesla (µT)
#                  -           when the absolute magnetism value is less than 1.0 mT.
# Version: 3.0.24 - MODIFIED: Now uses the absolute value of the magnetism reading as input
#                  -           for the magnetism AI model. The signed value is still
#                  -           displayed in the results for user context.
# Version: 3.0.23 - MODIFIED: Replaced single AI model with a hierarchical ensemble of three
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

# =========================================
# ### NEW ### - Hierarchical Model Setup
# =========================================
# --- Model Filenames ---
MODEL_VISUAL_FILENAME = "visual_model.tflite"
MODEL_MAGNETISM_FILENAME = "magnetism_model.tflite"
MODEL_RESISTIVITY_FILENAME = "resistivity_model.tflite"
LABELS_FILENAME = "material_labels.txt"

# --- Model Paths ---
MODEL_VISUAL_PATH = os.path.join(BASE_PATH, MODEL_VISUAL_FILENAME)
MODEL_MAGNETISM_PATH = os.path.join(BASE_PATH, MODEL_MAGNETISM_FILENAME)
MODEL_RESISTIVITY_PATH = os.path.join(BASE_PATH, MODEL_RESISTIVITY_FILENAME)
LABELS_PATH = os.path.join(BASE_PATH, LABELS_FILENAME)

# --- Hierarchical Weights (Must sum to 1.0) ---
MODEL_WEIGHTS = {
    'visual': 1.0,
    'magnetism': 0.0,
    'resistivity': 0.0
}

# --- Hardcoded Scaler Parameters ---
# NOTE: Replace these placeholder values with the actual mean and scale
#       values from your trained scalers. Each list should have one value
#       per feature the model expects (e.g., [value1] for 1 feature).
SCALER_PARAMS = {
    'magnetism': {
        'mean': [0.00048415711947626843],  # Example: Mean of magnetism training data
        'scale': [0.0007762457818081904]  # Example: Std Dev of magnetism training data
    },
    'resistivity': {
        'mean': [61000.82880523732], # Example: Mean of LDC RP training data
        'scale': [1362.7716526399417]   # Example: Std Dev of LDC RP training data
    }
}
# =========================================

TESTING_FOLDER_NAME = "testing" # Folder to save screenshots

AI_IMG_WIDTH = 224
AI_IMG_HEIGHT = 224

HALL_ADC_CHANNEL = ADS.P0 if I2C_ENABLED else None
SENSITIVITY_V_PER_TESLA = 0.0004
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

# ============================
# === Global Objects/State ===
# ============================
camera = None
i2c = None
ads = None
hall_sensor = None
spi = None
ldc_initialized = False

# ### MODIFIED ### - Globals for the three AI models
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
g_low_pulse_counter = 0         # Counts consecutive LOW reads for calibration trigger

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
def initialize_ai():
    global loaded_labels
    global interpreter_visual, input_details_visual, output_details_visual
    global interpreter_magnetism, input_details_magnetism, output_details_magnetism
    global interpreter_resistivity, input_details_resistivity, output_details_resistivity

    print("\n--- Initializing AI Components (Hierarchical) ---")
    
    # --- Load Labels (common for all models) ---
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

    # --- Helper function to load a single model ---
    def load_model(model_name, model_path):
        print(f"\n--- Loading {model_name} Model ---")
        try:
            interpreter = Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            # Basic validation
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

    # --- Load all three models ---
    interpreter_visual, input_details_visual, output_details_visual = load_model("Visual", MODEL_VISUAL_PATH)
    interpreter_magnetism, input_details_magnetism, output_details_magnetism = load_model("Magnetism", MODEL_MAGNETISM_PATH)
    interpreter_resistivity, input_details_resistivity, output_details_resistivity = load_model("Resistivity", MODEL_RESISTIVITY_PATH)

    # --- Final Check ---
    all_models_loaded = all([interpreter_visual, interpreter_magnetism, interpreter_resistivity])
    if not all_models_loaded:
        print("\n--- AI Initialization Failed: One or more models could not be loaded. ---")
        return False
    else:
        # Check weight sum
        if not math.isclose(sum(MODEL_WEIGHTS.values()), 1.0):
             print(f"WARNING: Model weights sum to {sum(MODEL_WEIGHTS.values())}, not 1.0. This may cause unexpected results.")
        print("\n--- All AI Models Initialized Successfully ---")
        return True

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
    # ### IMPROVED: Use median to be more robust against outlier noise spikes ###
    if readings: return statistics.median(readings)
    else: return None

def get_averaged_rp_data(num_samples=NUM_SAMPLES_PER_UPDATE):
    if not ldc_initialized: return None
    readings = []
    for _ in range(num_samples):
        rp_value = get_ldc_rpdata()
        if rp_value is not None: readings.append(rp_value)
    # ### IMPROVED: Use median to be more robust against outlier noise spikes ###
    if readings: return statistics.median(readings)
    else: return None

# ==========================
# === AI Processing ========
# ==========================
def preprocess_visual_input(image_pil, input_details):
    """Prepares image data for the visual model."""
    if not input_details: return None
    try:
        img_resized = image_pil.resize((AI_IMG_WIDTH, AI_IMG_HEIGHT), Image.Resampling.LANCZOS)
        image_np = np.array(img_resized.convert('RGB'), dtype=np.float32)
        image_np /= 255.0  # Normalize to [0, 1]
        image_input = np.expand_dims(image_np, axis=0)
        # Ensure data type matches model's expectation
        return image_input.astype(input_details[0]['dtype'])
    except Exception as e:
        print(f"ERROR: Visual preprocessing failed: {e}")
        return None

def preprocess_numerical_input(value, model_type, input_details):
    """Prepares and scales a single numerical value for a sensor model."""
    if value is None or not input_details: return None
    try:
        # Get scaler params for the specific model
        params = SCALER_PARAMS.get(model_type)
        if not params:
            print(f"ERROR: No scaler parameters defined for model type '{model_type}'.")
            return None
        
        raw_value = np.array([[float(value)]], dtype=np.float32)
        
        # Manual scaling: (value - mean) / scale
        mean = np.array(params['mean'], dtype=np.float32)
        scale = np.array(params['scale'], dtype=np.float32)
        scaled_value = (raw_value - mean) / scale
        
        return scaled_value.astype(input_details[0]['dtype'])
    except Exception as e:
        print(f"ERROR: Numerical preprocessing for {model_type} failed: {e}")
        return None

def run_single_inference(interpreter, input_details, processed_input):
    """Runs inference on a single TFLite interpreter."""
    if interpreter is None or processed_input is None:
        return None
    try:
        interpreter.set_tensor(input_details[0]['index'], processed_input)
        interpreter.invoke()
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Return the raw probability array (e.g., [[0.1, 0.8, 0.1]])
        return output_data
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        traceback.print_exc()
        return None

def postprocess_hierarchical_output(outputs):
    global loaded_labels, MODEL_WEIGHTS
    print("\n--- Postprocessing AI Outputs (Hierarchical) ---")
    if not loaded_labels:
        print("ERROR: No labels loaded for postprocessing.")
        return "Error", 0.0

    # Ensure we have the same number of outputs as weights
    if len(outputs) != len(MODEL_WEIGHTS):
        print(f"ERROR: Mismatch between number of model outputs ({len(outputs)}) and weights ({len(MODEL_WEIGHTS)}).")
        return "Mismatch Err", 0.0

    # Initialize a zero array for the final probabilities
    num_classes = len(loaded_labels)
    final_probabilities = np.zeros(num_classes, dtype=np.float32)
    
    # Apply weights and sum the probabilities
    for model_type, raw_output in outputs.items():
        weight = MODEL_WEIGHTS.get(model_type, 0)
        if weight == 0:
            print(f"Warning: No weight found for model '{model_type}', skipping its output.")
            continue
        
        if raw_output is not None and raw_output.shape == (1, num_classes):
            probabilities = raw_output[0] # Extract the 1D array from [[...]]
            weighted_probs = probabilities * weight
            final_probabilities += weighted_probs
            print(f"  -> Applied weight {weight:.2f} to {model_type} output.")
        else:
            print(f"Warning: Skipping invalid or missing output from {model_type} model.")
    
    print(f"Final combined probabilities: {final_probabilities}")
    
    # Get the final prediction from the combined probabilities
    try:
        predicted_index = np.argmax(final_probabilities)
        # The confidence is the value of the highest probability in the combined result
        confidence = float(final_probabilities[predicted_index])
        predicted_label = loaded_labels[predicted_index]
        
        print(f"Final Prediction: '{predicted_label}', Confidence: {confidence:.4f}")
        print("--- Postprocessing Complete ---")
        return predicted_label, confidence
    except Exception as e:
        print(f"ERROR: Postprocessing failed: {e}")
        return "Post Err", 0.0

# ==================================
# === Sorting Signal Functions ===
# ==================================
def send_sorting_signal(material_label):
    if not SORTING_GPIO_ENABLED: print("Sorting Signal: GPIO for sorting not enabled. Skipping send."); return
    if not RPi_GPIO_AVAILABLE: print("Sorting Signal: RPi.GPIO library not available. Cannot send signal."); return

    print(f"\n--- Sending Sorting Signal for: {material_label} ---")
    mid_val, lsb_val = GPIO.LOW, GPIO.LOW
    signal_desc = "Others (00)"
    if material_label == "Aluminum": mid_val, lsb_val, signal_desc = GPIO.LOW, GPIO.HIGH, "Aluminum (01)"
    elif material_label == "Copper": mid_val, lsb_val, signal_desc = GPIO.HIGH, GPIO.LOW, "Copper (10)"
    elif material_label == "Steel": mid_val, lsb_val, signal_desc = GPIO.HIGH, GPIO.HIGH, "Steel (11)"

    try:
        GPIO.output(SORTING_DATA_READY_PIN, GPIO.LOW)
        time.sleep(0.01)
        GPIO.output(SORTING_DATA_PIN_MID, mid_val)
        GPIO.output(SORTING_DATA_PIN_LSB, lsb_val)
        print(f"Set GPIO Pins: MID={mid_val}, LSB={lsb_val} for {signal_desc}")
        time.sleep(0.01)
        GPIO.output(SORTING_DATA_READY_PIN, GPIO.HIGH)
        print(f"Pulsed {SORTING_DATA_READY_PIN} HIGH (Data Ready)")
        time.sleep(0.05)
        GPIO.output(SORTING_DATA_READY_PIN, GPIO.LOW)
        print(f"Set {SORTING_DATA_READY_PIN} LOW (Data Transmitted)")
        time.sleep(0.01)
        GPIO.output(SORTING_DATA_PIN_MID, GPIO.LOW)
        GPIO.output(SORTING_DATA_PIN_LSB, GPIO.LOW)
        print(f"Data pins ({SORTING_DATA_PIN_MID}, {SORTING_DATA_PIN_LSB}) reset to LOW after signal.")
        print(f"--- Sorting signal {signal_desc} sent ---")
    except Exception as e:
        print(f"ERROR: Failed to send sorting signal via GPIO: {e}")
        try:
            if RPi_GPIO_AVAILABLE:
                GPIO.output(SORTING_DATA_READY_PIN, GPIO.LOW)
                GPIO.output(SORTING_DATA_PIN_MID, GPIO.LOW)
                GPIO.output(SORTING_DATA_PIN_LSB, GPIO.LOW)
                print("Ensured sorting pins are LOW after error.")
        except Exception as e_cleanup: print(f"Warning: Could not reset pins after error: {e_cleanup}")

# ==============================
# === View Switching Logic ===
# ==============================
def calibrate_and_show_live_view():
    """Re-arms the trigger, runs a calibration, and shows the live view."""
    global g_accepting_triggers
    print("\n--- 'Classify Another' clicked: Re-arming GPIO trigger ---")
    g_accepting_triggers = True # Re-arm the system
    calibrate_sensors(is_manual_call=True) 
    show_live_view()

def show_live_view():
    global live_view_frame, results_view_frame
    if results_view_frame and results_view_frame.winfo_ismapped():
        results_view_frame.pack_forget()
    if live_view_frame and not live_view_frame.winfo_ismapped():
        live_view_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def show_results_view():
    global live_view_frame, results_view_frame
    if live_view_frame and live_view_frame.winfo_ismapped():
        live_view_frame.pack_forget()
    if results_view_frame and not results_view_frame.winfo_ismapped():
        results_view_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# ============================
# === Screenshot Function ===
# ============================
def save_result_screenshot(image_pil, prediction, confidence, mag_text, ldc_text):
    """Creates and saves a composite image of the results."""
    global BASE_PATH, TESTING_FOLDER_NAME, RESULT_IMG_DISPLAY_WIDTH

    print("\n--- Saving Result Screenshot ---")
    testing_folder = os.path.join(BASE_PATH, TESTING_FOLDER_NAME)

    try:
        os.makedirs(testing_folder, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Cannot create/access testing folder '{testing_folder}': {e}")
        messagebox.showerror("Save Error", f"Could not create/access folder:\n{testing_folder}")
        return

    i = 1
    while True:
        filename = os.path.join(testing_folder, f"data_{i}.png")
        if not os.path.exists(filename):
            break
        i += 1
        if i > 9999:
            print("ERROR: More than 9999 result files exist. Cannot save.")
            messagebox.showerror("Save Error", "Too many result files exist.")
            return

    # --- Create Composite Image ---
    IMG_WIDTH = 400; IMG_HEIGHT = 600; BG_COLOR = "white"; TEXT_COLOR = "black"
    MARGIN = 20; IMG_DISPLAY_WIDTH_SS = RESULT_IMG_DISPLAY_WIDTH; FONT_SIZE_TITLE = 20
    FONT_SIZE_TEXT = 16; LINE_SPACING = 5

    try:
        try:
            font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", FONT_SIZE_TITLE)
            font_text = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE_TEXT)
        except IOError:
            print("Warning: DejaVu fonts not found, using default PIL font.")
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()

        ss_img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(ss_img)
        y_pos = MARGIN
        title_w, title_h = draw.textsize("Classification Result", font=font_title)
        draw.text(((IMG_WIDTH - title_w) / 2, y_pos), "Classification Result", fill=TEXT_COLOR, font=font_title)
        y_pos += title_h + 15
        w, h_img = image_pil.size
        aspect = h_img / w if w > 0 else 0.75
        display_h = int(IMG_DISPLAY_WIDTH_SS * aspect) if aspect > 0 else int(IMG_DISPLAY_WIDTH_SS * 0.75)
        img_disp = image_pil.resize((IMG_DISPLAY_WIDTH_SS, max(1, display_h)), Image.Resampling.LANCZOS)
        img_x = (IMG_WIDTH - IMG_DISPLAY_WIDTH_SS) // 2
        ss_img.paste(img_disp, (img_x, y_pos))
        y_pos += display_h + 20
        details = [
            (f"Material:", f"{prediction}", font_title),
            (f"Confidence:", f"{confidence:.1%}", font_text),
            ("--- Sensor Values ---", "", font_text),
            (f" Magnetism:", f"{mag_text}", font_text),
            (f" LDC Reading:", f"{ldc_text}", font_text),
        ]
        max_label_w = 0
        for label, _, font in details:
            lw, _ = draw.textsize(label, font=font)
            max_label_w = max(max_label_w, lw)
        value_x = MARGIN + max_label_w + 10
        for label, value, font in details:
            _, lh = draw.textsize("A", font=font)
            if value:
                draw.text((MARGIN, y_pos), label, fill=TEXT_COLOR, font=font)
                draw.text((value_x, y_pos), value, fill=TEXT_COLOR, font=font)
            else:
                draw.text((MARGIN, y_pos), label, fill=TEXT_COLOR, font=font)
            y_pos += lh + LINE_SPACING
        ss_img.save(filename)
        print(f"Result saved successfully to: {filename}")
    except Exception as e:
        print(f"ERROR: Failed to create or save screenshot: {e}")
        traceback.print_exc()
        messagebox.showerror("Save Error", f"Failed to save screenshot:\n{e}")

# ======================
# === GUI Functions ===
# ======================
def create_placeholder_image(width, height, color='#E0E0E0', text="No Image"):
    try:
        pil_img = Image.new('RGB', (width, height), color); tk_img = ImageTk.PhotoImage(pil_img)
        return tk_img
    except Exception as e: print(f"Warning: Failed to create placeholder image: {e}"); return None

def clear_results_display():
    global rv_image_label, rv_prediction_label, rv_confidence_label, rv_magnetism_label, rv_ldc_label, placeholder_img_tk
    if rv_image_label:
        if placeholder_img_tk: rv_image_label.config(image=placeholder_img_tk, text=""); rv_image_label.img_tk = placeholder_img_tk
        else: rv_image_label.config(image='', text="No Image"); rv_image_label.img_tk = None
    default_text = "---"
    if rv_prediction_label: rv_prediction_label.config(text=default_text)
    if rv_confidence_label: rv_confidence_label.config(text=default_text)
    if rv_magnetism_label: rv_magnetism_label.config(text=default_text)
    if rv_ldc_label: rv_ldc_label.config(text=default_text)

def capture_and_classify():
    global window, camera, IDLE_VOLTAGE, IDLE_RP_VALUE
    global rv_image_label, rv_prediction_label, rv_confidence_label, rv_magnetism_label, rv_ldc_label
    global save_output_var, g_accepting_triggers
    global interpreter_visual, interpreter_magnetism, interpreter_resistivity

    # --- Disarm the trigger as soon as we start processing ---
    g_accepting_triggers = False
    print("\n" + "="*10 + " Automatic Classification Triggered (System Paused) " + "="*10)

    # --- Pre-flight Checks ---
    if not all([interpreter_visual, interpreter_magnetism, interpreter_resistivity]):
        messagebox.showerror("Error", "One or more AI Models are not initialized. Cannot classify.")
        print("Classification aborted: AI not ready."); show_live_view(); return
    if not camera or not camera.isOpened():
        messagebox.showerror("Error", "Camera is not available. Cannot capture image.")
        print("Classification aborted: Camera not ready."); show_live_view(); return

    window.update_idletasks()

    # --- Capture Image ---
    ret, frame = camera.read()
    if not ret or frame is None:
        messagebox.showerror("Capture Error", "Failed to capture image from camera.")
        print("ERROR: Failed to read frame from camera."); show_live_view(); return
    try:
        img_captured_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        messagebox.showerror("Image Error", f"Failed to process captured image: {e}")
        print(f"ERROR: Failed converting captured frame to PIL Image: {e}"); show_live_view(); return

    # --- Capture Sensor Data ---
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
    avg_rp_val = get_averaged_rp_data(num_samples=NUM_SAMPLES_CALIBRATION)
    if avg_rp_val is not None:
        current_rp_raw = avg_rp_val
    else:
        print("ERROR: LDC read failed during capture.")

    # --- Preprocess Data for Each Model ---
    print("\n--- Preprocessing all inputs ---")

    # Use the absolute value of magnetism for the AI input, but keep the
    # original signed value for display purposes.
    magnetism_for_ai = None
    if current_mag_mT is not None:
        magnetism_for_ai = abs(current_mag_mT)
        print(f"Original magnetism: {current_mag_mT:+.4f}mT, Using absolute value for AI: {magnetism_for_ai:.4f}mT")

    visual_input = preprocess_visual_input(img_captured_pil, input_details_visual)
    magnetism_input = preprocess_numerical_input(magnetism_for_ai, 'magnetism', input_details_magnetism)
    resistivity_input = preprocess_numerical_input(current_rp_raw, 'resistivity', input_details_resistivity)
    
    # --- Run Inference on Each Model ---
    print("\n--- Running inference on all models ---")
    output_visual = run_single_inference(interpreter_visual, input_details_visual, visual_input)
    output_magnetism = run_single_inference(interpreter_magnetism, input_details_magnetism, magnetism_input)
    output_resistivity = run_single_inference(interpreter_resistivity, input_details_resistivity, resistivity_input)

    # --- Combine Results with Hierarchical Logic ---
    model_outputs = {
        'visual': output_visual,
        'magnetism': output_magnetism,
        'resistivity': output_resistivity
    }
    predicted_label, confidence = postprocess_hierarchical_output(model_outputs)
    
    print(f"\n--- HIERARCHICAL RESULT: Prediction='{predicted_label}', Confidence={confidence:.1%} ---")

    # --- Handle Saving and Sorting ---
    mag_display_text = ""
    if current_mag_mT is not None:
        if abs(current_mag_mT) < 1:
            mag_display_text = f"{current_mag_mT * 1000:+.1f}µT"
        else:
            mag_display_text = f"{current_mag_mT:+.2f}mT"
    else:
        mag_display_text = "ReadErr"
        
    ldc_display_text = f"{int(round(current_rp_raw))}" if current_rp_raw is not None else "ReadErr"

    if save_output_var and save_output_var.get() == 1:
        save_result_screenshot(img_captured_pil, predicted_label, confidence, mag_display_text, ldc_display_text)

    send_sorting_signal(predicted_label)

    # --- Update Results Display ---
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

# ### MODIFIED FUNCTION ###
def calibrate_sensors(is_manual_call=False):
    global IDLE_VOLTAGE, IDLE_RP_VALUE, g_last_live_magnetism_mT
    global hall_sensor, ldc_initialized

    # NEW: Only perform auto-calibration if the sensor is truly idle
    # This prevents the baseline from being reset when a material is present.
    # A threshold of 2.0 µT (0.002 mT) is a safe "zero" point.
    if not is_manual_call and abs(g_last_live_magnetism_mT * 1000) > 1.2:
        return # Exit the function, do not recalibrate

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
                
                g_last_live_magnetism_mT = current_mT

                # Format the display text based on the magnitude
                if abs(current_mT) < 1:
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
    global lv_ldc_label, window, IDLE_RP_VALUE, ldc_initialized
    if not window or not window.winfo_exists(): return
    display_text = "N/A"
    if ldc_initialized:
        # Get a single, median-averaged reading for this update cycle.
        # This provides a bit of smoothing against noise spikes, but is "rawer"
        # because it doesn't smooth over multiple update cycles.
        avg_rp = get_averaged_rp_data(num_samples=NUM_SAMPLES_PER_UPDATE)
        if avg_rp is not None:
            # The RP_DISPLAY_BUFFER is no longer used, making the display more responsive.
            cur_rp = int(round(avg_rp))
            delta = cur_rp - IDLE_RP_VALUE
            display_text = f"{cur_rp}"
            if IDLE_RP_VALUE != 0:
                display_text += f" (Δ{delta:+,})"
            else:
                display_text += " (NoCal)"
        else:
            display_text = "ReadErr"
    if lv_ldc_label and lv_ldc_label.cget("text") != display_text:
        lv_ldc_label.config(text=display_text)
    if window and window.winfo_exists():
        window.after(GUI_UPDATE_INTERVAL_MS, update_ldc_reading)

def manage_automation_flow():
    """
    Checks the GPIO pin to manage calibration and classification.
    - On LOW->HIGH transition: Triggers classification ONLY if the system is armed.
    - After 10 consecutive LOW reads: Triggers an auto-calibration.
    """
    global window, g_previous_control_state, g_accepting_triggers, g_low_pulse_counter
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
            g_low_pulse_counter = 0 # Reset counter on a rising edge
            window.after(2000, capture_and_classify) # 2s delay
        
        # STATE IS HIGH (but not a rising edge): Reset counter
        elif current_state == GPIO.HIGH:
            g_low_pulse_counter = 0

        # STATE IS LOW: Increment counter and check for calibration
        elif current_state == GPIO.LOW:
            g_low_pulse_counter += 1
            if g_low_pulse_counter >= 10:
                # print("AUTOMATION: 10 consecutive LOW states detected. Auto-calibrating...") # Uncomment for debug
                calibrate_sensors(is_manual_call=False)
                g_low_pulse_counter = 0 # Reset after calibrating

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
    window.title("AI Metal Classifier v3.0.28 (RPi - Hierarchical Ensemble)")
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
    global camera, hall_sensor, ldc_initialized

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
    print("="*30 + "\n Starting AI Metal Classifier (RPi Hierarchical Ensemble) \n" + "="*30)
    hw_init_attempted = False
    try:
        initialize_hardware(); hw_init_attempted = True
        if initialize_ai(): # Only run the app if AI models load successfully
            run_application()
        else:
            print("\nApplication cannot start due to AI initialization failure.")
            messagebox.showerror("AI Init Error", "Could not load AI models. Please check console for details.")
            
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
