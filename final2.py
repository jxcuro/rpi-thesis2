# CODE 3.0.21 - AI Metal Classifier GUI with Gated Automation
# ... (previous version history) ...
# Version: 3.0.33 - MODIFIED: Increased sensor sensitivity for detecting weakly
#                  -           magnetic materials by adjusting key thresholds.
#                  -           1. Lowered the AI's magnetic detection threshold.
#                  -           2. Lowered the responsive flush threshold for the GUI.
#                  -           3. Increased calibration samples for a more precise baseline.
# Version: 3.0.32 - IMPROVED: Implemented "Responsive Smoothing" for the magnetism
#                  -           sensor. The smoothing buffer is now flushed on
#                  -           large signal changes, providing both idle stability
#                  -           against spikes and instant responsiveness for detection.

# ... (imports are the same) ...
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
# ...

# ==================================
# === Constants and Configuration ===
# ==================================
### KEPT FOR STABILITY ###
NUM_SAMPLES_PER_UPDATE = 7
### MODIFIED FOR SENSITIVITY ### - More samples for a more accurate baseline
NUM_SAMPLES_CALIBRATION = 50 
GUI_UPDATE_INTERVAL_MS = 60 
CAMERA_UPDATE_INTERVAL_MS = 50
MAGNETISM_SMOOTHING_BUFFER_SIZE = 10

### MODIFIED FOR SENSITIVITY ### - Parameters for Responsive Smoothing (Tunable)
# Lower thresholds make the GUI feel more responsive to weaker signals.
# 1. Greater than this absolute threshold (to ignore noise)
FLUSH_ABS_THRESHOLD_MT = 0.02 # Lowered from 0.08 mT
# 2. This many times larger than the current smoothed average
FLUSH_RATIO_THRESHOLD = 4.0 # Lowered from 5.0

# ... (Paths are the same) ...
try:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_PATH = os.getcwd()
    print(f"Warning: __file__ not defined, using current working directory as base path: {BASE_PATH}")

# =========================================
# ### Hierarchical Model Setup
# =========================================
# ... (Model filenames and paths are the same) ...

# --- Dynamic Hierarchical Weights ---
### MODIFIED FOR SENSITIVITY ### - Lowering this is the most critical change.
# This makes the AI consider much weaker signals as "magnetic".
MAGNETISM_THRESHOLD_mT = 0.0004 # Lowered from 0.0012 mT (now 0.4µT)

# Weights to use WHEN an object IS detected as magnetic
MAGNETIC_WEIGHTS = {
    'visual': 0.25,
    'magnetism': 0.60,
    'resistivity': 0.15
}

# Weights to use WHEN an object is NOT detected as magnetic
NON_MAGNETIC_WEIGHTS = {
    'visual': 0.6,
    'magnetism': 0.0,
    'resistivity': 0.4
}

# ... (The rest of the code is exactly the same as the previous version) ...
# --- Hardcoded Scaler Parameters ---
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
# =========================================

TESTING_FOLDER_NAME = "testing"

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

### KEPT FOR STABILITY ###
MAGNETISM_LIVE_BUFFER = deque(maxlen=MAGNETISM_SMOOTHING_BUFFER_SIZE)
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

# --- State for Gated GPIO Automation ---
g_accepting_triggers = True
g_previous_control_state = None
g_low_pulse_counter = 0

# ... (The rest of the file is identical to the previous version v3.0.32)
