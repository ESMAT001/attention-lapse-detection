SEED = 42

# Features
FEATURE_COLUMNS = [
    "ear",
    "mar",
    "roll",
    "pitch",
    "yaw",
    "perclos",
    "blinks",
    "face_present",
    "gaze_x",
    "gaze_y",
]

# Data
LABELS_SPLITS = {
    "Train": "TrainLabels.csv",
    "Validation": "ValidationLabels.csv",
    "Test": "TestLabels.csv",
}

IMPUTE_ON_ABSENT = ["ear", "mar", "roll", "pitch", "yaw", "gaze_x", "gaze_y"]

FPS_OPTIONS = (10, 15, 30)
WINDOW_SECONDS_OPTIONS = (5, 10)

DEFAULT_WINDOW_SECONDS = 5
STRIDE_SECONDS = 5
DEFAULT_FPS = 10


# Eye counts as closed at or below this EAR, PERCLOS is the closed fraction over
# the trailing window. Justification for the chosen threshold is provided in the thesis.
EAR_CLOSED_THRESHOLD = 0.20
PERCLOS_WINDOW_SECONDS = 10.0

WINDOW_METADATA_COLUMNS = [
    "sample_id",
    "video_id",
    "label_split",
    "window_index",
    "start_frame",
    "end_frame",
    "start_timestamp_ms",
    "end_timestamp_ms",
]