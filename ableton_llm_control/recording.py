import numpy as np
import sounddevice as sd

DEFAULT_INPUT_CHANNELS = 1
DEFAULT_SAMPLE_RATE = 16_000  # Whisper sample rate
DEFAULT_MAX_FRAMES = 5 * DEFAULT_SAMPLE_RATE  # Max recording time

sd.default.samplerate = DEFAULT_SAMPLE_RATE
sd.default.channels = DEFAULT_INPUT_CHANNELS


def init_recording(
    max_frames=DEFAULT_MAX_FRAMES, input_channels=DEFAULT_INPUT_CHANNELS
) -> np.ndarray:
    return np.zeros((max_frames, input_channels))


def start_recording() -> np.ndarray:
    recording = init_recording()
    sd.rec(out=recording)
    return recording


def trim_recording(recording: np.ndarray) -> np.ndarray:
    last_non_zero = np.max(np.where(recording.any(axis=1))[0]) + 1
    return recording[:last_non_zero]


def stop_recording(out: np.ndarray) -> np.ndarray:
    sd.stop()
    return trim_recording(out)
