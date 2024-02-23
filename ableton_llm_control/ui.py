from typing import Callable

import ipywidgets as widgets
import numpy as np

from ableton_llm_control.recording import (
    init_recording,
    start_recording,
    stop_recording,
)

recording = init_recording()


def create_record_button(callback: Callable[[np.ndarray], None]) -> widgets.Button:
    button = widgets.Button(
        description="Record",
        disabled=False,
        button_style="danger",
        tooltip="Record",
        icon="microphone",
    )

    def wrapper(b: widgets.Button) -> None:
        global recording
        if b.description == "Record":
            b.description = "Done"
            b.tooltip = "Done"
            b.button_style = "warning"
            b.icon = "microphone-slash"
            recording = start_recording()
        else:
            b.description = "Record"
            b.tooltip = "Record"
            b.button_style = "danger"
            b.icon = "microphone"
            b.disabled = True
            recording = stop_recording(recording)
            b.disabled = False
            callback(recording)

    button.on_click(wrapper)
    return button
