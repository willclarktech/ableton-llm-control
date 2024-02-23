from typing import Union

import live
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool

livequery = live.Query()

tools: dict[str, StructuredTool] = {}


def get_tools_list() -> list[StructuredTool]:
    return list(tools.values())


def test() -> str:
    """Test that the connection to Ableton Live is working (only use if specifically requested)"""
    result = livequery.query("/live/test")
    return (
        "The connection to Ableton Live is working"
        if result[0] == "ok"
        else "There is a problem with the connection to Ableton Live"
    )


tools[test.__name__] = StructuredTool.from_function(
    func=test,
    args_schema=None,
    return_direct=True,
)


def continue_playing() -> str:
    """Resume session playback from the current position"""
    livequery.cmd("/live/song/continue_playing")
    return "The song has successfully continued playing"


tools[continue_playing.__name__] = StructuredTool.from_function(
    func=continue_playing,
    args_schema=None,
    return_direct=True,
)


def start_playing() -> str:
    """Start session playback from the beginning of the song"""
    livequery.cmd("/live/song/start_playing")
    return "The song has successfully started playing"


tools[start_playing.__name__] = StructuredTool.from_function(
    func=start_playing,
    args_schema=None,
    return_direct=True,
)


def stop_playing() -> str:
    """Stop session playback"""
    livequery.cmd("/live/song/stop_playing")
    return "The song has successfully stopped playing"


tools[stop_playing.__name__] = StructuredTool.from_function(
    func=stop_playing,
    args_schema=None,
    return_direct=True,
)


def get_tempo() -> str:
    """Get the current song tempo in BPM"""
    result = livequery.query("/live/song/get/tempo")
    return f"The current tempo is {result[0]} BPM"


tools[get_tempo.__name__] = StructuredTool.from_function(
    func=get_tempo,
    args_schema=None,
    return_direct=True,
)


class SetTempoInput(BaseModel):
    tempo_bpm: float = Field(description="The new tempo as a float")


def set_tempo(tempo_bpm: Union[float, int]) -> str:
    """Set the current song tempo in BPM"""
    livequery.cmd("/live/song/set/tempo", float(tempo_bpm))
    return f"The tempo was successfully changed to {tempo_bpm} BPM"


tools[set_tempo.__name__] = StructuredTool.from_function(
    func=set_tempo,
    args_schema=SetTempoInput,
    return_direct=True,
)
