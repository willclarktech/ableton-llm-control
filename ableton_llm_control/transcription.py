import numpy as np
from transformers import (
    Pipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

DEFAULT_TRANSCRIPTION_MODEL = "openai/whisper-tiny.en"


def get_transcriber(model=DEFAULT_TRANSCRIPTION_MODEL) -> Pipeline:
    # TODO: Handle tuple return value
    processor: WhisperProcessor = WhisperProcessor.from_pretrained(model)
    speech_recognition_model = WhisperForConditionalGeneration.from_pretrained(model)
    return pipeline(
        "automatic-speech-recognition",
        model=speech_recognition_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )


default_transcriber = get_transcriber()


def transcribe(recording: np.ndarray, model=DEFAULT_TRANSCRIPTION_MODEL) -> str:
    transcriber = (
        default_transcriber
        if model == DEFAULT_TRANSCRIPTION_MODEL
        else get_transcriber(model)
    )
    result = transcriber(recording.squeeze())["text"].strip()
    print(f"[Transcriber]: {result}")
    return result
