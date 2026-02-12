import torch
import torchaudio
import tempfile
import os
import folder_paths

from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
from transformers.models.whisper.processing_whisper import WhisperProcessor
from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)


# Use ComfyUI checkpoints folder for model storage
checkpoints_dir = folder_paths.get_folder_paths("checkpoints")[0]


def _get_model_dirs():
    """
    Scan the checkpoints folder for model directories (all subdirectories).
    """
    dirs = []
    for search_dir in folder_paths.get_folder_paths("checkpoints"):
        if os.path.exists(search_dir):
            for name in sorted(os.listdir(search_dir)):
                full = os.path.join(search_dir, name)
                if os.path.isdir(full) and name not in dirs:
                    dirs.append(name)
    return dirs if dirs else ["(none found)"]


class HeartTranscriptorLoader:
    """
    Load a HeartTranscriptor (Whisper-based) model for lyrics transcription.
    Place your model directory inside: ComfyUI/models/checkpoints/
    Or provide an absolute path for testing via model_path_override.
    """

    _pipeline_cache = {}

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        available = _get_model_dirs()
        return {
            "required": {
                "model_name": (available, {
                    "tooltip": "Select a model directory from ComfyUI checkpoints folder.",
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                }),
                "dtype": (["float16", "float32", "bfloat16"], {
                    "default": "float16",
                }),
            },
            "optional": {
                "model_path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: absolute path to model directory (overrides model_name). "
                               "Useful for testing.",
                }),
            },
        }

    RETURN_TYPES = ("HEART_TRANSCRIPTOR_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "HeartMuLa"

    def load_model(self, model_name, device, dtype, model_path_override=""):
        # Determine model path
        if model_path_override and model_path_override.strip():
            model_dir = model_path_override.strip()
        else:
            # Search through all checkpoint paths
            model_dir = None
            for search_dir in folder_paths.get_folder_paths("checkpoints"):
                candidate = os.path.join(search_dir, model_name)
                if os.path.isdir(candidate):
                    model_dir = candidate
                    break
            if model_dir is None:
                model_dir = os.path.join(checkpoints_dir, model_name)

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}\n"
                f"Please place your HeartTranscriptor model directory in: "
                f"{checkpoints_dir}/<model_name>/"
            )

        # Map dtype string
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map[dtype]
        torch_device = torch.device(device)

        # Cache to avoid reloading
        cache_key = (model_dir, device, dtype)
        if cache_key in HeartTranscriptorLoader._pipeline_cache:
            print(f"[HeartTranscriptorLoader] Using cached pipeline: {model_dir}")
            return (HeartTranscriptorLoader._pipeline_cache[cache_key],)

        print(f"[HeartTranscriptorLoader] Loading model from {model_dir} ...")
        model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        processor = WhisperProcessor.from_pretrained(model_dir)

        pipe = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=torch_device,
            torch_dtype=torch_dtype,
            chunk_length_s=30,
            batch_size=16,
        )

        HeartTranscriptorLoader._pipeline_cache[cache_key] = pipe
        print(f"[HeartTranscriptorLoader] Model loaded successfully.")
        return (pipe,)


class HeartTranscriptor:
    """
    Lyrics transcription node using HeartTranscriptor (Whisper-based ASR optimized for music).
    Filters complex background instrument interference for highly accurate lyrics recognition.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HEART_TRANSCRIPTOR_PIPELINE",),
                "audio": ("AUDIO",),
                "max_new_tokens": ("INT", {
                    "default": 256,
                    "min": 16,
                    "max": 1024,
                    "step": 16,
                    "tooltip": "Maximum number of new tokens to generate per chunk.",
                }),
                "num_beams": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of beams for beam search decoding.",
                }),
                "task": (["transcribe", "translate"], {
                    "default": "transcribe",
                    "tooltip": "'transcribe' keeps the original language, "
                               "'translate' translates to English.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature. A fallback sequence "
                               "(0.0 -> temperature) is built internally.",
                }),
                "no_speech_threshold": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Threshold for detecting no-speech segments.",
                }),
                "compression_ratio_threshold": ("FLOAT", {
                    "default": 1.8,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Threshold for compression ratio filtering.",
                }),
                "logprob_threshold": ("FLOAT", {
                    "default": -1.0,
                    "min": -5.0,
                    "max": 0.0,
                    "step": 0.1,
                    "tooltip": "Log-probability threshold for filtering "
                               "low-confidence outputs.",
                }),
            },
            "optional": {
                "condition_on_prev_tokens": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to condition on previously generated tokens.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "transcribe"
    CATEGORY = "HeartMuLa"
    OUTPUT_NODE = True

    def transcribe(self, pipeline, audio, max_new_tokens, num_beams, task,
                   temperature, no_speech_threshold, compression_ratio_threshold,
                   logprob_threshold, condition_on_prev_tokens=False):
        waveform = audio["waveform"]  # [B, C, T]
        sample_rate = audio["sample_rate"]

        if waveform.dim() == 3:
            waveform = waveform[0]  # [C, T]

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                torchaudio.save(tmp_path, waveform.cpu(), sample_rate)

            # Build temperature fallback tuple
            if temperature <= 0.0:
                temp_tuple = (0.0,)
            else:
                temp_values = [0.0]
                step = 0.1
                val = step
                while val < temperature:
                    temp_values.append(round(val, 2))
                    val += step
                temp_values.append(round(temperature, 2))
                temp_tuple = tuple(temp_values)

            print(f"[HeartTranscriptor] Transcribing audio "
                  f"({sample_rate}Hz, {waveform.shape[-1] / sample_rate:.1f}s) ...")

            with torch.no_grad():
                result = pipeline(
                    tmp_path,
                    **{
                        "max_new_tokens": max_new_tokens,
                        "num_beams": num_beams,
                        "task": task,
                        "condition_on_prev_tokens": condition_on_prev_tokens,
                        "compression_ratio_threshold": compression_ratio_threshold,
                        "temperature": temp_tuple,
                        "logprob_threshold": logprob_threshold,
                        "no_speech_threshold": no_speech_threshold,
                    },
                )

            lyrics_text = result.get("text", "") if isinstance(result, dict) else str(result)
            lyrics_text = lyrics_text.strip()

            print(f"[HeartTranscriptor] Done. {len(lyrics_text)} chars.")
            return (lyrics_text,)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


# Node registration
NODE_CLASS_MAPPINGS = {
    "HeartTranscriptorLoader": HeartTranscriptorLoader,
    "HeartTranscriptor": HeartTranscriptor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartTranscriptorLoader": "HeartTranscriptor Loader 🎤",
    "HeartTranscriptor": "HeartTranscriptor (Lyrics ASR) 🎤",
}
