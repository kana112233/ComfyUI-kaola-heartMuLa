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


def _auto_device():
    """Auto-detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

    def load_model(self, model_name, dtype, model_path_override=""):
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
        torch_device = _auto_device()

        # Cache to avoid reloading
        cache_key = (model_dir, dtype)
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


class HeartMuLaLoader:
    """
    Load the HeartMuLa music generation pipeline.
    Requires: HeartMuLa model, HeartCodec model, tokenizer.json, gen_config.json.

    Expected structure in checkpoints folder:
      <model_name>/          — HeartMuLa model weights (e.g. HeartMuLa-oss-3B/)
      <codec_name>/          — HeartCodec model weights (e.g. HeartCodec-oss/)
      <gen_dir>/             — Containing tokenizer.json and gen_config.json (e.g. HeartMuLaGen/)
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
                    "tooltip": "HeartMuLa model directory (e.g. HeartMuLa-oss-3B).",
                }),
                "codec_name": (available, {
                    "tooltip": "HeartCodec model directory (e.g. HeartCodec-oss).",
                }),
                "gen_config_dir": (available, {
                    "tooltip": "Directory containing tokenizer.json and gen_config.json "
                               "(e.g. HeartMuLaGen).",
                }),
                "version": (["3B", "7B"], {
                    "default": "3B",
                    "tooltip": "HeartMuLa model version.",
                }),
                "mula_dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Dtype for HeartMuLa model. bfloat16 recommended.",
                }),
                "codec_dtype": (["float32", "float16", "bfloat16"], {
                    "default": "float32",
                    "tooltip": "Dtype for HeartCodec. float32 recommended for quality.",
                }),
                "lazy_load": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load models on demand and unload after inference to save VRAM.",
                }),
            },
            "optional": {
                "model_path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: absolute path to the parent directory containing "
                               "all model subfolders. Overrides checkpoint folder scanning.",
                }),
            },
        }

    RETURN_TYPES = ("HEART_MULA_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "HeartMuLa"

    def load_pipeline(self, model_name, codec_name, gen_config_dir, version,
                      mula_dtype, codec_dtype, lazy_load, model_path_override=""):
        from heartlib import HeartMuLaGenPipeline

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }

        # Determine base path
        if model_path_override and model_path_override.strip():
            base_path = model_path_override.strip()
        else:
            # Build a virtual base path by resolving each component
            base_path = None

        cache_key = (model_name, codec_name, gen_config_dir, version,
                     mula_dtype, codec_dtype, lazy_load, model_path_override)
        if cache_key in HeartMuLaLoader._pipeline_cache:
            print(f"[HeartMuLaLoader] Using cached pipeline.")
            return (HeartMuLaLoader._pipeline_cache[cache_key],)

        if base_path:
            # User provided a single parent directory containing all subfolders
            pretrained_path = base_path
        else:
            # Resolve each component from checkpoints folders
            mula_dir = _resolve_checkpoint_path(model_name)
            codec_dir = _resolve_checkpoint_path(codec_name)
            gen_dir = _resolve_checkpoint_path(gen_config_dir)

            # HeartMuLaGenPipeline expects a parent directory with specific subdirs.
            # We create a temporary structure using symlinks.
            import tempfile
            pretrained_path = tempfile.mkdtemp(prefix="heartmula_")
            version_dir_name = f"HeartMuLa-oss-{version}"
            os.symlink(mula_dir, os.path.join(pretrained_path, version_dir_name))
            os.symlink(codec_dir, os.path.join(pretrained_path, "HeartCodec-oss"))
            # Copy tokenizer.json and gen_config.json
            tokenizer_src = os.path.join(gen_dir, "tokenizer.json")
            gen_config_src = os.path.join(gen_dir, "gen_config.json")
            if os.path.exists(tokenizer_src):
                os.symlink(tokenizer_src, os.path.join(pretrained_path, "tokenizer.json"))
            if os.path.exists(gen_config_src):
                os.symlink(gen_config_src, os.path.join(pretrained_path, "gen_config.json"))

        device_auto = _auto_device()

        print(f"[HeartMuLaLoader] Loading pipeline from {pretrained_path} ...")
        print(f"[HeartMuLaLoader] HeartMuLa: {mula_dtype}, HeartCodec: {codec_dtype}, "
              f"lazy_load: {lazy_load}")

        pipe = HeartMuLaGenPipeline.from_pretrained(
            pretrained_path,
            device={
                "mula": device_auto,
                "codec": device_auto,
            },
            dtype={
                "mula": dtype_map[mula_dtype],
                "codec": dtype_map[codec_dtype],
            },
            version=version,
            lazy_load=lazy_load,
        )

        HeartMuLaLoader._pipeline_cache[cache_key] = pipe
        print(f"[HeartMuLaLoader] Pipeline loaded successfully.")
        return (pipe,)


def _resolve_checkpoint_path(name):
    """Find a model directory by name across all checkpoint paths."""
    for search_dir in folder_paths.get_folder_paths("checkpoints"):
        candidate = os.path.join(search_dir, name)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Model directory '{name}' not found in any checkpoint folder."
    )


class HeartMuLaGenerator:
    """
    Generate music from lyrics and tags using the HeartMuLa pipeline.
    Outputs ComfyUI AUDIO format (48kHz).
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HEART_MULA_PIPELINE",),
                "lyrics": ("STRING", {
                    "default": "[Verse]\nThe sun creeps in across the floor\n"
                               "I hear the traffic outside the door\n"
                               "[Chorus]\nEvery day the light returns\n"
                               "Every day the fire burns\n",
                    "multiline": True,
                    "tooltip": "Lyrics text with section markers like [Verse], [Chorus], etc.",
                }),
                "tags": ("STRING", {
                    "default": "piano,happy,romantic",
                    "multiline": False,
                    "tooltip": "Comma-separated style tags without spaces. "
                               "e.g. piano,happy,wedding,synthesizer",
                }),
                "max_audio_length_ms": ("INT", {
                    "default": 240000,
                    "min": 10000,
                    "max": 600000,
                    "step": 10000,
                    "tooltip": "Maximum audio length in milliseconds.",
                }),
                "topk": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Top-k sampling parameter.",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature for generation.",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"
    OUTPUT_NODE = True

    def generate(self, pipeline, lyrics, tags, max_audio_length_ms,
                 topk, temperature, cfg_scale):
        import tempfile

        print(f"[HeartMuLaGenerator] Generating music...")
        print(f"[HeartMuLaGenerator] Tags: {tags}")
        print(f"[HeartMuLaGenerator] Lyrics length: {len(lyrics)} chars")
        print(f"[HeartMuLaGenerator] Max duration: {max_audio_length_ms / 1000:.0f}s")

        # Generate to temporary file
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            with torch.no_grad():
                pipeline(
                    {
                        "lyrics": lyrics,
                        "tags": tags,
                    },
                    max_audio_length_ms=max_audio_length_ms,
                    save_path=tmp_path,
                    topk=topk,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                )

            # Load the generated audio into ComfyUI AUDIO format
            import torchaudio
            waveform, sample_rate = torchaudio.load(tmp_path)
            # AUDIO format: {"waveform": [B, C, T], "sample_rate": int}
            audio_output = {
                "waveform": waveform.unsqueeze(0),  # [1, C, T]
                "sample_rate": sample_rate,
            }

            print(f"[HeartMuLaGenerator] Done. Duration: "
                  f"{waveform.shape[-1] / sample_rate:.1f}s @ {sample_rate}Hz")
            return (audio_output,)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


# Node registration
NODE_CLASS_MAPPINGS = {
    "HeartTranscriptorLoader": HeartTranscriptorLoader,
    "HeartTranscriptor": HeartTranscriptor,
    "HeartMuLaLoader": HeartMuLaLoader,
    "HeartMuLaGenerator": HeartMuLaGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartTranscriptorLoader": "HeartTranscriptor Loader 🎤",
    "HeartTranscriptor": "HeartTranscriptor (Lyrics ASR) 🎤",
    "HeartMuLaLoader": "HeartMuLa Loader 🎵",
    "HeartMuLaGenerator": "HeartMuLa Generator 🎵",
}

