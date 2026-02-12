# ComfyUI-kaola-heartMuLa

ComfyUI custom node for **HeartTranscriptor** — a Whisper-based ASR model from the [HeartMuLa](https://github.com/HeartMuLa/heartlib) project, specifically tuned for **lyrics transcription** in music scenes.

HeartTranscriptor can effectively filter complex background instrument interference and achieve highly accurate lyrics recognition.

## Features

- 🎤 **Lyrics Transcription**: Accurate lyrics recognition from music audio
- 🎵 **Music-Optimized**: Tuned to handle background instruments and complex audio mixes
- 🌍 **Multilingual**: Supports English, Chinese, Japanese, Korean, Spanish and more
- 🔄 **Translate**: Optional translation to English

## Installation

### 1. Clone into ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kana112233/ComfyUI-kaola-heartMuLa.git
cd ComfyUI-kaola-heartMuLa
pip install -r requirements.txt
```

### 2. Download Model

Download the HeartTranscriptor-oss model and place it under `ComfyUI/models/checkpoints/`:

```bash
cd ComfyUI/models/checkpoints
hf download HeartMuLa/HeartTranscriptor-oss --local-dir ./HeartTranscriptor-oss
```

Expected directory structure:
```
ComfyUI/models/checkpoints/
└── HeartTranscriptor-oss/   # or any name you like
    ├── config.json
    ├── model.safetensors
    ├── preprocessor_config.json
    ├── tokenizer.json
    └── ...
```

## Nodes

### HeartTranscriptor Loader 🎤

Loads the HeartTranscriptor model. Works like CheckpointLoaderSimple — dropdown lists available model directories from `ComfyUI/models/checkpoints/`.

| Parameter             | Description                                          |
|-----------------------|------------------------------------------------------|
| `model_name`          | Select model from dropdown                           |
| `device`              | `cuda` or `cpu`                                      |
| `dtype`               | `float16`, `float32`, or `bfloat16`                  |
| `model_path_override` | (Optional) Absolute path for testing                 |

### HeartTranscriptor (Lyrics ASR) 🎤

Transcribes lyrics from audio input.

| Parameter                      | Default    | Description                              |
|-------------------------------|------------|------------------------------------------|
| `pipeline`                    | —          | Pipeline from the Loader node            |
| `audio`                       | —          | ComfyUI AUDIO input                      |
| `max_new_tokens`              | 256        | Max tokens per chunk                     |
| `num_beams`                   | 2          | Beam search width                        |
| `task`                        | transcribe | `transcribe` or `translate` (to English) |
| `temperature`                 | 0.0        | Sampling temperature                     |
| `no_speech_threshold`         | 0.4        | No-speech detection threshold            |
| `compression_ratio_threshold` | 1.8        | Compression ratio filter                 |
| `logprob_threshold`           | -1.0       | Log-probability filter                   |
| `condition_on_prev_tokens`    | False      | Condition on previous tokens             |

## Example Workflow

An example workflow is included in the `examples/` folder:

- **[heart_transcriptor_basic.json](examples/heart_transcriptor_basic.json)** — Basic lyrics transcription: LoadAudio → HeartTranscriptor Loader → HeartTranscriptor → ShowText

Drag and drop the JSON file into ComfyUI to load the workflow.

## Tips

- **Best Results**: Use source separation tools (e.g. Demucs) to extract vocal tracks before transcription.
- **Memory**: The model uses ~1.5GB VRAM in float16 mode.

## License

Apache-2.0
