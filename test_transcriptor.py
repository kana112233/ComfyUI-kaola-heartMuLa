import sys
import os
import types

# 1. Mock ComfyUI 'folder_paths' module BEFORE importing nodes
mock_folder_paths = types.ModuleType("folder_paths")
def get_folder_paths(key):
    return ["./models/checkpoints"]
mock_folder_paths.get_folder_paths = get_folder_paths
sys.modules["folder_paths"] = mock_folder_paths

# 2. Mock 'nodes' module imports that might be missing or complex
# We assume torch/torchaudio/transformers are available, but if not we could mock them too.
# For now, let's rely on installed packages.

import torch

# 3. Import the node class
# We need to add current dir to path to import nodes.py
sys.path.append(os.getcwd())
try:
    from nodes import HeartTranscriptor
except ImportError:
    # If nodes.py is not reachable or fails to import
    print("Could not import HeartTranscriptor from nodes.py")
    sys.exit(1)

# 4. Mock Pipeline Object
class MockPipeline:
    def __call__(self, audio_path, **kwargs):
        print(f"\n[MockPipeline] Called with audio_path: {audio_path}")
        print(f"[MockPipeline] Kwargs received: {kwargs}")
        
        # Check return_timestamps
        if "return_timestamps" in kwargs and kwargs["return_timestamps"]:
            print("[MockPipeline] return_timestamps=True verified (Chunking Enabled)")
        else:
            print("[MockPipeline] return_timestamps NOT set (Chunking Disabled)")
            
        # Verify file exists and is a valid WAV
        import wave
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            with wave.open(audio_path, 'rb') as f:
                params = f.getparams()
                print(f"[MockPipeline] WAV params: {params}")
                # Expect 1 channel, 2 bytes/sample (16-bit), 16000Hz (or whatever input passed)
                assert params.nchannels == 1, f"Expected 1 channel, got {params.nchannels}"
                assert params.sampwidth == 2, f"Expected 2 bytes width, got {params.sampwidth}"
        except Exception as e:
            print(f"[MockPipeline] ERROR reading WAV: {e}")
            raise

        return {"text": "Hello World", "chunks": []}

def test_heart_transcriptor():
    print("----------------------------------------------------------------")
    print("Testing HeartTranscriptor Node Logic")
    print("----------------------------------------------------------------")
    
    node = HeartTranscriptor()
    
    # Dummy Audio
    sample_rate = 16000
    duration = 1.0 
    # [1, 1, 16000] tensor
    waveform = torch.randn(1, 1, int(sample_rate * duration))
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}
    
    pipeline = MockPipeline()
    
    print("\n[Test 1] Run with Defaults (Chunking Enabled, Auto Language)")
    res = node.transcribe(
        pipeline=pipeline,
        audio=audio_input,
        max_new_tokens=256,
        num_beams=2,
        task="transcribe",
        temperature=0.0,
        no_speech_threshold=0.4,
        compression_ratio_threshold=1.8,
        logprob_threshold=-1.0,
        language="auto",
        enable_chunking=True,
        condition_on_prev_tokens=False
    )
    print(f"Result 1: {res}")
    
    print("\n[Test 2] Run with Chunking Disabled (Mac Workaround)")
    res2 = node.transcribe(
        pipeline=pipeline,
        audio=audio_input,
        max_new_tokens=256,
        num_beams=2,
        task="transcribe",
        temperature=0.0,
        no_speech_threshold=0.4,
        compression_ratio_threshold=1.8,
        logprob_threshold=-1.0,
        language="zh",
        enable_chunking=False,
        condition_on_prev_tokens=False
    )
    print(f"Result 2: {res2}")

if __name__ == "__main__":
    test_heart_transcriptor()
