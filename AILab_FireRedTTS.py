# ComfyUI-FireRedTTS V0.1.1
# A ComfyUI integration for FireRedTTS‑2, a real-time multi-speaker TTS system enabling high-quality,
# emotionally expressive dialogue and monologue synthesis. Leveraging a streaming architecture and context-aware prosody modeling,
# it supports natural speaker turns and stable long-form generation, ideal for interactive chat and podcast applications.
#
# Models License Notice:
# - FireRedTTS: Apache-2.0 License (https://huggingface.co/FireRedTeam/FireRedTTS2)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-FireRedTTS

import os
import os.path as osp
import torch
import torchaudio
import tempfile
import shutil
import folder_paths
from huggingface_hub import snapshot_download
from pathlib import Path
from typing import Optional, List

# Import the FireRedTTS2 model
from fireredtts2.fireredtts2 import FireRedTTS2

# Setup temporary directory
now_dir = osp.dirname(osp.abspath(__file__))
tmp_dir = osp.join(now_dir, "tmp")

def get_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] CUDA available - Using GPU: {device_name}")
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("[INFO] MPS available - Using Apple Silicon GPU")
        return "mps"
    else:
        print("[INFO] Using CPU (no GPU acceleration available)")
        return "cpu"

device = get_device()
dialogue_model = None
monologue_model = None
model_path = None

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def check_gpu_memory():
    if device == "cuda" and torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        print(f"[INFO] GPU Memory - Total: {total_memory:.1f}GB, Allocated: {allocated_memory:.1f}GB, Cached: {cached_memory:.1f}GB")
        return total_memory, allocated_memory, cached_memory
    return None, None, None

def safe_generate_with_fallback(model, generate_func, *args, **kwargs):
    try:
        clear_gpu_memory()
        return generate_func(*args, **kwargs)
    except RuntimeError as e:
        if "CUDA" in str(e) or "out of memory" in str(e).lower():
            print(f"[WARNING] GPU memory error: {str(e)}")
            print("[INFO] Attempting to clear GPU cache and retry...")
            clear_gpu_memory()
            try:
                return generate_func(*args, **kwargs)
            except RuntimeError as e2:
                print(f"[WARNING] GPU retry failed: {str(e2)}")
                print("[INFO] Falling back to CPU generation...")
                global dialogue_model, monologue_model
                if hasattr(model, 'device'):
                    original_device = model.device
                    model.to('cpu')
                    try:
                        result = generate_func(*args, **kwargs)
                        model.to(original_device)
                        return result
                    except Exception as e3:
                        print(f"[ERROR] CPU fallback also failed: {str(e3)}")
                        raise e3
                else:
                    raise e2
        else:
            raise e

def get_model_path():
    global model_path
    
    if model_path is None:
        models_dir = folder_paths.models_dir
        tts_dir = os.path.join(models_dir, "TTS")
        fireredtts2_dir = os.path.join(tts_dir, "FireRedTTS2")
        
        if not os.path.exists(fireredtts2_dir) or not os.path.exists(os.path.join(fireredtts2_dir, "config_llm.json")):
            print("[INFO] Downloading FireRedTTS2 model...")
            try:
                os.makedirs(tts_dir, exist_ok=True)
                os.makedirs(fireredtts2_dir, exist_ok=True)
                
                snapshot_download(
                    repo_id="FireRedTeam/FireRedTTS2",
                    local_dir=fireredtts2_dir,
                    local_dir_use_symlinks=False
                )
                print("[INFO] FireRedTTS2 model downloaded successfully!")
            except Exception as e:
                print(f"[ERROR] Failed to download FireRedTTS2 model: {str(e)}")
                raise e
        else:
            print("[INFO] Using existing FireRedTTS2 model")
        
        model_path = fireredtts2_dir
    
    return model_path

def get_dialogue_model():
    global dialogue_model
    
    if dialogue_model is None:
        try:
            model_dir = get_model_path()
            print(f"[INFO] Loading dialogue model on {device}...")
            check_gpu_memory()
            dialogue_model = FireRedTTS2(
                pretrained_dir=model_dir,
                gen_type="dialogue",
                device=device
            )
            print("[INFO] Dialogue model loaded successfully!")
            check_gpu_memory()
        except Exception as e:
            print(f"[ERROR] Failed to load dialogue model: {str(e)}")
            import traceback
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            raise e
    
    return dialogue_model

def get_monologue_model():
    global monologue_model
    
    if monologue_model is None:
        try:
            model_dir = get_model_path()
            print(f"[INFO] Loading monologue model on {device}...")
            monologue_model = FireRedTTS2(
                pretrained_dir=model_dir,
                gen_type="monologue", 
                device=device
            )
            print("[INFO] Monologue model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load monologue model: {str(e)}")
            raise e
    
    return monologue_model

def parse_multiline_string(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    
    lines = [line.strip() for line in text.strip().split('\n')]
    return [line for line in lines if line]

def to_comfyui_audio(tensor: torch.Tensor, sample_rate: int = 24000):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    
    result = {"waveform": tensor, "sample_rate": sample_rate}
    return result

def parse_dialogue_text(text_list_str: str) -> List[str]:
    if not text_list_str or not text_list_str.strip():
        return []
    
    lines = parse_multiline_string(text_list_str)
    parsed_lines = []
    
    for i, line in enumerate(lines):
        has_speaker_label = any(f"[S{j}]" in line for j in range(1, 3))
        
        if not has_speaker_label:
            speaker_num = (i % 2) + 1
            formatted_line = f"[S{speaker_num}]{line}"
            print(f"[INFO] Auto-formatted line {i+1}: Added [S{speaker_num}] to '{line[:50]}...'")
            parsed_lines.append(formatted_line)
        else:
            parsed_lines.append(line)
    
    return parsed_lines

class FireRedTTS2_Dialogue:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_list": ("STRING", {"multiline": True, "default": "[S1]Hello, how are you?\n[S2]I'm doing great, thanks!\n[S1]That's wonderful to hear."}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.01}),
                "topk": ("INT", {"default": 30, "min": 1, "max": 100}),
            },
            "optional": {
                "S1": ("AUDIO",),
                "S1_text": ("STRING", {"multiline": False, "default": ""}),
                "S2": ("AUDIO",),
                "S2_text": ("STRING", {"multiline": False, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_dialogue"
    CATEGORY = "🧪AILab/🔊TTS/FireRedTTS"
    
    def generate_dialogue(self, text_list, temperature, topk, 
                         S1=None, S1_text="", S2=None, S2_text=""):
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            model = get_dialogue_model()
            
            text_lines = parse_dialogue_text(text_list)
            if not text_lines:
                raise ValueError("Dialogue text cannot be empty")
            
            prompt_wavs = []
            prompt_texts = []
            
            speakers = [
                ("[S1]", S1, S1_text),
                ("[S2]", S2, S2_text),
            ]
            
            for speaker_label, prompt_wav, prompt_text in speakers:
                if prompt_wav is not None and prompt_text.strip():
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=tmp_dir) as f:
                        waveform = prompt_wav["waveform"].squeeze(0)
                        torchaudio.save(f.name, waveform, prompt_wav["sample_rate"])
                        prompt_wavs.append(f.name)
                        prompt_texts.append(f"{speaker_label}{prompt_text.strip()}")
            
            final_prompt_wavs = prompt_wavs if prompt_wavs else None
            final_prompt_texts = prompt_texts if prompt_texts else None
            
            if final_prompt_wavs:
                print(f"[INFO] Using {len(final_prompt_wavs)} speaker prompts for voice cloning")
            else:
                print("[INFO] No prompts provided, using random voices")
            
            print(f"[INFO] Generating dialogue with {len(text_lines)} text segments")
            print(f"[INFO] Using device: {device}")
            
            audio_tensor = safe_generate_with_fallback(
                model, 
                model.generate_dialogue,
                text_list=text_lines,
                prompt_wav_list=final_prompt_wavs,
                prompt_text_list=final_prompt_texts,
                temperature=temperature,
                topk=topk
            )
            
            res_audio = to_comfyui_audio(audio_tensor, 24000)
            
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            
            print("[INFO] Dialogue generation completed successfully")
            return (res_audio,)
            
        except Exception as e:
            print(f"[ERROR] FireRedTTS2 Dialogue generation failed: {str(e)}")
            import traceback
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            empty_audio = torch.zeros(1, 1, 1000)
            return (to_comfyui_audio(empty_audio.squeeze(), 24000),)

class FireRedTTS2MonologueNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "这是一段测试文本"}),
                "temperature": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 2.0, "step": 0.01}),
                "topk": ("INT", {"default": 20, "min": 1, "max": 100}),
            },
            "optional": {
                "prompt_wav": ("AUDIO",),
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_monologue"
    CATEGORY = "🧪AILab/🔊TTS/FireRedTTS"
    
    def generate_monologue(self, text, temperature, topk, prompt_wav=None, prompt_text=""):
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            model = get_monologue_model()
            
            if not text.strip():
                raise ValueError("Text cannot be empty")
            
            prompt_wav_path = None
            prompt_text_content = None
            
            if prompt_wav is not None and prompt_text.strip():
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=tmp_dir) as f:
                    waveform = prompt_wav["waveform"].squeeze(0)
                    torchaudio.save(f.name, waveform, prompt_wav["sample_rate"])
                    prompt_wav_path = f.name
                    prompt_text_content = prompt_text.strip()
                    print(f"[INFO] Using voice cloning with reference audio and text")
            else:
                print(f"[INFO] Using random voice generation (no prompts provided)")
            
            print(f"[INFO] Generating monologue for text: {text[:50]}...")
            print(f"[INFO] Parameters - temperature: {temperature}, topk: {topk}")
            
            audio_tensor = safe_generate_with_fallback(
                model,
                model.generate_monologue,
                text=text,
                prompt_wav=prompt_wav_path,
                prompt_text=prompt_text_content,
                temperature=temperature,
                topk=topk
            )
            print(f"[INFO] Generated audio tensor shape: {audio_tensor.shape}")
            
            res_audio = to_comfyui_audio(audio_tensor, 24000)
            
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            
            print("[INFO] Monologue generation completed successfully")
            return (res_audio,)
            
        except Exception as e:
            print(f"[ERROR] FireRedTTS2 Monologue generation failed: {str(e)}")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            empty_audio = torch.zeros(1, 1, 1000)
            return (to_comfyui_audio(empty_audio.squeeze(), 24000),)

NODE_CLASS_MAPPINGS = {
    "FireRedTTS2_Dialogue": FireRedTTS2_Dialogue,
    "FireRedTTS2MonologueNode": FireRedTTS2MonologueNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FireRedTTS2_Dialogue": "FireRedTTS2 Dialogue",
    "FireRedTTS2MonologueNode": "FireRedTTS2 Monologue",
}