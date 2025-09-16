# ComfyUI-FireRedTTS

A ComfyUI integration for FireRedTTS2, providing high-quality dialogue and monologue speech synthesis capabilities.

## Features

- **Dialogue Generation**: Multi-speaker conversation audio generation
- **Monologue Generation**: Single-speaker narrative audio generation  
- **Voice Cloning**: Zero-shot voice cloning functionality
- **Multi-language Support**: Chinese, English, Japanese, Korean, French, German, Russian
- **Automatic Model Download**: Models download automatically on first use
- **Device Adaptive**: Automatically selects optimal device (CUDA/MPS/CPU)

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. open [ComfyUI Manager]
2. Search for "ComfyUI-FireRedTTS" in ComfyUI Manager
3. Click Install

### Method 2: Manual Installation

1. Clone this repository to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-FireRedTTS.git
```

2. Install dependencies:
```bash
cd ComfyUI-FireRedTTS
pip install -r requirements.txt
```

3. Restart ComfyUI

### Model Download

On first use, the system will automatically download the FireRedTTS2 model from Hugging Face:
- Model source: [FireRedTeam/FireRedTTS2](https://huggingface.co/FireRedTeam/FireRedTTS2)
- Storage location: `ComfyUI\models\TTS\FireRedTTS2`
- Download size: ~2GB

A progress bar will show during download. Once complete, the model is cached for future use.

## Nodes

### FireRedTTS2 Dialogue Node

Generates Multi-speaker dialogue audio.

**Inputs:**
- `text_list` (STRING): Dialogue text with speaker tags ([S1], [S2])
- `temperature` (FLOAT): Controls generation randomness (0.1-2.0, default: 0.9)
- `topk` (INT): Controls sampling range (1-100, default: 30)
- `S1` (AUDIO, optional): Reference audio for Speaker 1
- `S1_text` (STRING, optional): Reference text for Speaker 1
- `S2` (AUDIO, optional): Reference audio for Speaker 2
- `S2_text` (STRING, optional): Reference text for Speaker 2

**Outputs:**
- `audio` (AUDIO): Generated dialogue audio
- `sample_rate` (INT): Audio sample rate (24000Hz)

### FireRedTTS2 Monologue Node

Generates single-speaker monologue audio.

**Inputs:**
- `text` (STRING): Input text content
- `temperature` (FLOAT): Temperature parameter (0.1-2.0, default: 0.75)
- `topk` (INT): TopK parameter (1-100, default: 20)
- `prompt_wav` (STRING, optional): Reference audio file path
- `prompt_text` (STRING, optional): Reference text content

**Outputs:**
- `audio` (AUDIO): Generated monologue audio
- `sample_rate` (INT): Audio sample rate (24000Hz)

## Usage

### Speaker Tag Format

Use square brackets to mark different speakers in dialogue text:

```
[S1]Hello, what a nice day![S2]Yes, perfect for a walk.[S1]Shall we go to the park?[S2]Great idea!
```

**Supported speaker tags:**
- `[S1]` - Speaker 1
- `[S2]` - Speaker 2

### Voice Cloning Setup

For voice cloning, provide both audio and text for each speaker:

**Speaker 1 (S1):**
- Connect reference audio to `S1` input
- Enter reference text in `S1_text` field

**Speaker 2 (S2):**
- Connect reference audio to `S2` input  
- Enter reference text in `S2_text` field

## Examples

### Basic Dialogue Generation

1. Add "FireRedTTS2 Dialogue" node
2. Input in `text_list`:
   ```
   [S1]Welcome to our podcast![S2]Today we'll discuss AI development.[S1]That's a fascinating topic indeed.
   ```
3. Adjust `temperature` and `topk` parameters
4. Connect audio output to preview or save node

### Voice Cloning Dialogue

1. Prepare reference audio files for each speaker
2. Connect Speaker 1 reference audio to `S1` input
3. Enter Speaker 1 reference text in `S1_text`:
   ```
   This is a voice sample for speaker one
   ```
4. Connect Speaker 2 reference audio to `S2` input
5. Enter Speaker 2 reference text in `S2_text`:
   ```
   This is a voice sample for speaker two
   ```

### Monologue Generation

1. Add "FireRedTTS2 Monologue" node
2. Input long text content in `text` field
3. Optionally provide `prompt_wav` and `prompt_text` for voice cloning
4. Adjust parameters and generate audio

## Parameter Guide

### Temperature
- **Low (0.1-0.5)**: More stable, consistent speech
- **Medium (0.6-1.0)**: Balanced stability and naturalness
- **High (1.1-2.0)**: More variation and expressiveness, may be unstable

### TopK
- **Low (1-20)**: Conservative sampling, more stable speech
- **Medium (21-50)**: Balanced choice
- **High (51-100)**: More diverse sampling, increased variation

## Troubleshooting

### Common Issues

**Q: Model download fails**
A: Check network connection and Hugging Face access. Try using proxy or mirror sites.

**Q: CUDA out of memory**
A: 
- Reduce input text length
- Lower batch size
- Use CPU mode by setting `device="cpu"` in code

**Q: Poor audio quality**
A:
- Check input text format is correct
- Adjust temperature parameter (recommended 0.7-1.0)
- Ensure reference audio quality is good (if using voice cloning)

**Q: Speaker tags not working**
A:
- Ensure correct tag format: `[S1]`, `[S2]`, etc.
- Check for extra spaces around tags
- Confirm text contains corresponding speaker tags

**Q: Node loading fails**
A:
- Check dependencies are properly installed
- Verify ComfyUI version compatibility
- Check console for error messages

### Performance Optimization

**Memory Optimization:**
- Long texts are automatically split for processing
- Model instances are cached and reused
- Recommended single text length: under 500 characters

**Speed Optimization:**
- First use requires model download, subsequent uses are faster
- GPU acceleration significantly improves generation speed
- Batch processing multiple short texts is more efficient than single long text

### System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB storage space (for models)

**Recommended:**
- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU (4GB+ VRAM)
- SSD storage

## Support

If you encounter issues, please check:
1. Dependencies are fully installed
2. Models downloaded correctly
3. Input format meets requirements
4. System resources are sufficient

For more technical details, refer to the project source code and FireRedTTS2 official documentation.