#  Speech Recognition Model

This repository contains code for fine-tuning OpenAI's Whisper medium model for Dhivehi (Maldivian) language speech recognition using the Mozilla Common Voice dataset.
![image](https://github.com/user-attachments/assets/8f4c3ce0-2b42-4c5f-adc8-9e38264817ee)

## Project Overview

This project implements an Automatic Speech Recognition (ASR) system for the Dhivehi language by fine-tuning the Whisper medium model on Mozilla's Common Voice dataset. The fine-tuned model can transcribe Dhivehi speech audio to text with improved accuracy compared to the base model.

## Dataset

The project uses the Dhivehi (dv) language subset from Mozilla Common Voice:
- **Dataset version**: Common Voice 11.0 and 13.0
- **Data splits**: Training data combines train and validation splits, with a separate test split
- **Features used**: Audio files and their corresponding text transcriptions ('sentence')

## Model Architecture

- **Base model**: `openai/whisper-medium`
- **Task**: Speech transcription
- **Language**: Initially configured for Sinhalese but fine-tuned for Dhivehi

## Setup and Installation

```bash
# Install required packages
pip install datasets
pip install transformers
pip install evaluate
pip install jiwer
```

## Data Preparation

The code performs several preprocessing steps:
1. Loading the Common Voice dataset for Dhivehi
2. Selecting only the audio and sentence columns
3. Resampling audio to match the Whisper model's required sampling rate
4. Processing data with the Whisper processor to create model inputs
5. Filtering out audio samples longer than 30 seconds

## Training Configuration

- **Batch size**: 2 samples per device
- **Gradient accumulation steps**: 8
- **Learning rate**: 1e-5
- **Training steps**: 500 (can be increased to 4000 for better results)
- **Optimization**: Mixed precision (fp16)
- **Evaluation**: Word Error Rate (WER) with and without text normalization

## Usage

### Loading the Fine-tuned Model

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load processor and model
processor = WhisperProcessor.from_pretrained("YOUR_USERNAME/whisper-small-dv")
model = WhisperForConditionalGeneration.from_pretrained("YOUR_USERNAME/whisper-small-dv")

# Transcribe audio
def transcribe_audio(audio_path):
    # Load audio
    from datasets import Audio
    audio_loader = Audio(sampling_rate=16000)
    audio = audio_loader.decode_audio(audio_path)
    
    # Process audio
    input_features = processor(audio["array"], 
                               sampling_rate=audio["sampling_rate"], 
                               return_tensors="pt").input_features
    
    # Generate transcription
    predicted_ids = model.generate(input_features, task="transcribe", language="dv")
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription[0]
```

## Performance Metrics

The model is evaluated using Word Error Rate (WER) in two ways:
1. **WER Ortho**: Standard WER without text normalization
2. **WER Norm**: WER after applying basic text normalization

Lower WER indicates better performance, with the model selecting the checkpoint with the lowest WER as the best model.

## Limitations

- The training dataset size is relatively small (~4,863 samples), which may limit performance
- The model uses Sinhalese tokenization as the base, which may not be optimal for Dhivehi
- Audio samples are limited to 30 seconds in length

## Citation

If you use this model, please cite:
```
@misc{whisper-small-dv,
  author = {Your Name},
  title = {Whisper Medium Model Fine-tuned for Dhivehi Speech Recognition},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/whisper-small-dv}
}
```

## Acknowledgments

- Mozilla Common Voice for providing the speech dataset
- OpenAI for the Whisper model
- Hugging Face for the Transformers library

## License

This project is available under [LICENSE NAME] - see the LICENSE file for details.
