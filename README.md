# Whisper ASR Fine-tuning for Dhivehi (Maldivian) Language

This repository contains code for fine-tuning OpenAI's Whisper medium model for Automatic Speech Recognition (ASR) in Dhivehi (Maldivian) language. The model is trained on the Mozilla Common Voice dataset.

![image](https://github.com/user-attachments/assets/43ad5f09-0046-47ad-a87c-146ed412e8c6)


## Overview

This project fine-tunes a pre-trained Whisper medium model to transcribe Dhivehi speech. The Whisper model, developed by OpenAI, is a powerful speech recognition system that can be adapted to various languages through fine-tuning. This implementation uses the Hugging Face Transformers library and the Mozilla Common Voice dataset.

## Performance Metrics

### Fine-tuned Model Performance
After fine-tuning for 500 steps (Epoch 1.63), the model achieved:
- Training Loss: 0.136
- Validation Loss: 0.1727
- WER Orthographic: 63.8972
- WER Normalized: 14.0661

### Pre-finetuning Baseline
Before fine-tuning, the base Whisper medium model performed significantly worse on Dhivehi:
- WER Orthographic: 167.2958
- WER Normalized: 125.6981

This represents a **61.8% improvement** in orthographic WER and an **88.8% improvement** in normalized WER after fine-tuning, demonstrating the effectiveness of the adaptation to the Dhivehi language.

## Word Error Rate (WER) Explanation

Word Error Rate is a common metric for evaluating speech recognition quality:

- **WER Orthographic** measures the direct character-for-character error rate on the raw transcription output, including punctuation and case sensitivity. The lower score of 63.8972 (down from 167.2958) indicates substantial improvement in raw transcription accuracy.

- **WER Normalized** measures the error rate after text normalization, which removes punctuation, standardizes case, and applies other text processing techniques to focus on semantic accuracy rather than exact character reproduction. The dramatic improvement to 14.0661 (from 125.6981) indicates that the fine-tuned model produces transcriptions that are semantically accurate and usable despite potential minor orthographic differences.

A normalized WER of approximately 14% is considered reasonable for a low-resource language like Dhivehi, where training data is limited compared to more widely spoken languages.

## Requirements

The following packages are required to run this code:
- datasets
- transformers
- evaluate
- torch
- jiwer
- huggingface_hub

You can install them using pip:
```
pip install datasets transformers evaluate torch jiwer huggingface_hub
```

## Dataset

The project uses Mozilla Common Voice dataset version 11.0 for Dhivehi (language code: 'dv'):
- Training data: Combination of train and validation splits
- Testing data: Test split

The dataset is preprocessed to select only the necessary columns (audio and sentence) and resampled to match the Whisper feature extractor's sampling rate. Audio clips longer than 30 seconds are filtered out.

## Model Architecture

The model architecture is based on OpenAI's Whisper medium model, which is a Transformer-based encoder-decoder model designed for ASR tasks. The model is configured for transcription in Sinhalese language mode, as it appears to be the closest available language to Dhivehi in the Whisper model.

## Training Configuration

The training configuration includes:
- Batch size: 2 per device
- Gradient accumulation steps: 8
- Learning rate: 1e-5
- Learning rate scheduler: Constant with warmup
- Warmup steps: 50
- Maximum steps: 500 (recommended to increase to 4000 for better results)
- Mixed precision training (FP16)

## Evaluation Metrics

The model performance is evaluated using Word Error Rate (WER) in two ways:
1. Orthographic WER: Standard WER calculation on raw output
2. Normalized WER: WER calculation after applying text normalization

Lower WER values indicate better performance, with 0% being perfect transcription.

## Usage

### Fine-tuning the Model

To fine-tune the model, simply run the script:

```python
# The script automatically:
# 1. Loads the dataset
# 2. Preprocesses the audio data
# 3. Sets up the model and training configuration
# 4. Trains the model
# 5. Pushes the trained model to Hugging Face Hub (requires login)
```

### Using the Fine-tuned Model

Once fine-tuned, you can use the model for inference:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Load model and processor
processor = WhisperProcessor.from_pretrained("your-username/whisper-small-dv")
model = WhisperForConditionalGeneration.from_pretrained("your-username/whisper-small-dv")

# Transcribe audio
def transcribe_audio(audio_path):
    # Load audio
    from datasets import Audio
    audio = Audio(sampling_rate=16000).decode_example({"path": audio_path})
    
    # Process audio
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], 
                              return_tensors="pt").input_features
    
    # Generate transcription
    predicted_ids = model.generate(input_features, 
                                   language="sinhalese", 
                                   task="transcribe")
    
    # Decode the predicted ids
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription
```

## Model Details

- Base model: openai/whisper-medium
- Fine-tuned for: Dhivehi language (dv)
- Task: Automatic Speech Recognition (transcription)
- Dataset: Mozilla Common Voice 11.0/13.0
- Metrics: Word Error Rate (orthographic and normalized)

## Further Improvement Opportunities

While the current model shows significant improvement over the base model, there are several approaches that could potentially enhance performance further:

1. **Extended training**: Increasing the number of training steps from 500 to 4000+ could significantly improve performance
2. **Data augmentation**: Applying techniques like speed perturbation, pitch shifting, and adding background noise to increase training data diversity
3. **Language model integration**: Implementing a Dhivehi language model for post-processing transcriptions
4. **Hyperparameter optimization**: Fine-tuning learning rates, batch sizes, and other training parameters

## Notes

- The code configures the Whisper model to use "Sinhalese" as the language setting since Dhivehi is not directly supported in the base Whisper model.
- For optimal results on production use, it's recommended to increase the training steps to 4000.
- The model is pushed to the Hugging Face Hub with appropriate metadata to facilitate discovery and reuse.

## References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Mozilla Common Voice Dataset](https://commonvoice.mozilla.org/)
