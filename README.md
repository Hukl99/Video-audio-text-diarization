# Video-audio-text-diarization
Offline tool for converting audio/video to text with speaker diarization using Whisper and a locally fine-tuned PyAnnote model. Ideal for privacy-focused transcription of meetings, interviews, and conversations.

# 🎙️ Offline Audio-to-Text with Speaker Diarization

This project transcribes audio/video files into text and tags each speaker using Whisper and a locally fine-tuned PyAnnote model. Everything runs **completely offline** to ensure data privacy.

## 📂 Sample Files
Test input video:


## ⚙️ Features
- Extracts audio from video using FFmpeg  
- Performs speaker diarization using local PyAnnote model  
- Transcribes speech with Whisper (`medium`)  
- Generates readable text output with speaker labels  
- All processing is done **offline**  

## 🔧 Installation
Make sure `ffmpeg` is installed and added to PATH.

Then install required Python packages:
```bash
pip install -r requirements.txt
