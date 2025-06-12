import subprocess
import os
import wave
import whisper
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio import Pipeline
import torch

# Paths
VIDEO_FILE_PATH = "video-2.mp4"  # Input video file
TEXT_OUTPUT_PATH = ""  # This will be dynamically set
AUDIO_FILE_PATH = ""  # This will be dynamically set

# Load Whisper model from local directory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("medium", device=DEVICE)

# Initialize the PyAnnote speaker diarization pipeline
PATH_TO_CONFIG = "pyannote_diarization_config.yaml"
pipeline = Pipeline.from_pretrained(PATH_TO_CONFIG)

# Function to extract the base name of the input video
def get_base_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

# Function to extract audio from a video file using FFmpeg
def extract_audio_from_video(video_file_path, audio_file_path):
    command = [
        'ffmpeg', 
        '-i', video_file_path,
        '-vn',                   # No video
        '-acodec', 'pcm_s16le',  # PCM audio codec
        '-ar', '16000',          # Sample rate
        '-ac', '1',              # Mono audio
        audio_file_path
    ]
    subprocess.run(command, check=True)
    print(f"Audio extracted and saved to {audio_file_path}")

# Function to perform diarization using pyannote
def diarize_audio(file_path):
    if isinstance(pipeline, SpeakerDiarization):
        diarization = pipeline({'uri': 'test_audio', 'audio': file_path})
        return diarization
    else:
        print("Pipeline is not a SpeakerDiarization pipeline.")
        return None

# Function to transcribe with Whisper and match with diarization
TOLERANCE = 0.5

def transcribe_with_diarization(file_path,   diarization):
    result = whisper_model.transcribe(file_path)
    transcribed_segments = result["segments"]
    
    segments = [(segment.start, segment.end, speaker) for segment, _, speaker in diarization.itertracks(yield_label=True)]
    
    speaker_map = {}
    speaker_count = 1
    for _, _, speaker in segments:
        if speaker not in speaker_map:
            speaker_map[speaker] = f"User{speaker_count}"
            speaker_count += 1
    
    output_lines = []
    last_assigned_speaker = None
    
    for segment in transcribed_segments:
        text_time = segment["start"]
        text = segment["text"]
        assigned_speaker = None

        for segment_start, segment_end, speaker in segments:
            if segment_start - TOLERANCE <= text_time <= segment_end + TOLERANCE:
                assigned_speaker = speaker_map[speaker]
                last_assigned_speaker = assigned_speaker
                break

        if not assigned_speaker and last_assigned_speaker:
            assigned_speaker = last_assigned_speaker
        
        if not assigned_speaker:
            assigned_speaker = "Unknown"
        
        output_lines.append(f"{text_time:.2f} - {assigned_speaker}: {text}")
    
    with open(TEXT_OUTPUT_PATH, "w") as f:
        for line in output_lines:
            f.write(line + "\n")
    
    print(f"Corrected transcript saved at {TEXT_OUTPUT_PATH}")

# Function to save diarization to text file
def diarization_to_text(diarization, output_file_path):
    speaker_map = {}
    speaker_counter = 1
    
    with open(output_file_path, "w") as f:
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_map:
                speaker_map[speaker] = f'user{speaker_counter}'
                speaker_counter += 1
            
            speaker_name = speaker_map[speaker]
            f.write(f"{speaker_name}: [{segment.start:.3f} --> {segment.end:.3f}] {speaker}\n")
    
    print(f"Diarization results saved to {output_file_path}")

# Main function
def main():
    base_name = get_base_name(VIDEO_FILE_PATH)

    global AUDIO_FILE_PATH, TEXT_OUTPUT_PATH
    AUDIO_FILE_PATH = f"{base_name}_extracted_audio.wav"
    TEXT_OUTPUT_PATH = f"{base_name}_transcription.txt"

    extract_audio_from_video(VIDEO_FILE_PATH, AUDIO_FILE_PATH)
    
    diarization_result = diarize_audio(AUDIO_FILE_PATH)
    if diarization_result:
        transcribe_with_diarization(AUDIO_FILE_PATH, diarization_result)
        diarization_to_text(diarization_result, f"{base_name}_diarization.txt")

if __name__ == "__main__":
    main()
