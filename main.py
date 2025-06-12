import subprocess
import os
import whisper
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
import torch
import yaml

# === Paths ===
VIDEO_FILE_PATH = "path/to/your/audio/or/video/file"
TEXT_OUTPUT_PATH = ""
AUDIO_FILE_PATH = ""
CONFIG_PATH = "pyannote_diarization_config.yaml"  # Your local config path


# === Device Setup ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load Whisper ===
try:
    whisper_model = whisper.load_model("medium", device=DEVICE)
except Exception as e:
    print(f"[ERROR] Failed to load Whisper model: {e}")
    exit(1)

# === Load PyAnnote Local Diarization Pipeline ===
try:
    pipeline = Pipeline.from_pretrained(CONFIG_PATH)
except Exception as e:
    print(f"[ERROR] Failed to load local diarization model or config: {e}")
    exit(1)

# === Helpers ===
def get_base_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def extract_audio_from_video(video_file_path, audio_file_path):
    command = [
        'ffmpeg',
        '-i', video_file_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        audio_file_path
    ]
    try:
        subprocess.run(command, check=True)
        print(f"[INFO] Audio extracted to: {audio_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg failed: {e}")
        exit(1)

def diarize_audio(file_path):
    try:
        if isinstance(pipeline, SpeakerDiarization):
        diarization = pipeline({'uri': 'test_audio', 'audio': file_path})
        return diarization
    else:
        print("Pipeline is not a SpeakerDiarization pipeline.")
        return None
    except Exception as e:
        print(f"[ERROR] Diarization failed: {e}")
        return None

TOLERANCE = 0.5

def transcribe_with_diarization(file_path, diarization):
    try:
        result = whisper_model.transcribe(file_path)
    except Exception as e:
        print(f"[ERROR] Whisper transcription failed: {e}")
        return

    transcribed_segments = result["segments"]
    diarization_segments = [(segment.start, segment.end, speaker) for segment, _, speaker in diarization.itertracks(yield_label=True)]

    speaker_map = {}
    speaker_count = 1
    for _, _, speaker in diarization_segments:
        if speaker not in speaker_map:
            speaker_map[speaker] = f"User{speaker_count}"
            speaker_count += 1

    output_lines = []
    last_assigned_speaker = None

    for segment in transcribed_segments:
        text_start = segment["start"]
        text_end = segment["end"]
        text = segment["text"]
        assigned_speaker = None

        for seg_start, seg_end, speaker in diarization_segments:
            if seg_start - TOLERANCE <= text_start <= seg_end + TOLERANCE:
                assigned_speaker = speaker_map[speaker]
                last_assigned_speaker = assigned_speaker
                break

        if not assigned_speaker and last_assigned_speaker:
            assigned_speaker = last_assigned_speaker

        if not assigned_speaker:
            assigned_speaker = "Unknown"

        output_lines.append(f"[{text_start:.2f}s] {assigned_speaker}: {text.strip()}")

    try:
        with open(TEXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(line + "\n")
        print(f"[INFO] Transcription with speaker diarization saved at: {TEXT_OUTPUT_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to write output file: {e}")

def diarization_to_text(diarization, output_file_path):
    speaker_map = {}
    speaker_counter = 1
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_map:
                    speaker_map[speaker] = f"user{speaker_counter}"
                    speaker_counter += 1
                speaker_name = speaker_map[speaker]
                f.write(f"{speaker_name}: [{segment.start:.2f} --> {segment.end:.2f}]\n")
        print(f"[INFO] Diarization saved at: {output_file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save diarization output: {e}")

def main():
    base_name = get_base_name(VIDEO_FILE_PATH)

    global AUDIO_FILE_PATH, TEXT_OUTPUT_PATH
    AUDIO_FILE_PATH = f"{base_name}_extracted_audio.wav"
    TEXT_OUTPUT_PATH = f"{base_name}_transcription.txt"

    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"[ERROR] Video file not found: {VIDEO_FILE_PATH}")
        return

    extract_audio_from_video(VIDEO_FILE_PATH, AUDIO_FILE_PATH)

    diarization_result = diarize_audio(AUDIO_FILE_PATH)
    if diarization_result:
        transcribe_with_diarization(AUDIO_FILE_PATH, diarization_result)
        diarization_to_text(diarization_result, f"{base_name}_diarization.txt")
    else:
        print("[ERROR] Skipping transcription due to diarization failure.")

if __name__ == "__main__":
    main()
