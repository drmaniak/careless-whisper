import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, Features, Value, load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# Assume you already have the following helper functions
def merge_transcripts(utterances):
    """
    Merge the text from a list of utterances.
    Insert the special token <|speakerturn|> whenever the speaker_id changes.
    """
    merged_text = ""
    prev_speaker = None
    for utt in utterances:
        current_speaker = utt["speaker_id"]
        if prev_speaker is not None and current_speaker != prev_speaker:
            merged_text += "<|speakerturn|> "
        merged_text += utt["text"].strip() + " "
        prev_speaker = current_speaker
    return merged_text.strip()


def merge_audio(utterances, sampling_rate=16000, gap_duration=0.1):
    """
    Merge the audio arrays from a list of utterances.
    Insert a short silence (gap) between utterances.
    """
    gap_samples = int(gap_duration * sampling_rate)
    silence = np.zeros(gap_samples, dtype=np.float32)
    audio_segments = []

    for utt in utterances:
        audio_array = utt["audio"]["array"]
        audio_segments.append(audio_array)
        audio_segments.append(silence)

    if audio_segments:
        audio_segments = audio_segments[:-1]  # remove final silence
    return np.concatenate(audio_segments)


def segment_meeting(utterances, target_duration=25.0, max_duration=26.0):
    """
    Group utterances into segments of roughly target_duration to max_duration seconds.
    """
    segments = []
    current_segment = []
    current_duration = 0.0

    for utt in utterances:
        utt_duration = utt["duration"]
        if current_duration + utt_duration <= max_duration:
            current_segment.append(utt)
            current_duration += utt_duration
            if current_duration >= target_duration:
                segments.append(current_segment)
                current_segment = []
                current_duration = 0.0
        else:
            if current_duration >= target_duration:
                segments.append(current_segment)
                current_segment = [utt]
                current_duration = utt_duration
            else:
                current_segment.append(utt)
                current_duration += utt_duration
                segments.append(current_segment)
                current_segment = []
                current_duration = 0.0
    if current_segment:
        segments.append(current_segment)
    return segments


def save_audio(merged_audio, sampling_rate, output_dir, meeting_id, segment_idx):
    """
    Save merged audio as a WAV file and return its file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{meeting_id}_segment{segment_idx}.wav")
    sf.write(file_path, merged_audio, sampling_rate)
    return file_path


def process_meetings(
    meetings,
    output_dir="./merged_audio_clips",
    sampling_rate=16000,
    target_duration=24.0,
    max_duration=26.0,
):
    """
    Process a meetings dictionary into training samples.

    Args:
        meetings (dict): Keys are meeting IDs and values are lists of utterance dicts.

    Returns:
        List[dict]: Each dict contains keys: meeting_id, position, audio_file, text, duration.
    """
    training_samples = []
    for meeting_id, utterances in tqdm(
        meetings.items(), desc="Processing meeting utterances"
    ):
        # Ensure utterances are sorted by begin_time:
        utterances.sort(key=lambda x: x["begin_time"])
        segments = segment_meeting(utterances, target_duration, max_duration)
        for i, segment in enumerate(segments):
            merged_text = merge_transcripts(segment)
            merged_audio = merge_audio(segment, sampling_rate=sampling_rate)
            segment_duration = sum(utt["duration"] for utt in segment)
            audio_file = save_audio(
                merged_audio, sampling_rate, output_dir, meeting_id, i
            )
            training_samples.append(
                {
                    "meeting_id": meeting_id,
                    "position": i,
                    "audio_file": audio_file,
                    "text": merged_text,
                    "duration": segment_duration,
                }
            )
    return training_samples


def create_dataset_from_samples(samples):
    """
    Create a Hugging Face Dataset from a list of training samples.
    Audio is loaded lazily.
    """
    features = Features(
        {
            "meeting_id": Value("string"),
            "position": Value("int32"),
            "audio_file": Audio(sampling_rate=16000),
            "text": Value("string"),
            "duration": Value("float32"),
        }
    )
    ds = Dataset.from_list(samples, features=features)
    return ds


def get_ami_dataset(
    url: str = "edinburghcstr/ami",
    subset: str = "ihm",
    split: Literal["train", "test", "validation"] = "train",
) -> Dataset:
    dataset: DatasetDict = load_dataset(url, subset)

    meetings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Arrange the examples by their meeting id
    for example in tqdm(
        dataset[split], desc=f"Grouping {split} examples by meeting_id"
    ):
        example["duration"] = abs(example["end_time"] - example["begin_time"])
        meetings[example["meeting_id"]].append(example)

    # Sort the meeting_id clips by begin_time
    for meeting_id in tqdm(meetings, f"Sorting {split} meetings by time"):
        meetings[meeting_id].sort(key=lambda x: x["begin_time"])

    # Create training samples
    training_samples = process_meetings(
        meetings, output_dir=str(DATA_DIR / f"merged_audio_clips_{split}")
    )

    # Create dataset from training samples
    ds = create_dataset_from_samples(training_samples)

    return ds


if __name__ == "__main__":
    for split in ["train", "test", "validation"]:
        print(f"Begining processing of {split} split")
        ds = get_ami_dataset(split=split)  # type: ignore
        datapath = DATA_DIR / f"ami_dataset_{split}"
        ds.save_to_disk(datapath)
        print(f"Saved {split} split to {datapath}")
