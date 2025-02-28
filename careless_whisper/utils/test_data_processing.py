import numpy as np
import pytest


def merge_transcripts(utterances):
    """
    Merge the text from a list of utterances.
    Insert the special token <|speakerturn|> whenever the speaker_id changes.

    Args:
        utterances (list of dict): Each dict must have keys "text" and "speaker_id".

    Returns:
        str: The merged transcript.
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

    Args:
        utterances (list of dict): Each dict must have an "audio" key, where audio is a dict with key "array".
        sampling_rate (int): Sampling rate of the audio.
        gap_duration (float): Duration of silence (in seconds) to insert between utterances.

    Returns:
        numpy.ndarray: The concatenated audio array.
    """
    gap_samples = int(gap_duration * sampling_rate)
    silence = np.zeros(gap_samples, dtype=np.float32)
    audio_segments = []

    for utt in utterances:
        audio_array = utt["audio"]["array"]
        audio_segments.append(audio_array)
        audio_segments.append(silence)

    if audio_segments:
        audio_segments = audio_segments[:-1]  # Remove the last inserted silence
    return np.concatenate(audio_segments)


# ---------------------------
# Unit tests for the functions
# ---------------------------


def test_merge_transcripts_single_speaker():
    utterances = [
        {"speaker_id": "A", "text": "Hello"},
        {"speaker_id": "A", "text": "How are you?"},
    ]
    result = merge_transcripts(utterances)
    expected = "Hello How are you?"
    assert result == expected


def test_merge_transcripts_multiple_speakers():
    utterances = [
        {"speaker_id": "A", "text": "Hello"},
        {"speaker_id": "B", "text": "Hi there"},
        {"speaker_id": "B", "text": "How are you?"},
        {"speaker_id": "A", "text": "I'm fine"},
    ]
    result = merge_transcripts(utterances)
    expected = "Hello <|speakerturn|> Hi there How are you? <|speakerturn|> I'm fine"
    assert result == expected


def test_merge_audio():
    # Create dummy audio arrays for testing
    utt1 = {"audio": {"array": np.array([0.1, 0.2, 0.3], dtype=np.float32)}}
    utt2 = {"audio": {"array": np.array([0.4, 0.5], dtype=np.float32)}}
    utterances = [utt1, utt2]

    # Use a small sampling rate and gap_duration for test purposes
    result = merge_audio(
        utterances, sampling_rate=1000, gap_duration=0.1
    )  # gap = 100 samples

    # Expected length: len(utt1)=3 + gap=100 + len(utt2)=2 => 105 samples total
    assert len(result) == 3 + 100 + 2
    # Verify that the gap region is all zeros
    assert np.all(result[3:103] == 0.0)
