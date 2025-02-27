#!/usr/bin/env python3
"""
Real-time audio transcription using Hugging Face's implementation of Whisper model.
Captures audio from the microphone and prints transcriptions to STDOUT.
"""

import queue
import threading
import numpy as np
import sounddevice as sd
import torch
from datetime import datetime, timedelta
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHUNK_SIZE = 1024 * 3  # Number of samples per chunk
SILENCE_THRESHOLD = 0.01  # Threshold for silence detection (lowered to be more sensitive)
SILENCE_DURATION = 0.5  # Seconds of silence to trigger processing (reduced)
MODEL_NAME = "openai/whisper-tiny.en"  # Hugging Face model ID
DEBUG = True  # Enable debug output

class HFAudioTranscriber:
    def __init__(self):
        # Initialize the Whisper model from Hugging Face
        print(f"Loading Whisper model '{MODEL_NAME}' from Hugging Face...")
        self.processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"Model loaded on {self.device}!")
        
        # Test the microphone
        print("Testing microphone...")
        try:
            test_duration = 0.1  # seconds
            test_data = sd.rec(
                int(test_duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            audio_level = np.abs(test_data).mean()
            print(f"Microphone test: Audio level = {audio_level:.6f}")
            if audio_level < 0.001:
                print("WARNING: Very low audio level detected. Please check your microphone.")
        except Exception as e:
            print(f"Microphone test error: {e}")
        
        # Audio processing variables
        self.audio_queue = queue.Queue()
        self.last_audio_time = datetime.now()
        self.is_running = False
        self.is_processing = False
        self.current_segment = []
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice to capture audio"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert audio to mono and the right format
        audio_data = indata[:, 0]  # Take first channel if stereo
        self.audio_queue.put(audio_data.copy())
    
    def process_audio_queue(self):
        """Process audio from the queue"""
        segment_counter = 0
        while self.is_running:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=0.5)
                self.current_segment.extend(audio_data)
                
                # Check if we have silence
                audio_level = np.abs(audio_data).mean()
                if audio_level < SILENCE_THRESHOLD:
                    time_since_last_audio = datetime.now() - self.last_audio_time
                    if time_since_last_audio > timedelta(seconds=SILENCE_DURATION) and not self.is_processing:
                        # Process the accumulated audio
                        if len(self.current_segment) > SAMPLE_RATE:  # At least 1 second of audio
                            segment_counter += 1
                            print(f"Detected silence. Processing segment #{segment_counter} ({len(self.current_segment)/SAMPLE_RATE:.2f} sec)")
                            self.process_segment()
                else:
                    self.last_audio_time = datetime.now()
                    # Periodically process audio even without silence if it gets too long
                    if len(self.current_segment) > SAMPLE_RATE * 10 and not self.is_processing:  # 10 seconds max
                        segment_counter += 1
                        print(f"Max segment length reached. Processing segment #{segment_counter} ({len(self.current_segment)/SAMPLE_RATE:.2f} sec)")
                        self.process_segment()
                
            except queue.Empty:
                # Periodically check if we have accumulated audio to process
                if len(self.current_segment) > SAMPLE_RATE * 2 and not self.is_processing:  # At least 2 seconds
                    segment_counter += 1
                    print(f"Queue empty but have audio. Processing segment #{segment_counter} ({len(self.current_segment)/SAMPLE_RATE:.2f} sec)")
                    self.process_segment()
    
    def process_segment(self):
        """Process a segment of audio with Whisper via Hugging Face"""
        self.is_processing = True
        
        # Convert to the format expected by the model
        audio_array = np.array(self.current_segment, dtype=np.float32)
        
        # Add debug info about the audio segment
        audio_duration = len(audio_array) / SAMPLE_RATE
        print(f"Processing audio segment: {audio_duration:.2f} seconds, max amplitude: {np.abs(audio_array).max():.4f}")
        
        # Normalize audio
        if np.abs(audio_array).max() > 0:
            audio_array = audio_array / np.abs(audio_array).max()
        
        # Transcribe with Whisper via Hugging Face
        try:
            # Process the audio input
            print("Processing audio features...")
            input_features = self.processor(
                audio_array, 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate token ids
            print("Generating transcription...")
            predicted_ids = self.model.generate(input_features)
            
            # Decode token ids to text
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            # Print the transcription if not empty
            if transcription.strip():
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Transcription: {transcription.strip()}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No transcription detected (empty result)")
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
        
        # Reset for next segment
        self.current_segment = []
        self.is_processing = False
    
    def start(self):
        """Start capturing and transcribing audio"""
        self.is_running = True
        
        # Start the processing thread
        processing_thread = threading.Thread(target=self.process_audio_queue)
        processing_thread.daemon = True
        processing_thread.start()
        
        print("Starting real-time transcription with Hugging Face Whisper. Press Ctrl+C to stop.")
        print("Speak into your microphone...")
        print("(If you don't see any output, try speaking louder or adjusting your microphone)")
        
        try:
            # Start capturing audio
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self.audio_callback,
                blocksize=CHUNK_SIZE
            ):
                # Keep the main thread alive
                while self.is_running:
                    threading.Event().wait(0.1)
        except KeyboardInterrupt:
            print("\nStopping transcription...")
        finally:
            self.is_running = False
            print("Transcription stopped.")

if __name__ == "__main__":
    transcriber = HFAudioTranscriber()
    transcriber.start()
