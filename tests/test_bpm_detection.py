import pytest
from pathlib import Path
import numpy as np
import librosa
import os
import shutil

@pytest.fixture
def test_audio_path(tmp_path):
    """Create a temporary copy of the test audio file."""
    # Original file path
    original_path = Path("test.test_mp3")
    
    # Create a temporary copy
    temp_path = tmp_path / "test.mp3"
    shutil.copy(original_path, temp_path)
    
    return temp_path

def test_bpm_detection_accuracy(test_audio_path):
    """Test that BPM detection is accurate within acceptable range."""
    # Load audio with higher sample rate
    y, sr = librosa.load(str(test_audio_path), sr=44100)
    
    # Get multiple tempo estimates using different methods
    tempos = []
    
    # Method 1: Basic onset strength and beat tracking
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo1, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempos.append(float(tempo1))
    
    # Method 2: Multi-band onset strength
    onset_env_multi = librosa.onset.onset_strength_multi(y=y, sr=sr, channels=[0, 1, 2])
    onset_env_mean = np.mean(onset_env_multi, axis=0)
    tempo2, _ = librosa.beat.beat_track(onset_envelope=onset_env_mean, sr=sr)
    tempos.append(float(tempo2))
    
    # Method 3: Tempogram-based estimation
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tempo3 = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
    tempos.append(float(tempo3))
    
    # Convert all tempos to float and round to nearest 0.5
    tempos = [round(t * 2) / 2 for t in tempos]
    
    # Calculate tempo confidence scores
    tempo_scores = {}
    for tempo in tempos:
        # Score based on how many times this tempo appears
        count = tempos.count(tempo)
        # Additional score for tempos that are close to other estimates
        close_tempos = sum(1 for t in tempos if abs(t - tempo) <= 1.0)
        tempo_scores[tempo] = count + (close_tempos * 0.5)
    
    # Select the tempo with the highest confidence score
    final_tempo = max(tempo_scores.items(), key=lambda x: x[1])[0]
    
    # Test assertions
    assert final_tempo is not None, "BPM should not be None"
    assert isinstance(final_tempo, (float, int)), "BPM should be a number"
    assert 40 <= final_tempo <= 200, "BPM should be within reasonable range (40-200)"
    assert abs(final_tempo - 110) <= 1.0, f"BPM {final_tempo} should be close to 110"

def test_audio_loading(test_audio_path):
    """Test that audio file can be loaded correctly."""
    y, sr = librosa.load(str(test_audio_path), sr=44100)
    assert sr == 44100, "Sample rate should be 44100 Hz"
    assert len(y) > 0, "Audio data should not be empty"
    assert isinstance(y, np.ndarray), "Audio data should be a numpy array"

def test_invalid_audio_file(tmp_path):
    """Test handling of invalid audio file."""
    invalid_path = tmp_path / "invalid.mp3"
    with pytest.raises(Exception):
        librosa.load(str(invalid_path)) 