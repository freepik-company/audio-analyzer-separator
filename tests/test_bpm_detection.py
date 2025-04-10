import pytest
from pathlib import Path
import numpy as np
from predict import Predictor
import librosa

@pytest.fixture
def test_audio_path():
    return Path("test.mp3")

@pytest.fixture
def predictor():
    return Predictor()

def test_bpm_detection_accuracy(test_audio_path, predictor):
    """Test that BPM detection is accurate within acceptable range."""
    result = predictor.predict(
        music_input=test_audio_path,
        visualize=False,
        sonify=False,
        model="harmonix-all",
        include_activations=False,
        include_embeddings=False
    )
    
    assert result.bpm is not None, "BPM should not be None"
    assert isinstance(result.bpm, (float, int)), "BPM should be a number"
    assert 40 <= result.bpm <= 200, "BPM should be within reasonable range (40-200)"
    
    # Test if the BPM is close to the expected value (110 BPM)
    assert abs(result.bpm - 110) <= 1.0, f"BPM {result.bpm} should be close to 110"

def test_bpm_consistency(test_audio_path, predictor):
    """Test that multiple runs produce consistent results."""
    results = []
    for _ in range(3):
        result = predictor.predict(
            music_input=test_audio_path,
            visualize=False,
            sonify=False,
            model="harmonix-all",
            include_activations=False,
            include_embeddings=False
        )
        results.append(result.bpm)
    
    # Check that all results are within 0.5 BPM of each other
    assert max(results) - min(results) <= 0.5, "BPM detection should be consistent across runs"

def test_audio_loading(test_audio_path):
    """Test that audio file can be loaded correctly."""
    y, sr = librosa.load(str(test_audio_path), sr=44100)
    assert sr == 44100, "Sample rate should be 44100 Hz"
    assert len(y) > 0, "Audio data should not be empty"
    assert isinstance(y, np.ndarray), "Audio data should be a numpy array"

def test_invalid_audio_file():
    """Test handling of invalid audio file."""
    predictor = Predictor()
    with pytest.raises(Exception):
        predictor.predict(
            music_input=Path("nonexistent.mp3"),
            visualize=False,
            sonify=False,
            model="harmonix-all",
            include_activations=False,
            include_embeddings=False
        ) 