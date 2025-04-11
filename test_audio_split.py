from pathlib import Path
import librosa
import numpy as np
from predict import Predictor
import os
import shutil

def test_audio_split():
    """Test the audio splitting functionality."""
    # Initialize predictor
    predictor = Predictor()
    
    # Test file path
    test_file = Path("test.test_mp3")
    
    # Run prediction with audio separation
    result = predictor.predict(
        music_input=test_file,
        visualize=True,
        sonify=True,
        model="harmonix-all",
        include_activations=True,
        include_embeddings=True
    )
    
    # Expected output stems
    expected_stems = ['vocals', 'drums', 'bass', 'other']
    
    # Check if output directory exists
    output_dir = Path('output')
    assert output_dir.exists(), "Output directory should be created"
    
    # Check if all stems were created
    for stem in expected_stems:
        stem_path = output_dir / f"{test_file.stem}_{stem}.wav"
        print(f"Checking {stem_path}...")
        
        # Check if stem file exists
        assert stem_path.exists(), f"Stem file {stem} should exist"
        
        # Load and validate stem audio
        y, sr = librosa.load(str(stem_path))
        
        # Basic audio quality checks
        assert len(y) > 0, f"Stem {stem} should not be empty"
        assert sr == 44100, f"Sample rate should be 44100 Hz for {stem}"
        assert not np.all(y == 0), f"Stem {stem} should not be silent"
        assert not np.any(np.isnan(y)), f"Stem {stem} should not contain NaN values"
        assert not np.any(np.isinf(y)), f"Stem {stem} should not contain Inf values"
        
        # Check audio properties
        rms = np.sqrt(np.mean(y**2))
        print(f"{stem} RMS level: {rms:.4f}")
        assert rms > 0.001, f"Stem {stem} RMS level too low"
        
        # Check peak levels
        peak = np.max(np.abs(y))
        print(f"{stem} Peak level: {peak:.4f}")
        assert peak < 1.0, f"Stem {stem} should not clip"
        
    print("\nAll stems passed quality checks!")

if __name__ == "__main__":
    try:
        test_audio_split()
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Error during testing: {e}")
    else:
        print("\nAll tests passed successfully!") 