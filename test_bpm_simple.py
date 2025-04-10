import librosa
import numpy as np
from pathlib import Path

def detect_bpm(audio_path):
    # Load audio with higher sample rate
    y, sr = librosa.load(str(audio_path), sr=44100)
    
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
    
    return {
        "final_bpm": final_tempo,
        "all_estimates": tempos,
        "confidence_scores": tempo_scores
    }

def main():
    # Test file path
    test_file = Path("test.mp3")
    
    # Run BPM detection
    results = detect_bpm(test_file)
    
    # Print results
    print("\nBPM Detection Results:")
    print("---------------------")
    print(f"Final BPM: {results['final_bpm']}")
    print("\nAll tempo estimates:")
    print(results['all_estimates'])
    print("\nTempo confidence scores:")
    for tempo, score in results['confidence_scores'].items():
        print(f"{tempo} BPM: {score}")

if __name__ == "__main__":
    main() 