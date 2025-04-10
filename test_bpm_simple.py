import librosa
import numpy as np

def test_bpm_detection():
    # Load the audio file
    y, sr = librosa.load('test.mp3')
    
    # Detect the BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Convert tempo to float (properly handling numpy array)
    tempo = tempo.item() if isinstance(tempo, np.ndarray) else float(tempo)
    
    print(f"Detected BPM: {tempo:.1f}")

if __name__ == "__main__":
    test_bpm_detection() 