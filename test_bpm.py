from predict import Predictor
from pathlib import Path

def test_bpm_detection():
    # Initialize the predictor
    predictor = Predictor()
    predictor.setup()
    
    # Test with test.mp3 from root folder
    test_audio = Path("test.mp3")
    
    # Run prediction
    result = predictor.predict(
        music_input=test_audio,
        visualize=False,
        sonify=False,
        model="harmonix-all",
        include_activations=False,
        include_embeddings=False,
        audioSeparator=False
    )
    
    # Print the BPM result
    print(f"Detected BPM: {result.bpm}")

if __name__ == "__main__":
    test_bpm_detection() 