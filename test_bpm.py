from pathlib import Path
from predict import Predictor
import json

def main():
    # Initialize predictor
    predictor = Predictor()
    
    # Test file path
    test_file = Path("test.mp3")
    
    # Run prediction
    result = predictor.predict(
        music_input=test_file,
        visualize=True,
        sonify=True,
        model="harmonix-all",
        include_activations=True,
        include_embeddings=True
    )
    
    # Print detailed results
    print("\nBPM Detection Results:")
    print("---------------------")
    print(f"Detected BPM: {result.bpm}")
    print("\nRaw tempo estimates:")
    print(json.dumps(result.tempo_estimates, indent=2))
    print("\nTempo confidence scores:")
    print(json.dumps(result.tempo_scores, indent=2))

if __name__ == "__main__":
    main() 