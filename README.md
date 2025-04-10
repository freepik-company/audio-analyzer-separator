# Audio Analyzer & BPM Detector

[![Python Tests](https://github.com/yourusername/audio-analyzer-separator/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/audio-analyzer-separator/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/yourusername/audio-analyzer-separator/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/audio-analyzer-separator)

A Python-based audio analysis tool that provides accurate BPM (Beats Per Minute) detection using multiple estimation methods.

## Features

- High-precision BPM detection using multiple analysis methods:
  - Basic onset strength and beat tracking
  - Multi-band onset strength analysis
  - Tempogram-based estimation
- Confidence scoring system for BPM estimates
- Support for various audio formats (mp3, wav, etc.)
- High sample rate processing (44.1kHz) for improved accuracy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-analyzer-separator.git
cd audio-analyzer-separator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic BPM Detection

```python
from predict import Predictor

predictor = Predictor()
result = predictor.predict(
    music_input="path/to/your/audio.mp3",
    model="harmonix-all"
)
print(f"Detected BPM: {result.bpm}")
```

### Running Tests

```bash
python -m pytest
```

## Testing

The project includes automated tests to ensure accurate BPM detection. To run the tests:

```bash
python -m pytest tests/
```

### Test Coverage

- Unit tests for BPM detection methods
- Integration tests with sample audio files
- Confidence score validation

## Development

### Running Tests Locally

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_bpm.py
```

### Code Style

This project follows PEP 8 guidelines. To check code style:

```bash
flake8 .
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

AI Music Structure Analyzer + Stem Splitter using Demucs & Mdx-Net with Python-Audio-Separator | Cog | Replicate

https://github.com/karaokenerds/python-audio-separator/

https://github.com/mir-aidj/all-in-one

How to deploy to Replicate: https://replicate.com/docs/guides/push-a-model


Model Names:
https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json
Model Files: https://huggingface.co/seanghay/uvr_models/tree/main


cog predict -i music_input=@bolsoremix.wav -i audioSeparator=True -i sonify=True -i visualize=True

Original commit: https://github.com/mir-aidj/all-in-one/tree/ac942b8663b69f972407c79c28ff09986fad63c3

Using cog on Windows 11 with WSL 2: https://github.com/replicate/cog/blob/main/docs/wsl2/wsl2.md
