
# Audio Processing and Clustering Pipeline

This repository contains Python scripts for processing audio files, extracting features, generating embeddings, and clustering them. The project uses machine learning models for transcription and clustering.

## Prerequisites

- Python 3.12 or later
- pip (Python package manager)
- [ffmpeg](https://ffmpeg.org/download.html)

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://www.github.com/sajitsasi/audio_pipeline
   cd audio_pipeline
   ```

2. Create a virtual environment to manage dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required Python modules from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Setting Up Environment Variables

The scripts require environment variables to run properly. Use the provided `sample_env` file to set up your environment:

1. Copy the `sample_env` file to create a `.env` file:
   ```bash
   cp sample_env .env
   ```

2. Edit the `.env` file to configure any necessary values.

## Scripts Overview

### 1. `audio_pipeline.py`
This script handles the audio processing pipeline:
- Loads audio files.
- Extracts features using models from the `transformers` library.
- Transcribes audio data.

#### How to Run:
```bash
python audio_pipeline.py
```

### 2. `cluster_audio.py`
This script performs clustering on audio embeddings:
- Clusters audio features and embeddings using `K-Means`.
- Visualizes results with matplotlib.

#### How to Run:
```bash
python cluster_audio.py
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Feel free to open issues or submit pull requests for enhancements or bug fixes.
