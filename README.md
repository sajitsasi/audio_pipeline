
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

3. Copy the `sample_cluster_label_categories.json` file to create a `cluster_label_categories.json` file:
   ```bash
   cp sample_cluster_label_categories.json cluster_label_categories.json
   ```

4. Edit the `sample_cluster_label_categories.json` file to configure any necessary values.

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

### 3. `llm_label.py`
This script labels the calls by sending the transcript of the call to OpenAI
- Loads clustered data from the [previous step](#2-cluster_audiopy). The name of the file must be defined in the `.env` file as `CLUSTERED_AUDIO_FILE`
- Loads categories defined in the `.env` file. This is loaded from `CLUSTER_LABEL_DEFINITION_FILE`. 
- Writes the labelled data to `LABELLED_AUDIO_FILE` defined in `.env`

#### How to Run:
```bash
python llm_label.py
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Feel free to open issues or submit pull requests for enhancements or bug fixes.
