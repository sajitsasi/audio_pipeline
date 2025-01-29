
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

4. Edit the `cluster_label_categories.json` file to configure any necessary values.

## Scripts Overview

### 1. `audio_pipeline.py`

This script handles the audio processing pipeline:

- Loads audio files.
- Extracts features using models from the `transformers` library.
- Transcribes audio data.

This script also assumes that the files are named in the format `<src_number>_<dst_number>_<epoch>.wav`. This generates metadata that has the following information:

- `src_number`  : The src_number from the filename
- `dst_number`  : The dst_number from the filename
- `call_time`   : The time in ISO format from the `epoch` in filename
- `is_us_number`: Boolean on whether this is a 12-digit number or not

If the filename format doesn't meet the above description, the fileds will be filled as:

- `src_number`  : 0
- `dst_number`  : 0
- `call_time`   : The time in ISO format from the when the filename was created (ctime)
- `is_us_number`: False

#### How to Run:
```bash
python audio_pipeline.py
```

#### Output will be written to `PROCESSED_AUDIO_FILE` defined in your `.env`

### 2. `cluster_audio.py`

This script performs K-Means clustering on audio embeddings:

- Loads processed audio data from the [previous step](#1-audio_pipelinepy). The name of the file must be defined in the `.env` file as `PROCESSED_AUDIO_FILE`
- Calculates and plots elbow and silhouette scores on both `audio_embedding` and `transcription_embedding` and gives the user the ability to choose the right number of clusters
- Clusters audio features and embeddings using `K-Means`.
- Visualizes clustered results with matplotlib.

#### How to Run:
```bash
python cluster_audio.py
```

#### Output will be written to `CLUSTERED_AUDIO_FILE` defined in your `.env`

### 3. `llm_label.py`

This script labels the calls by sending the transcript of the call to OpenAI

- Loads clustered data from the [previous step](#2-cluster_audiopy). The name of the file must be defined in the `.env` file as `CLUSTERED_AUDIO_FILE`
- Loads categories defined in the `.env` file. This is loaded from `CLUSTER_LABEL_DEFINITION_FILE`. 
- Writes the labelled data to `LABELLED_AUDIO_FILE` defined in `.env`

#### How to Run:
```bash
python llm_label.py
```

#### Output will be written to `LABELLED_AUDIO_FILE` defined in your `.env`

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Feel free to open issues or submit pull requests for enhancements or bug fixes.
