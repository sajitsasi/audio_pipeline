import os
import re
import json
import subprocess
import logging
import signal
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch
import librosa
import tempfile
import time
from dotenv import load_dotenv
import termios
import sys
import platform
import multiprocessing
from tqdm import tqdm

if platform.system() == "Darwin":
    multiprocessing.set_start_method("fork", force=True)


# Global flag for graceful shutdown
shutdown_flag = False
executor = None


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global shutdown_flag
    if shutdown_flag:  # If ctrl-c is pressed twice, exit immediately
        logger.warning("Forced shutdown requested. Exiting immediately.")
        sys.exit(1)
    
    logger.warning("Interrupt received. Shutting down gracefully... (Press Ctrl+C again to force quit)")
    shutdown_flag = True
    
    if executor:
        logger.info("Cancelling pending tasks...")
        executor.shutdown(wait=False, cancel_futures=True)

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = os.getcwd() + "/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"audio_pipeline_{timestamp}.log")
    
    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s]: %(message)s',
        handlers=[
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger()


logger = setup_logging()
# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Load models globally for efficiency
try:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
    transcript_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    sys.exit(1)

def check_terminal_state():
    try:
        if sys.stdin.isatty():
            attrs = termios.tcgetattr(sys.stdin.fileno())
            logger.debug(f"Terminal settings: {attrs}")
    except Exception as e:
        logger.debug(f"Unable to check terminal state: {e}")

def reset_terminal():
    try:
        if sys.stdin.isatty():
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, termios.tcgetattr(sys.stdin))
    except Exception as e:
        logger.debug(f"Unable to reset terminal: {e}")

def has_audio_content(audio_array, threshold_db=-60):
    """
    Check if the audio file contains actual audio content.
    """
    try:
        # Convert to dB scale
        db = librosa.amplitude_to_db(np.abs(audio_array), ref=np.max)
        
        # Calculate the percentage of samples above threshold
        non_silence = np.mean(db > threshold_db)
        
        # Log the audio statistics
        logger.debug(f"Audio stats - Mean dB: {np.mean(db):.2f}, "
                    f"Max dB: {np.max(db):.2f}, "
                    f"Non-silence ratio: {non_silence:.2%}")
        
        # Consider audio valid if at least 1% of samples are above threshold
        return non_silence > 0.01
        
    except Exception as e:
        logger.error(f"Error checking audio content: {e}")
        return False

def is_valid_transcription(transcription):
    """
    Check if a transcription contains any words.
    Returns tuple of (has_words, cleaned_text)
    """
    if not transcription:
        return False, ""
    
    # Remove punctuation and whitespace
    cleaned_text = re.sub(r'[^\w\s]', '', transcription).strip()
    
    # Check if there are any words
    has_words = bool(cleaned_text and cleaned_text.split())
    
    return has_words, cleaned_text

def extract_metadata(filename):

    metadata = {
        "src_number": "0",
        "dst_number": "0",
        "call_time": "",
        "is_us_number": False,
        "file_name": os.path.basename(filename)
    }
    pattern = r"(?P<src_number>\+\d+)_(?P<dst_number>\+\d+)_(?P<epoch>\d+)\.wav"
    match = re.match(pattern, filename)
    if match:
        metadata["src_number"]   = match.group("src_number")
        metadata["dst_number"]   = match.group("dst_number")
        metadata["call_time"]    = datetime.fromtimestamp(int(match.group("epoch")) / 1e6).isoformat()
        metadata["is_us_number"] = metadata["src_number"].startswith("+1") and len(metadata["src_number"]) == 12
    else:
        metadata["call_time"] = datetime.fromtimestamp(os.path.getctime(os.path.basename(filename))).isoformat()
    return metadata

def preprocess_audio(file_path):
    try:
        # Create unique temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            output_path = temp_file.name

        # Re-encode audio file to temp file
        command = [
            "ffmpeg", "-y", "-i", file_path,
            "-ar", "16000", "-ac", "1", output_path
        ]
        check_terminal_state()
        start_time = time.time()
        with open(os.devnull, 'wb') as devnull:
            subprocess.run(command, check=True, stdout=devnull, stderr=subprocess.STDOUT)
        elapsed_time = time.time() - start_time
        check_terminal_state()
        logger.info(f"Preprocessing audio completed in {elapsed_time:.2f} seconds")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to preprocess {file_path}: {e}")
        return None

def transcribe_and_embed_audio(audio, sampling_rate=16000):
    try:
        start_time = time.time()
        inputs = whisper_processor(audio, sampling_rate=sampling_rate, return_tensors="pt", language="en").input_features
        with torch.no_grad():
            predicted_ids = whisper_model.generate(inputs)
            transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        elapsed_time = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
        start_time = time.time()
        embedding = transcript_model.encode(transcription)
        elapsed_time = time.time() - start_time
        logger.info(f"Transcription embedding completed in {elapsed_time:.2f} seconds")
        return transcription, embedding
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None

def generate_audio_embeddings(audio, sampling_rate=16000):
    try:
        start_time = time.time()
        input_values = feature_extractor(audio, return_tensors="pt", sampling_rate=sampling_rate).input_values
        with torch.no_grad():
            embeddings = wav2vec_model(input_values).last_hidden_state.mean(dim=1).squeeze().numpy()

        elapsed_time = time.time() - start_time
        logger.info(f"Audio embedding completed in {elapsed_time:.2f} seconds")
        return embeddings.tolist()  # Convert to list for JSON compatibility
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None

def process_audio_file(file_path):
    try:
        filename = os.path.basename(file_path)
        metadata = extract_metadata(filename)
        if not metadata:
            return None

        # Preprocess audio
        preprocessed_path = preprocess_audio(file_path)
        if not preprocessed_path:
            return None

        try:
            # Load preprocessed audio
            audio, sr = librosa.load(preprocessed_path, sr=16000)
            
            # Check for audio content
            if not has_audio_content(audio):
                logger.warning(f"Skipping {file_path}: No significant audio content detected")
                return None
                
        finally:
            # Ensure temporary file is removed even if loading fails
            if os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)

        # Generate embeddings first
        audio_embedding = generate_audio_embeddings(audio, sampling_rate=sr)
        if not audio_embedding:
            logger.warning(f"Skipping {file_path}: Failed to generate embeddings")
            return None

        # Try to generate transcription
        transcription, embedding = transcribe_and_embed_audio(audio, sampling_rate=sr)
        has_words = False
        word_count = 0
        
        if transcription:
            has_words, cleaned_text = is_valid_transcription(transcription)
            if has_words:
                word_count = len(cleaned_text.split())
                logger.info(f"Valid transcription with {word_count} words")
            else:
                logger.info("No words found in transcription")
        else:
            logger.info("Failed to generate transcription")

        # Update metadata with all available information
        metadata.update({
            "transcription": transcription if transcription else "",
            "transcription_embedding": embedding.tolist() if embedding is not None and embedding.size > 0 else [],
            "audio_embedding": audio_embedding,
            "empty_audio": not has_words,
            "word_count": word_count,
            "processing_status": "success"
        })
        
        logger.info(f"Processed {file_path} successfully - Empty audio: {not has_words}, Word count: {word_count}")
        return metadata
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def process_directory(directory, max_workers=4):
    global executor
    
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".wav")]
    total_files = len(all_files)
    processed_files = 0
    failed_files = 0
    results = []
    
    logger.info(f"Starting batch processing of {total_files} files with {max_workers} workers")
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as exec:
            executor = exec
            futures = {executor.submit(process_audio_file, path): path for path in all_files}
            
            for future in tqdm(as_completed(futures), total=total_files, desc="Processing files"):
                if shutdown_flag:
                    logger.info("Graceful shutdown initiated. Stopping processing...")
                    break
                    
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        processed_files += 1
                    else:
                        failed_files += 1
                    
                    total_processed = processed_files + failed_files
                    logger.info(f"Progress: {total_processed}/{total_files} files "
                              f"(Success: {processed_files}, Failed: {failed_files})")
                    
                except Exception as e:
                    failed_files += 1
                    logger.error(f"Error processing {file_path}: {e}")
    
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
    finally:
        executor = None
    
    status = "completed" if not shutdown_flag else "interrupted"
    logger.info(f"Batch processing {status}. "
               f"Total files: {total_files}, "
               f"Succeeded: {processed_files}, "
               f"Failed: {failed_files}")
    
    return results

def cleanup():
    """Perform cleanup operations"""
    logger.info("Performing cleanup...")
    reset_terminal()
    logger.info("Cleanup completed")

if __name__ == "__main__":
    try:
        load_dotenv()
        input_directory = os.getenv('WAV_FILE_PATH')
        output_file = os.getenv('PROCESSED_AUDIO_FILE')

        logger.info(f"Starting processing with input directory: {input_directory}")
        results = process_directory(input_directory)

        if results:  # Only save results if we have any
            summary = {
                "total_files_processed": len(results),
                "processing_timestamp": datetime.now().isoformat(),
                "processing_status": "interrupted" if shutdown_flag else "completed",
                "results": results
            }

            # Save results to JSON file
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=4)

            logger.info(f"Processing complete. Results saved to {output_file}")
        else:
            logger.warning("No results to save")
            
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
    finally:
        cleanup()
