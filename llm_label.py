import json
from openai import OpenAI
import os
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import platform
import multiprocessing
from tqdm import tqdm


if platform.system() == "Darwin":
    multiprocessing.set_start_method("fork", force=True)


load_dotenv()
# Set OpenAI API key and custom LLM proxy base URL
base_url = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

if not base_url:
    client = OpenAI(api_key=api_key)
else:
    client = OpenAI(base_url=base_url, api_key=api_key)


def load_category_mapping(file_path):
    with open(file_path, "r") as f:
        return {item['Cluster Label']: item['Category'] for item in json.load(f)}

category_mapping = load_category_mapping(os.getenv("CATEGORY_MAPPING_FILE", "cluster_label_categories.json"))

def load_clustered_data(file_path):
    """Load clustered audio data from JSON."""
    with open(file_path, "r") as f:
        return json.load(f)

def label_call_with_llm(transcription, metadata, cluster_id):
    labels = list(set(category_mapping.keys()))
    labels_list = "\n- " + "\n- ".join(labels)
    """Use GPT-4 (via proxy) to label the call based on transcription and metadata."""
#If the call doesn't have any words, it should be labeled 'unknown'.
#If the call does not fit any of the labels, it should be labeled as 'unknown'
    messages = [
        {"role": "system", "content": "You are an expert call classifier."},
        {
            "role": "user",
            "content": f"""
Label the following call into one of the labels below. 

Labels are:
{labels_list}

Call Details:
- Transcription: "{transcription}"
- Metadata: {metadata}
- Cluster: {cluster_id}

Provide the label as a single word from the list above
""",
        },
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )
        label = response.choices[0].message.content.strip()
        return label
    except Exception as e:
        print(f"Error labeling call: {e}")
        return "Unclassified"

def label_call(item):
    """Label a single call and return the updated item."""
    transcription = item.get("transcription", "")
    metadata = {
        "source_number": item.get("src_number"),
        "destination_number": item.get("dst_number"),
        "call_time": item.get("call_time"),
    }
    cluster_id = item.get("audio_embedding_cluster", -1)
    label = label_call_with_llm(transcription, metadata, cluster_id)
    item["call_label"] = label
    item["call_category"] = category_mapping.get(label, "unknown")
    return item

def label_calls_multithreaded(data, max_workers=5):
    """Label all calls using multithreading."""
    updated_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(label_call, item) for item in data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Labelling calls"):
            updated_data.append(future.result())
    return updated_data

def save_classified_data(data, output_file):
    """Save classified data to a new JSON file."""
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

# Main classification workflow
if __name__ == "__main__":
    clustered_file_name = os.getenv("CLUSTERED_AUDIO_FILE", "./output/clustered_audio.json")
    classified_file_name = os.getenv("LABELLED_AUDIO_FILE", "./output/labelled_audio.json")
    
    clustered_data_file = os.path.join(working_directory, clustered_file_name)
    output_file = os.path.join(working_directory, classified_file_name)

    print(f"Loading clustered data from {clustered_data_file}...")
    clustered_data = load_clustered_data(clustered_data_file)

    print("Labeling calls with the custom LLM proxy using multithreading...")
    labelled_data = label_calls_multithreaded(clustered_data, max_workers=30)

    print(f"Saving labelled data to {output_file}...")
    save_classified_data(labelled_data, output_file)

    print("Labelling complete!")