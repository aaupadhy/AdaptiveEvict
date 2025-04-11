import os
import requests
import zstandard as zstd
import json

# Base URL for the dataset files
base_url = "https://huggingface.co/datasets/allenai/peS2o/resolve/main/data/v3/"

# List of file names to download
file_names = [
    "train-0000-of-0136.zst",
    "train-0001-of-0136.zst",
    "train-0002-of-0136.zst",
    "train-0003-of-0136.zst",
    "train-0004-of-0136.zst",
    "train-0005-of-0136.zst",
    "train-0006-of-0136.zst"
    # Add additional files as needed
]

def prepare_data(data_path='data', data_file='data.txt'):
    '''
    Function to check if the data file already exists; otherwise, call "download_data" function for downloading and preprocessing the data.
    '''
    if not os.path.isfile(os.path.join(data_path, data_file)):
        print("Preparing dataset.")
        download_data(data_path, data_file)
    print(f"Using data from {os.path.join(data_path, data_file)}")

def download_data(data_path='data', data_file='final_data.txt'):
    '''
    Function to download and preprocess the dataset.
    '''

    # Create data directory
    os.makedirs(data_path, exist_ok=True)

    # Iterate over each file to download and process
    for file_name in file_names:
        file_url = base_url + file_name
        local_file_path = os.path.join(data_path, file_name)
        print(local_file_path)
        # Download file if not already present
        if not os.path.isfile(local_file_path):
            print(f"Downloading {file_name}...")
            response = requests.get(file_url, stream=True)
            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"Using existing file: {local_file_path}")

        # Decompress and process the .zst file
        print(f"Processing {file_name}...")
        with open(local_file_path, 'rb') as compressed:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed) as reader:
                text_writer = open(os.path.join(data_path, data_file), 'a', encoding='utf-8')
                while True:
                    chunk = reader.read()  # Read in 16KB chunks
                    if not chunk:
                        break
                    # Assuming each line in the decompressed file is a JSON object
                    for line in chunk.decode('utf-8').splitlines():
                        record = json.loads(line)
                        text_writer.write("<sot>\n")  # Start of text indicator
                        text_writer.write(record.get('text', '') + "\n")
                        text_writer.write("<eot>\n")  # End of text indicator
                text_writer.close()
        print(f"Finished processing {file_name}.")

    print("Data preparation complete.")
