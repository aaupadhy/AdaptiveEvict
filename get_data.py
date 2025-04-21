import os
import logging
import random
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(data_path='data', data_file='data.txt', num_docs=None, random_seed=42):
    if not os.path.isfile(os.path.join(data_path, data_file)):
        logger.info("Preparing dataset.")
        download_data(data_path, data_file, num_docs, random_seed)
    logger.info(f"Using data from {os.path.join(data_path, data_file)}")

def download_data(data_path='data', data_file='data.txt', num_docs=None, random_seed=42):
    os.makedirs(data_path, exist_ok=True)
    output_file_path = os.path.join(data_path, data_file)
    
    logger.info("Loading dataset from Hugging Face...")
    try:
        set_verbosity_error()
        dataset = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train", streaming=True, trust_remote_code=True)
        
        if num_docs is not None:
            logger.info(f"Selecting {num_docs} documents from the dataset")
            random.seed(random_seed)
            dataset = dataset.take(num_docs)
            logger.info(f"Selected {num_docs} documents")
        else:
            logger.info("Using all available documents in the dataset")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise
    
    logger.info("Processing dataset...")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as text_writer:
            for item in dataset:
                text_writer.write("<sot>\n")
                text_writer.write(item['text'] + "\n")
                text_writer.write("<eot>\n")
    except Exception as e:
        logger.error(f"Failed to write dataset: {str(e)}")
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        raise
    
    logger.info("Data preparation complete.")
