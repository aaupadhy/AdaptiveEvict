import os
import zipfile
import requests
import unicodedata

# Dataset location and attributes
dataset_url = "https://www.kaggle.com/api/v1/datasets/download/ishikajohari/taylor-swift-all-lyrics-30-albums"
download_file_name = "taylor_swift_lyrics.zip"
extract_folder_name = "taylor_swift_lyrics"

def prepare_data(data_path='./data', data_file='data.txt'):
	'''
	function to check if the data file already exits else call "download data" function for downloading and preprocessing the data.
	'''
	if not os.path.isfile(os.path.join(data_path, data_file)):
		print("Preparing dataset.")
		download_data(data_path, data_file)
	print(f"Using data from {os.path.join(data_path, data_file)}")

def download_data(data_path='./data', data_file='final_data.csv'):
	'''
	function to download and preprocess the dataset.
	'''

	# Create data directory
	os.makedirs(data_path,  exist_ok=True)					

	# Download raw file if it not found.
	if not os.path.isfile(os.path.join(data_path, download_file_name)):
		print("Downloading raw data file.")
		response = requests.get(dataset_url)
		with open(os.path.join(data_path, download_file_name), 'wb') as f:
			f.write(response.content)
	else:
		print(f"Using downloaded file. {os.path.join(data_path, download_file_name)}")


	# unzip content from the downloaded file.
	if (not os.path.isdir(os.path.join(data_path, extract_folder_name))):
		print("Extracting data from zip file.")
		with zipfile.ZipFile(os.path.join(data_path, download_file_name), "r") as zipf:
		    zipf.extractall(os.path.join(data_path, extract_folder_name))
	

	# Find all lyrics file together. 
	files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(data_path, extract_folder_name)) for f in filenames if os.path.splitext(f)[1] == '.txt']
	files.sort()

	# Combine all the lyrics and write to the output file.
	with open(os.path.join(data_path, data_file), "w", encoding='utf-8') as out_file:
		for file in files:
			with open(file, encoding='utf-8') as f:
				lyrics = f.readlines()[1:-1]																			# The first and the last lines are removed due to noise.
				out_file.write("<sol>\n")																				# Start of new lyrics indicator
				for line in lyrics:
					line_ = line.replace("See Taylor Swift LiveGet tickets as low as $60", "")							# Found this Ad in the lyrics
					out_file.write(unicodedata.normalize('NFKD', line_).encode('ascii', 'ignore').decode('utf-8') )		# Unicodes conversion
				out_file.write("\n<eol>\n")																				# End of the lyrics indicator
	print("Data is ready.")
