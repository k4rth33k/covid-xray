import pandas as pd
from urllib.request import urlopen
from azure.storage.file import FileService
from azure.storage.file import ContentSettings

file_service = FileService(account_name='covidmodels', account_key='')

def upload_to_azure(url, filename):
	print(f'Uploading {filename}')

	with urlopen(url+filename) as fp:
		byte = fp.read()

	file_service.create_file_from_bytes(
	    'covid-share',
	    'data', 
	    filename,
	    byte,
	    content_settings=ContentSettings(content_type='image/jpeg'))

def get_files_from_blob():
	generator = file_service.list_directories_and_files('covid-share/data')
	
	return [file_or_dir.name for file_or_dir in generator]

metadata = pd.read_csv('https://github.com/ieee8023/covid-chestxray-dataset/raw/master/metadata.csv')


def main():
	url = 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/'
	dataset = get_files_from_blob()

	for index, row in metadata.iterrows():
	    if row['finding'] == 'COVID-19':
	    	file = row['filename']
	    	if file not in dataset:
	    		upload_to_azure(url, file)
	    	# break


if __name__ == '__main__':
	main()



























