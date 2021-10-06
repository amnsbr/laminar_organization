import urllib.request
import shutil
from urllib.parse import urlparse
import os

def download(url, file_name=None):
	"""
	Download the file from `url` and save it locally under `file_name`
	"""
	if not file_name:
		file_name = os.path.basename(urlparse(url).path)
	with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
	    shutil.copyfileobj(response, out_file)
