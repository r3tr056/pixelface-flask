

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

class GoogleDriveFileClient:
	def __init__(self):
		self.gauth = GoogleAuth()
		self.gauth.LocalWebserverAuth()
		self.drive = GoogleDrive(self.gauth)

	def get(self, file_id, destination_path):
		""" Download a file from Google Drive """
		file = self.drive.CreateFile({'id': file_id})
		file.GetContentFile(destination_path)
		return destination_path

		