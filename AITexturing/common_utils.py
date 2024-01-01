import os
import sys

def get_persistent_directory(folder_name):
    if sys.platform == "win32":
        folder = os.path.join(os.path.expanduser("~"), "AppData", "Local", folder_name)
    else:
        folder = os.path.join(os.path.expanduser("~"), "." + folder_name)
    
    os.makedirs(folder, exist_ok=True)
    return folder