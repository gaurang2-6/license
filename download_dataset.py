import socket
import urllib3.util.connection as urllib_connection

# Force IPv4 to prevent IPv6 drops on Windows
def allowed_gai_family():
    return socket.AF_INET

urllib_connection.allowed_gai_family = allowed_gai_family

import os
import kagglehub

print("Downloading dataset with IPv4 forced...")
path = kagglehub.dataset_download("saisirishan/indian-vehicle-dataset")
print("Path to dataset files:", path)

for root, dirs, files in os.walk(path):
    rel = os.path.relpath(root, path)
    print(f"Dir: {rel} - {len(dirs)} subdirs, {len(files)} files")
    if files:
        print("  Sample files:", files[:5])
