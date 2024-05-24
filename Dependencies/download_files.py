import os
import gdown

def download_dependencies():
    # URLs of the files to be downloaded
    urls = {
        "file1": "https://drive.google.com/uc?id=1EyW6lGwZbhuB4uTRRoDrphpT6WR5oPkQ",
        "file2": "https://drive.google.com/uc?id=1UCXvz-2NGjjDJhmXnPRsPtoHAFs3YaxB"
    }

    # Destination directories
    dest_dirs = {
        "file1": "deep_sort_pytorch/deep_sort/deep/checkpoint/",
        "file2": "Dependencies/"  # Different directory for file2
    }

    # File names to save
    file_names = {
        "file1": "ckpt.t7",
        "file2": "Test.mp4"
    }

    # Ensure the destination directories exist
    for dir_path in dest_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Check if files already exist, and download if necessary
    for filename, url in urls.items():
        # Set destination directory
        dest_dir = dest_dirs[filename]

        # Set file name
        file_name = file_names[filename]

        # Set file path
        file_path = os.path.join(dest_dir, file_name)

        # Check if file already exists
        if os.path.exists(file_path):
            print(f"File {file_name} already exists. Skipping download.")
        else:
            # Download the file
            gdown.download(url, file_path, quiet=False)
            print(f"File downloaded and saved to {file_path}")