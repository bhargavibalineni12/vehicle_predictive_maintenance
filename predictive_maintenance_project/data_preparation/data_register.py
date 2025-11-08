from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "Bhargavi329/vehicle-predictive-maintenance"
repo_type = "dataset"

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set.")

api = HfApi(token=hf_token)

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Step 2: Upload folder
try:
    api.upload_folder(
        folder_path="predictive_maintenance_project/data",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Data uploaded successfully.")
except Exception as e:
    print(f"Error uploading data: {e}")
