import os
from huggingface_hub import HfApi, login

# -------------------------------
# Load Hugging Face token from environment
# -------------------------------
hf_token = os.getenv("HF_TOKEN")  # set this in Colab using: %env HF_TOKEN=your_token
if not hf_token:
    raise ValueError("HF_TOKEN not found. Please set it in your environment using '%env HF_TOKEN=...' in Colab.")

# Login to Hugging Face
login(token=hf_token)

# Initialize API
api = HfApi()

# -------------------------------
# Upload Deployment Files to Hugging Face Space
# -------------------------------
repo_id = "Bhargavi329/vehicle-predictive-maintenance"  # Your Space repo name
folder_path = "predictive_maintenance_project/deployment"          # Local deployment folder

api.upload_folder(
    folder_path=folder_path,   # folder containing app.py, requirements.txt, Dockerfile
    repo_id=repo_id,
    repo_type="space",         # uploading to Space
    path_in_repo=""            # keep files at the root of the repo
)

print("Deployment files uploaded successfully!")
