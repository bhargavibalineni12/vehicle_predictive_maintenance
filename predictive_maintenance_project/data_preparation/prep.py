import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ------------------------------
# Configurations
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_PATH = "hf://datasets/Bhargavi329/vehicle-predictive-maintenance/engine_data.csv"
REPO_ID = "Bhargavi329/vehicle-predictive-maintenance"
OUTPUT_DIR = "processed_data"

# Initialize Hugging Face API
api = HfApi(token=HF_TOKEN)


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from Hugging Face hub."""
    try:
        df = pd.read_csv(path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess predictive maintenance dataset."""
    # Drop any unnecessary columns (if exist)
    drop_cols = [col for col in df.columns if "Unnamed" in col or col.strip() == ""]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped unnecessary columns: {drop_cols}")

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())  # numeric → median
        else:
            df[col] = df[col].fillna(df[col].mode()[0])  # categorical → mode

    print("Preprocessing complete (missing values handled).")
    return df


def split_and_save(df: pd.DataFrame, target_col: str):
    """Split dataset and save train/test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

    print("Train/test datasets saved locally.")
    return [
        f"{OUTPUT_DIR}/X_train.csv",
        f"{OUTPUT_DIR}/X_test.csv",
        f"{OUTPUT_DIR}/y_train.csv",
        f"{OUTPUT_DIR}/y_test.csv"
    ]


def upload_files(files: list, repo_id: str, repo_type: str = "dataset"):
    """Upload train/test files to Hugging Face dataset repo."""
    for file_path in files:
        if os.path.exists(file_path):
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=repo_id,
                    repo_type=repo_type
                )
                print(f"Uploaded: {file_path}")
            except Exception as e:
                print(f"Failed to upload {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")


def main():
    df = load_dataset(DATASET_PATH)
    df = preprocess_data(df)
    files = split_and_save(df, target_col="Engine Condition")
    upload_files(files, REPO_ID)
    print(" Predictive Maintenance data preparation and upload completed successfully!")


if __name__ == "__main__":
    main()
