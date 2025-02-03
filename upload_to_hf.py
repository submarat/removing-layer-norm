from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi, login
from huggingface_hub.utils import RepositoryNotFoundError

# 1. First, login to Hugging Face
login()  # You'll need your API token from huggingface.co/settings/tokens

# 2. Set the model namespace
model_name = "submarat/model-without-ln"

# 3. Create repository and upload
api = HfApi()

try:
    repo_info = api.repo_info(model_name, repo_type="model")
    print(f"Repository {model_name} exists. Preparing to update...")
except RepositoryNotFoundError:
    print(f"Repository {model_name} not found. Creating new repository...")
    api.create_repo(model_name, repo_type="model", private=False)
    
api.upload_folder(
    folder_path="./model-without-ln",
    repo_id=model_name,
    repo_type="model"
)
