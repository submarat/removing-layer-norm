from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi, login

# 1. First, login to Hugging Face
login()  # You'll need your API token from huggingface.co/settings/tokens

# 2. Save your model and tokenizer locally
model_name = "submarat/model-without-ln"

# 3. Create repository and upload
api = HfApi()
api.create_repo(model_name)
api.upload_folder(
    folder_path="./model-without-ln",
    repo_id=model_name,
    repo_type="model"
)
