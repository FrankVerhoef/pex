# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")

# model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-xlarge-mnli")

from huggingface_hub import hf_hub_download
r = hf_hub_download(repo_id="microsoft/deberta-xlarge-mnli", filename="pytorch_model.bin")
print("result:", r)