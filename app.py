from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

label2id = {"positive" : 1, "negative" : 0, "neutral" : 2}
id2label = {0 : "negative", 1 : "positive", 2 : "neutral"}
model_path = "results/checkpoint-2560"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, label2id=label2id, id2label=id2label)
model.eval()

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: InputText):
    inputs = tokenizer(input_data.text, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        predicted_label = id2label[int(predicted_class)]
    return {
        "label" : predicted_label,
        "confidence" : probs[0][predicted_class].item()
    }