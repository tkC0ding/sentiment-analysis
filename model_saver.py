from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

base_model_name = "distilbert-base-uncased"
lora_checkpoint_path = "results/checkpoint-2560"
label2id = {"positive" : 1, "negative" : 0, "neutral" : 2}
id2label = {0 : "negative", 1 : "positive", 2 : "neutral"}

base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=3, label2id=label2id, id2label=id2label)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)

model = model.merge_and_unload()

output_dir = "final_model"
model.save_pretrained(output_dir)
model.config.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)