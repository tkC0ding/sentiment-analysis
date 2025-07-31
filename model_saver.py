from transformers import AutoModelForSequenceClassification, AutoTokenizer

label2id = {"positive" : 1, "negative" : 0, "neutral" : 2}
id2label = {0 : "negative", 1 : "positive", 2 : "neutral"}
model = AutoModelForSequenceClassification.from_pretrained("results/checkpoint-2560", num_labels=3, label2id=label2id, id2label=id2label)
tokenizer = AutoTokenizer.from_pretrained("results/checkpoint-2560")

model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")