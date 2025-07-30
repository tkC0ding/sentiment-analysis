from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import os

label2id = {"positive" : 1, "negative" : 0, "neutral" : 2}
id2label = {0 : "negative", 1 : "positive", 2 : "neutral"}
data_path = "data/preprocessed_data"
train_path = os.path.join(data_path, 'train_df.csv')
test_path = os.path.join(data_path, 'test_df.csv')
val_path = os.path.join(data_path, 'val_df.csv')

dataset = load_dataset('csv', data_files={'train':train_path, 'test':test_path, 'val':val_path})

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenizer_function(inp):
    return tokenizer(inp["Sentence"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenizer_function, batched=True, batch_size=32)
tokenized_dataset.set_format(type="torch", columns=['labels', 'input_ids', 'attention_mask'])

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    label2id = label2id,
    id2label = id2label
)

lora_config = LoraConfig(
    r = 8,
    lora_alpha=8,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='./logs',
)

accuracy = evaluate.load("accuracy")

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return accuracy.compute(predictions=preds, references=p.label_ids)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate(tokenized_dataset["test"])
print("Test Results:", results)