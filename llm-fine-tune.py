# !pip install bitsandbytes trl torch wandb

import kagglehub

# Download latest version
path = kagglehub.dataset_download("rohitgrewal/weather-data")

print("Path to dataset files:", path)

import os
files = os.listdir(path)

# Assuming there is a CSV file in the list
for file in files:
    if file.endswith(".csv"):
        csv_file_path = os.path.join(path, file)
        break
else:
    raise FileNotFoundError("No CSV file found in the dataset directory.")

import pandas as pd
df = pd.read_csv(csv_file_path)

df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Downsample overrepresented classes
df = df[~df['Weather'].str.contains(',')]
balanced_df = (
    df.groupby('Weather', group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), 400), random_state=42))
)

# Step 4: Create prompt-completion pairs
samples = []

for i, row in balanced_df.iterrows():
    prompt = (
        f"At {row['Date/Time'].strftime('%Y-%m-%d %H:%M')}, "
        f"the temperature was {row['Temp_C']}°C, "
        f"dew point was {row['Dew Point Temp_C']}°C, "
        f"humidity was {row['Rel Hum_%']}%, "
        f"wind speed was {row['Wind Speed_km/h']} km/h, "
        f"visibility was {row['Visibility_km']} km, "
        f"pressure was {row['Press_kPa']} kPa. "
        f"The weather was"
    )

    completion = f" {row['Weather']}"  # Note: space before value

    samples.append({
        "prompt": prompt,
        "completion": completion
    })
    percent = (i + 1) / len(df) * 100
    print(f'\rPreparing: {percent:.2f}%', end='')

import json
# Step 5: Save to JSONL
total = len(samples)
with open("weather_finetune.jsonl", "w") as f:
    for i, entry in enumerate(samples):
        f.write(json.dumps(entry) + "\n")
        percent = (i + 1) / total * 100
        print(f'\rProgress: {percent:.2f}%', end='')

from sklearn.model_selection import train_test_split

# Split the prompt-completion data into 90% training and 10% validation
train_samples, val_samples = train_test_split(samples, test_size=0.1, random_state=42, shuffle=True)

# Save training data to JSONL
with open("train.jsonl", "w") as f_train:
    for entry in train_samples:
        f_train.write(json.dumps(entry) + "\n")

# Save validation data to JSONL
with open("valid.jsonl", "w") as f_val:
    for entry in val_samples:
        f_val.write(json.dumps(entry) + "\n")

balanced_df['Weather'].value_counts()

# pick the right quantization
from transformers import BitsAndBytesConfig
import torch
QUANT_4_BIT = True
if QUANT_4_BIT:
  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
  )
else:
  quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
  )

#check the performance of LLM before fine tuning
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

from google.colab import userdata
from huggingface_hub import login
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# Load model and tokenizer
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config,
    torch_dtype=torch.float16
)

print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.1f} MB")

from transformers import LogitsProcessor

class AllowedWordsLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_words, tokenizer):
        self.allowed_ids = set()
        self.tokenizer = tokenizer

        for word in allowed_words:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 1:
                self.allowed_ids.add(token_ids[0])

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        for idx in self.allowed_ids:
            mask[:, idx] = scores[:, idx]
        return mask


# Load some samples
with open("valid.jsonl") as f:
    samples = [json.loads(line) for line in f]

weather_labels = [
    "Blowing Snow", "Clear", "Cloudy", "Drizzle", "Fog", "Freezing Drizzle",
    "Freezing Fog", "Freezing Rain", "Haze", "Heavy Rain Showers", "Ice Pellets",
    "Mainly Clear", "Moderate Rain", "Moderate Rain Showers", "Moderate Snow",
    "Mostly Cloudy", "Rain", "Rain Showers", "Snow", "Snow Grains",
    "Snow Pellets", "Snow Showers", "Thunderstorms"
]

processor = AllowedWordsLogitsProcessor(weather_labels, tokenizer)

label_list = ", ".join(weather_labels)

correct = 0
total = 0
# Predict using few examples
for sample in samples:
    input_text = sample["prompt"]
    prompt = (
        "Classify the weather condition. " + sample['prompt'] + ". \Label:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=1,
        logits_processor=[processor],
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    predicted = generated.replace(prompt, "").strip()
    actual = sample["completion"].strip()

    if predicted.lower() == actual.lower():
        correct += 1
    total += 1

    print(f"🧠 {total} Prediction: {predicted}\n✅ Actual: {actual}\n{'-'*50}")

print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load your JSONL data
dataset = load_dataset('json', data_files={'train': 'train.jsonl', 'validation': 'valid.jsonl'})

# Tokenize prompts
def tokenize(example):
    return tokenizer(example['prompt'] + example['completion'], truncation=True)

tokenized = dataset.map(tokenize)

# Hyperparameters for QLoRA
from datetime import datetime
RUN_NAME =  f"{datetime.now():%Y_%m_%d_%H_%M_%S}"
LORA_R = 32
LORA_ALPHA = 64
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LORA_DROPOUT = 0.1

PROJECT_NAME = "Weather_prediction"

import wandb
LOG_TO_WANDB = True

# Log in to Weights & Biases
wandb_api_key = userdata.get('WANDB_API_KEY')
os.environ["WANDB_API_KEY"] = wandb_api_key
wandb.login()

if LOG_TO_WANDB:
  wandb.init(project=PROJECT_NAME, name=RUN_NAME)

PROJECT_RUN_NAME = PROJECT_NAME + RUN_NAME

#PEFT configuration
from peft import LoraConfig
lora_parameters = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)

from trl import SFTTrainer, SFTConfig

# Training setup
training_args = SFTConfig(
    output_dir="./weather-lora-model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=4,
    learning_rate=3e-4,
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=200,
    logging_steps=50,
    report_to="wandb" if LOG_TO_WANDB else None,
    bf16=False,
    fp16=True,
    save_total_limit=2
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    peft_config=lora_parameters,
    args=training_args
)

# Start training
trainer.train()

# Save model
trainer.save_model("./weather-lora-4bit")
tokenizer.save_pretrained("./weather-lora-4bit")

# Push our fine-tuned model to Hugging Face
trainer.model.push_to_hub(PROJECT_RUN_NAME, private=True)

from transformers import pipeline
from peft import PeftModel

# Load the fine-tuned adapter weights
fine_tuned_model_path = "./weather-lora-4bit"
fine_tuned_model = PeftModel.from_pretrained(model, fine_tuned_model_path)

print(f"Memory footprint for fine tuned model: {fine_tuned_model.get_memory_footprint() / 1e6:.1f} MB")

# Merge the adapter weights into the base model
merged_model = fine_tuned_model.merge_and_unload()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

# Create the pipeline with the merged model
pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer, device_map="auto")

# Use the pipeline for inference
prompt_text = "At 2012-10-05 22:00, the temperature was 17.4\u00b0C, dew point was 12.1\u00b0C, humidity was 71%, wind speed was 19 km/h, visibility was 25.0 km, pressure was 100.54 kPa. The weather was"
prompt = (
    "Classify the weather condition. " + prompt_text + ". \Label:"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(
    **inputs,
    max_new_tokens=1,
    logits_processor=[processor],
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

generated = tokenizer.decode(output[0], skip_special_tokens=True)
predicted = generated.replace(prompt, "").strip()
actual = sample["completion"].strip()
print(f"{predicted} - {actual}")

if LOG_TO_WANDB:
  wandb.finish()

correct = 0
total = 0
# Predict using few examples
for sample in samples:
    input_text = sample["prompt"]
    prompt = (
        "Classify the weather condition. " + sample['prompt'] + ". \Label:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=1,
        logits_processor=[processor],
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    predicted = generated.replace(prompt, "").strip()
    actual = sample["completion"].strip()

    if predicted.lower() == actual.lower():
        correct += 1
    total += 1

    print(f"🧠 {total} Prediction: {predicted}\n✅ Actual: {actual}\n{'-'*50}")

print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")