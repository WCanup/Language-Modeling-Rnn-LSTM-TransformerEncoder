import torch
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu
from LSTM_Module import LSTM, load_tokenizer
from TextTools import TextDataset

# Download NLTK BLEU resources if not already present
# nltk.download('punkt')

prompts = []
completions = []
with open("train.jsonl", 'r') as file:
    for line in file:
        qaPair = json.loads(line)
        prompts.append(qaPair.get("prompt"))
        completions.append(qaPair.get("completion"))

model = LSTM()
model.load_state_dict(torch.load("best_LSTM_model.pt"))
model.eval()

tokenizer = load_tokenizer("bpe_tokenizer.model")
vocab_size = tokenizer.GetPieceSize()

train_dataset = TextDataset("train.jsonl", tokenizer, 128)
val_dataset = TextDataset("test.jsonl", tokenizer, 128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

predicted_words = []
ground_truth_words = []
i = 0
for (prompt, completion) in zip(prompts, completions):
    print("{}/{}".format(i, len(prompts)))

    predicted_word = model.prompt(tokenizer=tokenizer,prompt=prompt,max_length=1)
    predicted_words.append(predicted_word)
    ground_truth_words.append(completion)
    i += 1

# Compute BLEU score with NLTK
bleu_score = corpus_bleu(ground_truth_words, predicted_words, weights=[1.0,0.0,0.0,0.0])
print(f"BLEU Score (NLTK): {bleu_score:.4f}")

prompt = "“Which do you prefer? Dogs or cats?"
model_response = model.prompt(tokenizer=tokenizer, prompt=prompt)
print("\n")
print(prompt)
print("\n")
print(model_response)

custom_prompt = "Hello there, how are you?"
model_response2 = model.prompt(tokenizer=tokenizer, prompt=prompt)
print("\n")
print(custom_prompt)
print("\n")
print(model_response2)