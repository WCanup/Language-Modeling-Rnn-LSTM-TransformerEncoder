import math
import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
import sentencepiece as spm
from TextTools import TextDataset


class TransformerModule(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_size=256, num_heads=4, max_seq_len = 512):
        super(TransformerModule, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, dropout=0.1, max_len=512)
        # self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            activation='relu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder, num_layers=1)
        self.output = nn.Linear(embed_dim,vocab_size)
        self.max_seq_len = max_seq_len



    def generate_causal_mask(self, seq_len, device):
        # Creates a causal (lower triangular) mask
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, x):

        batch_size, seq_len = x.size()
        x = self.embedding(x)
        x = self.pos_encoder(x)
        mask = self.generate_causal_mask(seq_len, x.device)
        x = self.encoder(x, mask=mask)

        return self.output(x)



    def predict_next_token(self, input_ids, temperature=0.8):

        logits = self.forward(input_ids)
        logits = logits/temperature
        probs = torch.softmax(logits, dim=-1)[0, -1]  # softmax
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        return next_token_id

    def prompt(self, tokenizer, prompt, max_length=50, eos_token_id=3, device='cuda'):
        """
        :param tokenizer: The trained SentencePiece tokenizer
        :param prompt: The input prompt (plain text string)
        :param max_length: Maximum number of tokens to generate autoregressively before stopping
        :param eos_token_id: The token ID of the EOS token
        :param device: Device we are using to run the model
        :return:
        """

        self.eval() #set the model to evaluation mode
        input_ids = tokenizer.encode(prompt, out_type=int) # Encode the input string into token IDs
        #convert token ID list to tensor, move to device memory, and adding a batch dimension (1D to 2d)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_ids = [] #this will store the generated token IDs


        # loop over max output tokens
        for _ in range(max_length):
            next_token_id = self.predict_next_token(input_tensor)
            # exit early if the model generated <eos> token ID
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            # keep track of generated tokens
            generated_ids.append(next_token_id)
            # the input to the next step is just the new token and the hidden state
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long,device=device)
        # decode generated token IDs into tokens
        return tokenizer.decode(generated_ids, out_type=str)


def train_model(BATCH_SIZE, train_dataset, val_dataset):

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader

def load_tokenizer(TOKENIZER_PATH):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(TOKENIZER_PATH)
    return tokenizer


def collate_fn(batch):
    """
    Ensure batch is appropriately sized and padded for efficient training
    :param batch: batch from DataLoader, which will be a list of Tuples of token ID tensors (which
        could be different sizes)
    :return: collated input and target batch
    """

    input_batch, target_batch = zip(*batch)
    input_batch = nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=3)
    target_batch = nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=3)
    return input_batch, target_batch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

if __name__ == "__main__":

    TRAIN_FILE = "train.jsonl"
    VAL_FILE = "test.jsonl"
    MAX_SEQ_LEN = 512 #Had to do this for Bleu score
    BATCH_SIZE = 128
    TOKENIZER_PATH = "bpe_tokenizer.model"
    NUM_EPOCHS = 30

    # instantiate our model and move it to the correct device memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.GetPieceSize()

    train_dataset = TextDataset(TRAIN_FILE, tokenizer, MAX_SEQ_LEN)
    val_dataset = TextDataset(VAL_FILE, tokenizer, MAX_SEQ_LEN)

    train_loader, val_loader = train_model(BATCH_SIZE, train_dataset, val_dataset)

    model = TransformerModule().to(device)

    # Using AdamW optimizer on the trainable params.
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    # going to use a learning rate scheduler that reduces LR by half after stagnation for 1 epoch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=3)

    best_val_loss = float('inf')
    no_improve_epochs = 0
    train_losses, val_losses = [], []
    avg_train_loss, avg_val_loss = 0, 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        #loop through each sample batch in training
        for input_ids, target_ids in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):

            # move input and target tensors to device memory
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # reset gradients between batches
            optimizer.zero_grad()

            #compute output logits
            logits = model(input_ids)

            #apply cross entropy
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        #Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)


                logits = model(input_ids)

                val_loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss/ len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_Transformer_model.pt")
                print("Saved new best model")

    perplexity_train = math.exp(avg_train_loss)
    perplexity_val = math.exp(avg_val_loss)

    print("Perplexity of training set = {}".format(perplexity_train))
    print("Perplexity of Validation set = {}".format(perplexity_val))

    plt.plot(train_losses, label='Training', color = 'b')
    plt.plot(val_losses, label= 'Validation', color = 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve', size=16)
    plt.legend()
    plt.show()