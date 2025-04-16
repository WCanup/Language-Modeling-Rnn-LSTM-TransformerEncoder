import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
import tqdm
import sentencepiece as spm
from TextTools import TextDataset


class LSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, output_size=10000, embed_dim=128, pad_token_id=3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        # Connects the input to the input gate output
        self.Wxi = nn.Linear(input_size, hidden_size)
        # Connects the previous hidden state to the input gate output
        self.Whi = nn.Linear(hidden_size, hidden_size, bias=False)

        # Connects the input to the forget gate output
        self.Wxf = nn.Linear(input_size, hidden_size)
        # Connects the previous hidden state to the forget gate output
        self.Whf = nn.Linear(hidden_size, hidden_size, bias=False)

        # Connects the input to the memory cell (input node, in ref. to class slides) output
        self.Wxc = nn.Linear(input_size, hidden_size)
        # Connects the previous hidden state to the memory cell output
        self.Whc = nn.Linear(hidden_size, hidden_size, bias=False)

        # Connects the input to the output gate's output
        self.Wxo = nn.Linear(input_size, hidden_size)
        # Connects the previous hidden state to the output gate's output
        self.Who = nn.Linear(hidden_size, hidden_size, bias=False)

        # Simple linear layer that computes the LSTM module's final output (only used in the final LSTM module)
        self.Why = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.embedding = nn.Embedding(output_size, embed_dim, padding_idx=pad_token_id)


    def forward(self, x, hidden_state, cell_state):
        """
        Compute the hidden state, memory cell internal state, and output of the LSTM module
        :param x: Input at current timestep (batch_size, input_size)
        :param hidden: Previous hidden state (batch_size, hidden_size)
        :param cell: Previous cell state (batch_size, hidden_size)
        :return: Output, new hidden state, new cell state
        """
        #Include this line for bleu score evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
        x = x.long()

        #RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat2 in method wrapper_CUDA_mm)
        x = x.to(device)
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)

        embedded = self.embedding(x)
        seq_len = embedded.size(1)
        outputs = []
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            # Compute gate outputs
            input_gate = self.sigmoid(self.Wxi(x_t) + self.Whi(hidden_state))
            forget_gate = self.sigmoid(self.Wxf(x_t) + self.Whf(hidden_state))
            output_gate = self.sigmoid(self.Wxo(x_t) + self.Who(hidden_state))


            # Compute the cell node output (candidate cell state)
            cell_node = self.tanh(self.Wxc(x_t) + self.Whc(hidden_state))
            # Update cell state
            cell_state = forget_gate * cell_state + input_gate * cell_node

            # Update hidden state
            hidden_state = output_gate * self.tanh(cell_state)

            # Compute final output of the network
            output = self.Why(hidden_state)
            outputs.append(output)

        logits = torch.stack(outputs, dim=1)
        return logits, hidden_state, cell_state

    def init_hidden_cell(self, batch_size):
        """
        Initializes hidden and cell states to just zeros
        :param batch_size: Number of samples in the batch
        :return: Initial hidden state, Initial cell state (both of shape: batch_size x hidden_size)
        """
        return torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)

    def predict_next_token(self, input_ids, hidden, cell_state, temperature=0.8):
        logits, _, cell_state = self.forward(input_ids, hidden,cell_state)
        logits = logits/temperature
        probs = torch.softmax(logits, dim=-1)[0, -1] #softmax
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        return next_token_id, hidden, cell_state

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
        hidden, cell  = self.init_hidden_cell(batch_size=1) # ensure it's on the same device

        # loop over max output tokens
        for _ in range(max_length):
            next_token_id, hidden, cell = self.predict_next_token(input_tensor, hidden, cell)
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

if __name__ == "__main__":

    TRAIN_FILE = "train.jsonl"
    VAL_FILE = "test.jsonl"
    MAX_SEQ_LEN = 128
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

    model = LSTM().to(device)

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

            hidden, cell = model.init_hidden_cell(input_ids.size(0))


            #compute output logits
            logits, _ , cell_state = model(input_ids, hidden, cell)

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

                hidden, cell = model.init_hidden_cell(input_ids.size(0))

                logits, _, cell_state = model(input_ids, hidden, cell)

                val_loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss/ len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_LSTM_model.pt")
                print("Saved new best model!")

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
