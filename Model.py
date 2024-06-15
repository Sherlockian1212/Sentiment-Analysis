import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

class PhoBERT_LSTM(nn.Module):
    def __init__(self, tokenizer, model, hidden_size, num_layers, num_classes):
        super(PhoBERT_LSTM, self).__init__()

        # Load pre-trained PhoBERT model and tokenizer
        self.tokenizer = tokenizer
        self.bert_model = model

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.bert_model.config.hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Dense layers
        self.fc1 = nn.Linear(hidden_size * 2, 256)  # hidden_size * 2 for bidirectional LSTM
        self.fc2 = nn.Linear(256, num_classes)

        # Activation function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_texts):

        # Tokenize the input texts
        inputs = self.tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.bert_model.device)
        attention_mask = inputs['attention_mask'].to(self.bert_model.device)

        # Get the embeddings
        with torch.no_grad():
            features = self.bert_model(input_ids, attention_mask=attention_mask)
            hidden_states = features.last_hidden_state


        # LSTM layer
        h0 = torch.zeros(self.lstm.num_layers * 2, hidden_states.size(0), self.lstm.hidden_size).to(hidden_states.device)  # Initial hidden state for bidirectional LSTM
        c0 = torch.zeros(self.lstm.num_layers * 2, hidden_states.size(0), self.lstm.hidden_size).to(hidden_states.device)  # Initial cell state for bidirectional LSTM

        lstm_out, _ = self.lstm(hidden_states, (h0, c0))  # shape: [batch_size, sequence_length, hidden_size * 2]

        # Use the output of the last time step for each sequence in the batch
        lstm_out = lstm_out[:, -1, :]  # shape: [batch_size, hidden_size * 2]

        # Dropout layer
        dropout_out = self.dropout(lstm_out)

        # Fully connected layers
        fc1_out = self.relu(self.fc1(dropout_out))  # shape: [batch_size, 256]
        logits = self.fc2(fc1_out)  # shape: [batch_size, num_classes]

        # Apply softmax activation to get the probability distribution
        output = self.softmax(logits)

        return output

# # Example usage
# hidden_size = 256
# num_layers = 2
# num_classes = 2
#
# save_directory = "./phobert_model"
# phobert = AutoModel.from_pretrained(save_directory)
# tokenizer = AutoTokenizer.from_pretrained(save_directory)
# print("LOAD phoBERT DONE")
#
# model = PhoBERT_LSTM(tokenizer, phobert, hidden_size, num_layers, num_classes)
#
# # Dummy input: a list of 16 sentences
# input_texts = ["Đây là số " + str(i) for i in range(16)]
# output = model(input_texts)
#
# print(output)  # Should print torch.Size([16, num_classes])