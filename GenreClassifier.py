import torch
import torch.nn as nn

class ClassificationModel(nn.Module):
    # Instantiate layers for your model-
    # 
    # Your model architecture will be an optionally bidirectional LSTM,
    # followed by a linear + sigmoid layer.
    #
    # You'll need 4 nn.Modules
    # 1. An embeddings layer (see nn.Embedding)
    # 2. A bidirectional LSTM (see nn.LSTM)
    # 3. A Linear layer (see nn.Linear)
    # 4. A sigmoid output (see nn.Sigmoid)
    #
    # 
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, \
                 num_layers=1, bidirectional=True):
        super().__init__()
        # Embeddings layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        # Bidirectional LSTM layer
        # Required parameters: input_size, hidden_size
            # input_size - The number of expected features in the input x
            # hidden_size - The number of features in the hidden state h
        # Optional parameters: num_layers, bidirectional
            # num_layers - number of recurrent layers
            # bidirectional - whether or not it's bidirectional
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        
        # Linear layer
        # Required parameters: in_features, out_features
            # in_features - size of each input sampleemhin
            # out_features - size of each output sample
        self.linear = nn.Linear(in_features=hidden_dim*(1+bidirectional), out_features=1)

        # Softmax output layer
        self.softmax = nn.Softmax(output_dim)
        
    # Complete the forward pass of the model.
    #
    # Use the last hidden timestep of the LSTM as input
    # to the linear layer. When completing the forward pass,
    # concatenate the last hidden timestep for both the foward,
    # and backward LSTMs.
    # 
    # args:
    # x - 2D LongTensor of shape (BATCH_SIZE, max len of all tokenized_word_tensor))
    #     This is the same output that comes out of the collate_fn function you completed-
    def forward(self, x):
        x = self.embedding(x)
        length = len(x[0])
        lengths = [length]*len(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x, (h0, _) = self.lstm(x)
        x = self.linear(torch.cat((h0[-2,:,:], h0[-1,:,:]), dim = 1))
        x = self.softmax(x)
        return x
    