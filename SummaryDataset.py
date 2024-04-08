import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
PADDING_VALUE = 0

def split_train_val_test(df, props=[.8, .1, .1]):
    assert round(sum(props), 2) == 1 and len(props) >= 2
    train_df, test_df, val_df = None, None, None
    
    df_size = len(df)
    train_end = int(df_size*props[0])
    val_end = int(train_end + (df_size*props[1]))
    # Select the training dataset.
    train_df = df.iloc[:train_end]
    # Select the validation dataset.
    val_df = df.iloc[train_end:val_end]
    # Select the testing dataset.
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

# SummaryDataset
# This class takes a Pandas DataFrame and wraps in a Torch dataset.
class SummaryDataset(Dataset):
    # initialize this class with appropriate instance variables
    def __init__(self, vocab, df):
        self.vocab = vocab
        self.df = df
        return 

    def __len__(self):
        return len(self.df)
    
    # __getitem__
    # 
    # Converts two dataframe rows (row["tokenized_summary"] and row["tokenized_title"])
    # to an encoded torch LongTensor,
    # using our vocab map we created using generate_vocab_map. Restricts the encoded 
    # headline length to max_length.
    # 
    # The purpose of this method is to convert the row - a list of words - into
    # a corresponding list of numbers.
    #
    # i.e. using a map of {"hi": 2, "hello": 3, "UNK": 0}
    # this list ["hi", "hello", "NOT_IN_DICT"] will turn into [2, 3, 0]
    #
    # returns: 
    # tokenized_word_tensor - torch.LongTensor 
    #                         A 1D tensor of type Long, that has each
    #                         token in the dataframe mapped to a number.
    #                         These numbers are retrieved from the vocab_map
    #                         we created in generate_vocab_map. 
    # 
    #                         **IMPORTANT**: if we filtered out the word 
    #                         because it's infrequent (and it doesn't exist 
    #                         in the vocab) we need to replace it w/ the UNK 
    #                         token
    # 
    # curr_label            - int
    #                         Binary 0/1 label retrieved from the DataFrame.
    # 
    def __getitem__(self, index: int):
        tokenized_word_tensor = None
        curr_label = None
        tokens = self.df.iloc[index]['tokenized_title']
        tokens.extend(self.df.iloc[index]['tokenized_summary'])
        tokenized_word_tensor = []
        for t in tokens:
          if t in self.vocab.keys():
            tokenized_word_tensor.append(self.vocab[t])
          else:
             tokenized_word_tensor.append(self.vocab['UNK'])
        tokenized_word_tensor = torch.tensor(tokenized_word_tensor, dtype=torch.long)

        curr_label = self.df.iloc[index]['formatted_genre']
        return tokenized_word_tensor, curr_label
    

# collate_fn
# This function is passed as a parameter to Torch DataSampler. collate_fn collects
# batched rows, in the form of tuples, from a DataLoader and applies some final 
# pre-processing.
#
# Objective:
# In our case, we need to take the batched input array of 1D tokenized_word_tensors, 
# and create a 2D tensor that's padded to be the max length from all our tokenized_word_tensors 
# in a batch. We're moving from a Python array of tuples, to a padded 2D tensor. 
#
# args: 
# batch - PythonArray[tuple(tokenized_word_tensor: 1D Torch.LongTensor, curr_label: int)]
#         len(batch) == BATCH_SIZE
# 
# returns:
# padded_tokens - 2D LongTensor of shape (BATCH_SIZE, max len of all tokenized_word_tensor))
# y_labels      - 2D FloatTensor of shape (BATCH_SIZE, max number of genres)
# 
def collate_fn(batch, padding_value=PADDING_VALUE):
    padded_tokens, y_labels = None, None
    padded_tokens = pad_sequence(sequences=list(zip(*batch))[0], padding_value=padding_value, batch_first=True)
    #y_labels = torch.tensor(list(zip(*batch))[1], dtype=torch.long)
    genre_labels = []
    for x in batch[1]:
       x_t = torch.tensor(x, dtype=torch.long)
       genre_labels.append(x_t)
    y_labels = pad_sequence(sequences=genre_labels, padding_value=padding_value, batch_first=True)
    return padded_tokens, y_labels