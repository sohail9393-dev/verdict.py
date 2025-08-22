# Open the file 'the-verdict.txt' in read mode using UTF-8 encoding
with open("the-verdict.txt", "r", encoding="utf-8") as f:  # Open file for reading
    raw_text = f.read()  # Read the entire content of the file into the variable 'raw_text'

# Print the total number of characters in the loaded text
print("Total number of character:", len(raw_text))  # Print character count
# Print the first 99 characters of the loaded text to preview its content
print(raw_text[:99])  # Print preview of text

# Import the regular expressions module for text processing
import re  # Import regex module

# Example text for demonstrating tokenization
text = "Hello, world. This, is a test."  # Example string for splitting
# Split the text by whitespace, keeping the delimiter in the result
result = re.split(r'(\s)', text)  # Split by whitespace, keep delimiter

# Print the result of splitting by whitespace
print(result)  # Print split result
# Split the text by comma, period, or whitespace, keeping the delimiter in the result
result = re.split(r'([,.]|\s)', text)  # Split by punctuation or whitespace

# Print the result of splitting by punctuation and whitespace
print(result)  # Print split result
# Remove empty strings and whitespace-only items from the result
result = [item for item in result if item.strip()]  # Remove empty/whitespace items
# Print the cleaned result after removing empty/whitespace-only items
print(result)  # Print cleaned result
# Another example text for tokenization demonstration
text = "Hello, world. Is this-- a test?"  # New example string
# Split the text by punctuation, double dash, or whitespace, keeping the delimiter in the result
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)  # Split by more punctuation
# Remove empty strings and whitespace-only items from the result
result = [item.strip() for item in result if item.strip()]  # Clean split result
# Print the cleaned result after removing empty/whitespace-only items
print(result)  # Print cleaned result
# Strip whitespace from each item and then filter out any empty strings.
result = [item for item in result if item.strip()]  # Remove empty/whitespace items
print(result)  # Print cleaned result
text = "Hello, world. Is this-- a test?"  # Example string

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)  # Split by punctuation/whitespace
result = [item.strip() for item in result if item.strip()]  # Clean result
print(result)  # Print cleaned result
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)  # Split raw text
preprocessed = [item.strip() for item in preprocessed if item.strip()]  # Clean split
print(preprocessed[:30])  # Print first 30 tokens
print(len(preprocessed))  # Print total number of tokens
all_words = sorted(set(preprocessed))  # Get sorted unique tokens
vocab_size = len(all_words)  # Get vocabulary size

print(vocab_size)  # Print vocabulary size
vocab = {token:integer for integer,token in enumerate(all_words)}  # Map tokens to integers
for i, item in enumerate(vocab.items()):  # Iterate over vocab items
    print(item)  # Print vocab item
    if i >= 50:  # Stop after 50 items
        break
    class SimpleTokenizerV1:
     def __init__(self, vocab):  # Constructor
        self.str_to_int = vocab  # Token to integer mapping
        self.int_to_str = {i:s for s,i in vocab.items()}  # Integer to token mapping
    
     def encode(self, text):  # Encode text to token ids
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)  # Split text
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()  # Clean split
        ]
        ids = [self.str_to_int[s] for s in preprocessed]  # Map tokens to ids
        return ids  # Return ids
        
     def decode(self, ids):  # Decode ids to text
        text = " ".join([self.int_to_str[i] for i in ids])  # Join tokens
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # Clean up spaces
        return text  # Return decoded text
tokenizer = SimpleTokenizerV1(vocab)  # Instantiate tokenizer

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

class SimpleTokenizerV2:
    def __init__(self, vocab):  # Constructor
        self.str_to_int = vocab  # Token to integer mapping
        self.int_to_str = { i:s for s,i in vocab.items()}  # Integer to token mapping
    
    def encode(self, text):  # Encode text to token ids
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)  # Split text
        preprocessed = [item.strip() for item in preprocessed if item.strip()]  # Clean split
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed  # Replace unknown tokens
        ]

        ids = [self.str_to_int[s] for s in preprocessed]  # Map tokens to ids
        return ids  # Return ids
        
    def decode(self, ids):  # Decode ids to text
        text = " ".join([self.int_to_str[i] for i in ids])  # Join tokens
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)  # Clean up spaces
        return text  # Return decoded text
tokenizer = SimpleTokenizerV2(vocab)  # Instantiate tokenizer

text1 = "Hello, do you like tea?"  # Example text 1
text2 = "In the sunlit terraces of the palace."  # Example text 2

text = " <|endoftext|> ".join((text1, text2))  # Concatenate texts with special token

print(text)  # Print concatenated text

ids = tokenizer.encode(text)  # Encode text to token ids
print(ids)  # Print token ids
print(tokenizer.decode(ids))  # Print decoded text

# LECTURE NO. 8 STARTS FROM HERE.

import importlib.metadata  # Import metadata module
import tiktoken  # Import tiktoken library
print("tiktoken version:", importlib.metadata.version("tiktoken"))  # Print tiktoken version

tokenizer = tiktoken.get_encoding("gpt2")  # Get GPT-2 tokenizer
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."  # Example text
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})  # Encode text

print(integers)  # Print token ids

strings = tokenizer.decode(integers)  # Decode token ids

print(strings)  # Print decoded text

integers = tokenizer.encode("Akwirw ier")  # Encode example text
print(integers)  # Print token ids

strings = tokenizer.decode(integers)  # Decode token ids
print(strings)  # Print decoded text

# LECTURE NUMBER 9 STARTS FROM HERE

with open("the-verdict.txt", "r", encoding="utf-8") as f:  # Open file for reading
    raw_text = f.read()  # Read file content

enc_text = tokenizer.encode(raw_text)  # Encode raw text
print(len(enc_text))  # Print length of encoded text

# This code removes the first 50 tokens it is done to make lecture more interesting
# No need to do it
enc_sample = enc_text[50:]  # Remove first 50 tokens

context_size = 4  # Set context size

x = enc_sample[:context_size]  # Get first context_size tokens
y = enc_sample[1:context_size+1]  # Get next context_size tokens

print(f"x: {x}")  # Print x
print(f"y:      {y}")  # Print y
# Here we are printing our encoded token numbers
for i in range(1, context_size+1):  # Loop over context sizes
    context = enc_sample[:i]  # Get context tokens
    desired = enc_sample[i]  # Get desired token

    print(context, "---->", desired)  # Print context and desired token

#  Here we are printing our decoded text
for i in range(1, context_size+1):  # Loop over context sizes
    context = enc_sample[:i]  # Get context tokens
    desired = enc_sample[i]  # Get desired token

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))  # Print decoded context and token

from torch.utils.data import Dataset, DataLoader  # Import Dataset and DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:  # Open file for reading
    raw_text = f.read()  # Read file content
# It only gives us the input output pair


import torch  # Import torch
print("PyTorch version:", torch.__version__)  # Print PyTorch version
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)  # Create dataloader

data_iter = iter(dataloader)  # Get iterator
first_batch = next(data_iter)  # Get first batch
print(first_batch)  # Print first batch

second_batch = next(data_iter)  # Get second batch
print(second_batch)  # Print second batch

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)  # Create dataloader

data_iter = iter(dataloader)  # Get iterator
inputs, targets = next(data_iter)  # Get inputs and targets
print("Inputs:\n", inputs)  # Print inputs
print("\nTargets:\n", targets)  # Print targets

# LECTURE NUMBER 10 STARTS FROM HERE
# IN THIS LEC. WE ARE GOING TO CREATE TOKEN EMBEDDINGS
input_ids = torch.tensor([2, 3, 5, 1])  # Example token ids

vocab_size = 6  # Set vocabulary size
output_dim = 3  # Set output dimension

torch.manual_seed(123)  # Set random seed
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)  # Create embedding layer

print(embedding_layer.weight)  # Print embedding weights

print(embedding_layer(torch.tensor([3])))  # Print embedding for token 3

print(embedding_layer(input_ids))  # Print embeddings for input ids

# LECTURE NUMBER 11 STARTS FROM HERE
# IN THIS LEC. WE ARE GOING TO CREATE POSITIONAL VECTOR EMBEDDINGS

vocab_size = 50257  # Set vocabulary size
output_dim = 256  # Set output dimension
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)  # Create token embedding layer

max_length = 4  # Set max sequence length
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False 
)  # Create dataloader
data_iter = iter(dataloader)  # Get iterator
inputs, targets = next(data_iter)  # Get inputs and targets

print("Token IDs:\n", inputs)  # Print token ids
print("\nInputs shape:\n", inputs.shape)  # Print input shape

token_embeddings = token_embedding_layer(inputs)  # Get token embeddings
print(token_embeddings.shape)  # Print shape of token embeddings

context_length = max_length  # Set context length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)  # Create positional embedding layer


pos_embeddings = pos_embedding_layer(torch.arange(max_length))  # Get positional embeddings
print(pos_embeddings.shape)  # Print shape of positional embeddings

input_embeddings = token_embeddings + pos_embeddings  # Add token and positional embeddings
print(input_embeddings.shape)  # Print shape of input embeddings

# LECTURE NUMBER 12 STARTS FROM HERE
import torch  # Import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)  # Example input embeddings

import matplotlib.pyplot as plt  # Import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit

# Corresponding words
words = ['Your', 'journey', 'starts', 'with', 'one', 'step']  # List of words

# Extract x, y, z coordinates
x_coords = inputs[:, 0].numpy()  # Get x coordinates
y_coords = inputs[:, 1].numpy()  # Get y coordinates
z_coords = inputs[:, 2].numpy()  # Get z coordinates

# Create 3D plot
fig = plt.figure()  # Create figure
ax = fig.add_subplot(111, projection='3d')  # Add 3D subplot

# Plot each point and annotate with corresponding word
for x, y, z, word in zip(x_coords, y_coords, z_coords, words):  # Loop over points
    ax.scatter(x, y, z)  # Plot point
    ax.text(x, y, z, word, fontsize=10)  # Annotate point

# Set labels for axes
ax.set_xlabel('X')  # Set x label
ax.set_ylabel('Y')  # Set y label
ax.set_zlabel('Z')  # Set z label

plt.title('3D Plot of Word Embeddings')  # Set plot title
plt.show()  # Show plot

attn_scores = torch.empty(6, 6)  # Create empty attention scores tensor

for i, x_i in enumerate(inputs):  # Loop over inputs
    for j, x_j in enumerate(inputs):  # Loop over inputs
        attn_scores[i, j] = torch.dot(x_i, x_j)  # Compute dot product

print(attn_scores)  # Print attention scores

attn_weights = torch.softmax(attn_scores, dim=-1)  # Compute softmax weights
print(attn_weights)  # Print attention weights


row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])  # Sum of row 2
print("Row 2 sum:", row_2_sum)  # Print row 2 sum
print("All row sums:", attn_weights.sum(dim=-1))  # Print all row sums
# nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn

all_context_vecs = attn_weights @ inputs  # Compute context vectors
print(all_context_vecs)  # Print context vectors

context_vec_2 = all_context_vecs[1]  # Get context vector for second input
print("Previous 2nd context vector:", context_vec_2)  # Print context vector

import torch  # Import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)  # Example input embeddings

x_2 = inputs[1] #A  # Get second input
d_in = inputs.shape[1] #B  # Get input dimension
d_out = 2 #C  # Set output dimension

torch.manual_seed(123)  # Set random seed
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # Create query weights
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # Create key weights
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # Create value weights
print(W_query)  # Print query weights

print(W_key)  # Print key weights
print(W_value)  # Print value weights

query_2 = x_2 @ W_query  # Compute query for second input
key_2 = x_2 @ W_key  # Compute key for second input
value_2 = x_2 @ W_value  # Compute value for second input
print(query_2)  # Print query

keys = inputs @ W_key  # Compute keys for all inputs
values = inputs @ W_value  # Compute values for all inputs
queries = inputs @ W_query  # Compute queries for all inputs
print("keys.shape:", keys.shape)  # Print shape of keys

print("values.shape:", values.shape)  # Print shape of values

print("queries.shape:", queries.shape)  # Print shape of queries

keys_2 = keys[1] #A  # Get key for second input
attn_score_22 = query_2.dot(keys_2)  # Compute attention score for second input
print(attn_score_22)  # Print attention score

attn_scores_2 = query_2 @ keys.T # All attention scores for given query  # Compute attention scores for second input
print(attn_scores_2)  # Print attention scores

attn_scores = queries @ keys.T # omega  # Compute attention scores for all inputs
print(attn_scores)  # Print attention scores

d_k = keys.shape[-1]  # Get key dimension
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)  # Compute softmax weights for second input
print(attn_weights_2)  # Print attention weights
print(d_k)  # Print key dimension

import torch  # Import torch

tensor = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])  # Example tensor

softmax_result = torch.softmax(tensor, dim=-1)  # Compute softmax
print("Softmax without scaling:", softmax_result)  # Print result

scaled_tensor = tensor * 8  # Scale tensor
softmax_scaled_result = torch.softmax(scaled_tensor, dim=-1)  # Compute softmax
print("Softmax after scaling (tensor * 8):", softmax_scaled_result)  # Print result

import numpy as np  # Import numpy

# Function to compute variance before and after scaling
def compute_variance(dim, num_trials=1000):  # Define function
    dot_products = []  # List for dot products
    scaled_dot_products = []  # List for scaled dot products

    # Generate multiple random vectors and compute dot products
    for _ in range(num_trials):  # Loop over trials
        q = np.random.randn(dim)  # Generate random vector
        k = np.random.randn(dim)  # Generate random vector
        
        # Compute dot product
        dot_product = np.dot(q, k)
        dot_products.append(dot_product)
        
        # Scale the dot product by sqrt(dim)
        scaled_dot_product = dot_product / np.sqrt(dim)
        scaled_dot_products.append(scaled_dot_product)
    
    # Calculate variance of the dot products
    variance_before_scaling = np.var(dot_products)
    variance_after_scaling = np.var(scaled_dot_products)

    return variance_before_scaling, variance_after_scaling  # Return variances

# For dimension 5
variance_before_5, variance_after_5 = compute_variance(5)
print(f"Variance before scaling (dim=5): {variance_before_5}")  # Print variance
print(f"Variance after scaling (dim=5): {variance_after_5}")  # Print variance

# For dimension 20
variance_before_100, variance_after_100 = compute_variance(100)
print(f"Variance before scaling (dim=100): {variance_before_100}")  # Print variance
print(f"Variance after scaling (dim=100): {variance_after_100}")  # Print variance

context_vec_2 = attn_weights_2 @ values  # Compute context vector for second input
print(context_vec_2)  # Print context vector

import torch.nn as nn  # Import torch.nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
    
    torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)  # Example input embeddings

queries = sa_v2.W_query(inputs) #A  # Compute queries
keys = sa_v2.W_key(inputs)  # Compute keys
attn_scores = queries @ keys.T  # Compute attention scores
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)  # Compute softmax weights
print(attn_weights)  # Print attention weights

torch.ones(context_length, context_length)  # Create tensor of ones

context_length = attn_scores.shape[0]  # Get context length
mask_simple = torch.tril(torch.ones(context_length, context_length))  # Create lower triangular mask
print(mask_simple)  # Print mask

masked_simple = attn_weights*mask_simple  # Apply mask
print(masked_simple)  # Print masked weights

row_sums = masked_simple.sum(dim=1, keepdim=True)  # Compute row sums
masked_simple_norm = masked_simple / row_sums  # Normalize masked weights
print(masked_simple_norm)  # Print normalized weights

print(attn_scores)  # Print attention scores

torch.triu(torch.ones(context_length, context_length))  # Create upper triangular mask

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)  # Create upper triangular mask
print(mask)  # Print mask

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)  # Create upper triangular mask
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)  # Apply mask
print(masked)  # Print masked scores

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)  # Compute softmax weights
print(attn_weights)  # Print attention weights

example = torch.ones(6, 6) #B  # Create tensor of ones
print(example)  # Print tensor

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) #A  # Create dropout layer
example = torch.ones(6, 6) #B  # Create tensor of ones
print(dropout(example))  # Print dropout output
torch.manual_seed(123)
print(dropout(attn_weights))  # Print dropout output

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)  # Example input embeddings
batch = torch.stack((inputs, inputs), dim=0)  # Stack inputs to create batch
print(batch.shape)  # Print batch shape

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec
    
    print(d_in)

    print(d_out)

    torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)

print(context_vecs)

class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)  # Example input embeddings
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)  # Print batch shape

torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens = 6
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
    
torch.manual_seed(123)

inputs = torch.tensor(
    [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # Row 1
     [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # Row 2
     [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]  # Row 3
)  # Example tensor

batch = torch.stack((inputs, inputs), dim=0)  # Stack inputs to create batch
print(batch.shape)  # Print batch shape


batch_size, context_length, d_in = batch.shape  # Get batch size, context length, input dim
d_out = 6  # Set output dimension
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)  # Instantiate multi-head attention
context_vecs = mha(batch)  # Compute context vectors
print(context_vecs)  # Print context vectors
print("context_vecs.shape:", context_vecs.shape)  # Print shape of context vectors


import torch


GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x