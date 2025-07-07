
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch as pt
from torch import nn, optim
from transformers import AutoProcessor
text1 = "This is a sample text for tokenization."
# Load the tokenizer
processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")

# Tokenize text using the tokenizer
processor.encode(text1)