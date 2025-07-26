import re
import pandas as pd
import torch
import tiktoken

def read_file(file_path):
    #read a file
    with open(file_path, 'r') as f:
        text = f.read()
    print("length of dataset in characters: ", len(text))
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    return text, chars, vocab_size

def creat_mapping(chars):
    stoi = { ch:i for i, ch in enumerate(chars) }
    itos = { i:ch for i, ch in enumerate(chars) }
    return stoi, itos

def encode(text, stoi):
    return [stoi[c] for c in text]

def decode(int_list, itos):
    return ''.join([itos[i] for i in int_list]) 

def tiktoken_encode(text):
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer.encode(text)

def inspect_tiktoken_vocabulary():
    tokenizer = tiktoken.get_encoding("gpt2")
    
    vocab_size = tokenizer.max_token_value + 1
    print(f"GPT-2 vocabulary size: {vocab_size}")
    
    print("\nFirst 50 tokens:")
    for i in range(50):
        try:
            token_text = tokenizer.decode([i])
            print(f"Token {i}: '{repr(token_text)}'")
        except:
            print(f"Token {i}: <cannot decode>")
    
    print("\nSome common word encodings:")
    test_words = ["hello", "world", "the", "and", "python", "AI","I love machine learning"]
    for word in test_words:
        tokens = tokenizer.encode(word)
        decoded = tokenizer.decode(tokens)
        print(f"'{word}' -> {tokens} -> '{decoded}'")

if __name__ == "__main__":
    text, chars, vocab_size = read_file('data/english/input.txt')
    stoi, itos = creat_mapping(chars)
    encode_text = encode(text, stoi)
    data = torch.tensor(encode_text, dtype=torch.long)
    tiktoken_encode_text = tiktoken_encode(text)
    inspect_tiktoken_vocabulary()
    import pdb; pdb.set_trace()


