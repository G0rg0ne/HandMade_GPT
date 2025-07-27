import torch

from tokenizer import read_file, creat_mapping, encode, decode, tiktoken_encode
from models.bigram import BigramLanguageModel

torch.manual_seed(1337)

def split_data(data, train_split=0.9, val_split=0.1):
    train_data = data[:int(train_split*len(data))]
    val_data = data[int(train_split*len(data)):]
    return train_data, val_data

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def show_input_prediction_sequence(x, y, itos):
    print("input/prediction sequence process:")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            context = x[i,:j+1]
            gt = y[i,j]
            print(f"when input is {context.tolist()} the target: {gt}")
        print("-"*100)

if __name__ == "__main__":
    text, chars, vocab_size = read_file('data/english/input.txt')
    stoi, itos = creat_mapping(chars)
    encode_text = encode(text, stoi)
    data_tokenized = torch.tensor(encode_text, dtype=torch.long)
    train_data, val_data =split_data(data_tokenized)
    train_batch_x, train_batch_y = get_batch(train_data, 8, 4)

    #Model
    model = BigramLanguageModel(vocab_size)
    generate_text = model.generate(train_batch_x, 10)

    for batch_idx in range(train_batch_x.shape[0]):
        generate_text_decode = decode(generate_text[batch_idx].tolist(), itos)
        print(f"input: {decode(train_batch_x[batch_idx].tolist(), itos)}")
        print(f"generate: {generate_text_decode}")
        print("-"*20)
    import pdb; pdb.set_trace()