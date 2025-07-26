import torch

from tokenizer import read_file, creat_mapping, encode, decode, tiktoken_encode

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

if __name__ == "__main__":
    text, chars, vocab_size = read_file('data/english/input.txt')
    stoi, itos = creat_mapping(chars)
    encode_text = encode(text, stoi)
    data = torch.tensor(encode_text, dtype=torch.long)
    train_data, val_data =split_data(data)
    train_batch_x, train_batch_y = get_batch(train_data, 8, 4)

    print("input/prediction sequence process:")
    for i in range(train_batch_x.shape[0]):
        for j in range(train_batch_x.shape[1]):
            context =  train_batch_x[i,:j+1]
            gt = train_batch_y[i,j]
            print(f"when input is {context.tolist()} the target: {gt}")
        print("-"*100)	
    import pdb; pdb.set_trace()