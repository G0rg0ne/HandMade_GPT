import torch

from tokenizer import read_file, creat_mapping, encode, decode, tiktoken_encode
from models.bigram import BigramLanguageModel
from utils import plot_loss

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


def estimate_loss(model,train_data,val_data,eval_iters,block_size,batch_size):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        if split == 'train' : 
            split_data = train_data
        else : 
            split_data = val_data
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split_data,block_size,batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(model,train_data,val_data):
    batch_size = 32
    block_size = 8
    epochs = 10000
    learning_rate = 1e-3
    eval_iters = 200
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_values = []
    for iter in range(epochs):
        #estimate the loss
        estimated_loss = estimate_loss(model,train_data,val_data,eval_iters,block_size,batch_size)
        if iter % 10 == 0:
            print(f"iter {iter} train loss: {estimated_loss['train']} val loss: {estimated_loss['val']}")
        #sample a batch of data
        xb, yb = get_batch(train_data, block_size, batch_size)
        #evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_values.append(estimated_loss['train'])
    #i want to smoothen the loss values
    plot_loss(loss_values, title='Training Loss', xlabel='Iterations', ylabel='Loss')
    return model
    
def generate_text(train_batch_x,itos , model):
    generate_text = model.generate(train_batch_x, 10)
    for batch_idx in range(train_batch_x.shape[0]):
        generate_text_decode = decode(generate_text[batch_idx].tolist(), itos)
        print(f"input: {decode(train_batch_x[batch_idx].tolist(), itos)}")
        print(f"generate: {generate_text_decode}")
        print("-"*20)

if __name__ == "__main__":
    #Loading data and tokenizing
    text, chars, vocab_size = read_file('data/english/input.txt')
    stoi, itos = creat_mapping(chars)
    encode_text = encode(text, stoi)
    data_tokenized = torch.tensor(encode_text, dtype=torch.long)
    train_data, val_data =split_data(data_tokenized)
    train_batch_x, train_batch_y = get_batch(train_data, 8, 4)

    #Model
    model = BigramLanguageModel(vocab_size)
    trained_model = train_model(model,train_data,val_data)
    generate_text(train_batch_x, itos, trained_model)
    import pdb; pdb.set_trace()