import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)    
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    model = BigramLanguageModel(vocab_size=65)
    x_input = torch.tensor(
        [
            [24, 43, 58,  5, 57,  1, 46, 43],
            [44, 53, 56,  1, 58, 46, 39, 58],
            [52, 58,  1, 58, 46, 39, 58,  1],
            [25, 17, 27, 10,  0, 21,  1, 54]
        ], dtype=torch.long
        )
    y_target = torch.tensor(
        [
            [43, 58,  5, 57,  1, 46, 43, 39],
            [53, 56,  1, 58, 46, 39, 58,  1],
            [58,  1, 58, 46, 39, 58,  1, 46],
            [17, 27, 10,  0, 21,  1, 54, 39]
        ], dtype=torch.long
        )
    logits_example, loss_example = model.forward(x_input, y_target)
    import pdb; pdb.set_trace()