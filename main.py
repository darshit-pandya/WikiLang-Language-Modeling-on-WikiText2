
# # Installing and importing


# installing required packages
get_ipython().system('pip install pytorch_lightning')
get_ipython().system('pip install nltk')
get_ipython().system('python -m nltk.downloader punkt')



# imports
import pytorch_lightning as pl
import torch
import math
from torch import nn
from torch import optim
import pytorch_lightning.loggers as pl_loggers
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
import random
import operator
import re


# ## 1a
# 
# ###Architecture:
# 1. Embedding Layer - words are converted to vectors. The vector has 100 values.
# 2. RNN Layer - the RNN layer catures the sequential context of the text.
# 3. Output Layer - Softmax is used to generate the probability distribution of the words in the vocab.
# 4. Dropout - 25% dropout is applied to avoid overfitting.
# 5. Loss Function - Since a language model is similar to a classification task, I have used Cross Entropy Loss as the loss function.
# 6. Optimizer - SGD is used.
# 
# ### Design Choices:
# 1. Embedding is used to standardize the input size.
# 2. The hidden state at t=0 is initialized at 0. Indicating there is no context in the previous states.
# 3. Batch processing is used to improve efficieny in the training.
# 4. PyTorch Ligthning is used to speed up the training process.
# 5. Pre-processing:
#   * Tokenization
#   * Creating Vocabulary
# 6. Preplexity is used as the evalution metric.

trpe = []
valpe = []
x = 0
class RNN(pl.LightningModule):
    def __init__(self, vocab_size,n_layers, hidden_size, embeddding_dim):
        super().__init__()
        # setting parameters and initialization
        self.num_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_dim
        self.vocab_size = vocab_size
        self.training_step_outputs = []
        self.validation_step_outputs = []
        # embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # RNN layer
        self.rnn = nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        self.out_fc = nn.Linear(self.hidden_size, vocab_size)
        # calculating the loss
        self.loss = nn.CrossEntropyLoss()
        # 25% dropout
        self.dropout = nn.Dropout(0.25)
        self.hidden = None

    def forward(self, data):
        # Initialize hidden state if not yet initialized or if batch size changes
        batch_size = data.size(0)
        if self.hidden is None or self.hidden.size(1) != batch_size:
            self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(data.device)
        else:
            # Use the hidden state from the previous time step
            self.hidden = self.hidden.detach()
        embedding = self.dropout(self.embedding(data))
        output, self.hidden = self.rnn(embedding, self.hidden)
        output = self.out_fc(output)
        return output.view(-1, self.vocab_size)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=5e-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        # Forward pass
        output = self.forward(x)
        loss = self.loss(output, y)
        perplexity = torch.exp(loss)
        self.training_step_outputs.append(perplexity)
        return {"loss": loss}

    def on_train_epoch_end(self):
        global x,trpe
        x+=1
        trpe.append(self.training_step_outputs[-1].item())
        print("Epoch "+str(x)+" completed.")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        # Forward pass
        output = self.forward(x)
        loss = self.loss(output, y)
        perplexity = torch.exp(loss)
        self.validation_step_outputs.append(perplexity)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        global valpe
        valpe.append(self.validation_step_outputs[-1].item())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        # Forward pass
        output = self.forward(x)
        loss = self.loss(output, y)
        perplexity = torch.exp(loss)
        tensorboard_logs = {'perplexity': {'test': perplexity}, 'loss': {'test': loss.detach()}}
        self.log("loss/test", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("perplexity/test", perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def init_hidden(self, batch_size = 20):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden


class Prep:
    def __init__(self):
        with open("wiki2.test.txt") as f:
            self.test = f.read()
        with open("wiki2.train.txt") as f:
            self.train = f.read()
        with open("wiki2.valid.txt") as f:
            self.valid = f.read()
        self.word_freqs = {"<oov>":1}

    def tokenize(self, corpus):
        sent_tokens = [word_tokenize(t) for t in sent_tokenize(corpus)]
        random.shuffle(sent_tokens)
        word_tokens = [[w.lower() for w in s] for s in sent_tokens]
        word_tokens = [["<s>"] + s + ["</s>"] if s[-1].isalnum() else ["<s>"] + s[:-1] + ["</s>"] for s in word_tokens]
        corpus = []
        for s in word_tokens:
            corpus.extend(s)
        return corpus

    def building_vocab(self, corpus):
        for w in corpus:
            if w in self.word_freqs:
                self.word_freqs[w] += 1
            else:
                self.word_freqs[w] = 1

class Vocab(object):
    def __init__(self, freqs, corpus, window_size):
        super().__init__()
        self.indix2token = tuple(freqs)
        self.token2index = {k: v for v, k in enumerate(self.indix2token)}
        self.corpus = corpus
        self.window_size = window_size
        self.encoded_list = []
        self.data, self.target = self.encoding()

    def __len__(self):
        return len(self.encoded_list)

    def __getitem__(self, key):
        return torch.tensor(self.data[key]),torch.tensor(self.target[key])

    def encoding(self):
        def retrive(key):
            if isinstance(key, int):
                return None
            else:
                return self.token2index[key]
        encoded_list = [retrive(i) for i in self.corpus]
        self.encoded_list = [encoded_list[i:i + self.window_size] for i in range(0, len(encoded_list), self.window_size) if len(encoded_list[i:i + self.window_size])==self.window_size]
        data = [s[:-1] for s in self.encoded_list]
        target = [s[1:] for s in self.encoded_list]
        return data, target

    def decoding(self):
        def retrive(self, key):
            if isinstance(key, int):
                return self.indix2token[key]
            else:
                return None
        decoded_list = [[retrive(w) for w in s] for s in self.corpus]



p = Prep()
train_corpus = p.tokenize(p.train)
p.building_vocab(train_corpus)

valid_corpus = p.tokenize(p.valid)
p.building_vocab(valid_corpus)

test_corpus = p.tokenize(p.test)
p.building_vocab(test_corpus)

word_freqs = p.word_freqs

#hyperparameters
vocab_size = len(word_freqs)
batch_size = 100
n_layers = 1
hidden_size = 100
embedding_dim = 100
seq_len = 30
total_epochs = 20

train = Vocab(word_freqs, train_corpus, seq_len)
valid = Vocab(word_freqs, valid_corpus, seq_len)
test = Vocab(word_freqs, test_corpus, seq_len)

train_loader = torch.utils.data.DataLoader(train, batch_size, num_workers=16, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(valid, batch_size, num_workers=16, shuffle=False, drop_last=True)
test_loader = torch.utils.data.DataLoader(test, batch_size, num_workers=16, shuffle=False, drop_last=True)


# In[ ]:


vocab_size = len(word_freqs)
model = RNN(vocab_size, n_layers,hidden_size,embedding_dim)
trainer = pl.Trainer(max_epochs = total_epochs, enable_progress_bar=False)
trainer.fit(model, train_loader,val_loader)


# ## Learning Curves
# 

# In[ ]:


valpe.pop(0)
train_perplexity = trpe
valid_perplexity = valpe

# Plot learning curves
plt.figure(figsize=(8, 4))
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], train_perplexity, label="Training Perplexity")
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], valid_perplexity, label="Validation Perplexity")
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("Learning Curves")
plt.legend()
plt.show()


result = trainer.test(model, test_loader, verbose=False)
print()
print("Test Set Perplexity: " + str(result[0]['perplexity/test']))


# There are many ways to improve the vanilla RNN:
# 
# 1. LSTM - The most efficient way to upgrade the RNN model is using LSTM instead. It captures more context and uses different gating mechanisms that help in forgetting information that is not relevant and focusing on the important bits only.
# 
# 2. Tuning the hyperparameters - I can try increasing the size of the hidden layer so that it captures the data better.
# 
# 3. Stacking multiple RNN - Using more number of RNN layers instead of just 1.
# 
# 4. Normalization - Implementing batch normalization might stablize the network and improve its convergence.

# # 
# > LSTM Implementation (Improved Version of RNN)
# 
# > Hidden Nodes - 128
# 
# > Epochs - 30

# In[ ]:


trpe = []
valpe = []
x = 0
hidden_size = 128
total_epochs = 30
class ImprovedRNN(pl.LightningModule):
    def __init__(self, vocab_size, n_layers, hidden_size, embedding_dim):
        super().__init__()
        self.num_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_dim
        self.vocab_size = vocab_size

        self.loss = nn.CrossEntropyLoss()
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        self.out_fc = nn.Linear(self.hidden_size, self.vocab_size)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.dropout = nn.Dropout(0.25)

    def forward(self, data):
        embedding = self.dropout(self.embedding(data))
        lstm_out, _ = self.lstm(embedding)
        output = self.out_fc(lstm_out)
        return output.view(-1, self.vocab_size)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=5e-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        # Forward pass
        output = self.forward(x)
        loss = self.loss(output, y)
        perplexity = torch.exp(loss)
        self.training_step_outputs.append(perplexity)
        return {"loss": loss}

    def on_train_epoch_end(self):
        global x,trpe
        x+=1
        trpe.append(self.training_step_outputs[-1].item())
        print("Epoch "+str(x)+" completed")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        # Forward pass
        output = self.forward(x)
        loss = self.loss(output, y)
        perplexity = torch.exp(loss)
        self.validation_step_outputs.append(perplexity)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        global valpe
        valpe.append(self.validation_step_outputs[-1].item())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        # Forward pass
        output = self.forward(x)
        loss = self.loss(output, y)
        perplexity = torch.exp(loss)
        tensorboard_logs = {'perplexity': {'test': perplexity}, 'loss': {'test': loss.detach()}}
        self.log("loss/test", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("perplexity/test", perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def init_hidden(self, batch_size = 20):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden



# Instantiate the model with the hyperparameters and train the model
improved_model = ImprovedRNN(vocab_size, n_layers, hidden_size, embedding_dim)
trainer = pl.Trainer(max_epochs=total_epochs, enable_progress_bar=False)
trainer.fit(improved_model, train_loader, val_loader)



valpe.pop(0)
train_perplexity = trpe
valid_perplexity = valpe

# Plot learning curves
plt.figure(figsize=(8, 4))
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], train_perplexity, label="Training Perplexity")
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], valid_perplexity, label="Validation Perplexity")
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("Learning Curves")
plt.legend()
plt.show()


# In[ ]:


test_result = trainer.test(improved_model, test_loader)
print()
print("Test Set Perplexity: " + str(test_result[0]['perplexity/test']))

