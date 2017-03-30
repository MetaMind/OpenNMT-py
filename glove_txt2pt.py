import torch
stoi = {}
itos = []
vectors = []
with open('glove.840B.300d.txt', 'rb') as f:
    for l in f:
        l = l.strip().split(b' ')
        word, vector = l[0], l[1:]
        try:
            word = word.decode()
        except:
            print('non-UTF8 token', repr(word), 'ignored')
            continue
        stoi[word] =  len(itos)
        itos.append(word)
        vectors.append([float(x) for x in vector])
d = {'stoi': stoi, 'itos': itos, 'vectors': torch.FloatTensor(vectors)}
torch.save(d, 'glove.840B.300d.pt')
