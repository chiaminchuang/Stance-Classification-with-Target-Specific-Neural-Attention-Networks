import pandas as pd
import numpy as np
import jieba
import re

# utils
def read_train_data(train_file):
    train_df = pd.read_csv(train_file, usecols=[3, 4, 7], keep_default_na=False)
    train_df = train_df.drop(train_df[(train_df['title1_zh'] == '') | (train_df['title2_zh'] == '')].index)
    
    train_df['title1_zh'] = train_df['title1_zh'].map(segment_word)
    train_df['title2_zh'] = train_df['title2_zh'].map(segment_word)
    train_df['label'] = train_df['label'].map({'agreed': 0, 'disagreed': 1, 'unrelated': 2})
    
    return train_df


def read_test_data(test_file):
    test_df = pd.read_csv(test_file, usecols=[0, 3, 4], keep_default_na=False)
    
    test_df['title1_zh'] = test_df['title1_zh'].map(segment_word)
    test_df['title2_zh'] = test_df['title2_zh'].map(segment_word)
    
    return test_df

    
def segment_word(sentence):
    words = list(jieba.cut(sentence.strip()))
    return remove_punctuation(words)


def remove_punctuation(words):
    def remove_emptystring(words):
        return [w for w in words if w]
    
    return remove_emptystring(re.sub(r'[^\w]', '', w) for w in words)


def load_wordvector(wordvec_file, UNK, PAD):
    wordvector, vocab = [], []
    word2index = {}
    with open(wordvec_file, 'r', encoding='utf-8') as file:
        next(file)  # skip header
        for i, row in enumerate(file):
            row = row.strip().split(' ')
            wordvector.append(row[1:])
            vocab.append(row[0])
            word2index[row[0]] = i
    
    wordvector.extend([[0] * len(wordvector[0])] * 2)  # <UNK>, <PAD>
    vocab.extend([UNK, PAD])
    word2index.update({UNK: len(wordvector) - 2, PAD: len(wordvector) - 1})
    return np.array(wordvector).astype(float), word2index, vocab

    
def process_unknown(sentA, sentB, vocab, UNK):
    new_sentA, new_sentB = [], []
    for A, B in zip(sentA, sentB):
        new_sentA.append([word if word in vocab else UNK for word in A])
        new_sentB.append([word if word in vocab else UNK for word in B])
    return new_sentA, new_sentB


def word_to_index(sentA, sentB, word2index):
    new_sentA, new_sentB = [], []
    for A, B in zip(sentA, sentB):
        new_sentA.append([word2index[word] for word in A])
        new_sentB.append([word2index[word] for word in B])
    return new_sentA, new_sentB


def train_val_split(sentA, sentB, seq_lenA, seq_lenB, label, train_ratio=0.7):
    train_len  = int(len(sentA) * train_ratio)
    train_data = [sentA[: train_len], sentB[: train_len], seq_lenA[: train_len], seq_lenB[: train_len], label[: train_len]]
    val_data   = [sentA[train_len: ], sentB[train_len: ], seq_lenA[train_len: ], seq_lenB[train_len: ], label[train_len: ]]
    return train_data, val_data


def shuffle_data(data):
    indice = np.arange(len(data[0]))
    np.random.shuffle(indice)
    
    return [data[0][indice], data[1][indice], data[2][indice], data[3][indice], data[4][indice]]


def next_batch(data, batch_size, word2index):
    def pad(sequence, max_len):
        return np.array([seq + [word2index['<PAD>']] * (max_len - len(seq)) for seq in sequence])
    
    sentA, sentB, seq_lenA, seq_lenB, label = data[0], data[1], data[2], data[3], data[4]
    nbatch = len(data[0]) // batch_size
    for i in range(nbatch):
        offset = i * batch_size
        
        batch_seq_lenA = seq_lenA[offset: offset + batch_size]
        batch_seq_lenB = seq_lenB[offset: offset + batch_size]
        
        batch_sentA = pad(sentA[offset: offset + batch_size], max(batch_seq_lenA))
        batch_sentB = pad(sentB[offset: offset + batch_size], max(batch_seq_lenB))
        batch_label = label[offset: offset + batch_size] if label.any() else []
        
        yield batch_sentA, batch_sentB, batch_seq_lenA, batch_seq_lenB, batch_label
    
    offset = nbatch * batch_size
    
    batch_seq_lenA = seq_lenA[offset: offset + batch_size]
    batch_seq_lenB = seq_lenB[offset: offset + batch_size]
    batch_sentA = pad(sentA[offset: offset + batch_size], max(batch_seq_lenA))
    batch_sentB = pad(sentB[offset: offset + batch_size], max(batch_seq_lenB))
    batch_label = label[offset: offset + batch_size] if label.any() else []
    
    return batch_sentA, batch_sentB, batch_seq_lenA, batch_seq_lenB, batch_label
    
    
def accuracy(probability, label):
    prediction = np.round(probability)
    return np.mean(np.equal(prediction, label))


def prediction_to_csv(test_id, prediction, filepath):
    prediction_df = pd.DataFrame({'Id': test_id, 'Category': prediction})
    prediction_df['Category'] = prediction_df['Category'].map({0: 'agreed', 1: 'disagreed', 2: 'unrelated'})
    prediction_df.to_csv(filepath, sep=',', index=False, encoding='utf-8')


if __name__ == '__main__':
    pass