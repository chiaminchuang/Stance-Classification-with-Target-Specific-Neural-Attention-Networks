import tensorflow as tf
from tensorflow.nn import embedding_lookup, softmax
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.nn import bidirectional_dynamic_rnn, softmax_cross_entropy_with_logits_v2
from tqdm import tqdm_notebook
import os, json

from utils import *

class TAN:
    def __init__(self, config, emb_matrix):
        tf.reset_default_graph()
        self.config = config
        self.num_class = config['num_class']
        self.lstm_unit = config['lstm_unit']
        self.emb_dim = config['emb_dim']
        self.vocab_size = config['vocab_size']
        self.learning_rate = config['learning_rate']
                
        self.sentA = tf.placeholder(tf.int32, shape=(None, None))  # (batch_size, sent_len)
        self.sentB = tf.placeholder(tf.int32, shape=(None, None))  # (batch_size, sent_len)
        self.seq_lenA = tf.placeholder(tf.int32, shape=(None, ))   # (batch_size, )
        self.seq_lenB = tf.placeholder(tf.float32, shape=(None, 1))   # (batch_size, 1)
        self.stance = tf.placeholder(tf.int32, shape=(None, config['num_class']))  # (batach_size, num_class)
        
        emb_shape = (self.vocab_size, self.emb_dim)
        self.emb_matrix = tf.get_variable(shape=emb_shape, 
                                    initializer=tf.constant_initializer(emb_matrix, dtype=tf.float32),
                                    dtype=tf.float32,
                                    trainable=config['emb_trainable'],
                                    name='embedding_matrix')
        
        self.Wa = tf.get_variable('attenion_weight', (self.emb_dim * 2, 1)) 
        self.ba = tf.get_variable('attention_bias', (1, ))
        self.Wo = tf.get_variable('output_weight', (self.lstm_unit * 2, self.num_class))
        self.bo = tf.get_variable('output_bias', (self.num_class, ))        

        
    def embedding_layer(self, sequence):
        return embedding_lookup(self.emb_matrix, sequence)
    
    
    def BiLSTM(self, sequence, seq_len):
        # sequence shape: (batch_size, sent_len, emb_dim)
        # seq_len shape: (batch_size)
        
        cell_fw = LSTMCell(num_units=self.lstm_unit)
        cell_bw = LSTMCell(num_units=self.lstm_unit)
        
        ((output_fw, output_bw), _) = bidirectional_dynamic_rnn(cell_fw, cell_bw, sequence, dtype=tf.float32, sequence_length=seq_len)
        
        # output_fw shape: (batch_size, sent_len, unit)
        # output_bw shape: (batch_size, sent_len, unit)
        context = tf.concat([output_fw, output_bw], axis=2)  # (batch_size, sent_len, unit * 2)
        return context
    
    def attention_layer(self, sequence):
        # sequence shape: (batch_size, sent_len, emb_dim * 2)
        
        def fn(sent):
            # sent shape: (sent_len, emb_dim * 2)
            return tf.matmul(sent, self.Wa) + self.ba  # (sent_len, 1)
        
        # iterate each batch sentence
        return softmax(tf.map_fn(fn, sequence)) # (batch_size, sent_len, 1)

    
    def build(self):
        
        ##### Context Representation #####
        self.x = self.embedding_layer(self.sentA)  # (batch_size, sent_lenA, emb_dim)
        self.h = self.BiLSTM(self.x, self.seq_lenA)     # (batch_size, sent_lenA, lstm_unit * 2)
        
        ##### Target-augmented Embedding #####
        sent_lenA = tf.shape(self.x)[1]
        target_info = self.embedding_layer(self.sentB)  # (batch_size, sent_lenB, emb_dim)
        
        z = tf.reduce_sum(target_info, axis=1)          # (batch_size, emb_dim)
        z = tf.divide(z, self.seq_lenB)                 # (batch_size, emb_dim)
        z = tf.expand_dims(z, axis=1)                   # (batch_size, 1, emb_dim)
        self.z = tf.tile(z, [1, sent_lenA, 1])               # (batch_size, sent_lenA, emb_dim)
        self.e = tf.concat([self.x, self.z], axis=2)                   # (batch_size, sent_lenA, emb_dim * 2)
        
        ##### Target-specific Attention Extraction #####
        self.a = self.attention_layer(self.e)  # (batch_size, sent_len, 1)        
        
        ##### Stance Classification #####
        s = self.h * self.a  # (batch_size, sent_len, lstm_unit * 2)
        self.s = tf.reduce_mean(s, axis=1)  # (batch_size, lsmt_unit * 2)
        p = tf.matmul(self.s, self.Wo) + self.bo  # (batch_size, num_class)
        self.output = softmax(p)
        
        ##### Cross Entropy #####
        cross_entropy = softmax_cross_entropy_with_logits_v2(labels=self.stance, logits=p)  # (batch_size, )
        self.loss = tf.reduce_mean(cross_entropy)  # ()
        
        ##### L2 Regularization #####
        L2_lambda = 0.01
        L2 = L2_lambda * tf.reduce_sum([tf.nn.l2_loss(tf_var) 
                                        for tf_var in tf.trainable_variables() 
                                        if not('bias' in tf_var.name)]) # or 'embedding' in tf_var.name)])
        self.loss += L2
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
    
    def fit(self, train_data, val_data, epoch_size, batch_size, word2index, model_name):
        def learn(data, epoch, mode):
            tn = tqdm_notebook(total=len(data[0]))
            nbatch, epoch_loss, epoch_acc = 0, 0, 0 
            for sentA, sentB, seq_lenA, seq_lenB, label in next_batch(data, batch_size, word2index):
                feed_dict = {
                    self.sentA: sentA,
                    self.sentB: sentB, 
                    self.seq_lenA: seq_lenA,
                    self.seq_lenB: seq_lenB.reshape((-1, 1)),
                    self.stance: label
                }
                if mode == 'train':
                    fetches = [self.loss, self.output, self.optimizer, self.h, self.x]
                    loss, output, _, h, x = self.sess.run(fetches, feed_dict)
                    tn.set_description('Epoch: {}/{}'.format(epoch, epoch_size))
                elif mode == 'validate':                    
                    fetches = [self.loss, self.output]
                    loss, output = self.sess.run(fetches, feed_dict)
                
                acc = accuracy(output, label)
                tn.set_postfix(loss=loss, accuracy=acc, mode=mode)
                tn.update(n=batch_size)
            
            return [loss, acc]
                
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
                
        train_log, val_log = [], []
        print('Train on {} samples, validate on {} samples'.format(len(train_data[0]), len(val_data[0])))
        for epoch in range(1, epoch_size + 1):       
            train_data = shuffle_data(train_data)
            # train
            train_log.append(learn(train_data, epoch, 'train'))

            # validate
            if len(val_data[0]) > 0:
                val_log.append(learn(val_data, epoch, 'validate')) 
        
        self.save(model_name, train_log, val_log)
     
    
    def predict(self, data, word_to_index):
        
        y_empty = np.empty(0)
        batch_size, i = 500, 0
        tn = tqdm_notebook(total=len(data[0]))
        prediction = np.empty((len(data[0]), 3))
        for sentA, sentB, seq_lenA, seq_lenB, _ in next_batch(data, batch_size, word_to_index):
            fetches = [self.output]
            feed_dict = {
                self.sentA: sentA,
                self.sentB: sentB, 
                self.seq_lenA: seq_lenA,
                self.seq_lenB: seq_lenB.reshape((-1, 1)),
            }
            output = self.sess.run(fetches, feed_dict)[0]
            prediction[i * batch_size: i * batch_size + len(output)] = output
            
            tn.set_postfix(mode='predict')
            tn.update(n=batch_size)
            
            i += 1
        
        
        return np.argmax(prediction, axis=1)
    
    
    def save(self, model_name, train_log, val_log):
        model_dir = 'models/{}'.format(model_name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
            os.mkdir('{}/result'.format(model_dir))
        
        # save model
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, '{}/{}.ckpt'.format(model_dir, model_name))
        
        # save config
        with open('{}/config.json'.format(model_dir), 'w', encoding='utf-8') as file:
            json.dump(self.config, file)
            
        # save log
        with open('{}/log'.format(model_dir), 'w', encoding='utf-8') as file:
            for i in range(len(train_log)):
                tlog = train_log[i]
                vlog = val_log[i] if len(val_log) > 0 else []
                log_str = 'Epoch {}: train_acc={}, train_loss={}'.format(i+1, tlog[0], tlog[1])
                log_str += ', val_acc={}, val_loss={}'.format(vlog[0], vlog[1]) if vlog else ''
                file.write(log_str + '\n')
            
        print('Model was saved in {}'.format(save_path))
    
    
    def restore(self, model_path):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, model_path)
            


if __name__ == '__main__':
    pass   