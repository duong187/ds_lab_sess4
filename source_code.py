import os
from collections import defaultdict
import re
import tensorflow as tf
import numpy as np
import random

NUM_CLASSES=20
MAX_DOC_LENGTH=500
unknown_id=0
padding_id=1
def gen_data_and_vocab():
    def collect_data_from(parents_path,newsgroup_list,words_count=None):
        data=[]
        for group_id,newsgroup in enumerate(newsgroup_list):
            dir_path=parents_path+newsgroup+'/'
            files=[(file_name,dir_path+file_name) for file_name in os.listdir(dir_path) if os.path.isfile(dir_path+file_name)]
            files.sort()
            label=group_id
            print('Proccessing: {}-{}'.format(group_id,newsgroup))
            for file_name,file_path in files:
                with open(file_path) as f:
                    text=f.read().lower()
                    words=re.split('\W+',text)
                    if(words_count is not None):
                        for word in words:
                            words_count[word]+=1
                    content=' '.join(words)
                    assert len(content.splitlines())==1
                    data.append(str(label)+'<fff>'+file_name+'<fff>'+content)
        return data            

    words_count=defaultdict(int)
    path='../20news-bydate/'
    parts=[path+dir_name+'/' for dir_name in os.listdir(path) if not os.path.isfile(path+dir_name)]
    train_path,test_path=(parts[0],parts[1]) if 'train' in parts[0] else (parts[1],parts[0])
    newsgroup_list=[newsgroup for newsgroup in os.listdir(train_path)]
    newsgroup_list.sort()
    train_data=collect_data_from(parents_path=train_path,newsgroup_list=newsgroup_list,words_count=words_count)
    vocab=[word for word,freq in zip(words_count.keys(),words_count.values()) if freq>10]
    vocab.sort()
    with open('../vocab_raw.txt','w') as f:
        f.write('\n'.join(vocab))
    test_data=collect_data_from(parents_path=test_path,newsgroup_list=newsgroup_list)
    with open('../w2v/20news_train_raw.txt','w') as f:
        f.write('\n'.join(train_data))
    with open('../w2v/20news_test_raw.txt','w') as f:
        f.write('\n'.join(test_data))

def encode_data(data_path,vocab_path):
    with open (vocab_path) as f:
        vocab=dict([(word,word_id+2)for (word_id,word) in enumerate(f.read().splitlines())])
    with open(data_path) as f:
        documents=[(line.split('<fff>')[0],line.split('<fff>')[1],line.split('<fff>')[2]) for line in f.read().splitlines()]
    encoded_data=[]
    for document in documents:
        label,doc_id,content=document
        words=content.split()[:MAX_DOC_LENGTH] 
        sentence_len=len(words)
        encoded_text=[]
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(str(unknown_id))
        if sentence_len<MAX_DOC_LENGTH:
            num_padding=MAX_DOC_LENGTH-sentence_len
            for _ in range (num_padding):
                encoded_text.append(str(padding_id))
        encoded_data.append(str(label)+'<fff>'+str(doc_id)+'<fff>'+str(sentence_len)+'<fff>'+' '.join(encoded_text))
    file_path='/'.join(data_path.split('/')[:-1])
    file_name='_'.join(data_path.split('/')[-1].split('_')[:-1])+'_encoded.txt'
    with open(file_path+'/'+file_name,'w') as f:
        f.write('\n'.join(encoded_data))

#encode_data('D:/Python/machine_learning/dslab_sess4/w2v/20news_train_raw.txt','D:/Python/machine_learning/dslab_sess4/w2v/vocab_raw.txt')
#encode_data('D:/Python/machine_learning/dslab_sess4/w2v/20news_test_raw.txt','D:/Python/machine_learning/dslab_sess4/w2v/vocab_raw.txt')


class DataReader:
    def __init__(self, data_path, batch_size,vocab_size):
        self._batch_size= batch_size
        self._vocab_size=vocab_size
        with open(data_path) as f:
            lines= f.read().splitlines()
        
        self._data=[]
        self._labels=[]
        self._sentence_lengths=[]
        for data_id, line in enumerate(lines):
            features= line.split('<fff>')
            label, doc_id, sentence_length= int(features[0]), int(features[1]), int(features[2])
            content= features[3].split()
            vector= [int(num) for num in content]
            self._data.append(vector)
            self._labels.append(label)
            self._sentence_lengths.append(sentence_length)

        self._data= np.array(self._data)
        self._labels= np.array(self._labels)
        self._sentence_lengths= np.array(self._sentence_lengths)

        self._num_epoch=0
        self._batch_id=0

    def next_batch(self):
        start= self._batch_id * self._batch_size
        end= start + self._batch_size
        self._batch_id +=1

        if end + self._batch_size > len(self._data):
            end= len(self._data)
            self._num_epoch+=1
            self._batch_id=0
        
            
            indices= list(range(len(self._data)))
            random.seed(2020)
            random.shuffle(indices)
            self._data, self._labels,self._sentence_lengths = self._data[indices], self._labels[indices], self._sentence_lengths[indices]

        return self._data[start:end], self._labels[start:end], self._sentence_lengths[start:end]

class RNN:
    def __init__(self,vocab_size,embedding_size,lstm_size,batch_size):
        self._vocab_size=vocab_size
        self._embedding_size=embedding_size
        self._lstm_size=lstm_size
        self._batch_size=batch_size

        self._data=tf.compat.v1.placeholder(tf.int32,shape=[batch_size,MAX_DOC_LENGTH])
        self._labels=tf.compat.v1.placeholder(tf.int32,shape=[batch_size,])
        self._sentence_lengths=tf.compat.v1.placeholder(tf.int32,shape=[batch_size,])
        
    def embedding_layer(self,indices):
        pretrained_vector=[]
        pretrained_vector.append(np.zeros(self._embedding_size))
        np.random.seed(2018)
        for _ in range(self._vocab_size+1):
            pretrained_vector.append(np.random.normal(loc=0.,scale=1.,size=self._embedding_size))
        pretrained_vector=np.array(pretrained_vector)
        self._embedding_matrix=tf.compat.v1.get_variable(
            name='embedding',
            shape=(self._vocab_size+2,self._embedding_size),
            initializer=tf.constant_initializer(pretrained_vector)
        )
        return tf.nn.embedding_lookup(self._embedding_matrix,indices)
    def LSTM_layer(self,embeddings):
        lstm_cell=tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self._lstm_size)
        zero_state=tf.zeros(shape=(self._batch_size,self._lstm_size))
        initial_state=tf.compat.v1.nn.rnn_cell.LSTMStateTuple(zero_state,zero_state)
        lstm_input=tf.unstack(tf.transpose(embeddings,perm=[1,0,2]))
        lstm_outputs,last_state=tf.compat.v1.nn.static_rnn(
            cell=lstm_cell,
            input=lstm_input,
            initial_state=initial_state,
            sequence_length=self._sentence_lengths
        )
        lstm_outputs=tf.unstack(tf.transpose(lstm_outputs,perm=[1,0,2]))
        lstm_outputs=tf.concat(lstm_outputs,axis=0)
        mask=tf.sequence_mask(
            length=self._sentence_lengths,
            maxlen=MAX_DOC_LENGTH,
            dtype=tf.float32
        )
        mask=tf.concat(tf.unstack(mask,axis=0),axis=0)
        mask=tf.expand_dims(mask,-1)
        lstm_outputs=mask*lstm_outputs
        lstm_outputs_split=tf.split(lstm_outputs,self._batch_size)
        lstm_outputs_sum=tf.reduce_sum(lstm_outputs_split,axis=1)
        lstm_outputs_average=lstm_outputs_sum/tf.expand_dims(tf.cast(self._sentence_lengths,tf.float32),-1)
        return lstm_outputs_average

    def build_graph(self):
        embeddings=self.embedding_layer(self._data)
        LSTM_outputs=self.LSTM_layer(embeddings)

        weights=tf.compat.v1.get_variable(
            name='final_layer_weights',
            shape=(self._lstm_size,NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2018)
        )
        biases=tf.compat.v1.get_variable(
            name='final_layer_biases',
            shape=(NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2018)
        )
        logits=tf.linalg.matmul(weights,LSTM_outputs)+biases
        labels_one_hot=tf.one_hot(indices=self._labels,depth=NUM_CLASSES,dtype=tf.float32)
        loss=tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot,logits=logits)
        loss=tf.reduce_mean(loss)
        probs=tf.nn.softmax(logits)
        predicted_labels=tf.argmax(probs)
        predicted_labels=tf.squeeze(predicted_labels)
        return predicted_labels,loss
        
    def trainer(self,loss,learning_rate):
        train_op=tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op
    
def train_and_evaluate_RNN():
    with open ('../vocab_raw.txt') as f:
        vocab_size=len(f.read().splitlines())
    tf.compat.v1.set_random_seed(2018)
    rnn=RNN(vocab_size=vocab_size,embedding_size=300,lstm_size=50,batch_size=50)
    predicted_labels,loss=rnn.build_graph()
    train_op=rnn.trainer(loss=loss,learning_rate=0.01)
    with tf.compat.v1.Session() as sess:
        train_data_reader=DataReader(data_path='../20news_train_encoded.txt',batch_size=50,vocab_size=vocab_size)
        test_data_reader=DataReader(data_path='../20news_test_encoded.txt',batch_size=50,vocab_size=vocab_size)
        step=0
        MAX_STEP=1000
        sess.run(tf.compat.v1.global_variables_initializer())
        while (step<MAX_STEP):
            next_train_batch=train_data_reader.next_batch()
            train_data,train_labels,train_setence_length=next_train_batch
            plabels_eval,loss_eval,_=sess.run([predicted_labels,loss,train_op],
            feed_dict={
                rnn._data:train_data,
                rnn._labels:train_labels,
                rnn._sentence_lengths:train_setence_length
            }
            )
            step+=1
            if(step%20==0):
                print('Step:{}'.format(step))
                print('Loss:{}',format(loss_eval))
            if train_data_reader._batch_id==0:
                num_true_preds=0
                while True:
                    next_test_batch=test_data_reader.next_batch()
                    test_data,test_labels,test_sentence_length=next_test_batch
                    test_plabels_eval=sess.run(predicted_labels,
                    feed_dict={
                        rnn._data:test_data,
                        rnn._labels:test_labels,
                        rnn._sentence_lengths:test_sentence_length
                    })
                    matches=np.equal(test_plabels_eval,test_labels)
                    num_true_preds+=np.sum(matches.astype(float))
                    if(test_data_reader._batch_id==0):
                        break
                print('Epoch:{}'.format(train_data_reader._num_epoch))
                print('Accuracy:{}'.format(num_true_preds*100/len(test_data_reader._data)))
if __name__=='__main__':
  train_and_evaluate_RNN()








    

