from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
import tensorflow as tf
import numpy as np
import sys
import time


class HParam():
    def __init__(self):
        self.batch_size = 32
        self.n_epoch = 100
        self.learning_rate = 0.01
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.grad_clip = 5

        self.state_size = 100
        self.num_layers = 3
        self.seq_length = 20
        self.log_dir = './logs'


class DataGenerator():
    def __init__(self, datafiles, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        with open(datafiles, encoding='utf-8') as f:
            self.data = f.read()
        
        self.total_len = len(self.data) # total data length
        self.words = list(set(self.data))
        self.words.sort()
         # vocabulary
        self.vocab_size = len(self.words) # vocabulary size
        print('Vocabulary Size: ', self.vocab_size)
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}
        
        # pointer position to generate current batch
        self._pointer = 0
        
    
    def char2id(self, c):
        return self.char2id_dict[c]


    def id2char(self, id):
        return self.id2char_dict[id]


    def next_batch(self):
        x_batches = []
        y_batches = []
        for i in range(self.batch_size):
            if self._pointer + self.seq_length + 1 >= self.total_len: 
                self._pointer = 0
            bx = self.data[self._pointer: self._pointer + self.seq_length]
            by = self.data[self._pointer + 1 : self._pointer + self.seq_length + 1]
            self._pointer += self.seq_length # update pointer position

            # convert to ids
            bx = [self.char2id(c) for c in bx]
            by = [self.char2id(c) for c in by]
            x_batches.append(bx)
            y_batches.append(by)

        return x_batches, y_batches

    
        

class Model():
    """
    The core recurrent neural network model.
    """

    def __init__(self, args, data, infer=False):
        if infer:
            args.batch_size = 1
            args.seq_length = 1
        with tf.name_scope('inputs'):
            self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.target_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        with tf.name_scope('model'):
            self.cell = rnn_cell.BasicLSTMCell(args.state_size)
            self.cell = rnn_cell.MultiRNNCell([self.cell] * args.num_layers)
            self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)

            with tf.variable_scope('rnnlm'):
                w = tf.get_variable('softmax_w', [args.state_size, data.vocab_size])
                b = tf.get_variable('softmax_b', [data.vocab_size])
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable('embedding', [data.vocab_size, args.state_size])
                    inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            outputs, last_state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self.initial_state)

        with tf.name_scope('loss'):
            output = tf.reshape(outputs, [-1, args.state_size])

            self.logits = tf.matmul(output, w) + b
            self.probs = tf.nn.softmax(self.logits)
            self.last_state = last_state

            targets = tf.reshape(self.target_data, [-1])
            loss = seq2seq.sequence_loss_by_example([self.logits],
                                                    [targets],
                                                    [tf.ones_like(targets, dtype=tf.float32)])
            self.cost = tf.reduce_sum(loss)/args.batch_size      
            tf.scalar_summary('loss', self.cost)
        with tf.name_scope('optimize'):
            self.lr = tf.placeholder(tf.float32, [])
            tf.scalar_summary('learning_rate', self.lr)

            optimizer = tf.train.AdamOptimizer(self.lr)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.cost, tvars)
            for g in grads:
                tf.histogram_summary(g.name, g)
            grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.merged_op = tf.merge_all_summaries()

    

def sample(data, model, num=400):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint('./')
        print(ckpt)
        saver.restore(sess, ckpt)

        # initial phrase to warm RNN
        prime = u'你要离开我知道很简单'
        state = sess.run(model.cell.zero_state(1, tf.float32))

        for word in prime[:-1]:
            x = np.zeros((1,1))
            x[0,0] = data.char2id(word)
            feed = {model.input_data: x, model.initial_state: state}
            state = sess.run(model.last_state, feed)

        word = prime[-1]
        lyrics = prime
        for i in range(num):
            x = np.zeros([1,1])
            x[0,0] = data.char2id(word)
            feed_dict={model.input_data: x, model.initial_state: state}
            probs, state = sess.run([model.probs, model.last_state], feed_dict)
            p = probs[0]
            word = data.id2char(np.argmax(p))
            print(word, end='')
            sys.stdout.flush()
            time.sleep(0.05)
            lyrics += word
        return lyrics

        

def main(infer):
    
    args = HParam()
    data = DataGenerator('JayLyrics.txt', args)
    model = Model(args, data, infer=infer)

    if infer:
        sample(data, model, 1000)
    else:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            writer = tf.train.SummaryWriter(args.log_dir, sess.graph)
  
            max_iter = args.n_epoch * (data.total_len // args.seq_length) // args.batch_size
            for i in range(max_iter):
                learning_rate = args.learning_rate * (args.decay_rate ** (i//args.decay_steps))
                x_batch, y_batch = data.next_batch()
                feed_dict={model.input_data: x_batch, model.target_data: y_batch, model.lr: learning_rate}
                train_loss, summary, _, _ = sess.run([model.cost, model.merged_op, model.last_state, model.train_op],
                                                     feed_dict)

                if i % 10 == 0:
                    writer.add_summary(summary, global_step=i)
                    print('Step:{}/{}, training_loss:{:4f}'.format(i, max_iter, train_loss))
                if i % 10000 == 0 or (i+1) == max_iter:
                    saver.save(sess, 'lyrics_model', global_step=i)


if __name__ == '__main__':
    msg = """
    Usage:
    Training: 
        python3 gen_lyrics.py 0
    Sampling:
        python3 gen_lyrics.py 1
    """

    if len(sys.argv) == 2:
        infer = int(sys.argv[-1])
        print('--Sampling--' if infer else '--Training--')
        main(infer)
    else:
        print(msg)
        sys.exit(1)
        
            



