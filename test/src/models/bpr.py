import tensorflow as tf
import numpy as np
class BPR_MF(object):

    def __init__(self, sess, user_count, item_count, hidden_dim, lr):
      gpu_config = tf.ConfigProto()
      gpu_config.gpu_options.allow_growth = True
      gpu_config.allow_soft_placement = True
      gpu_config.intra_op_parallelism_threads = 16
      gpu_config.inter_op_parallelism_threads = 16

      
      self.sess = sess

      if sess is None:
          self.sess = tf.Session(config=gpu_config)


      
      self.lr = lr
      self.u = tf.placeholder(tf.int32, [None])
      self.i = tf.placeholder(tf.int32, [None])
      self.j = tf.placeholder(tf.int32, [None])

      user_emb_w = tf.get_variable("user_emb_w", [user_count + 1, hidden_dim],
                                   initializer=tf.random_normal_initializer(0, 0.1))
      item_emb_w = tf.get_variable("item_emb_w", [item_count + 1, hidden_dim],
                                   initializer=tf.random_normal_initializer(0, 0.1))
      self.user_emb_w = user_emb_w
      u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)
      i_emb = tf.nn.embedding_lookup(item_emb_w, self.i)

      j_emb = tf.nn.embedding_lookup(item_emb_w, self.j)
      x = tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keepdims=True)
      l2_norm = tf.add_n([tf.reduce_sum(tf.multiply(u_emb, u_emb)),
                          tf.reduce_sum(tf.multiply(i_emb, i_emb)),
                          tf.reduce_sum(tf.multiply(j_emb, j_emb))])

      regulation_rate = 0.0001
      self.bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))
      self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.bprloss)
      u_emb2 = tf.reshape(u_emb, (-1, 1, hidden_dim))
      i_emb2 = tf.reshape(i_emb, (-1, 1, hidden_dim))
      self.pred = tf.matmul(u_emb2, i_emb2, transpose_b=True)

    def train(self, batch_users, batch_items, batch_neg_items):

      feed_dict = {self.u: batch_users,
                   self.i: batch_items,
                   self.j: batch_neg_items}

      _, current_loss = self.sess.run([self.train_op, self.bprloss], feed_dict=feed_dict)

      return current_loss

    def predict(self, data, batch_size=0, verbose=0):      
      users, items = data
      feed_dict = {self.u: users, self.i: items}
      pred = self.sess.run(self.pred, feed_dict=feed_dict)
      pred = pred.reshape([1, len(items)])

      return pred
    def get_userembeddings(self):
      W = self.sess.run(self.user_emb_w)
      return W
    def init(self):

      self.sess.run(tf.global_variables_initializer())



