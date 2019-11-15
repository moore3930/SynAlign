
import tensorflow as tf
from nn.directed_gcn import DirectedGCN
import numpy as np

# a = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[2, 3])
# b = tf.constant([[1, 2, 3], [4, 5, 6]])
# c = a * b
# with tf.Session() as sess:
#     print(sess.run(c))

inputs = tf.random.uniform([1, 10, 128], dtype=tf.float32)

adj_indices = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
adj_values = np.ones([9], dtype=np.float32)
adj_shape = [10, 10]
adj = tf.SparseTensor(adj_indices, adj_values, adj_shape)   # [10, 10]

labels_indices = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
labels_values = [0, 1, 2, 0, 1, 2, 0, 1, 2]
labels_shape = [10, 10]
labels = tf.SparseTensor(labels_indices, labels_values, labels_shape)

adj_inv = tf.sparse.transpose(adj)
labels_inv = tf.sparse.transpose(labels)

# labels_indices = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
# labels_values = [0, 1, 2, 0, 1, 2, 0, 1, 2]
# labels_shape = [10, 10]
# labels_inv = tf.SparseTensor(labels_indices, labels_values, labels_shape)

print(adj)
print(adj.values)
print(labels)
print(labels.values)

train_mode = tf.constant(True, dtype=tf.bool)
gcn_layer = DirectedGCN(128, 3, train_mode=train_mode)
re = gcn_layer(inputs, adj, labels, adj_inv, labels_inv)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(re))
    # print(sess.run(adj.values))
    # print(sess.run(labels.values))


