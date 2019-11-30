import tensorflow as tf

# def rm_zeros(pred):
#     pred = tf.cast(pred, tf.float32)
#     # num_non_zero element in every row
#     num_non_zero = tf.count_nonzero(pred, -1)  #[3 2 3]
#     # flat input and remove all zeros
#     flat_pred = tf.reshape(pred, [-1])
#     mask = tf.math.logical_not(tf.equal(flat_pred, tf.zeros_like(flat_pred)))
#     flat_pred_without_zero = tf.boolean_mask(flat_pred, mask) #[2. 3. 4. 1. 5. 2. 3. 1.]
#     # create a ragged tensor and change it to tensor, rows will be padded to max length
#     ragged_pred = tf.RaggedTensor.from_row_lengths(values=flat_pred_without_zero, row_lengths=num_non_zero)
#     paded_pred = ragged_pred.to_tensor(default_value=0.)
#     paded_pred = tf.cast(paded_pred, tf.int32)
#     return paded_pred

def rm_zeros(pred):
    pred = tf.cast(pred, tf.float32)
    # num_non_zero element in every row
    num_non_zero = tf.count_nonzero(pred, -1)  #[3 2 3]
    # flat input and remove all zeros
    flat_pred = tf.reshape(pred, [-1])
    mask = tf.math.logical_not(tf.equal(flat_pred, tf.zeros_like(flat_pred)))
    flat_pred_without_zero = tf.boolean_mask(flat_pred, mask) #[2. 3. 4. 1. 5. 2. 3. 1.]
    # create a ragged tensor and change it to tensor, rows will be padded to max length
    ragged_pred = tf.RaggedTensor.from_row_lengths(values=flat_pred_without_zero, row_lengths=num_non_zero)
    paded_pred = ragged_pred.to_tensor(default_value=0.)
    paded_pred = tf.cast(paded_pred, tf.int32)
    padding = [[0, 0], [0, 64-tf.shape(paded_pred)[1]]]
    paded_pred = tf.pad(paded_pred, padding, 'CONSTANT', constant_values=0)
    return paded_pred

a = tf.constant([[0, 2, 3, 4], [0, 1, 0, 5], [2, 3, 1, 0]])
with tf.Session() as sess:
    print(sess.run(rm_zeros(a)))
