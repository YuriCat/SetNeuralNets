import numpy as np
import tensorflow as tf


def self_attention(query, keys, out_features, out_heads,
                   projection=True, layer_norm=True, residual=True):
    """
    Args:
        query (torch.Tensor): [batch, element_size, in_features]
        keys  (torch.Tensor): [batch, element_size, in_features]
    Returns:
        torch.Tensor: [batch, element_size, out_features]
    """
    assert out_features % out_heads == 0

    element_size = query.shape[1]
    in_features = query.shape[2]
    chunk_size = out_features // out_heads

    # [batch, element_size, out_features] に変換
    Q = tf.layers.dense(query, out_features, use_bias=False)
    K = tf.layers.dense(keys, out_features, use_bias=False)
    V = tf.layers.dense(keys, out_features, use_bias=False)
    Q = tf.transpose(tf.reshape(Q, [-1, element_size, out_heads, chunk_size]), [0, 2, 1, 3])
    K = tf.transpose(tf.reshape(K, [-1, element_size, out_heads, chunk_size]), [0, 2, 3, 1])
    V = tf.transpose(tf.reshape(V, [-1, element_size, out_heads, chunk_size]), [0, 2, 1, 3])

    # attentionは [batch, out_heads, element_size, element_size]
    attention = tf.matmul(Q, K)
    attention *= float(int(in_features)) ** -0.5
    attention = tf.nn.softmax(attention, axis=-1)

    # output は [batch, element_size, out_heads, chunk_size] に変換
    output = tf.matmul(attention, V)
    output = tf.transpose(output, [0, 2, 1, 3])

    if projection:
        output = tf.reshape(output, [-1, out_features])
        output = tf.layers.dense(output, out_features)

    output = tf.reshape(output, [-1, element_size, out_features])

    if residual:
        output += query
    if layer_norm:
        output = tf.contrib.layers.layer_norm(output, trainable=False)

    return output

def graph(x):
    h = x
    '''filters = 128
    h = tf.nn.relu(self_attention(h, h, filters, 16, residual=False))
    for _ in range(8):
      h = tf.nn.relu(self_attention(h, h, filters, 16))'''
    h = self_attention(h, h, 2, 2, layer_norm=False, residual=False)
    return h

def loss(y, y_):
    return tf.reduce_mean(tf.abs(y - y_))

sess = tf.Session()

elements = 8
x_ = tf.placeholder(tf.float32, [None, elements, 2])
y_ = tf.placeholder(tf.float32, [None, 1, 2])

g = graph(x_)
l = loss(g, y_)
#optimization = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(l)
optimization = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(l)

print([int(np.prod(v.shape)) for v in tf.trainable_variables()])

sess.run(tf.global_variables_initializer())

x = np.random.random((1, elements, 2))
print(sess.run(g, feed_dict={x_:x}))
print(sess.run(g, feed_dict={x_:x[:, ::-1, :]}))

points  = np.random.randn(100, elements, 2)
centers = np.mean(points, axis=1).reshape((-1, 1, 2))

points_v  = np.random.randn(100, elements, 2)
centers_v = np.mean(points_v, axis=1).reshape((-1, 1, 2))

print(points.shape, centers.shape)
print(points[0])
print(centers[0])

for _ in range(10000):
    print(sess.run(l, feed_dict={x_:points, y_:centers}), sess.run(l, feed_dict={x_:points_v, y_:centers_v}))
    print(sess.run(g, feed_dict={x_:points[0:1]}))
    print(centers[0])
    sess.run(optimization, feed_dict={x_:points, y_:centers})
