import numpy as np
import tensorflow as tf

def single_head_self_attention(query, keys, out_features, mask=None, self_specialization=True,
                               projection=False, layer_norm=True, residual=True):
    """
    Args:
        query (torch.Tensor): [batch, element_size, in_features]
        keys  (torch.Tensor): [batch, element_size, in_features]
    Returns:
        torch.Tensor: [batch, element_size, out_features]
    """

    element_size = query.shape[1]
    in_features = query.shape[2]

    # [batch, element_size, out_features] に変換
    Q = tf.layers.dense(query, out_features, use_bias=False)
    K = tf.layers.dense(keys, out_features, use_bias=False)
    V = tf.layers.dense(keys, out_features, use_bias=False)
    K = tf.transpose(K, [0, 2, 1])

    # attentionは [batch, element_size, element_size]
    attention = tf.matmul(Q, K)
    if self_specialization:
        attention *= 1 - tf.eye(int(element_size))
        attention += tf.matrix_diag(tf.squeeze(tf.layers.dense(query, 1), -1))
    attention *= float(int(in_features)) ** -0.5
    if mask is not None:
        # 大きな負の値にしてsoftmaxで0が付くように
        attention += tf.expand_dims(1 - mask, -2) * -2e32
    attention = tf.nn.softmax(attention, axis=-1)
    if mask is not None:
        attention *= tf.expand_dims(mask, -1)

    # output は [batch, element_size, out_features] に変換
    output = tf.matmul(attention, V)

    if projection:
        output = tf.layers.dense(output, out_features)
    if residual:
        output += query
    if layer_norm:
        output = tf.contrib.layers.layer_norm(output, trainable=False)

    return output

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
    output = tf.reshape(tf.transpose(output, [0, 2, 1, 3]), [-1, element_size, out_features])

    if projection:
        output = tf.layers.dense(output, out_features)
    if residual:
        output += query
    if layer_norm:
        output = tf.contrib.layers.layer_norm(output, trainable=False)

    return output

def graph(x, mask):
    h = x
    filters = 128
    #h = tf.nn.relu(self_attention(h, h, filters, 16, mask, layer_norm=False, residual=False))
    #for _ in range(2):
    #  h = tf.nn.relu(self_attention(h, h, filters, 16, layer_norm=False))
    #h = self_attention(h, h, 2, 2, layer_norm=False, residual=False)
    h = single_head_self_attention(h, h, 2, mask, layer_norm=False, residual=False)
    return h

def dot(a, b):
    return tf.squeeze(tf.matmul(tf.expand_dims(a, -2), tf.expand_dims(b, -1)), -1)

def mask_max(a, b):
    pass #return a - 

def loss(y, y_, mask):
    norm = tf.pow(tf.reduce_sum(tf.square(y - y_), axis=-1) + 1e-16, 0.5)
    #norm = tf.reduce_sum(tf.square(y - y_), axis=-1)
    return tf.reduce_mean(norm * mask)

sess = tf.Session()

elements = 3
x_ = tf.placeholder(tf.float32, [None, elements, 2])
y_ = tf.placeholder(tf.float32, [None, elements, 2])
mask_ = tf.placeholder(tf.float32, [None, elements])

g = graph(x_, mask_)
l = loss(g, y_, mask_)
#optimization = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(l)
optimization = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(l)

print([int(np.prod(v.shape)) for v in tf.trainable_variables()])

sess.run(tf.global_variables_initializer())

x = np.random.random((1, elements, 2))
msk = np.random.randint(0, 2, (1, elements))
print(x)
print(msk)
print(sess.run(g, feed_dict={x_:x, mask_:msk}))
print(sess.run(g, feed_dict={x_:x[:, ::-1, :], mask_:msk[:,::-1]}))

points   = np.clip(np.random.randn(100, elements, 2), -10, 10)
points_v = np.clip(np.random.randn(100, elements, 2), -10, 10)
mask     = np.random.randint(0, 2, (100, elements))
mask_v   = np.random.randint(0, 2, (100, elements))

def mask_mean(a, mask):
    sum_mask = np.expand_dims(mask.sum(axis=-1), -1)
    mean_mask = np.true_divide(mask, sum_mask)
    mean_mask[~np.isfinite(mean_mask)] = 0
    return np.sum(a * np.expand_dims(mean_mask, -1), axis=-2)

# 恒等写像
def identity(points, mask):
    return points

# 重心
def center(points, mask):
    return np.tile(mask_mean(points, mask).reshape((-1, 1, 2)), [1, elements, 1])

# 最近傍点
def nearest(points, mask):
    distance = np.stack([np.linalg.norm(points[:,i:(i+1),:] - points, axis=-1) for i in range(elements)], axis=1)
    nearest_idx = np.argmin(distance + np.diag(np.ones(elements) * float('inf')), axis=-1)
    # https://stackoverflow.com/questions/23435782/numpy-selecting-specific-column-index-per-row-by-using-a-list-of-indexes
    return np.stack([points[np.arange(len(points)), nearest_idx[:,i]] for i in range(elements)], axis=1)

func = nearest
x, y = points, func(points, mask)
x_v, y_v = points_v, func(points_v, mask_v)

for _ in range(10000):
    loss_train = sess.run(l, feed_dict={x_:x, y_:y, mask_:mask})
    loss_val = sess.run(l, feed_dict={x_:x_v, y_:y_v, mask_:mask_v})
    print(loss_train, loss_val)
    if np.isnan(loss_train):
        for i in range(100):
            print(x[i])
            print(mask[i])
            print(y[i])
        input()
    print(x[:1])
    print(mask[:1])
    print(sess.run(g, feed_dict={x_:x[:1], mask_:mask[:1]}))
    print(y[:1])
    sess.run(optimization, feed_dict={x_:x, y_:y, mask_:mask})
