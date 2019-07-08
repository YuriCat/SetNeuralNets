import numpy as np
import tensorflow as tf

class GraphConvLayer:
    def __init__(
            self,
            input_dim,
            output_dim,
            activation=None,
            use_bias=False,
            name="graph_conv"):
        """Initialise a Graph Convolution layer.
        Args:
            input_dim (int): The input dimensionality.
            output_dim (int): The output dimensionality, i.e. the number of
                units.
            activation (callable): The activation function to use. Defaults to
                no activation function.
            use_bias (bool): Whether to use bias or not. Defaults to `False`.
            name (str): The name of the layer. Defaults to `graph_conv`.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.name = name

        with tf.variable_scope(self.name):
            self.w = tf.get_variable(
                name='w',
                shape=(self.input_dim, self.output_dim),
                initializer=tf.initializers.glorot_uniform())

            if self.use_bias:
                self.b = tf.get_variable(
                    name='b',
                    initializer=tf.constant(0.1, shape=(self.output_dim,)))

    def call(self, adj_norm, x, sparse=False):
        x = tf.einsum('ijk,kl->ijl', x, self.w)  # XW
        #x = matmul(x=adj_norm, y=x, sparse=sparse)  # AXW
        x = tf.einsum('jk,ikl->ijl', adj_norm, x)  # AXW

        if self.use_bias:
            x = tf.add(x, self.use_bias)          # AXW + B

        if self.activation is not None:
            x = self.activation(x)                # activation(AXW + B)

        return x

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

# グラフデータ
g0 = np.array([0, 1, 2]).reshape([1, -1, 1])
g1 = np.array([0, 1, 3]).reshape([1, -1, 1])
adj = (np.ones((3, 3)) - np.eye(3)).reshape(3, 3)

print(adj.shape)

G0 = tf.constant(g0, dtype=tf.float32)
G1 = tf.constant(g1, dtype=tf.float32)
A = tf.constant(adj, dtype=tf.float32)

g = GraphConvLayer(1, 1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(G0))
print(sess.run(G1))
print(sess.run(g(A, G0)))
print(sess.run(g(A, G1)))
print(sess.run(g(A, tf.concat([G0, G1], axis=0))))