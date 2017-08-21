import tensorflow as tf
import numpy as np

def init_memory(N, W, R):
    """
    returns the initial values of the memory matrix, usage vector,
    precedence vector, link matrix, read weightings, write weightings,
    and the read vectors
    """

    M0 = tf.fill([N, W], 1e-6)
    u0 = tf.zeros([N])
    p0 = tf.zeros([N])
    L0 = tf.zeros([N, N])
    wr0 = tf.fill([N, R], 1e-6)  # initial read weightings
    ww0 = tf.fill([N], 1e-6)  # initial write weightings
    r0 = tf.fill([W, R], 1e-6)  # initial read vector

    return M0, u0, p0, L0, wr0, ww0, r0


def parse_interface(zeta, N, W, R):
    """
    returns the individual components of the interface vector
    """
    cursor = 0  # keeps track of how far we parsed into zeta
    kr, cursor = tf.reshape(zeta[cursor:cursor + W*R], [W, R]), cursor + W*R
    br, cursor = zeta[cursor:cursor + R], cursor + R
    kw, cursor = tf.reshape(zeta[cursor: cursor + W], [W, 1]), cursor + W
    bw, cursor = zeta[cursor], cursor + 1
    e, cursor = zeta[cursor: cursor + W], cursor + W
    v, cursor = zeta[cursor: cursor + W], cursor + W
    f, cursor = zeta[cursor: cursor + R], cursor + R
    ga, cursor = zeta[cursor], cursor + 1
    gw, cursor = zeta[cursor], cursor + 1
    pi = tf.reshape(zeta[cursor:], [3, R])

    # transforming the parsed components into their correct values
    oneplus = lambda z: 1 + tf.nn.softplus(z)

    e = tf.nn.sigmoid(e)
    f = tf.nn.sigmoid(f)
    ga = tf.nn.sigmoid(ga)
    gw = tf.nn.sigmoid(gw)
    br = oneplus(br)
    bw = oneplus(bw)
    pi = tf.nn.softmax(pi, 0)

    return kr, br, kw, bw, e, v, f, ga, gw, pi


def C(M, k, b):
    """
    Content-based addressing weightings
    """
    M_normalized = tf.nn.l2_normalize(M, 1)
    k_normalized = tf.nn.l2_normalize(k, 0)
    similarity = tf.matmul(M_normalized, k_normalized)

    return tf.nn.softmax(similarity * b, 0)


def ut(u, f, wr, ww):
    """
    returns the updated usage vector given the previous one along with
    free gates and previous read and write weightings
    """
    psi_t = tf.reduce_prod(1 - f * wr, 1)
    return (u + ww - u * ww) * psi_t


def at(ut, N):
    """
    returns the allocation weighting given the updated usage vector
    """
    sorted_ut, free_list = tf.nn.top_k(-1 * ut, N)
    sorted_ut *= -1  # brings the usages to the original positive values

    # the exclusive argument makes the first element in the cumulative
    # product a 1 instead of the first element in the given tensor
    sorted_ut_cumprod = tf.cumprod(sorted_ut, exclusive=True)
    out_of_location_at = (1 - sorted_ut) * sorted_ut_cumprod

    empty_at_container = tf.TensorArray(tf.float32, N)
    full_at_container = empty_at_container.scatter(free_list, out_of_location_at)

    return full_at_container.pack()


def wwt(ct, at, gw, ga):
    """
    returns the upadted write weightings given allocation and content-based
    weightings along with the write and allocation gates
    """
    ct = tf.squeeze(ct)
    return gw * (ga * at + (1 - ga) * ct)


def Lt(L, wwt, p, N):
    """
    returns the updated link matrix given the previous one along with
    the updated write weightings and the previous precedence vector
    """
    def pairwise_add(v):
        """
        returns the matrix of pairwe-adding the elements of v to themselves
        """
        n = v.get_shape().as_list()[0]
        V = tf.concat(1, [v] * n)  # a NxN matrix of duplicates of u along the columns
        return V + V

    # expand dimensions of wwt and p to make matmul behave as outer product
    wwt = tf.expand_dims(wwt, 1)
    p = tf.expand_dims(p, 0)

    I = tf.constant(np.identity(N, dtype=np.float32))
    return ((1 - pairwise_add(wwt)) * L + tf.matmul(wwt, p)) * (1 - I)


def pt(wwt, p):
    """
    returns the updated precedence vector given the new write weightings and
    the previous precedence vector
    """
    return (1 - tf.reduce_sum(wwt)) * p + wwt


def Mt(M, wwt, e, v):
    """
    returns the updated memory matrix given the previous one, the new write
    weightings, and the erase and write vectors
    """
    # expand the dims of wwt, e, and v to make matmul
    # behave as outer product
    wwt = tf.expand_dims(wwt, 1)
    e = tf.expand_dims(e, 0)
    v = tf.expand_dims(v, 0)

    return M * (1 - tf.matmul(wwt, e)) + tf.matmul(wwt, v)


def wrt(wr, Lt, ct, pi):
    """
    returns the updated read weightings given the previous ones, the new link
    matrix, a content-based weighting, and the read modes
    """
    ft = tf.matmul(Lt, wr)
    bt = tf.matmul(Lt, wr, transpose_a=True)

    return pi[0] * bt + pi[1] * ct + pi[2] * ft


def rt(Mt, wrt):
    """
    returns the new read vectors given the new memory matrix and the new read
    weightings
    """
    return tf.matmul(Mt, wrt, transpose_a=True)
