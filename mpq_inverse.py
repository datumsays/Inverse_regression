from gmpy2 import mpz,mpq,mpfr,mpc
import numpy as np

size = 105
foo = np.array([[mpq(v) for v in r]
                for r in np.random.randn(size, size)])

def invert_gaussian_mpq(matrix):
    
    mat = np.array(matrix)
    identity = np.array([[mpq(v) for v in r] for r in np.identity(len(mat))])
    m_id = np.concatenate([mat, identity], axis=1)
    size = len(mat[0])
    
    for c_i in xrange(size - 1):
        swap = np.argmax(np.abs(m_id[c_i:, c_i])) + c_i
        m_id[[swap, c_i]] = m_id[[c_i, swap]]
        row = m_id[c_i, :] / m_id[c_i, c_i]
        for r_i in xrange(c_i+1, size):
            del_row = row * m_id[r_i, c_i]
            m_id[r_i, :] -= del_row

    inv_diag = np.diag((m_id.diagonal() ** -1))
    m_id = inv_diag.dot(m_id)
    
    for c_i in xrange(size - 1, 0, -1):
        row = m_id[c_i, :] / m_id[c_i, c_i]
        for r_i in xrange(c_i - 1, -1, -1):
            del_row = row * m_id[r_i, c_i]
            m_id[r_i, :] -= del_row

    output = m_id[:, len(mat):]
    
    return output

# Assert that it works, should print the sum of the identity matrix, which
# is the size of the matrix `size`
# print np.sum(invert_gaussian_mpq(foo).dot(foo))
