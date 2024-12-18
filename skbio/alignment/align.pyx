# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as cnp
from skbio.alignment import TabularMSA, AlignPath
from skbio.sequence import DNA
from libc.stdint cimport int32_t, uint8_t, INT32_MIN
cnp.import_array()

# Scoring constants
cdef uint8_t GAP = 1
cdef int32_t NEG_INF = INT32_MIN + 100

def align_wrapper(seq1, seq2, subMatrix, gap_open, gap_extend):
    seq1_idx = subMatrix._char_hash[DNA(seq1)._bytes]
    seq2_idx = subMatrix._char_hash[DNA(seq2)._bytes]
    result, score = align(seq1_idx, seq2_idx, subMatrix._data.astype(int), gap_open, gap_extend)
    path = AlignPath.from_bits(np.vstack(result))
    asDNA = [DNA(seq1), DNA(seq2)]
    return TabularMSA.from_path_seqs(path, asDNA), score

cdef align(const cnp.uint8_t[::1] seq1, const cnp.uint8_t[::1] seq2, const cnp.int64_t[:, :] subMatrix, int gap_open, int gap_extend):
    cdef Py_ssize_t m = seq1.shape[0]
    cdef Py_ssize_t n = seq2.shape[0]
    cdef int score
    cdef uint8_t current_matrix
    cdef Py_ssize_t max_len = m + n
    cdef D = np.empty((m + 1, n + 1), dtype=np.int32)
    cdef P = np.empty((m + 1, n + 1), dtype=np.int32)
    cdef Q = np.empty((m + 1, n + 1), dtype=np.int32)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] aligned_seq1 = np.zeros(max_len, dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] aligned_seq2 = np.zeros(max_len, dtype=np.uint8)

    cdef int32_t[:, :] D_view = D
    cdef int32_t[:, :] P_view = P
    cdef int32_t[:, :] Q_view = Q

    align_fill(D_view, P_view, Q_view, subMatrix, seq1, seq2, gap_open, gap_extend)

    cdef Py_ssize_t idx = align_trace(D_view, P_view, Q_view, m, n, max_len, aligned_seq1, aligned_seq2)

    return (aligned_seq1[idx+1:aligned_seq1.size], aligned_seq2[idx+1:aligned_seq2.size]), max(D_view[m, n], P_view[m, n], Q_view[m, n])

cdef void align_fill(int32_t[:, :] D_view, int32_t[:, :] P_view, int32_t[:, :] Q_view, const cnp.int64_t[:, :] subMatrix, const cnp.uint8_t[::1] seq1, const cnp.uint8_t[::1] seq2, const int GAP_OPEN_PENALTY, const int GAP_EXTEND_PENALTY) noexcept nogil:
    cdef Py_ssize_t i, j, idx
    cdef Py_ssize_t m = seq1.shape[0]
    cdef Py_ssize_t n = seq2.shape[0]

    D_view[0, 0] = 0
    P_view[0, 0] = NEG_INF
    Q_view[0, 0] = NEG_INF

    for i in range(1, m + 1):
        D_view[i, 0] = GAP_OPEN_PENALTY + GAP_EXTEND_PENALTY * (i-1)
        Q_view[i, 0] = NEG_INF

    for j in range(1, n + 1):
        D_view[0, j] = GAP_OPEN_PENALTY + GAP_EXTEND_PENALTY * (j-1)
        P_view[0, j] = NEG_INF

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            P_view[i, j] = max(
                D_view[i - 1, j] + GAP_OPEN_PENALTY + GAP_EXTEND_PENALTY * 0,
                P_view[i - 1, j] + GAP_EXTEND_PENALTY
            )

            Q_view[i, j] = max(
                D_view[i, j - 1] + GAP_OPEN_PENALTY + GAP_EXTEND_PENALTY * 0,
                Q_view[i, j - 1] + GAP_EXTEND_PENALTY
            )

            D_view[i, j] = max(
                D_view[i - 1, j - 1] + subMatrix[seq1[i-1], seq2[j-1]],
                P_view[i, j],
                Q_view[i, j]
            )

cdef int align_trace(int32_t[:, :] D_view, int32_t[:, :] P_view, int32_t[:, :] Q_view, Py_ssize_t m, Py_ssize_t n, Py_ssize_t max_len, aligned_seq1, aligned_seq2) noexcept :
    cdef Py_ssize_t i, j, idx
    cdef uint8_t[::1] aligned_seq1_view = aligned_seq1
    cdef uint8_t[::1] aligned_seq2_view = aligned_seq2

    if D_view[m, n] >= P_view[m, n] and D_view[m, n] >= Q_view[m, n]:
        current_matrix = 0
    elif P_view[m, n] >= Q_view[m, n]:
        current_matrix = 1
    else:
        current_matrix = 2

    # Traceback
    i = m
    j = n
    idx = max_len - 1

    while i > 0 or j > 0:
        if current_matrix == 0 and D_view[i-1][j-1] >= P_view[i][j] and D_view[i-1][j-1] >= Q_view[i][j]:
            i -= 1
            j -= 1
        elif (current_matrix == 0 and P_view[i][j] >= Q_view[i][j]) or current_matrix == 1:
            aligned_seq2_view[idx] = GAP
            if D_view[i-1][j] > P_view[i-1][j]:
                current_matrix = 0
            else:
                current_matrix = 1
            i -= 1
        elif (current_matrix == 0 and Q_view[i][j] > P_view[i][j]) or current_matrix == 2:
            aligned_seq1_view[idx] = GAP
            if D_view[i][j-1] > Q_view[i][j-1]:
                current_matrix = 0
            else:
                current_matrix = 2    
            j -= 1
        idx -= 1
        
    while i > 0:
        aligned_seq2_view[idx] = GAP
        i -= 1
        idx -= 1
    while j > 0:
        aligned_seq1_view[idx] = GAP
        j -= 1
        idx -= 1

    return idx