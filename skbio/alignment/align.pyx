# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as cnp
from skbio.alignment import TabularMSA, AlignPath
from skbio.sequence import DNA
from numpy cimport int32_t, uint8_t, int8_t, int64_t, float32_t
cnp.import_array()

#TODO: float support, always output float!
#TODO: float_t/int_t or float/int?
#TODO: just add local swap
#TODO: local alignnment where no negatives in D and traceback from max to 0.
#TODO: add verbose mode

# Scoring constants
cdef uint8_t GAP = 1
cdef uint8_t GAP_SCORE = 255
cdef int32_t NEG_INF = -2147483648 + 10

cdef struct Index:
    Py_ssize_t i 
    Py_ssize_t j

cdef struct TracebackRes:
    int8_t score
    Py_ssize_t idx
    Index start
    Index end

# move it to a python file and work with arg_pass
def align_wrapper(seq1, seq2, subMatrix, gap_open, gap_extend, scope):
    # print(gap_open, gap_extend, scope, seq1, seq2)
    #TODO: DNA conversion could cause overhead
    cdef uint8_t local = scope.lower() == "local"
    asDNA = [DNA(seq1), DNA(seq2)]
    seq1_idx = subMatrix._char_hash[asDNA[0]._bytes]
    seq2_idx = subMatrix._char_hash[asDNA[1]._bytes]
    max_len = seq1_idx.shape[0] + seq2_idx.shape[0]
    aligned_seq1 = np.zeros(max_len, dtype=np.uint8)
    aligned_seq2 = np.zeros(max_len, dtype=np.uint8)

    # gap_open = gap_open - gap_extend

    args = (seq1_idx, seq2_idx, aligned_seq1, aligned_seq2, subMatrix._data.astype(int), gap_open, gap_extend)
    if(local):
        args += (local_align_fill, local_align_trace)
    else:
        args += (global_align_fill, global_align_trace)

    res = arg_pass(*args)

    result = (aligned_seq1[res['idx']+1:aligned_seq1.size], aligned_seq2[res['idx']+1:aligned_seq2.size])
    if result[0].size == 0 and result[1].size == 1:
        raise Exception("No local alignment found!")
    path = AlignPath.from_bits(np.vstack(result))
    if(local):
        asDNA = [DNA(seq1[res['start']['i']:res['end']['i']]), DNA(seq2[res['start']['j']:res['end']['j']])]
    # return path instead of MSA, convert to biopy.
    return TabularMSA.from_path_seqs(path, asDNA), res['score']

def arg_pass(const uint8_t[::1] seq1, const uint8_t[::1] seq2, cnp.ndarray aligned_seq1, cnp.ndarray aligned_seq2, const int64_t[:, ::1] subMatrix, const int8_t gap_open, const int8_t gap_extend, fill_func, trace_func):
    return align_main(seq1, seq2, aligned_seq1, aligned_seq2, subMatrix, gap_open, gap_extend, fill_func, trace_func)

cdef TracebackRes align_main(const uint8_t[::1] seq1, const uint8_t[::1] seq2, cnp.ndarray aligned_seq1, cnp.ndarray aligned_seq2, const int64_t[:, ::1] subMatrix, const int8_t gap_open, const int8_t gap_extend, fill_func, trace_func) noexcept:
    cdef Py_ssize_t m = seq1.shape[0]
    cdef Py_ssize_t n = seq2.shape[0]
    cdef Py_ssize_t idx
    cdef TracebackRes res
    cdef uint8_t score
    cdef Py_ssize_t max_len = m + n
    # cdef D = np.empty((m + 1, n + 1), dtype=np.int32)
    # cdef P = np.empty((m + 1, n + 1), dtype=np.int32)
    # cdef Q = np.empty((m + 1, n + 1), dtype=np.int32)
    cdef D = np.zeros((m + 1, n + 1), dtype=np.int32)
    cdef P = np.zeros((m + 1, n + 1), dtype=np.int32)
    cdef Q = np.zeros((m + 1, n + 1), dtype=np.int32)

    cdef int32_t[:, ::1] D_view = D
    cdef int32_t[:, ::1] P_view = P
    cdef int32_t[:, ::1] Q_view = Q

    cdef uint8_t[::1] aligned_seq1_view = aligned_seq1
    cdef uint8_t[::1] aligned_seq2_view = aligned_seq2

    cdef Index loc = Index(i = m, j = n)

    fill_func(D_view, P_view, Q_view, subMatrix, seq1, seq2, gap_open, gap_extend)

    # print("P_view:")
    # for a in P_view:
    #     for b in a:
    #         c = str(b)
    #         if abs(b) < 100:
    #             c = ' ' + c
    #         if abs(b) < 10:
    #             c = ' ' + c
    #         if b >= 0:
    #             c = ' ' + c
    #         print(c, end=' ')
    #     print()
    # print("D_view:")
    # for a in D_view:
    #     for b in a:
    #         c = str(b)
    #         if abs(b) < 100:
    #             c = ' ' + c
    #         if abs(b) < 10:
    #             c = ' ' + c
    #         if b >= 0:
    #             c = ' ' + c
    #         print(c, end=' ')
    #     print()
    # print("Q_view")
    # for a in Q_view:
    #     for b in a:
    #         c = str(b)
    #         if abs(b) < 100:
    #             c = ' ' + c
    #         if abs(b) < 10:
    #             c = ' ' + c
    #         if b >= 0:
    #             c = ' ' + c
    #         print(c, end=' ')
    #     print()
    
    res = trace_func(D_view, P_view, Q_view, loc, max_len, aligned_seq1_view, aligned_seq2_view, seq1, seq2, subMatrix, gap_open, gap_extend)
    return res

cdef void global_align_fill(int32_t[:, :] D_view, int32_t[:, :] P_view, int32_t[:, :] Q_view, const cnp.int64_t[:, :] subMatrix, const cnp.uint8_t[::1] seq1, const cnp.uint8_t[::1] seq2, const int8_t GAP_OPEN_PENALTY, const int8_t GAP_EXTEND_PENALTY) noexcept :
    cdef Py_ssize_t i, j
    cdef Py_ssize_t m = seq1.shape[0]
    cdef Py_ssize_t n = seq2.shape[0]

    D_view[0, 0] = 0
    P_view[0, 0] = NEG_INF
    Q_view[0, 0] = NEG_INF

    #TODO: look at numpy arrange/line space to fill the first row/col for D, Q, and P (even spacing)

    for i in range(1, m + 1):
        D_view[i, 0] = GAP_OPEN_PENALTY + GAP_EXTEND_PENALTY * i
        Q_view[i, 0] = NEG_INF
        P_view[i, 0] = 0

    #TODO: can set i,0 to 0,j using numpy to improve performance more!

    for j in range(1, n + 1):
        D_view[0, j] = GAP_OPEN_PENALTY + GAP_EXTEND_PENALTY * j
        P_view[0, j] = NEG_INF
        Q_view[0, j] = 0

    #TODO: split it here, initial fill vs rest of fill
    #TODO: look into numpy.nditer for iteration through the matrix

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            P_view[i, j] = max(
                D_view[i - 1, j] + GAP_OPEN_PENALTY + GAP_EXTEND_PENALTY,# if D_view[i - 1, j] > P_view[i - 1, j] else NEG_INF,
                P_view[i - 1, j] + GAP_EXTEND_PENALTY
            )

            Q_view[i, j] = max(
                D_view[i, j - 1] + GAP_OPEN_PENALTY + GAP_EXTEND_PENALTY,# if D_view[i, j - 1] > Q_view[i, j - 1] else NEG_INF,
                Q_view[i, j - 1] + GAP_EXTEND_PENALTY
            )

            D_view[i, j] = max(
                D_view[i - 1, j - 1] + subMatrix[seq1[i-1], seq2[j-1]],
                P_view[i, j],
                Q_view[i, j]
            )

cdef void local_align_fill(int32_t[:, :] D_view, int32_t[:, :] P_view, int32_t[:, :] Q_view, const cnp.int64_t[:, :] subMatrix, const cnp.uint8_t[::1] seq1, const cnp.uint8_t[::1] seq2, const int8_t GAP_OPEN_PENALTY, const int8_t GAP_EXTEND_PENALTY) noexcept nogil:
    cdef Py_ssize_t i, j, idx
    cdef Py_ssize_t m = seq1.shape[0]
    cdef Py_ssize_t n = seq2.shape[0]

    D_view[0, 0] = 0
    P_view[0, 0] = NEG_INF
    Q_view[0, 0] = NEG_INF

    #TODO: look at numpy arrange/line space to fill the first row/col for D, Q, and P (even spacing)

    for i in range(1, m + 1):
        D_view[i, 0] = 0
        Q_view[i, 0] = NEG_INF
        P_view[i, 0] = 0

    #TODO: can set i,0 to 0,j using numpy to improve performance more!

    for j in range(1, n + 1):
        D_view[0, j] = 0
        P_view[0, j] = NEG_INF
        Q_view[0, j] = 0

    #TODO: split it here, initial fill vs rest of fill
    #TODO: look into numpy.nditer for iteration through the matrix

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            P_view[i, j] = max(
                D_view[i - 1, j] + GAP_OPEN_PENALTY,
                P_view[i - 1, j] + GAP_EXTEND_PENALTY
            )

            Q_view[i, j] = max(
                D_view[i, j - 1] + GAP_OPEN_PENALTY,
                Q_view[i, j - 1] + GAP_EXTEND_PENALTY
            )

            D_view[i, j] = max(
                D_view[i - 1, j - 1] + subMatrix[seq1[i-1], seq2[j-1]],
                P_view[i, j],
                Q_view[i, j],
                0
            )

cdef TracebackRes global_align_trace(int32_t[:, :] D_view, int32_t[:, :] P_view, int32_t[:, :] Q_view, Index loc, Py_ssize_t max_len, uint8_t[::1] aligned_seq1_view, uint8_t[::1] aligned_seq2_view, const cnp.uint8_t[::1] seq1, const cnp.uint8_t[::1] seq2, const cnp.int64_t[:, :] subMatrix, const int8_t GAP_OPEN_PENALTY, const int8_t GAP_EXTEND_PENALTY) noexcept:
    cdef int8_t score
    cdef uint8_t current_matrix
    cdef TracebackRes res
    cdef Py_ssize_t idx = max_len - 1

    if D_view[loc.i, loc.j] >= P_view[loc.i, loc.j] and D_view[loc.i, loc.j] >= Q_view[loc.i, loc.j]:
        score = D_view[loc.i, loc.j]
        current_matrix = 0
    elif P_view[loc.i, loc.j] >= Q_view[loc.i, loc.j]:
        score = P_view[loc.i, loc.j]
        current_matrix = 1
    else:
        score = Q_view[loc.i, loc.j]
        current_matrix = 2

    while loc.i > 0 and loc.j > 0:
        # print(loc.i, loc.j, current_matrix)
        #TODO: store them in var if needed
        if current_matrix == 0:
            if D_view[loc.i, loc.j] == D_view[loc.i-1, loc.j-1] + subMatrix[seq1[loc.i-1], seq2[loc.j-1]]:
                # print('diag', D_view[loc.i, loc.j], D_view[loc.i-1, loc.j-1] + subMatrix[seq1[loc.i-1], seq2[loc.j-1]], P_view[loc.i, loc.j], Q_view[loc.i, loc.j])
                loc.i -= 1
                loc.j -= 1
            elif D_view[loc.i, loc.j] == P_view[loc.i, loc.j]:
                current_matrix = 1
                continue
            else:
                current_matrix = 2
                continue
        elif current_matrix == 1:
            # print('up', P_view[loc.i, loc.j], D_view[loc.i-1, loc.j] + GAP_OPEN_PENALTY + GAP_EXTEND_PENALTY, P_view[loc.i-1, loc.j] + GAP_EXTEND_PENALTY)
            aligned_seq2_view[idx] = GAP
            if P_view[loc.i, loc.j] == D_view[loc.i-1, loc.j] + GAP_OPEN_PENALTY + GAP_EXTEND_PENALTY:
                current_matrix = 0
            loc.i -= 1
        else:
            # print('side', Q_view[loc.i, loc.j], D_view[loc.i, loc.j-1] + GAP_OPEN_PENALTY + GAP_EXTEND_PENALTY, Q_view[loc.i, loc.j-1] + GAP_EXTEND_PENALTY)
            aligned_seq1_view[idx] = GAP
            if Q_view[loc.i, loc.j] == D_view[loc.i, loc.j-1] + GAP_OPEN_PENALTY + GAP_EXTEND_PENALTY:
                current_matrix = 0 
            loc.j -= 1
        idx -= 1
        # print()
    
    # print(loc.i, loc.j)
        
    # could sub while loop because we know they're all gaps
    while loc.i > 0:
        aligned_seq2_view[idx] = GAP
        loc.i -= 1
        idx -= 1
    while loc.j > 0:
        aligned_seq1_view[idx] = GAP
        loc.j -= 1
        idx -= 1
    
    # print(loc.i, loc.j)
    
    res.score = score
    res.idx = idx
    return res

cdef TracebackRes local_align_trace(int32_t[:, :] D_view, int32_t[:, :] P_view, int32_t[:, :] Q_view, Index loc, Py_ssize_t max_len, uint8_t[::1] aligned_seq1_view, uint8_t[::1] aligned_seq2_view, const cnp.uint8_t[::1] seq1, const cnp.uint8_t[::1] seq2, const cnp.int64_t[:, :] subMatrix, const int8_t GAP_OPEN_PENALTY, const int8_t GAP_EXTEND_PENALTY) noexcept:
    cdef Index P_loc, D_loc, Q_loc, end
    cdef uint8_t current_matrix, score
    cdef TracebackRes res
    cdef Py_ssize_t idx = max_len - 1

    # change to a double for loop maybe to get max, maybe move this calculation to the outside
    P_loc.i, P_loc.j = np.unravel_index(np.argmax(P_view), np.shape(P_view))
    D_loc.i, D_loc.j = np.unravel_index(np.argmax(D_view), np.shape(D_view))
    Q_loc.i, Q_loc.j = np.unravel_index(np.argmax(Q_view), np.shape(Q_view))

    if D_view[D_loc.i, D_loc.j] >= P_view[P_loc.i, P_loc.j] and D_view[D_loc.i, D_loc.j] >= Q_view[Q_loc.i, Q_loc.j]:
        current_matrix = 0
        score = D_view[D_loc.i, D_loc.j]
        loc = D_loc
    elif P_view[P_loc.i, P_loc.j] >= Q_view[Q_loc.i, Q_loc.j]:
        current_matrix = 1
        score = P_view[P_loc.i, P_loc.j]
        loc = P_loc
    else:
        current_matrix = 2
        score = Q_view[Q_loc.i, Q_loc.j]
        loc = Q_loc
    
    end = loc

    while loc.i > 0 and loc.j > 0:
        #TODO: outside if/elif/else just for comparing current_matrix, then compare P_Q_D inside and store them in var if needed
        if current_matrix == 0:
            if D_view[loc.i-1, loc.j-1] >= P_view[loc.i, loc.j] and D_view[loc.i-1, loc.j-1] >= Q_view[loc.i, loc.j]:
                loc.i -= 1
                loc.j -= 1
            elif P_view[loc.i, loc.j] >= Q_view[loc.i, loc.j]:
                aligned_seq2_view[idx] = GAP
                if D_view[loc.i-1, loc.j] <= P_view[loc.i-1, loc.j]:
                    current_matrix = 1
                loc.i -= 1
            else:
                aligned_seq1_view[idx] = GAP
                if D_view[loc.i, loc.j-1] <= Q_view[loc.i, loc.j-1]:
                    current_matrix = 2
                loc.j -= 1
            if D_view[loc.i, loc.j] == 0:
                idx -= 1
                break
        elif current_matrix == 1:
            aligned_seq2_view[idx] = GAP
            if D_view[loc.i-1, loc.j] > P_view[loc.i-1, loc.j]:
                current_matrix = 0
            loc.i -= 1
        else:
            aligned_seq1_view[idx] = GAP
            if D_view[loc.i, loc.j-1] > Q_view[loc.i, loc.j-1]:
                current_matrix = 0 
            loc.j -= 1
        idx -= 1

    res.score = score
    res.idx = idx
    res.start = loc
    res.end = end
    return res

def align_score(seq1, seq2, subMatrix, gap_open, gap_extend):
    # Has sequences through substitution matrix
    seq1_idx = subMatrix._char_hash[DNA(seq1)._bytes]
    seq2_idx = subMatrix._char_hash[DNA(seq2)._bytes]
    # Run though helper function to get ouput
    return _align_score(seq1_idx, seq2_idx, subMatrix._data.astype(np.float32), gap_open, gap_extend)

cdef float32_t _align_score(const uint8_t[::1] seq1, const uint8_t[::1] seq2, const float32_t[:, :] subMatrix, const int8_t gap_open, const int8_t gap_extend) noexcept nogil:
    # Initialize  variables
    cdef uint8_t i
    cdef uint8_t state = 0
    cdef float32_t score = 0

    # Iterate through sequences
    for i in range(seq1.shape[0]):
        # Gap in seq1
        if seq1[i] == GAP_SCORE:
            # Either no current gap or gap in seq2
            if state == 0 or state == 2:
                score += gap_open + gap_extend
                state = 1
            else:
                score += gap_extend
        # Gap in seq2
        elif seq2[i] == GAP_SCORE:
            # Either no current gap or gap in seq1
            if state == 0 or state == 1:
                score += gap_open + gap_extend
                state = 2
            else:
                score += gap_extend
        # No gaps
        else:
            score += subMatrix[seq1[i], seq2[i]] 
            state = 0
    # Return final result
    return score

# TODO: change ints to floats
# TODO: make pull request for just alignment score function
# deadline march 20