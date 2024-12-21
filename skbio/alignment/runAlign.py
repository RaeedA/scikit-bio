import timeit
from align import align_wrapper
from skbio.sequence import SubstitutionMatrix as SubMatrix
from Bio.Align import substitution_matrices
from Bio import Align


def run(seq1, seq2):
    gap_open = 2
    gap_extend = 2
    seq1 = seq1.upper()
    seq2 = seq2.upper()
    submat = SubMatrix.by_name("NUC.4.4")
    scope = "global"

    res, score = align_wrapper(seq1, seq2, submat, -gap_open, -gap_extend, scope)
    print("Mine:", score)
    print(res)
    aligner = Align.PairwiseAligner(
        substitution_matrix=substitution_matrices.load("NUC.4.4"),
        open_gap_score=-gap_open,
        extend_gap_score=-gap_extend,
        mode=scope,
    ).align(seq1, seq2)
    print("BioPy:", aligner[0].score)
    print(aligner[0])
    # test = global_pairwise_align(DNA(seq1),
    #       DNA(seq2), gap_open, gap_extend, submat.to_dict())
    # print("SciKit:", test[1])
    # print(test[0])


def times(seq1, seq2):
    gap_open = 2
    gap_extend = 2
    seq1 = seq1.upper()
    seq2 = seq2.upper()
    submat = SubMatrix.by_name("NUC.4.4")
    num = 1
    print("Test with 1")
    result = timeit.Timer(
        lambda: align_wrapper(seq1, seq2, submat, -gap_open, -gap_extend)
    ).timeit(num)
    print(f"mine: {result/num:.5f}s")
    result = timeit.Timer(
        lambda: Align.PairwiseAligner(
            substitution_matrix=substitution_matrices.load("NUC.4.4"),
            open_gap_score=-gap_open,
            extend_gap_score=-gap_extend,
        ).align(seq1, seq2)
    ).timeit(num)
    print(f"biopy: {result/num:.5f}s")
    # num = 10000
    # print("Test with 10000")
    # result = timeit.Timer(
    #     lambda: align_wrapper(seq1, seq2, submat, -gap_open, -gap_extend)
    # ).timeit(num)
    # print(f"mine: {result/num:.5f}s")
    # result = timeit.Timer(
    #     lambda: Align.PairwiseAligner(
    #         substitution_matrix=substitution_matrices.load("NUC.4.4"),
    #         open_gap_score=-gap_open,
    #         extend_gap_score=-gap_extend,
    #     ).align(seq1, seq2)
    # ).timeit(num)
    # print(f"biopy: {result/num:.5f}s")


if __name__ == "__main__":
    run(
        "CCGTTA",
        "CCGATGTT",
    )
    # with open('skbio/alignment/a.txt', 'r') as a,
    # open('skbio/alignment/a.txt', 'r') as b:
    #     run(a.read(), b.read())
    # run('gcgt', 'gcgtt')
