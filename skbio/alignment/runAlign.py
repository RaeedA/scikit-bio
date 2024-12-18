import timeit
from skbio.sequence import DNA
from skbio.alignment import global_pairwise_align
from skbio.sequence import SubstitutionMatrix as SubMatrix
import numpy as np
from Bio.Align import substitution_matrices, Alignment
from biotite.sequence import NucleotideSequence
from Bio import Align
from biotite.sequence.align import align_optimal, SubstitutionMatrix
from skbio.alignment.align import align_wrapper


def run(seq1, seq2):
    gap_open = 2
    gap_extend = 2
    seq1 = seq1.upper()
    seq2 = seq2.upper()
    submat = SubMatrix.by_name("NUC.4.4")

    res, score = align_wrapper(seq1, seq2, submat, -gap_open, -gap_extend)
    print("Mine:", score)
    aligner = Align.PairwiseAligner(
        substitution_matrix=substitution_matrices.load("NUC.4.4"),
        open_gap_score=-gap_open,
        extend_gap_score=-gap_extend,
    ).align(seq1, seq2)
    print("BioPy:", aligner[0].score)
    # test = global_pairwise_align(DNA(seq1),
    #       DNA(seq2), gap_open, gap_extend, submat.to_dict())
    # print("SciKit:", test[1])
    # print(test[0])
    print(
        "Biotite:",
        align_optimal(
            NucleotideSequence(seq1),
            NucleotideSequence(seq2),
            SubstitutionMatrix.std_nucleotide_matrix(),
            (-gap_open, -gap_extend),
        )[0].score,
    )


def times(seq1, seq2):
    gap_open = 2
    gap_extend = 2
    seq1 = seq1.upper()
    seq2 = seq2.upper()
    submat = SubMatrix.by_name("NUC.4.4")
    num = 100
    print("Test with 100")
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
    result = timeit.Timer(
        lambda: align_optimal(
            NucleotideSequence(seq1),
            NucleotideSequence(seq2),
            SubstitutionMatrix.std_nucleotide_matrix(),
            (-gap_open, -gap_extend),
        )
    ).timeit(num)
    print(f"biotite: {result/num:.5f}s")
    num = 10000
    print("Test with 10000")
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


if __name__ == "__main__":
    run(
        "gcgcgtgcgcggaaggagccaaggtgaagttgtagcagtgtgtcagaagaggtgcgtggcaccatgctgtcccccgaggcggagcgggtgctgcggtacctggtcgaagtagaggagttgg",
        "gacttgtggaacctacttcctgaaaataaccttctgtcctccgagctctccgcacccgtggatgacctgctcccgtacacagatgttgccacctggctggatgaatgtccgaatgaagcg",
    )
    times(
        "gcgcgtgcgcggaaggagccaaggtgaagttgtagcagtgtgtcagaagaggtgcgtggcaccatgctgtcccccgaggcggagcgggtgctgcggtacctggtcgaagtagaggagttg",
        "gacttgtggaacctacttcctgaaaataaccttctgtcctccgagctctccgcacccgtggatgacctgctcccgtacacagatgttgccacctggctggatgaatgtccgaatgaagcg",
    )
    # with open('skbio/alignment/a.txt', 'r') as a,
    #      open('skbio/alignment/a.txt', 'r') as b:
    #     times(a.read(), b.read())
    # run('gcgt', 'gcgtt')
