from random import choice, randint, seed, uniform
import timeit

from skbio import DNA
from align import align_wrapper, align_score
from skbio.sequence import SubstitutionMatrix as SubMatrix
from Bio.Align import substitution_matrices
from Bio import Align


def run(seq1, seq2, gap_open, gap_extend, scope):
    submat = SubMatrix.by_name("NUC.4.4")
    # aligner = Align.PairwiseAligner(
    #     substitution_matrix=substitution_matrices.load("NUC.4.4"),
    #     open_gap_score=gap_open,
    #     extend_gap_score=gap_extend,
    #     mode=scope,
    # ).align(seq1, seq2)
    # print("BioPy:", aligner[0].score)
    # for align in aligner:
    #     print(align)
    res, score = align_wrapper(seq1, seq2, submat, gap_open, gap_extend, scope)
    print("Mine:", score)
    print()
    dnas = res.to_dict()
    seq1_aligned = dnas[0].values
    seq2_aligned = dnas[1].values
    print(
        "Score:",
        align_score(seq1_aligned, seq2_aligned, submat, gap_open, gap_extend),
    )
    print(res)


def times(times, length, diff=0):
    chars = ["A", "C", "G", "T"]
    seq1 = "".join(choice(chars) for j in range(length))
    seq2 = "".join(choice(chars) for j in range(length + randint(-diff, diff)))
    if randint(1, 2) == 1:
        scope = "local"
    else:
        scope = "global"
    gap_open = round(uniform(-10, 0), 2)
    gap_extend = round(uniform(-10, 0), 2)
    submat = SubMatrix.by_name("NUC.4.4")
    result = timeit.Timer(
        lambda: align_wrapper(seq1, seq2, submat, -gap_open, -gap_extend, scope)
    ).timeit(times)
    print(f"mine: {result/times:.5f}s")
    result = timeit.Timer(
        lambda: Align.PairwiseAligner(
            substitution_matrix=substitution_matrices.load("NUC.4.4"),
            open_gap_score=-gap_open,
            extend_gap_score=-gap_extend,
            mode=scope,
        ).align(seq1, seq2)
    ).timeit(times)
    print(f"biopy: {result/times:.5f}s")


def test(times, length, diff=0):
    submat1 = SubMatrix.by_name("NUC.4.4")
    submat2 = substitution_matrices.load("NUC.4.4")

    chars = ["A", "C", "G", "T"]
    # seed(572989)
    for i in range(times):
        a = "".join(choice(chars) for j in range(length))
        b = "".join(choice(chars) for j in range(length + randint(-diff, diff)))
        if randint(1, 2) == 1:
            scope = "local"
        else:
            scope = "global"
        gap_open = round(uniform(-10, 0), 2)
        gap_extend = round(uniform(-10, 0), 2)
        try:
            res, score1 = align_wrapper(a, b, submat1, gap_open, gap_extend, scope)
        except Exception:
            continue
        score2 = (
            Align.PairwiseAligner(
                substitution_matrix=submat2,
                open_gap_score=gap_open,
                extend_gap_score=gap_extend,
                mode=scope,
            )
            .align(a, b)[0]
            .score
        )
        dnas = res.to_dict()
        seq1_aligned = dnas[0].values
        seq2_aligned = dnas[1].values
        score3 = align_score(seq1_aligned, seq2_aligned, submat1, gap_open, gap_extend)
        # if score1 != score2 and score3 != score2:
        #     print("mismatch!", score2 > score1, score1 == score3, end=" ")
        #     print(f'"{a}", "{b}", {gap_open}, {gap_extend}, "{scope}"', end=" ")
        #     print("mine", score1, "func", score3, "biopy", score2, end=" ")
        #     print(str(seq1_aligned), str(seq2_aligned), end=" ")
        #     print()
        if score1 != score3:
            print("mismatch!", score1, score3, end=" ")
            print(f'"{a}", "{b}", {gap_open}, {gap_extend}, "{scope}"')
        # if score2 != score1:
        #     print(score2-score1, gap_open, gap_extend)


if __name__ == "__main__":
    # test(1000, 100)
    # run("TGATCCTA", "CCCGAATC", -1, -7, "local")
    # run("CGGAA", "CAACA", 0, -2, "global")
    times(100, 1000)
