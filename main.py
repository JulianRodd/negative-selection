import subprocess
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import subprocess

from utils import (
    analyze_language_comparison,
    analyze_language_data,
    analyze_syscalls_data,
)


def main():
    # part1_ex1()
    # part1_ex3(3)
    part2()


def part1_ex1():
    # Part 1: Analyze Language Data (English vs Tagalog)
    analyze_language_data()


def part1_ex3(r):
    english_test_file = "english.test"
    languages = {
        "Hiligaynon": "lang/hiligaynon.txt",
        "Middle English": "lang/middle-english.txt",
        "Plautdietsch": "lang/plautdietsch.txt",
        "Xhosa": "lang/xhosa.txt",
    }

    for language, test_file in languages.items():
        analyze_language_comparison(english_test_file, test_file, language, r)


def part2():
    train_file = "syscalls/snd-unm/snd-unm.train"
    test_file = "syscalls/snd-unm/snd-unm.1.test"
    labels_file = "syscalls/snd-unm/snd-unm.1.labels"

    for r in [3, 4, 5]:
        for chunk_size in [10, 15, 20]:
            analyze_syscalls_data(train_file, test_file, labels_file, chunk_size, r)


if __name__ == "__main__":
    main()
