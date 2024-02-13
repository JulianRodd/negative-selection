import subprocess
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import subprocess
import concurrent.futures
from utils import (
    analyze_language_comparison,
    analyze_language_data,
    analyze_syscalls_data,
    generate_combined_roc_curves,
)


def main():
    # part1_ex1()
    part1_ex3(3)
    #part2()


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

    data_sets = []

    for language, file_path in languages.items():
        fpr, tpr, roc_auc = analyze_language_comparison(english_test_file, file_path, language, r)
        if fpr is not None: 
            data_sets.append((fpr, tpr, roc_auc, r, f"English vs {language}"))

    if data_sets:
        generate_combined_roc_curves(data_sets)

def part2():
    train_file = "syscalls/snd-unm/snd-unm.train"
    test_file = "syscalls/snd-unm/snd-unm.1.test"
    labels_file = "syscalls/snd-unm/snd-unm.1.labels"

    # Define all combinations of r and chunk_size
    combinations = [(r, chunk_size) for r in [3, 4, 5] for chunk_size in [10, 15, 20]]

    # Use ThreadPoolExecutor to run in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(analyze_syscalls_data, train_file, test_file, labels_file, chunk_size, r) 
                   for r, chunk_size in combinations]

        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()  # This will also re-raise any exceptions caught during execution


if __name__ == "__main__":
    main()
