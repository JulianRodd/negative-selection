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
    plot_roc,
)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def main():
    part1_ex1()
    part1_ex3(3)
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

    data_sets = []

    for language, file_path in languages.items():
        fpr, tpr, roc_auc = analyze_language_comparison(english_test_file, file_path, language, r)
        if fpr is not None: 
            data_sets.append((fpr, tpr, roc_auc, r, f"English vs {language}"))

    if data_sets:
        generate_combined_roc_curves(data_sets, "Language Comparison")

import matplotlib.pyplot as plt

def part2():
    train_file = "syscalls/snd-unm/snd-unm.train"
    test_file = "syscalls/snd-unm/snd-unm.1.test"
    labels_file = "syscalls/snd-unm/snd-unm.1.labels"
    combinations = [(r, chunk_size) for r in [3, 4, 5, 6, 7, 8, 9] for chunk_size in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]]

    auc_scores = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(analyze_syscalls_data, train_file, test_file, labels_file, chunk_size, r) 
                   for r, chunk_size in combinations]

        for future in concurrent.futures.as_completed(futures):
            fpr, tpr, roc_auc, chunk_size, r_value = future.result()
            auc_scores.append((roc_auc, r_value, chunk_size))

    # Sort by AUC score in descending order and select top 4
    top_scores = sorted(auc_scores, key=lambda x: x[0], reverse=True)[:4]

    # Plotting AUC scores
    plt.figure()

    # Define a color map from green to bright red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_to_red", 
        [(0, "white"), (0.2, "#ffe6d3"), (0.5, "orange"), (0.7, "red"), (1, "#8B0000")]
    )

    # Normalize the data
    norm = mcolors.Normalize(vmin=min(auc_scores)[0], vmax=max(auc_scores)[0])

    for roc_auc, r_value, chunk_size in auc_scores:
        color = cmap(norm(roc_auc))
        plt.scatter(r_value, chunk_size, color=color, label=f'AUC={roc_auc:.2f}', s=150)  # s=100 for larger dots

    plt.xlabel('R-value')
    plt.ylabel('Chunk Size')
    plt.title('AUC Scores for Different R-value and Chunk Sizes')

    # Adding a colorbar based on the colormap
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='AUC Score')
    plt.yticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    plt.savefig('roc_curves/auc_scores_plot.png')
    plt.close()
    print("Saved AUC scores plot as 'roc_curves/auc_scores_plot.png'")

    # Generating combined ROC curves for top 4 configurations
    data_sets = []
    for roc_auc, r_value, chunk_size in top_scores:
        fpr, tpr, roc_auc, _, _ = analyze_syscalls_data(train_file, test_file, labels_file, chunk_size, r_value)
        data_sets.append((fpr, tpr, roc_auc, r_value, f"Chunk Size={chunk_size}, R-value={r_value}"))

    if data_sets:
        generate_combined_roc_curves(data_sets, "Top 4 Syscall Analysis")




if __name__ == "__main__":
    main()
