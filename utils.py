import subprocess
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import subprocess
import hashlib
import json
import os

def run_java_program(r_value, train_file, test_string):
    command = [
        "java",
        "-jar",
        "negsel2.jar",
        "-self",
        train_file,
        "-n",
        "10",
        "-r",
        str(r_value),
        "-c",
        "-l",
    ]
    print(
        f"Running Java program with r={r_value} on test string: {test_string[:30]}..."
    )  # Log part of the test string
    result = subprocess.run(command, input=test_string, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running Java program: {result.stderr}")
    else:
        print("Java program executed successfully.")
    return result.stdout


def parse_output(output):
    scores = []
    for line in output.split("\n"):
        if line:
            scores.extend([float(score) for score in line.split()])
    return scores



def compute_roc_auc(labels, scores):
    scores = np.nan_to_num(scores)  # Replace NaN with 0
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc



def plot_roc(fpr, tpr, roc_auc, r_value, additional_title=""):
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    title = f"ROC Curve with R-Value = {r_value} {additional_title}".strip()
    plt.title(title)
    plt.legend(loc="lower right")

    # Create the roc_curves directory if it doesn't exist
    os.makedirs('roc_curves', exist_ok=True)

    # Replace spaces and special characters in the title with underscores for the file name
    file_name = title.replace(" ", "_").replace("=", "").replace(",", "") + ".png"

    # Save the figure
    plt.savefig(f"roc_curves/{file_name}")
    plt.close()  # Close the plot to free up memory
    print(f"Saved ROC curve plot as roc_curves/{file_name}")



def load_test_data_and_labels(english_test_file, tagalog_test_file):
    # Load English and Tagalog test data
    # Here you need to implement the logic to read these files and return the test strings
    # Also, create a label array: 0 for English, 1 for Tagalog
    english_test_data = load_strings_from_file(english_test_file)
    tagalog_test_data = load_strings_from_file(tagalog_test_file)
    test_data = english_test_data + tagalog_test_data
    labels = [0] * len(english_test_data) + [1] * len(tagalog_test_data)
    return test_data, labels


def load_strings_from_file(file_path):
    """
    Reads a file and returns a list of strings, with each line in the file being a separate string.

    :param file_path: Path to the file containing the test strings.
    :return: List of strings from the file.
    """
    strings = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                strings.append(
                    line.strip()
                )  # .strip() removes any leading/trailing whitespace
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
    return strings


def analyze_language_data():
    english_test_file = "english.test"
    tagalog_test_file = "tagalog.test"
    test_data, labels = load_test_data_and_labels(english_test_file, tagalog_test_file)
    data_sets = []

    for r in range(1, 10):
        scores = get_scores_from_java_program(test_data, r, cache_name="language_analysis")
        fpr, tpr, roc_auc = compute_roc_auc(labels, scores)
        plot_roc(fpr, tpr, roc_auc, r)
        data_sets.append((fpr, tpr, roc_auc, r, f"r={r}"))
        
    if data_sets:
        generate_combined_roc_curves(data_sets, "Different R-values (1 to 9) on English vs. Tagalog")




def get_scores_from_java_program(test_data, r_value, cache_name=""):
    scores = []
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a unique cache filename based on the cache_name and r_value
    cache_file = os.path.join(cache_dir, f"{cache_name}_r{r_value}.json") if cache_name else ""

    if cache_name and os.path.exists(cache_file):
        print(f"Loading scores from cache: {cache_file}")
        with open(cache_file, "r") as file:
            scores = json.load(file)
    else:
        for line in test_data:
            output = run_java_program(r_value, "english.train", line)
            if not output:
                print(f"No output for line: {line}")
            score = parse_output(output)
            scores.extend(score)

        if cache_name:
            print(f"Saving scores to cache: {cache_file}")
            with open(cache_file, "w") as file:
                json.dump(scores, file)

    return scores




def generate_combined_roc_curves(data_sets, additional_title=""):
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown', 'pink', 'gray']

    for (fpr, tpr, roc_auc, r_value, title), color in zip(data_sets, colors):
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{title} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves ' + additional_title)
    plt.legend(loc='lower right', prop={'size': 10})
    plt.grid(True)

    # Save the combined plot
    os.makedirs('roc_curves', exist_ok=True)
    plt.savefig(f'roc_curves/combined_roc_curves_{additional_title}.png')
    plt.close()
    print(f"Saved combined ROC curve plot as 'roc_curves/combined_roc_curves_{additional_title}.png'")
    
    
  
def analyze_language_comparison(english_test_file, other_language_test_file, language_name, r):
    print(f"Analyzing language comparison between English and {language_name}...")

    test_data, labels = load_test_data_and_labels(english_test_file, other_language_test_file)
    print(f"Processing with r = {r}...")
    scores = get_scores_from_java_program(test_data, r, cache_name=f"language_comparison_{language_name}")

    if len(scores) != len(labels):
        print(f"Length mismatch for r={r}: Scores length: {len(scores)}, Labels length: {len(labels)}")
        return None, None, None

    fpr, tpr, roc_auc = compute_roc_auc(labels, scores)
    plot_roc(fpr, tpr, roc_auc, r, additional_title=f" (English vs {language_name})")
    print(f"Language: {language_name}, R-value = {r}, AUC = {roc_auc:.2f}")

    return fpr, tpr, roc_auc


def preprocess_data(file_path, chunk_size):
    """
    Reads a file containing system call sequences and converts them into fixed-length chunks.

    :param file_path: Path to the file containing system call sequences.
    :param chunk_size: The fixed length for each chunk.
    :return: A list of fixed-length chunks of system call sequences.
    """
    sequences = []
    chunk_counts = []  # Store the number of chunks per sequence

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            chunks = [line[i:i+chunk_size] for i in range(0, len(line), chunk_size)]
            sequences.extend(chunks)
            chunk_counts.append(len(chunks))

    return sequences, chunk_counts


def combine_scores(chunk_scores, chunk_counts):
    """
    Combines the chunk scores into a composite score for each sequence.

    :param chunk_scores: List of scores for each chunk.
    :param chunk_counts: List of the number of chunks per sequence.
    :return: Composite scores for each sequence.
    """
    composite_scores = []
    start_index = 0
    for count in chunk_counts:
        end_index = start_index + count
        sequence_scores = chunk_scores[start_index:end_index]
        composite_score = np.mean(sequence_scores) if sequence_scores else 0
        composite_scores.append(composite_score)
        start_index = end_index
    return composite_scores



def analyze_syscalls_data(train_file, test_file, labels_file, chunk_size, r_value):
    print(f"Analyzing syscall data with train file: {train_file}, test file: {test_file}")

    test_data_chunks, chunk_counts = preprocess_data(test_file, chunk_size)
    print(f"Preprocessed test data into {len(test_data_chunks)} chunks.")

    chunk_scores = get_scores_from_java_program(test_data_chunks, r_value=str(r_value), cache_name="syscall_analysis")
    print("Received scores for each chunk.")

    labels = np.loadtxt(labels_file)
    if len(labels) != len(chunk_counts):
        print(f"Warning: Number of labels {len(labels)} doesn't match number of sequences {len(chunk_counts)}")
    
    composite_scores = combine_scores(chunk_scores, chunk_counts)
    print("Combined chunk scores into composite scores.")

    fpr, tpr, roc_auc = compute_roc_auc(labels, composite_scores)
    
    return fpr, tpr, roc_auc, chunk_size, r_value
