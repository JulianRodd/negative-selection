import subprocess
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import subprocess


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
    result = subprocess.run(command, input=test_string, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running Java program: {result.stderr}")
    return result.stdout


def parse_output(output):
    return [float(line) for line in output.split("\n") if line]


def compute_roc_auc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc(fpr, tpr, roc_auc, r_value):
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic for r = {r_value}")
    plt.legend(loc="lower right")
    plt.show()


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

    for r in range(1, 10):
        scores = get_scores_from_java_program(test_data, r)
        fpr, tpr, roc_auc = compute_roc_auc(labels, scores)
        plot_roc(fpr, tpr, roc_auc, r)


def get_scores_from_java_program(test_data, r_value):
    scores = []
    for line in test_data:
        output = run_java_program(r_value, "english.train", line)
        if not output:
            print(f"No output for line: {line}")
        score = parse_output(output)
        scores.extend(score)
    return scores


def analyze_language_comparison(
    english_test_file, other_language_test_file, language_name
):
    test_data, labels = load_test_data_and_labels(
        english_test_file, other_language_test_file
    )

    for r in range(1, 10):
        scores = get_scores_from_java_program(test_data, r)
        if len(scores) != len(labels):
            print(
                f"Length mismatch for r={r}: Scores length: {len(scores)}, Labels length: {len(labels)}"
            )
        else:
            fpr, tpr, roc_auc = compute_roc_auc(labels, scores)
            plot_roc(fpr, tpr, roc_auc, r)
            print(f"Language: {language_name}, r = {r}, AUC = {roc_auc:.2f}")


def get_scores_from_java_program(test_data, r_value):
    scores = []
    for line in test_data:
        # Run the Java program for each line of test data
        output = run_java_program(r_value, "english.train", line)
        # Parse the output to get the scores
        score = parse_output(output)
        scores.extend(score)  # Use extend since parse_output returns a list
    return scores


def analyze_language_comparison(
    english_test_file, other_language_test_file, language_name
):
    test_data, labels = load_test_data_and_labels(
        english_test_file, other_language_test_file
    )

    for r in range(1, 10):
        scores = get_scores_from_java_program(test_data, r)
        if len(scores) != len(labels):
            print(
                f"Length mismatch for r={r}: Scores length: {len(scores)}, Labels length: {len(labels)}"
            )
        else:
            fpr, tpr, roc_auc = compute_roc_auc(labels, scores)
            plot_roc(fpr, tpr, roc_auc, r)
            print(f"Language: {language_name}, r = {r}, AUC = {roc_auc:.2f}")


def preprocess_data(file_path, chunk_size):
    """
    Reads a file containing system call sequences and converts them into fixed-length chunks.

    :param file_path: Path to the file containing system call sequences.
    :param chunk_size: The fixed length for each chunk.
    :return: A list of fixed-length chunks of system call sequences.
    """
    chunks = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            chunks.extend(
                [line[i : i + chunk_size] for i in range(0, len(line), chunk_size)]
            )
    return chunks


def combine_scores(chunk_scores, num_chunks_per_sequence):
    """
    Combines the chunk scores into a composite score for each sequence.

    :param chunk_scores: List of scores for each chunk.
    :param num_chunks_per_sequence: Number of chunks per original sequence.
    :return: Composite scores for each sequence.
    """
    composite_scores = []
    for i in range(0, len(chunk_scores), num_chunks_per_sequence):
        composite_scores.append(np.mean(chunk_scores[i : i + num_chunks_per_sequence]))
    return composite_scores


def analyze_syscalls_data(train_file, test_file, labels_file, chunk_size):
    # Preprocess test data
    test_data_chunks = preprocess_data(test_file, chunk_size)

    # Run Java program and get scores for each chunk
    chunk_scores = get_scores_from_java_program(test_data_chunks, r_value="4") # r_value can be adjusted

    # Read labels
    labels = np.loadtxt(labels_file)

    # Combine chunk scores into composite scores
    composite_scores = combine_scores(chunk_scores, len(test_data_chunks) // len(labels))

    # Compute ROC and AUC
    fpr, tpr, roc_auc = compute_roc_auc(labels, composite_scores)
    plot_roc(fpr, tpr, roc_auc, r_value="4") # r_value displayed in the plot

    print(f"ROC AUC: {roc_auc:.2f}")
