import subprocess
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import subprocess


def run_java_program(r_value, train_file, test_string):
    command = ["java", "-jar", "negsel2.jar", "-self", train_file, "-n", "10", "-r", str(r_value), "-c", "-l"]
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


def main():
    # Part 1: Analyze Language Data (English vs Tagalog)
    analyze_language_data()

    # Additional parts of the assignment would be implemented similarly.
    # For example, if you have a part 2 for Unix system calls data analysis,
    # you would implement it as another function and call it here.
    # analyze_system_calls_data()


def analyze_language_data():
    # Define the test files for English and Tagalog
    english_test_file = "english.test"  # Replace with the correct path
    tagalog_test_file = "tagalog.test"  # Replace with the correct path

    # Load test data and corresponding labels
    test_data, labels = load_test_data_and_labels(english_test_file, tagalog_test_file)

    for r in range(1, 10):
        scores = get_scores_from_java_program(test_data, r)
        if len(scores) != len(labels):
            print(f"Length mismatch for r={r}: Scores length: {len(scores)}, Labels length: {len(labels)}")
        else:
            fpr, tpr, roc_auc = compute_roc_auc(labels, scores)
            plot_roc(fpr, tpr, roc_auc, r)


def get_scores_from_java_program(test_data, r_value):
    scores = []
    for line in test_data:
        # Run the Java program for each line of test data
        output = run_java_program(
            r_value, "english.train", line
        )  
        # Parse the output to get the scores
        score = parse_output(output)
        scores.extend(score)  # Use extend since parse_output returns a list
    return scores


if __name__ == "__main__":
    main()
