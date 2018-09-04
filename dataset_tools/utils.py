"""Utilities for handling datasets."""


def read_file(file_path):
    """Reads a newline-delimited text file.

    Args:
        file_path: path to text file.

    Returns:
        [lines]: Array of lines for the text file."""
    with open(file_path) as fp:
        lines = fp.read().splitlines()
        return [(i, line) for i, line in enumerate(lines)]
