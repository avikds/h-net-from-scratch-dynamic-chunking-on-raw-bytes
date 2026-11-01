"""
H-Net from Scratch: Dynamic Chunking on Raw Bytes

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_corpus
import os
import tempfile
import urllib.request
import torch


def load_corpus(n_bytes=None):
    """
    Load TinyStories as a 1-D PyTorch tensor of raw UTF-8 bytes.

    Args:
        n_bytes (int, optional): Number of initial bytes to retain.

    Returns:
        torch.Tensor: 1-D tensor of dtype torch.long containing byte values.
    """

    # First try loading TinyStories through the dml_data package.
    try:
        import dml_data
        text = dml_data.load("tinystories").text

        if not isinstance(text, str):
            raise TypeError("TinyStories corpus must be a string.")

    except Exception:
        # Fall back to downloading the TinyStories validation corpus.
        url = (
            "https://huggingface.co/datasets/roneneldan/"
            "TinyStories/resolve/main/TinyStories-valid.txt"
        )

        # Store the downloaded corpus in the system temporary directory.
        file_path = os.path.join(
            tempfile.gettempdir(),
            "TinyStories-valid.txt"
        )

        # Download only if the file does not already exist.
        if not os.path.isfile(file_path):
            urllib.request.urlretrieve(url, file_path)

        # Read the corpus as UTF-8, ignoring invalid byte sequences.
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    # Encode the complete text into raw UTF-8 bytes.
    raw_bytes = text.encode("utf-8")

    # Retain only the first n_bytes bytes when requested.
    if n_bytes is not None:
        if n_bytes < 0:
            raise ValueError("n_bytes must be non-negative.")
        raw_bytes = raw_bytes[:n_bytes]

    # Convert byte values into a 1-D tensor of type torch.long.
    return torch.tensor(list(raw_bytes), dtype=torch.long)

# Step 2 - sample_batch
def sample_batch(data, batch_size, seq_len, generator):
    """
    Sample random windows for next-byte prediction.

    Args:
        data (torch.Tensor): 1-D tensor containing raw byte values.
        batch_size (int): Number of sequences to sample.
        seq_len (int): Length of each input sequence.
        generator (torch.Generator): Random number generator.

    Returns:
        tuple: (x, y), two tensors of shape (batch_size, seq_len).
               x contains input sequences.
               y contains the corresponding next-byte targets.
    """

    # Draw random starting positions using the supplied generator.
    starts = torch.randint(
        0,
        len(data) - seq_len - 1,
        (batch_size,),
        generator=generator
    )

    # Generate offsets for each position within a sequence.
    offsets = torch.arange(seq_len, device=data.device)

    # Extract input sequences.
    x = data[starts[:, None] + offsets]

    # Extract targets shifted forward by one byte.
    y = data[starts[:, None] + offsets + 1]

    # Ensure both outputs have dtype torch.long.
    x = x.long()
    y = y.long()

    return x, y

