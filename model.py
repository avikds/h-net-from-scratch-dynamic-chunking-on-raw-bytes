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

# Step 3 - bits_per_byte
import math
import torch.nn.functional as F


def bits_per_byte(logits, targets):
    """
    Calculate bits per byte (BPB) from next-byte prediction logits.

    Args:
        logits (torch.Tensor): Predicted logits of shape (B, L, 256).
        targets (torch.Tensor): Target byte values of shape (B, L).

    Returns:
        float: Mean cross-entropy loss converted from nats to bits.
    """

    # Extract dimensions and validate the input shapes.
    if logits.ndim != 3 or logits.shape[-1] != 256:
        raise ValueError("logits must have shape (B, L, 256).")

    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must have shape (B, L).")

    # Flatten batch and sequence dimensions for cross-entropy.
    logits_flat = logits.reshape(-1, 256)
    targets_flat = targets.reshape(-1).long()

    # Calculate mean cross-entropy loss in nats.
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="mean"
    )

    # Convert nats into bits and return a Python float.
    bpb = loss / math.log(2.0)

    return bpb.item()

# Step 4 - pool_boundaries
def pool_boundaries(ids, k):
    """
    Mark every k-th position as a chunk boundary.

    Args:
        ids (torch.Tensor): Input tensor of shape (B, L).
        k (int): Fixed chunk width.

    Returns:
        torch.Tensor: Boolean tensor of shape (B, L),
                      True at positions 0, k, 2k, ... .
    """

    if ids.ndim != 2:
        raise ValueError("ids must have shape (B, L).")

    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # Generate position indices on the same device as ids.
    positions = torch.arange(ids.shape[1], device=ids.device)

    # Mark positions divisible by k as chunk boundaries.
    boundaries = (positions % k == 0)

    # Expand the same boundary pattern across every batch row.
    return boundaries.unsqueeze(0).expand(ids.shape[0], -1)

# Step 5 - space_boundaries
def space_boundaries(ids):
    """
    Identify chunk boundaries using SpaceByte's space-like rule.

    Args:
        ids (torch.Tensor): Input byte tensor of shape (B, L).

    Returns:
        torch.Tensor: Boolean boundary tensor of shape (B, L).
    """

    if ids.ndim != 2:
        raise ValueError("ids must have shape (B, L).")

    # Identify uppercase and lowercase ASCII letters.
    is_upper = (ids >= 65) & (ids <= 90)
    is_lower = (ids >= 97) & (ids <= 122)
    is_letter = is_upper | is_lower

    # Identify ASCII digits.
    is_digit = (ids >= 48) & (ids <= 57)

    # Identify UTF-8 continuation bytes.
    is_continuation = (ids >= 128) & (ids <= 191)

    # A byte is spacelike if it is none of the above.
    spacelike = ~(is_letter | is_digit | is_continuation)

    # Shift spacelike indicators by one position to obtain
    # the previous byte's spacelike status.
    prev_spacelike = torch.zeros_like(spacelike)

    if ids.shape[1] > 1:
        prev_spacelike[:, 1:] = spacelike[:, :-1]

    # A boundary occurs when the current byte is spacelike
    # and the previous byte is not spacelike.
    boundaries = spacelike & ~prev_spacelike

    # Position zero is always considered a boundary.
    if ids.shape[1] > 0:
        boundaries[:, 0] = True

    return boundaries

