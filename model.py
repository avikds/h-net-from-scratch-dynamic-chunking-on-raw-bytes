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

# Step 6 - RoutingModule
class RoutingModule(torch.nn.Module):
    """
    Predict chunk boundaries using cosine similarity
    between adjacent projected hidden-state vectors.
    """

    def __init__(self, d):
        super().__init__()

        # Bias-free query and key projections.
        self.q_proj = torch.nn.Linear(d, d, bias=False)
        self.k_proj = torch.nn.Linear(d, d, bias=False)

        # Initialize both projection matrices to the identity.
        # This follows the initialization used in the paper's
        # released implementation.
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(d))
            self.k_proj.weight.copy_(torch.eye(d))

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input of shape (B, L, d).
            mask (torch.Tensor, optional): Boolean mask of shape (B, L).

        Returns:
            p (torch.Tensor): Boundary probabilities of shape (B, L).
            b (torch.Tensor): Boolean boundary decisions of shape (B, L).
        """

        B, L, d = x.shape

        if L == 0:
            raise ValueError("Sequence length must be greater than zero.")

        # Project input vectors into query and key representations.
        q = self.q_proj(x)
        k = self.k_proj(x)

        # The first position is always a chunk boundary.
        first_prob = x.new_ones((B, 1))

        if L > 1:
            # Compare each query vector with the preceding key vector.
            cosine_sim = F.cosine_similarity(
                q[:, 1:, :],
                k[:, :-1, :],
                dim=-1
            )

            # Convert cosine similarity into boundary probabilities.
            remaining_prob = 0.5 * (1.0 - cosine_sim)

            # Combine the first-position probability with the rest.
            p = torch.cat(
                [first_prob, remaining_prob],
                dim=1
            )
        else:
            p = first_prob

        # Keep all probabilities within the interval [0, 1].
        p = p.clamp(0.0, 1.0)

        # Convert probabilities into binary boundary decisions.
        b = p >= 0.5

        # Apply the optional mask to boundary decisions.
        # Masked positions must never become boundaries.
        if mask is not None:
            if mask.shape != (B, L):
                raise ValueError("mask must have shape (B, L).")

            b = b & mask.to(device=x.device, dtype=torch.bool)

        return p, b

# Step 7 - compact_chunks
def compact_chunks(x, b):
    """
    Keep boundary vectors and compact each batch row to the left.

    Args:
        x (torch.Tensor):
            Input tensor of shape (B, L) or (B, L, d).

        b (torch.Tensor):
            Boolean boundary tensor of shape (B, L).

    Returns:
        tuple:
            x_next: Compacted tensor of shape (B, M) or (B, M, d).
            mask_next: Boolean tensor of shape (B, M).
                       True indicates a valid compacted entry.
    """

    # Validate input dimensions.
    if x.ndim not in (2, 3):
        raise ValueError("x must have shape (B, L) or (B, L, d).")

    if b.ndim != 2 or b.shape != x.shape[:2]:
        raise ValueError("b must have shape (B, L).")

    if b.dtype != torch.bool:
        raise TypeError("b must be a boolean tensor.")

    if b.device != x.device:
        raise ValueError("x and b must be on the same device.")

    B, L = b.shape

    # Count the number of boundary vectors in each batch row.
    counts = b.sum(dim=1)

    # M is the maximum number of boundaries in any row.
    M = int(counts.max().item()) if counts.numel() > 0 else 0

    # Assign unique sorting keys to every position.
    # Boundary positions receive smaller keys than non-boundary positions.
    positions = torch.arange(L, device=x.device).unsqueeze(0)

    keys = positions + (~b).long() * L

    # Sort positions independently for every batch row.
    # Since all keys are unique, sorting does not depend on tie order.
    order = torch.argsort(keys, dim=1)

    # Select the first M positions after sorting.
    selected = order[:, :M]

    # Gather the original values using the sorted indices.
    if x.ndim == 2:
        x_next = torch.gather(x, dim=1, index=selected)

    else:
        # Expand indices across the feature dimension.
        d = x.shape[2]

        expanded_indices = selected.unsqueeze(-1).expand(-1, -1, d)

        x_next = torch.gather(
            x,
            dim=1,
            index=expanded_indices
        )

    # Create a mask identifying valid entries in each compacted row.
    # Positions beyond the original boundary count are marked False.
    mask_next = (
        torch.arange(M, device=x.device).unsqueeze(0)
        < counts.unsqueeze(1)
    )

    return x_next, mask_next

# Step 8 - smooth_chunks
def smooth_chunks(z, P):
    """
    Smooth main-network outputs across uncertain chunks using EMA.

    Args:
        z (torch.Tensor): Main-network outputs of shape (B, M, d).
        P (torch.Tensor): Router probabilities of shape (B, M).

    Returns:
        torch.Tensor: Smoothed outputs of shape (B, M, d).
    """

    if z.ndim != 3:
        raise ValueError("z must have shape (B, M, d).")

    if P.shape != z.shape[:2]:
        raise ValueError("P must have shape (B, M).")

    # Match the device and dtype of the main-network outputs.
    P = P.to(device=z.device, dtype=z.dtype)

    B, M, d = z.shape

    # Handle empty sequences.
    if M == 0:
        return torch.empty_like(z)

    # Initialize the previous smoothed vector to zero.
    z_prev = z.new_zeros((B, d))

    smoothed_outputs = []

    # Apply the EMA recurrence sequentially across chunks.
    for t in range(M):
        p_t = P[:, t].unsqueeze(-1)

        # Equation 5: combine the current output and previous EMA.
        z_t_bar = (
            p_t * z[:, t, :]
            + (1.0 - p_t) * z_prev
        )

        # Store the current smoothed output.
        smoothed_outputs.append(z_t_bar)

        # Update the previous smoothed vector.
        z_prev = z_t_bar

    # Stack all smoothed vectors along the sequence dimension.
    return torch.stack(smoothed_outputs, dim=1)

# Step 9 - causal_upsample
def causal_upsample(zbar, b):
    """
    Stretch compressed chunk vectors back to byte-level resolution.

    Args:
        zbar (torch.Tensor): Smoothed chunk representations of shape
                             (B, M, d).

        b (torch.Tensor): Boolean boundary tensor of shape (B, L),
                          with b[:, 0] being True.

    Returns:
        torch.Tensor: Upsampled representations of shape (B, L, d).
    """

    # Validate input dimensions.
    if zbar.ndim != 3:
        raise ValueError("zbar must have shape (B, M, d).")

    if b.ndim != 2:
        raise ValueError("b must have shape (B, L).")

    if b.shape[0] != zbar.shape[0]:
        raise ValueError("Batch dimensions of zbar and b must match.")

    if b.dtype != torch.bool:
        raise TypeError("b must be a boolean tensor.")

    if b.device != zbar.device:
        raise ValueError("zbar and b must be on the same device.")

    B, L = b.shape
    d = zbar.shape[2]

    # Ensure the first byte of every non-empty sequence is a boundary.
    if B > 0 and L > 0 and not b[:, 0].all():
        raise ValueError("The first position of every row must be True.")

    # Calculate the chunk index associated with every byte position.
    # Subtracting one converts the cumulative boundary count
    # into a zero-based chunk index.
    chunk_indices = b.long().cumsum(dim=1) - 1

    # Expand indices across the feature dimension.
    gather_indices = chunk_indices.unsqueeze(-1).expand(-1, -1, d)

    # Gather the most recent chunk representation for each byte.
    ztilde = torch.gather(
        zbar,
        dim=1,
        index=gather_indices
    )

    return ztilde

# Step 10 - straight_through_confidence
def straight_through_confidence(p, b):
    """
    Build a straight-through confidence multiplier.

    Forward value is 1 at every position, while the backward
    gradient is preserved from the confidence calculation.

    Args:
        p (torch.Tensor): Floating-point probability tensor.
        b (torch.Tensor): Boolean boundary tensor of the same shape.

    Returns:
        torch.Tensor: Straight-through confidence multiplier.
    """

    if p.shape != b.shape:
        raise ValueError("p and b must have the same shape.")

    if not p.is_floating_point():
        raise TypeError("p must be a floating-point tensor.")

    if b.dtype != torch.bool:
        raise TypeError("b must be a boolean tensor.")

    # Calculate confidence:
    # c = p for boundary positions, otherwise c = 1 - p.
    c = torch.where(b, p, 1.0 - p)

    # Straight-through estimator:
    # Forward value is always 1, but gradients flow through c.
    s = c + (1.0 - c).detach()

    return s

# Step 11 - ratio_loss
def ratio_loss(p, b, N, mask=None):
    """
    Calculate the dynamic chunking ratio loss (Equation 10).

    Args:
        p (torch.Tensor): Floating-point boundary probabilities.
        b (torch.Tensor): Boolean boundary decisions.
        N (int or float): Target compression factor (must be > 1).
        mask (torch.Tensor, optional): Boolean mask identifying valid
                                       positions.

    Returns:
        torch.Tensor: Scalar ratio loss with gradients through p.
    """

    # Validate inputs.
    if p.shape != b.shape:
        raise ValueError("p and b must have the same shape.")

    if not p.is_floating_point():
        raise TypeError("p must be a floating-point tensor.")

    if b.dtype != torch.bool:
        raise TypeError("b must be a boolean tensor.")

    if p.device != b.device:
        raise ValueError("p and b must be on the same device.")

    if N <= 1:
        raise ValueError("N must be greater than 1.")

    # Apply the optional mask to select valid positions.
    if mask is not None:
        if mask.shape != p.shape:
            raise ValueError("mask must have the same shape as p.")

        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor.")

        if mask.device != p.device:
            raise ValueError("mask must be on the same device as p.")

        p_valid = p[mask]
        b_valid = b[mask]

    else:
        p_valid = p.reshape(-1)
        b_valid = b.reshape(-1)

    # Ensure that at least one valid position exists.
    if p_valid.numel() == 0:
        raise ValueError("At least one valid position is required.")

    # F = mean of binary boundary decisions.
    # Since b is boolean, F does not carry gradients.
    F_ratio = b_valid.to(dtype=p.dtype).mean()

    # G = mean of predicted boundary probabilities.
    # Gradients must flow through G to train the router.
    G_ratio = p_valid.mean()

    # Implement Equation 10.
    loss = (
        N / (N - 1)
    ) * (
        (N - 1) * F_ratio * G_ratio
        + (1 - F_ratio) * (1 - G_ratio)
    )

    return loss

# Step 12 - RMSNorm
class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization with a learnable scale parameter.
    """

    def __init__(self, d, eps=1e-5):
        super().__init__()

        # Learnable scaling parameter, initialized to ones.
        self.weight = nn.Parameter(torch.ones(d))

        # Small constant for numerical stability.
        self.eps = eps

    def forward(self, x):
        """
        Apply RMS normalization along the last dimension.

        Args:
            x (torch.Tensor): Input tensor whose last dimension is d.

        Returns:
            torch.Tensor: RMS-normalized tensor with the same shape as x.
        """

        # Calculate the mean squared value along the last dimension.
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)

        # Calculate the inverse root mean square.
        inv_rms = torch.rsqrt(mean_square + self.eps)

        # Normalize the input and apply the learnable scale.
        return x * inv_rms * self.weight

