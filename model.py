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

# Step 13 - ssd_scan
def ssd_scan(u, dt, A, B_mat, C_mat, h0=None, chunk=64):
    """
    Run Mamba-2's state-space recurrence using chunked matrix operations.

    Args:
        u:     Input tensor of shape (B, L, H, P).
        dt:    Time-step tensor of shape (B, L, H).
        A:     State transition parameters of shape (H,).
        B_mat: Input state vectors of shape (B, L, N).
        C_mat: Output state vectors of shape (B, L, N).
        h0:    Optional initial state of shape (B, H, P, N).
        chunk: Maximum number of positions processed per chunk.

    Returns:
        y: Final output tensor of shape (B, L, H, P).
        h: Final state tensor of shape (B, H, P, N).
    """

    # Validate chunk size.
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise ValueError("chunk must be a positive integer.")

    # Validate input dimensions.
    if u.ndim != 4:
        raise ValueError("u must have shape (B, L, H, P).")

    if dt.ndim != 3:
        raise ValueError("dt must have shape (B, L, H).")

    if A.ndim != 1:
        raise ValueError("A must have shape (H,).")

    if B_mat.ndim != 3 or C_mat.ndim != 3:
        raise ValueError("B_mat and C_mat must have shape (B, L, N).")

    batch_size, seq_len, num_heads, head_dim = u.shape
    num_states = B_mat.shape[-1]

    if dt.shape != (batch_size, seq_len, num_heads):
        raise ValueError("dt must have shape (B, L, H).")

    if A.shape != (num_heads,):
        raise ValueError("A must have shape (H,).")

    if B_mat.shape != C_mat.shape:
        raise ValueError("B_mat and C_mat must have matching shapes.")

    if B_mat.shape[:2] != (batch_size, seq_len):
        raise ValueError("B_mat and C_mat must match the batch and sequence dimensions.")

    # Initialize the recurrent state.
    if h0 is None:
        h = u.new_zeros(
            batch_size,
            num_heads,
            head_dim,
            num_states
        )
    else:
        if h0.shape != (
            batch_size,
            num_heads,
            head_dim,
            num_states
        ):
            raise ValueError("h0 must have shape (B, H, P, N).")

        h = h0

    # An empty sequence returns an empty output and the initial state.
    if seq_len == 0:
        return u.new_empty(
            batch_size,
            0,
            num_heads,
            head_dim
        ), h

    # Store the output produced by each chunk.
    outputs = []

    # Process the sequence in chunks.
    for start in range(0, seq_len, chunk):

        end = min(start + chunk, seq_len)

        # Extract the current chunk.
        u_c = u[:, start:end]
        dt_c = dt[:, start:end]
        B_c = B_mat[:, start:end]
        C_c = C_mat[:, start:end]

        # Compute a_t = A * dt_t.
        a = dt_c * A.view(1, 1, num_heads)

        # Compute the cumulative state-transition exponent.
        # Shape: (B, T, H).
        s = torch.cumsum(a, dim=1)

        # Compute v_j = dt_j * u_j.
        # Shape: (B, T, H, P).
        v = dt_c.unsqueeze(-1) * u_c

        # --------------------------------------------------
        # 1. Within-chunk interactions
        # --------------------------------------------------

        # Calculate C_i^T * B_j for every pair of positions.
        # Shape: (B, T, T).
        CB = torch.einsum(
            "bin,bjn->bij",
            C_c,
            B_c
        )

        # Calculate s_i - s_j.
        # Shape: (B, T, T, H).
        delta = (
            s.unsqueeze(2)
            - s.unsqueeze(1)
        )

        T = end - start

        # Construct a lower-triangular causal mask.
        lower_mask = torch.tril(
            torch.ones(
                T,
                T,
                dtype=torch.bool,
                device=u.device
            )
        )

        # Suppress future-position interactions before exponentiation.
        # This also avoids unnecessary exponential overflow.
        delta = delta.masked_fill(
            ~lower_mask.unsqueeze(0).unsqueeze(-1),
            float("-inf")
        )

        # Calculate exp(s_i - s_j).
        # Shape: (B, T, T, H).
        decay = torch.exp(delta)

        # Combine the C-B interaction with the decay factor.
        # Shape: (B, T, T, H).
        weights = CB.unsqueeze(-1) * decay

        # Sum the contributions from all previous positions.
        # Shape: (B, T, H, P).
        y_chunk = torch.einsum(
            "bijh,bjhp->bihp",
            weights,
            v
        )

        # --------------------------------------------------
        # 2. Contribution from the incoming state
        # --------------------------------------------------

        # Calculate C_i^T * h_in.
        # Shape: (B, T, H, P).
        initial_output = torch.einsum(
            "bin,bhpn->bihp",
            C_c,
            h
        )

        # Multiply by exp(s_i).
        # Shape: (B, T, H, P).
        initial_output = (
            torch.exp(s).unsqueeze(-1)
            * initial_output
        )

        # Combine within-chunk and incoming-state contributions.
        y_chunk = y_chunk + initial_output

        outputs.append(y_chunk)

        # --------------------------------------------------
        # 3. Compute the outgoing state
        # --------------------------------------------------

        # The last cumulative exponent in this chunk.
        s_last = s[:, -1, :]

        # Calculate exp(s_last - s_j).
        # Shape: (B, T, H).
        final_decay = torch.exp(
            s_last.unsqueeze(1) - s
        )

        # Calculate the accumulated input contribution:
        #
        # sum_j exp(s_last - s_j) * B_j * v_j
        #
        # Result shape: (B, H, P, N).
        state_update = torch.einsum(
            "bjh,bjn,bjhp->bhpn",
            final_decay,
            B_c,
            v
        )

        # Propagate the incoming state through the chunk.
        # Shape: (B, H, P, N).
        propagated_state = (
            torch.exp(s_last)[:, :, None, None] * h
        )

        # Update the state for the next chunk.
        h = propagated_state + state_update

    # Concatenate chunk outputs along the sequence dimension.
    y = torch.cat(outputs, dim=1)

    return y, h

# Step 14 - MambaLayer
class MambaLayer(nn.Module):
    """
    Small Mamba-2 block with pre-normalization and a residual connection.
    """

    def __init__(
        self,
        d,
        expand=2,
        d_state=16,
        d_conv=4,
        head_dim=16,
        chunk=64
    ):
        super().__init__()

        # Validate architectural dimensions.
        if d <= 0 or expand <= 0:
            raise ValueError("d and expand must be positive.")

        if d_state <= 0 or d_conv <= 0 or head_dim <= 0:
            raise ValueError("d_state, d_conv and head_dim must be positive.")

        if chunk <= 0:
            raise ValueError("chunk must be positive.")

        # Inner dimension and number of heads.
        di = expand * d

        if di % head_dim != 0:
            raise ValueError(
                "expand * d must be divisible by head_dim."
            )

        H = di // head_dim

        # Store architectural dimensions.
        self.di = di
        self.H = H
        self.head_dim = head_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.chunk = chunk

        # --------------------------------------------------
        # 1. Normalization layers
        # --------------------------------------------------

        # Pre-normalization of the input.
        self.norm = RMSNorm(d)

        # Normalization of the gated output.
        self.gate_norm = RMSNorm(di)

        # --------------------------------------------------
        # 2. Input projection
        # --------------------------------------------------

        # x is projected into:
        # z:   di
        # xbc: di + 2*d_state
        # dt:  H
        self.in_proj = nn.Linear(
            d,
            2 * di + 2 * d_state + H,
            bias=False
        )

        # --------------------------------------------------
        # 3. Causal depthwise convolution
        # --------------------------------------------------

        n = di + 2 * d_state

        self.conv = nn.Conv1d(
            n,
            n,
            kernel_size=d_conv,
            groups=n,
            padding=d_conv - 1
        )

        # --------------------------------------------------
        # 4. Output projection
        # --------------------------------------------------

        self.out_proj = nn.Linear(
            di,
            d,
            bias=False
        )

        # --------------------------------------------------
        # 5. State transition initialization
        # --------------------------------------------------

        # Initialize A using log-uniformly sampled values
        # from the interval [1, 16].
        A_init = torch.empty(H).uniform_(1.0, 16.0)

        self.log_a = nn.Parameter(torch.log(A_init))

        # Sample initial time-step values from a log-uniform
        # distribution between 0.001 and 0.1.
        dt_init = torch.empty(H).uniform_(
            math.log(0.001),
            math.log(0.1)
        ).exp()

        # Inverse softplus:
        # softplus(dt_bias) = dt_init.
        # This is calculated in a numerically stable manner.
        dt_bias_init = (
            dt_init + torch.log(-torch.expm1(-dt_init))
        )

        self.dt_bias = nn.Parameter(dt_bias_init)

    def forward(self, x):
        """
        Apply the Mamba-2 block.

        Args:
            x (torch.Tensor): Input of shape (B, L, d).

        Returns:
            torch.Tensor: Output of shape (B, L, d).
        """

        if x.ndim != 3:
            raise ValueError("x must have shape (B, L, d).")

        if x.shape[-1] != self.norm.weight.numel():
            raise ValueError("Input feature dimension does not match d.")

        B, L, _ = x.shape

        # Nothing to process for an empty sequence.
        if L == 0:
            return x

        # Keep the original input for the residual connection.
        residual = x

        # --------------------------------------------------
        # 1. Normalize and project the input
        # --------------------------------------------------

        projected = self.in_proj(self.norm(x))

        n = self.di + 2 * self.d_state

        z, xbc, dt = torch.split(
            projected,
            [self.di, n, self.H],
            dim=-1
        )

        # --------------------------------------------------
        # 2. Causal depthwise convolution
        # --------------------------------------------------

        # Conv1d expects (B, channels, L).
        xbc = xbc.transpose(1, 2)

        # Apply convolution and retain only the first L positions.
        # This removes the right-padding outputs and ensures causality.
        xbc = self.conv(xbc)[..., :L]

        # Restore shape (B, L, channels).
        xbc = xbc.transpose(1, 2)

        # Apply SiLU activation.
        xbc = F.silu(xbc)

        # Split into input, input-state and output-state vectors.
        u, B_mat, C_mat = torch.split(
            xbc,
            [self.di, self.d_state, self.d_state],
            dim=-1
        )

        # --------------------------------------------------
        # 3. State-space scan
        # --------------------------------------------------

        # Convert time-step values using softplus.
        dt = F.softplus(dt + self.dt_bias)

        # State transition parameters must be negative.
        A = -torch.exp(self.log_a)

        # Reshape the input into multiple heads.
        u = u.reshape(
            B,
            L,
            self.H,
            self.head_dim
        )

        # Execute the chunked Mamba-2 SSD scan.
        y, _ = ssd_scan(
            u,
            dt,
            A,
            B_mat,
            C_mat,
            chunk=self.chunk
        )

        # Flatten the heads back into the inner dimension.
        y = y.reshape(B, L, self.di)

        # --------------------------------------------------
        # 4. Gating and output projection
        # --------------------------------------------------

        # Apply SiLU gating and RMS normalization.
        gated = self.gate_norm(
            y * F.silu(z)
        )

        # Project back to the original dimension.
        out = self.out_proj(gated)

        # --------------------------------------------------
        # 5. Residual connection
        # --------------------------------------------------

        return residual + out

# Step 15 - apply_rope
def apply_rope(x, offset=0, theta=10000.0):
    """
    Apply Rotary Positional Embeddings to attention head vectors.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, L, hd).
        offset (int): Starting position offset.
        theta (float): RoPE frequency base.

    Returns:
        torch.Tensor: Rotated tensor with the same shape and dtype as x.
    """

    # Validate input dimensions.
    if x.ndim != 4:
        raise ValueError("x must have shape (B, H, L, hd).")

    B, H, L, hd = x.shape

    # RoPE requires an even head dimension.
    if hd % 2 != 0:
        raise ValueError("The head dimension must be even.")

    if theta <= 0:
        raise ValueError("theta must be positive.")

    # Half of the head dimension.
    m = hd // 2

    # Use float64 for double inputs and float32 for other input types
    # to improve numerical stability when calculating frequencies.
    compute_dtype = (
        torch.float64 if x.dtype == torch.float64 else torch.float32
    )

    # Generate indices for the inverse frequencies.
    freq_indices = torch.arange(
        m,
        device=x.device,
        dtype=compute_dtype
    )

    # Calculate inverse frequencies.
    inv_freq = 1.0 / (theta ** (freq_indices / m))

    # Generate sequence positions, including the optional offset.
    positions = torch.arange(
        L,
        device=x.device,
        dtype=compute_dtype
    ) + offset

    # Calculate position-dependent angles.
    angles = positions[:, None] * inv_freq[None, :]

    # Calculate cosine and sine.
    cos_angles = torch.cos(angles).to(dtype=x.dtype)
    sin_angles = torch.sin(angles).to(dtype=x.dtype)

    # Reshape for broadcasting across batch and attention heads.
    cos_angles = cos_angles[None, None, :, :]
    sin_angles = sin_angles[None, None, :, :]

    # Split the head dimension into two equal halves.
    a = x[..., :m]
    b = x[..., m:]

    # Apply rotary transformation.
    first_half = a * cos_angles - b * sin_angles
    second_half = b * cos_angles + a * sin_angles

    # Concatenate the rotated halves.
    return torch.cat([first_half, second_half], dim=-1)

# Step 16 - TransformerLayer
class TransformerLayer(nn.Module):
    """
    Pre-norm Transformer layer with:
    - Multi-head causal self-attention
    - Rotary Positional Embeddings (RoPE)
    - Gated MLP
    - Residual connections
    """

    def __init__(self, d, heads=4, d_ff=None):
        super().__init__()

        # Validate the attention dimensions.
        if d <= 0:
            raise ValueError("d must be positive.")

        if heads <= 0:
            raise ValueError("heads must be positive.")

        if d % heads != 0:
            raise ValueError("d must be divisible by heads.")

        # Dimension of each attention head.
        self.heads = heads
        self.hd = d // heads

        # Default feed-forward dimension.
        if d_ff is None:
            d_ff = int(8 * d / 3)

        if d_ff <= 0:
            raise ValueError("d_ff must be positive.")

        # --------------------------------------------------
        # 1. Pre-normalization layers
        # --------------------------------------------------

        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)

        # --------------------------------------------------
        # 2. Multi-head attention projections
        # --------------------------------------------------

        # Combined query, key and value projection.
        self.qkv = nn.Linear(
            d,
            3 * d,
            bias=False
        )

        # Attention output projection.
        self.out = nn.Linear(
            d,
            d,
            bias=False
        )

        # --------------------------------------------------
        # 3. Gated MLP projections
        # --------------------------------------------------

        # First MLP projection.
        self.fc1 = nn.Linear(
            d,
            d_ff,
            bias=False
        )

        # Gating projection.
        self.fc_gate = nn.Linear(
            d,
            d_ff,
            bias=False
        )

        # MLP output projection.
        self.fc2 = nn.Linear(
            d_ff,
            d,
            bias=False
        )

    def forward(self, x):
        """
        Apply causal self-attention followed by a gated MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d).

        Returns:
            torch.Tensor: Output tensor of shape (B, L, d).
        """

        if x.ndim != 3:
            raise ValueError("x must have shape (B, L, d).")

        B, L, d = x.shape

        if L == 0:
            return x

        if d != self.heads * self.hd:
            raise ValueError("Input dimension does not match the layer.")

        # --------------------------------------------------
        # 1. Pre-normalization and QKV projection
        # --------------------------------------------------

        qkv = self.qkv(self.norm1(x))

        # Split into query, key and value tensors.
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape each tensor into (B, heads, L, head_dim).
        q = q.reshape(B, L, self.heads, self.hd).transpose(1, 2)
        k = k.reshape(B, L, self.heads, self.hd).transpose(1, 2)
        v = v.reshape(B, L, self.heads, self.hd).transpose(1, 2)

        # --------------------------------------------------
        # 2. Apply Rotary Positional Embeddings
        # --------------------------------------------------

        # Rotate queries and keys, but not values.
        q = apply_rope(q)
        k = apply_rope(k)

        # --------------------------------------------------
        # 3. Causal multi-head self-attention
        # --------------------------------------------------

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True
        )

        # Restore the original shape (B, L, d).
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .reshape(B, L, d)
        )

        # Apply output projection and residual connection.
        x = x + self.out(attn)

        # --------------------------------------------------
        # 4. Gated MLP with pre-normalization
        # --------------------------------------------------

        t = self.norm2(x)

        # Apply the gated feed-forward transformation.
        mlp_out = (
            F.silu(self.fc1(t))
            * self.fc_gate(t)
        )

        # Project back to dimension d and add residual.
        return x + self.fc2(mlp_out)

# Step 17 - build_stack
import re


def build_stack(layout, d, d_state=16, head_dim=16, chunk=64, heads=4):
    """
    Convert a layout string into an ordered neural network stack.

    Layout syntax:
        m<count> : MambaLayer
        T<count> : TransformerLayer

    Args:
        layout (str): Architecture description, e.g. "m1T2".
        d (int): Hidden dimension.
        d_state (int): Mamba state dimension.
        head_dim (int): Mamba head dimension.
        chunk (int): Mamba SSD chunk size.
        heads (int): Number of Transformer attention heads.

    Returns:
        nn.Sequential: Ordered layers followed by a final RMSNorm.
    """

    # Validate the layout string.
    if not isinstance(layout, str):
        raise TypeError("layout must be a string.")

    # Require one or more valid letter-count groups.
    if not re.fullmatch(r"(?:m\d+|T\d+)+", layout):
        raise ValueError(
            "Invalid layout. Expected groups such as 'm2', 'T3', or 'm1T2'."
        )

    layers = []

    # Extract each layer type and its repetition count.
    groups = re.findall(r"(m|T)(\d+)", layout)

    # Construct layers in the order specified by the layout.
    for layer_type, count_str in groups:

        count = int(count_str)

        if count <= 0:
            raise ValueError("Layer counts must be positive integers.")

        for _ in range(count):

            if layer_type == "m":
                # Create a Mamba-2 layer.
                layers.append(
                    MambaLayer(
                        d,
                        d_state=d_state,
                        head_dim=head_dim,
                        chunk=chunk
                    )
                )

            elif layer_type == "T":
                # Create a Transformer layer.
                layers.append(
                    TransformerLayer(
                        d,
                        heads=heads
                    )
                )

    # Append exactly one final RMSNorm layer.
    layers.append(RMSNorm(d))

    # Return the complete sequential stack.
    return nn.Sequential(*layers)

# Step 18 - HNetStage
class HNetStage(nn.Module):
    """
    One recursive level of the hierarchical H-Net architecture.

    Non-innermost stages perform:
        Encoder -> Router -> Compaction -> Recursive Substage
        -> Smoothing -> Upsampling -> Residual -> Decoder

    The innermost stage directly applies its main network.
    """

    def __init__(self, layout, widths, targets, cfg=None, s=0):
        super().__init__()

        # Default configuration.
        cfg = {} if cfg is None else cfg

        # Current stage hidden dimension.
        self.d = widths[s]

        # Determine whether this is the innermost stage.
        # A single layout entry represents the main network.
        self.inner = len(layout) == 1

        # --------------------------------------------------
        # 1. Optional dimension padding
        # --------------------------------------------------

        # When a stage is wider than its preceding stage,
        # introduce learnable zero-initialized padding.
        if s > 0 and widths[s] > widths[s - 1]:
            self.pad = nn.Parameter(
                torch.zeros(widths[s] - widths[s - 1])
            )
        else:
            self.pad = None

        # Stack configuration shared by the stage networks.
        stack_cfg = cfg.get("stack", {})

        # --------------------------------------------------
        # 2. Innermost stage
        # --------------------------------------------------

        if self.inner:

            # Build the main network.
            self.main = build_stack(
                layout[0],
                self.d,
                **stack_cfg
            )

            # No router, encoder, decoder or recursive
            # substage is required at the innermost level.
            return

        # --------------------------------------------------
        # 3. Non-innermost stage
        # --------------------------------------------------

        if len(layout) != 3:
            raise ValueError(
                "A non-innermost layout must contain "
                "[encoder, inner, decoder]."
            )

        if s >= len(targets):
            raise ValueError("Missing compression target for stage.")

        # Store configuration and compression target.
        self.N = targets[s]
        self.cfg = cfg

        self.no_smoothing = cfg.get("no_smoothing", False)
        self.no_ste = cfg.get("no_ste", False)

        # --------------------------------------------------
        # 4. Encoder
        # --------------------------------------------------

        self.encoder = build_stack(
            layout[0],
            self.d,
            **stack_cfg
        )

        # --------------------------------------------------
        # 5. Boundary router
        # --------------------------------------------------

        router_cls = cfg.get("router_cls", RoutingModule)

        self.router = router_cls(self.d)

        # --------------------------------------------------
        # 6. Recursive substage
        # --------------------------------------------------

        self.sub = HNetStage(
            layout[1],
            widths,
            targets,
            cfg,
            s + 1
        )

        # --------------------------------------------------
        # 7. Decoder
        # --------------------------------------------------

        self.decoder = build_stack(
            layout[2],
            self.d,
            **stack_cfg
        )

        # --------------------------------------------------
        # 8. Zero-initialized residual projection
        # --------------------------------------------------

        self.res = nn.Linear(
            self.d,
            self.d
        )

        # Initialize residual projection to zero.
        # Initially, the residual branch contributes zero.
        with torch.no_grad():
            self.res.weight.zero_()
            self.res.bias.zero_()

    def forward(self, x, mask, boundaries=None, stats=None):
        """
        Execute one recursive H-Net stage.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, L, D).

            mask (torch.Tensor):
                Boolean validity mask of shape (B, L).

            boundaries (torch.Tensor, optional):
                Predefined boolean chunk boundaries of shape (B, L).
                When supplied, the router is bypassed.

            stats (list, optional):
                List used to collect router and chunking statistics.

        Returns:
            torch.Tensor:
                Output tensor with the same shape as the original input.
        """

        # Preserve the incoming feature dimension.
        D = x.shape[-1]

        # --------------------------------------------------
        # 1. Optional dimension padding
        # --------------------------------------------------

        if self.pad is not None:

            # Expand the learnable padding vector across
            # all batch and sequence positions.
            pad = self.pad.view(1, 1, -1).expand(
                x.shape[0],
                x.shape[1],
                -1
            )

            # Append padding features.
            x = torch.cat([x, pad], dim=-1)

        # --------------------------------------------------
        # 2. Innermost stage
        # --------------------------------------------------

        if self.inner:

            # Apply the main network and restore the
            # original incoming feature dimension.
            return self.main(x)[..., :D]

        # --------------------------------------------------
        # 3. Encode input and calculate residual
        # --------------------------------------------------

        xhat = self.encoder(x)

        # Residual projection.
        r = self.res(xhat)

        # --------------------------------------------------
        # 4. Determine chunk boundaries
        # --------------------------------------------------

        if boundaries is not None:

            # Use externally supplied boundaries.
            # The mask suppresses invalid positions.
            b = boundaries & mask

            # Convert binary boundaries into floating-point
            # probabilities.
            p = b.to(dtype=xhat.dtype)

        else:

            # Predict probabilities and boundaries with the router.
            p, b = self.router(xhat, mask)

        # --------------------------------------------------
        # 5. Collect optional statistics
        # --------------------------------------------------

        if stats is not None:

            stats.append({
                "p": p,
                "b": b,
                "mask": mask,
                "N": self.N,
                "learned": boundaries is None
            })

        # --------------------------------------------------
        # 6. Compact the sequence
        # --------------------------------------------------

        # Keep boundary vectors and pack them to the left.
        x2, mask2 = compact_chunks(xhat, b)

        # Process the compressed sequence recursively.
        zhat = self.sub(
            x2,
            mask2,
            None,
            stats
        )

        # --------------------------------------------------
        # 7. Smooth uncertain chunks
        # --------------------------------------------------

        if boundaries is None and not self.no_smoothing:

            # Compact router probabilities using the same
            # boundary positions as the hidden vectors.
            P, _ = compact_chunks(p, b)

            # Apply exponential moving average smoothing.
            zbar = smooth_chunks(zhat, P)

        else:

            # Skip smoothing for predefined boundaries
            # or when smoothing is disabled.
            zbar = zhat

        # --------------------------------------------------
        # 8. Causal upsampling
        # --------------------------------------------------

        # Restore the original byte-level sequence length.
        zt = causal_upsample(zbar, b)

        # Apply confidence only when boundaries are router-predicted.
        if boundaries is None:

            if self.no_ste:

                # Ordinary confidence without the STE.
                confidence = torch.where(
                    b,
                    p,
                    1.0 - p
                )

            else:

                # Straight-through confidence estimator.
                confidence = straight_through_confidence(
                    p,
                    b
                )

            # Broadcast confidence over the feature dimension.
            zt = zt * confidence.unsqueeze(-1)

        # --------------------------------------------------
        # 9. Residual addition and decoding
        # --------------------------------------------------

        # Combine upsampled hierarchical representations
        # with the residual representation.
        combined = zt + r

        # Decode and restore the original incoming dimension.
        output = self.decoder(combined)[..., :D]

        return output

# Step 19 - HNetLM
class HNetLM(nn.Module):
    """
    Complete byte-level hierarchical language model.

    Architecture:
        Byte Embedding -> Recursive H-Net Stage -> Output Head

    The model predicts one of 256 possible byte values
    at every sequence position.
    """

    def __init__(self, layout, widths, targets, cfg=None):
        super().__init__()

        # Store the model architecture configuration.
        self.layout = layout
        self.widths = widths
        self.targets = targets

        # --------------------------------------------------
        # 1. Byte embedding
        # --------------------------------------------------

        # The vocabulary contains all 256 possible byte values.
        self.embed = nn.Embedding(
            num_embeddings=256,
            embedding_dim=widths[0]
        )

        # --------------------------------------------------
        # 2. Hierarchical H-Net stage
        # --------------------------------------------------

        # Construct the recursive H-Net architecture.
        self.stage = HNetStage(
            layout,
            widths,
            targets,
            cfg
        )

        # --------------------------------------------------
        # 3. Output projection
        # --------------------------------------------------

        # Project hidden representations into 256 byte logits.
        # Bias is disabled as required.
        self.head = nn.Linear(
            widths[0],
            256,
            bias=False
        )

    def forward(self, ids, boundaries=None):
        """
        Forward pass of the byte-level language model.

        Args:
            ids (torch.Tensor):
                Long tensor of shape (B, L) containing byte IDs.

            boundaries (torch.Tensor, optional):
                Optional predefined boolean chunk boundaries
                of shape (B, L).

        Returns:
            tuple:
                logits: Tensor of shape (B, L, 256).
                stats: List containing router statistics from
                       the hierarchical stages.
        """

        if ids.ndim != 2:
            raise ValueError("ids must have shape (B, L).")

        # --------------------------------------------------
        # 1. Create an all-True sequence validity mask
        # --------------------------------------------------

        mask = torch.ones(
            ids.shape,
            dtype=torch.bool,
            device=ids.device
        )

        # --------------------------------------------------
        # 2. Embed the raw byte IDs
        # --------------------------------------------------

        x = self.embed(ids)

        # --------------------------------------------------
        # 3. Run the hierarchical H-Net stage
        # --------------------------------------------------

        # Create a fresh statistics list for this forward pass.
        stats = []

        x = self.stage(
            x,
            mask,
            boundaries=boundaries,
            stats=stats
        )

        # --------------------------------------------------
        # 4. Predict byte-level logits
        # --------------------------------------------------

        logits = self.head(x)

        return logits, stats

# Step 20 - lr_multipliers
def lr_multipliers(Ns, Ds, n_gpt=4.6):
    """
    Calculate stage-wise learning-rate multipliers using Equation 11.

    Args:
        Ns (list): Compression targets of all non-innermost stages.
        Ds (list): Widths of all stages, including the innermost stage.
        n_gpt (float): GPT reference compression factor.

    Returns:
        list: Learning-rate multipliers, outermost stage first.
    """

    # Validate the input dimensions.
    if len(Ds) != len(Ns) + 1:
        raise ValueError(
            "Ds must contain exactly one more element than Ns."
        )

    if len(Ds) == 0:
        raise ValueError("Ds must not be empty.")

    # All compression targets and dimensions must be positive.
    if any(N <= 0 for N in Ns):
        raise ValueError("All compression targets must be positive.")

    if any(D <= 0 for D in Ds):
        raise ValueError("All stage widths must be positive.")

    if n_gpt <= 0:
        raise ValueError("n_gpt must be positive.")

    # The innermost stage has a compression target of 1.
    stage_targets = list(Ns) + [1.0]

    # Total number of stages (S + 1).
    S = len(stage_targets) - 1

    # Product of compression targets across all stages.
    total_product = math.prod(stage_targets)

    multipliers = []

    # Calculate one multiplier for each stage, outermost first.
    for s in range(S + 1):

        # Product of compression targets from stage s to S.
        suffix_product = math.prod(stage_targets[s:])

        # Apply Equation 11.
        multiplier = math.sqrt(
            n_gpt
            * (suffix_product / total_product)
            * (Ds[S] / Ds[s])
        )

        multipliers.append(float(multiplier))

    return multipliers

# Step 21 - stage_param_groups
def stage_param_groups(model, base_lr, weight_decay, mults):
    """
    Assign every HNetLM parameter to its corresponding hierarchical stage.

    Args:
        model: HNetLM model.
        base_lr (float): Base learning rate.
        weight_decay (float): Weight decay coefficient.
        mults (list): Stage-wise learning-rate multipliers.

    Returns:
        list: One optimizer parameter-group dictionary per stage.
    """

    # Collect all recursive H-Net stages, starting from stage 0.
    stages = []
    current = model.stage

    while current is not None:
        stages.append(current)
        current = getattr(current, "sub", None)

    # There must be exactly one multiplier per stage.
    if len(mults) != len(stages):
        raise ValueError(
            "The number of learning-rate multipliers must "
            "match the number of H-Net stages."
        )

    # Track parameter identities to prevent duplicate assignment.
    assigned = set()

    def collect_params(module, params):
        """Collect parameters from a module without duplicates."""
        for param in module.parameters():
            param_id = id(param)

            if param_id not in assigned:
                assigned.add(param_id)
                params.append(param)

    groups = []

    # --------------------------------------------------
    # Assign parameters stage by stage.
    # --------------------------------------------------

    for s, stage in enumerate(stages):

        stage_params = []

        # Stage 0 additionally owns the byte embedding and
        # the final byte-prediction head.
        if s == 0:
            collect_params(model.embed, stage_params)
            collect_params(model.head, stage_params)

        # Collect parameters registered directly on this stage.
        # This includes its optional padding parameter.
        for param in stage.parameters(recurse=False):
            param_id = id(param)

            if param_id not in assigned:
                assigned.add(param_id)
                stage_params.append(param)

        # Collect parameters from this stage's own modules.
        # Exclude 'sub', since it belongs to the next stage.
        for name, child in stage.named_children():

            if name == "sub":
                continue

            collect_params(child, stage_params)

        # Create the optimizer parameter group.
        lr_mult = float(mults[s])

        groups.append({
            "params": stage_params,
            "lr": base_lr * lr_mult,
            "lr_mult": lr_mult,
            "weight_decay": weight_decay
        })

    # --------------------------------------------------
    # Verify that every model parameter was assigned once.
    # --------------------------------------------------

    model_param_ids = {
        id(param) for param in model.parameters()
    }

    if assigned != model_param_ids:
        raise RuntimeError(
            "Parameter grouping failed: some model parameters "
            "were missed or incorrectly assigned."
        )

    return groups

# Step 22 - wsd_factor
def wsd_factor(step, total, warmup=0.1, decay=0.2, floor=0.1):
    """
    Calculate the learning-rate multiplier using a
    Warmup-Stable-Decay (WSD) schedule.

    Args:
        step (int): Current training step.
        total (int): Total number of training steps.
        warmup (float): Fraction of training allocated to warmup.
        decay (float): Fraction of training allocated to decay.
        floor (float): Minimum learning-rate multiplier.

    Returns:
        float: Learning-rate multiplier for the current step.
    """

    # Validate schedule parameters.
    if total <= 0:
        raise ValueError("total must be positive.")

    if step < 0:
        raise ValueError("step must be non-negative.")

    if not (0.0 <= warmup <= 1.0):
        raise ValueError("warmup must be between 0 and 1.")

    if not (0.0 <= decay <= 1.0):
        raise ValueError("decay must be between 0 and 1.")

    if warmup + decay > 1.0:
        raise ValueError("warmup + decay must not exceed 1.")

    if not (0.0 < floor <= 1.0):
        raise ValueError("floor must be greater than 0 and at most 1.")

    # Calculate warmup and decay durations.
    w = warmup * total
    d = decay * total

    # --------------------------------------------------
    # 1. Linear warmup phase
    # --------------------------------------------------

    if step < w:
        return float((step + 1) / w)

    # --------------------------------------------------
    # 2. Stable phase
    # --------------------------------------------------

    if step < total - d:
        return 1.0

    # If decay is disabled, retain the full learning rate.
    if d == 0:
        return 1.0

    # --------------------------------------------------
    # 3. Inverse-square-root decay phase
    # --------------------------------------------------

    tau = (step - (total - d)) / d

    # Keep the decay progress within the interval [0, 1].
    tau = max(0.0, min(1.0, tau))

    # Calculate the inverse-square-root decay factor.
    factor = 1.0 / math.sqrt(
        1.0 + (1.0 / (floor ** 2) - 1.0) * tau
    )

    return float(factor)

# Step 23 - layer_flops
def layer_flops(
    kind,
    L,
    d,
    heads=4,
    d_ff=None,
    expand=2,
    d_state=16,
    head_dim=16,
    d_conv=4
):
    """
    Calculate forward-pass FLOPs for one Transformer or Mamba-2 layer.

    Args:
        kind (str): "T" for Transformer or "m" for Mamba-2.
        L (int or float): Sequence length; may be fractional.
        d (int): Model hidden dimension.
        heads (int): Number of Transformer attention heads.
        d_ff (int, optional): Transformer feed-forward dimension.
        expand (int): Mamba expansion factor.
        d_state (int): Mamba state dimension.
        head_dim (int): Mamba head dimension.
        d_conv (int): Mamba convolution kernel size.

    Returns:
        int or float: Estimated forward-pass FLOPs.
    """

    # Validate common dimensions.
    if L < 0:
        raise ValueError("L must be non-negative.")

    if d <= 0:
        raise ValueError("d must be positive.")

    # --------------------------------------------------
    # 1. Transformer FLOPs
    # --------------------------------------------------

    if kind == "T":

        if heads <= 0:
            raise ValueError("heads must be positive.")

        # Default feed-forward dimension.
        f = d_ff if d_ff is not None else int(8 * d / 3)

        if f <= 0:
            raise ValueError("d_ff must be positive.")

        # QKV projections: 3 linear projections.
        qkv_flops = 2 * 3 * L * d * d

        # Attention logits: Q multiplied by K transpose.
        attention_logits = 2 * L**2 * d

        # Softmax computation across attention heads.
        softmax_flops = 3 * heads * L**2

        # Attention scores multiplied by V.
        attention_value = 2 * L**2 * d

        # Attention output projection.
        output_projection = 2 * L * d**2

        # Three gated MLP projections.
        mlp_projections = 2 * L * d * 3 * f

        # Gated MLP activation and element-wise operations.
        gating_flops = 5 * L * d

        total = (
            qkv_flops
            + attention_logits
            + softmax_flops
            + attention_value
            + output_projection
            + mlp_projections
            + gating_flops
        )

    # --------------------------------------------------
    # 2. Mamba-2 FLOPs
    # --------------------------------------------------

    elif kind == "m":

        if expand <= 0:
            raise ValueError("expand must be positive.")

        if d_state <= 0:
            raise ValueError("d_state must be positive.")

        if head_dim <= 0:
            raise ValueError("head_dim must be positive.")

        if d_conv <= 0:
            raise ValueError("d_conv must be positive.")

        # Expanded inner dimension.
        e = expand

        # Number of Mamba heads.
        H = (e * d) // head_dim

        if H <= 0:
            raise ValueError("Invalid Mamba head configuration.")

        # X and Z input projections.
        xz_projections = 2 * L * d * (2 * e * d)

        # B, C and delta-t projections.
        bcdt_projections = (
            2 * L * d * (2 * d_state + H)
        )

        # State Space Duality (SSD) computation.
        ssd_flops = 2 * 3 * L * (e * d) * d_state

        # Depthwise causal convolution.
        conv_flops = 2 * L * d_conv * d

        # Gating operations.
        gating_flops = 5 * L * d

        # Output projection back to dimension d.
        output_projection = 2 * L * d**2

        total = (
            xz_projections
            + bcdt_projections
            + ssd_flops
            + conv_flops
            + gating_flops
            + output_projection
        )

    else:
        raise ValueError(
            f"Unknown layer kind: {kind!r}. Use 'T' or 'm'."
        )

    return total

# Step 24 - flops_per_byte
def flops_per_byte(layout, widths, bytes_per_chunk, seq_len=128, **kw):
    """
    Calculate total forward FLOPs per input byte for a nested H-Net.

    Each stage's sequence length is calculated relative to the
    original input sequence length, not cumulatively.
    """

    # Validate architecture dimensions.
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")

    if not widths:
        raise ValueError("widths must not be empty.")

    if len(widths) != len(bytes_per_chunk) + 1:
        raise ValueError(
            "widths must contain exactly one more element "
            "than bytes_per_chunk."
        )

    if any(d <= 0 for d in widths):
        raise ValueError("All stage widths must be positive.")

    if any(b <= 0 for b in bytes_per_chunk):
        raise ValueError("All compression factors must be positive.")

    num_stages = len(widths) - 1

    # Calculate FLOPs for a stack such as m2, T3 or m1T2.
    def stack_flops(stack_layout, d, positions):

        if not isinstance(stack_layout, str):
            raise TypeError("Stack layout must be a string.")

        if not re.fullmatch(r"(?:m\d+|T\d+)+", stack_layout):
            raise ValueError(f"Invalid stack layout: {stack_layout!r}")

        total = 0.0

        for kind, count_str in re.findall(
            r"(m|T)(\d+)", stack_layout
        ):
            count = int(count_str)

            if count <= 0:
                raise ValueError("Layer counts must be positive.")

            total += count * layer_flops(
                kind,
                positions,
                d,
                **kw
            )

        return total

    # Recursively traverse the nested H-Net architecture.
    def traverse(node, stage):

        d = widths[stage]

        # IMPORTANT:
        # Every stage's positions are measured relative to seq_len.
        if stage == 0:
            positions = seq_len
        else:
            positions = seq_len / bytes_per_chunk[stage - 1]

        # Innermost main network represented as a stack string.
        if isinstance(node, str):

            if stage != num_stages:
                raise ValueError("Unexpected main-network depth.")

            return stack_flops(node, d, positions)

        if not isinstance(node, (list, tuple)):
            raise TypeError("Invalid nested layout.")

        # Innermost stage: [main]
        if len(node) == 1:

            if stage != num_stages:
                raise ValueError("Unexpected innermost stage.")

            return stack_flops(node[0], d, positions)

        # Non-innermost stage: [encoder, inner, decoder]
        if len(node) != 3:
            raise ValueError(
                "Expected [encoder, inner, decoder] layout."
            )

        if stage >= num_stages:
            raise ValueError("Missing compression factor.")

        encoder, inner, decoder = node

        # Encoder operates at the current stage's resolution.
        encoder_cost = stack_flops(
            encoder, d, positions
        )

        # Recursively calculate the next stage.
        # Its sequence length will be calculated directly
        # from the original seq_len.
        inner_cost = traverse(
            inner, stage + 1
        )

        # Decoder operates at the current stage's resolution.
        decoder_cost = stack_flops(
            decoder, d, positions
        )

        return encoder_cost + inner_cost + decoder_cost

    # Calculate the total hierarchical network cost.
    network_flops = traverse(layout, 0)

    # Embedding and output-head FLOPs.
    embed_head_flops = (
        2 * (2 * seq_len * 256 * widths[0])
    )

    total_flops = network_flops + embed_head_flops

    # Return average forward FLOPs per input byte.
    return total_flops / seq_len

# Step 25 - evaluate_bpb
def evaluate_bpb(
    model,
    data,
    chunker=None,
    seq_len=128,
    batch_size=32,
    n_batches=8,
    seed=999
):
    """
    Evaluate held-out bits per byte and actual compression.

    Args:
        model: HNetLM model to evaluate.
        data (torch.Tensor): 1-D tensor of raw byte values.
        chunker (callable, optional): Function mapping input IDs
            to a boolean boundary mask of shape (B, L).
        seq_len (int): Evaluation sequence length.
        batch_size (int): Number of sequences per batch.
        n_batches (int): Number of evaluation batches.
        seed (int): Seed for reproducible evaluation windows.

    Returns:
        dict:
            bpb (float): Mean negative log-likelihood in bits per byte.
            bytes_per_chunk (list): Actual cumulative compression
                for each chunking stage.
    """

    # Validate evaluation parameters.
    if data.ndim != 1:
        raise ValueError("data must be a 1-D tensor.")

    if seq_len <= 0 or batch_size <= 0 or n_batches <= 0:
        raise ValueError(
            "seq_len, batch_size and n_batches must be positive."
        )

    if len(data) <= seq_len + 1:
        raise ValueError("data must be longer than seq_len + 1.")

    # Obtain the model's device.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = data.device

    # Use CPU data for reproducible sampling with a CPU generator.
    sampling_data = data.detach().to(
        device="cpu",
        dtype=torch.long
    )

    # All model arms receive identical evaluation windows
    # when called with the same seed and arguments.
    generator = torch.Generator().manual_seed(seed)

    # Accumulators for evaluation loss and input bytes.
    total_nll = 0.0
    total_bytes = 0

    # Accumulate the number of positions retained by each
    # dynamic chunking stage across all evaluation batches.
    kept_positions = []

    # Evaluation mode disables training-specific behavior.
    model.eval()

    try:
        with torch.no_grad():

            for _ in range(n_batches):

                # Draw a reproducible batch of next-byte windows.
                x, y = sample_batch(
                    sampling_data,
                    batch_size,
                    seq_len,
                    generator
                )

                # Move evaluation data to the model's device.
                x = x.to(device)
                y = y.to(device)

                # Optionally generate predefined chunk boundaries.
                boundaries = None

                if chunker is not None:
                    boundaries = chunker(x)

                    if boundaries.shape != x.shape:
                        raise ValueError(
                            "chunker must return a boundary mask "
                            "with the same shape as the input IDs."
                        )

                    if boundaries.dtype != torch.bool:
                        raise TypeError(
                            "chunker must return a boolean boundary mask."
                        )

                # Run the complete H-Net model.
                logits, stats = model(
                    x,
                    boundaries=boundaries
                )

                # Validate output dimensions.
                if logits.shape != (
                    batch_size,
                    seq_len,
                    256
                ):
                    raise ValueError(
                        "Model logits must have shape (B, L, 256)."
                    )

                # ------------------------------------------
                # 1. Accumulate negative log-likelihood
                # ------------------------------------------

                batch_nll = F.cross_entropy(
                    logits.reshape(-1, 256),
                    y.reshape(-1),
                    reduction="sum"
                )

                total_nll += batch_nll.item()

                # Each target corresponds to one predicted byte.
                total_bytes += y.numel()

                # ------------------------------------------
                # 2. Accumulate actual compression statistics
                # ------------------------------------------

                for stage_idx, stage_stats in enumerate(stats):

                    # Initialize the accumulator for this stage.
                    if stage_idx >= len(kept_positions):
                        kept_positions.append(0.0)

                    # Count how many positions this stage retained.
                    kept_positions[stage_idx] += (
                        stage_stats["b"].sum().item()
                    )

    finally:
        # Restore training mode even if evaluation raises an error.
        model.train()

    # ----------------------------------------------
    # 3. Convert total NLL from nats into bits per byte
    # ----------------------------------------------

    bpb = total_nll / (math.log(2.0) * total_bytes)

    # ----------------------------------------------
    # 4. Calculate cumulative bytes per kept position
    # ----------------------------------------------

    bytes_per_chunk = []

    for kept in kept_positions:

        if kept <= 0:
            raise ValueError(
                "A chunking stage retained no positions."
            )

        # Use the total original input bytes as the numerator.
        # This produces cumulative compression relative to
        # the original byte sequence.
        ratio = total_bytes / kept

        bytes_per_chunk.append(float(ratio))

    return {
        "bpb": float(bpb),
        "bytes_per_chunk": bytes_per_chunk
    }

# Step 26 - train_hnet
def train_hnet(
    model,
    data,
    steps,
    chunker=None,
    batch_size=16,
    seq_len=128,
    lr=1e-3,
    weight_decay=0.1,
    alpha=0.03,
    seed=0,
    log_every=0,
    lr_mod=True
):
    """
    Train a byte-level H-Net language model end to end.

    Returns:
        dict containing:
            seconds: Total training duration.
            trace: Training statistics recorded during training.
    """

    # Validate inputs.
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer.")

    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive.")

    if log_every < 0:
        raise ValueError("log_every must be non-negative.")

    if data.ndim != 1:
        raise ValueError("data must be a 1-D tensor.")

    if len(data) <= seq_len + 1:
        raise ValueError("data must contain more than seq_len + 1 bytes.")

    # Initialize reproducible batch sampling.
    generator = torch.Generator().manual_seed(seed)

    sampling_data = data.detach().to(
        device="cpu",
        dtype=torch.long
    )

    # Identify the model's device.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = sampling_data.device

    # Calculate stage-wise learning-rate multipliers.
    if lr_mod:
        mults = lr_multipliers(
            model.targets,
            model.widths
        )
    else:
        mults = [1.0] * len(model.widths)

    # Build stage-specific optimizer parameter groups.
    param_groups = stage_param_groups(
        model,
        base_lr=lr,
        weight_decay=weight_decay,
        mults=mults
    )

    # Initialize AdamW.
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.95)
    )

    model.train()

    trace = []
    start_time = time.perf_counter()

    # Training loop: step indices are zero-based.
    for step in range(steps):

        # Update learning rates using the WSD schedule.
        schedule_factor = wsd_factor(step, steps)

        for group in optimizer.param_groups:
            group["lr"] = (
                lr * group["lr_mult"] * schedule_factor
            )

        # Sample a training batch.
        x, y = sample_batch(
            sampling_data,
            batch_size,
            seq_len,
            generator
        )

        x = x.to(device)
        y = y.to(device)

        # Generate optional predefined boundaries.
        boundaries = None

        if chunker is not None:
            boundaries = chunker(x)

            if boundaries.shape != x.shape:
                raise ValueError(
                    "chunker must return a boundary tensor "
                    "with the same shape as x."
                )

            if boundaries.dtype != torch.bool:
                raise TypeError(
                    "chunker must return a boolean boundary tensor."
                )

        optimizer.zero_grad(set_to_none=True)

        # Forward pass.
        logits, stats = model(
            x,
            boundaries=boundaries
        )

        # Mean next-byte cross-entropy in nats.
        ce_loss = F.cross_entropy(
            logits.reshape(-1, 256),
            y.reshape(-1),
            reduction="mean"
        )

        # Calculate ratio regularization for learned routers.
        ratio_total = ce_loss.new_zeros(())

        for s in stats:
            if s["learned"]:
                ratio_total = ratio_total + ratio_loss(
                    s["p"],
                    s["b"],
                    s["N"],
                    s["mask"]
                )

        # Combined training objective.
        loss = ce_loss + alpha * ratio_total

        # Backpropagation.
        loss.backward()

        # Clip the global gradient norm.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        # Record statistics periodically and always at the final step.
        is_final_step = (step == steps - 1)

        should_log = (
            (log_every > 0 and step % log_every == 0)
            or is_final_step
        )

        if should_log:

            # Convert cross-entropy from nats to bits per byte.
            batch_bpb = ce_loss.detach().item() / math.log(2.0)

            bytes_per_chunk = []
            mean_p = []

            for s in stats:

                # Cumulative bytes per retained position.
                kept = s["b"].sum().item()

                if kept <= 0:
                    raise RuntimeError(
                        "A chunking stage retained zero positions."
                    )

                compression = (
                    seq_len * batch_size / kept
                )

                bytes_per_chunk.append(float(compression))

                # Mean router probability over valid positions.
                p = s["p"]
                mask = s["mask"]

                if mask is not None:
                    valid_p = p[mask]
                else:
                    valid_p = p.reshape(-1)

                mean_probability = (
                    valid_p.mean().item()
                    if valid_p.numel() > 0
                    else 0.0
                )

                mean_p.append(float(mean_probability))

            # IMPORTANT: Record the zero-based step index.
            trace.append({
                "step": step,
                "bpb": float(batch_bpb),
                "bytes_per_chunk": bytes_per_chunk,
                "mean_p": mean_p
            })

    # Calculate total training duration.
    seconds = time.perf_counter() - start_time

    return {
        "seconds": float(seconds),
        "trace": trace
    }

# Step 27 - IndependentRouter
class IndependentRouter(nn.Module):
    """
    Independent boundary router.

    Predicts each boundary using only the current hidden-state
    vector, without comparing adjacent vectors using cosine similarity.
    """

    def __init__(self, d):
        super().__init__()

        # Predict one boundary logit from each individual vector.
        self.lin = nn.Linear(d, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d).
            mask (torch.Tensor, optional): Boolean mask of shape (B, L).

        Returns:
            p (torch.Tensor): Boundary probabilities of shape (B, L).
            b (torch.Tensor): Boolean boundary decisions of shape (B, L).
        """

        if x.ndim != 3:
            raise ValueError("x must have shape (B, L, d).")

        B, L, d = x.shape

        if d != self.lin.in_features:
            raise ValueError("Input feature dimension does not match d.")

        if mask is not None:
            if mask.shape != (B, L):
                raise ValueError("mask must have shape (B, L).")

            if mask.dtype != torch.bool:
                raise TypeError("mask must be a boolean tensor.")

            if mask.device != x.device:
                raise ValueError("mask and x must be on the same device.")

        # Handle empty sequences.
        if L == 0:
            p = x.new_empty((B, 0))
            b = torch.empty(
                (B, 0),
                dtype=torch.bool,
                device=x.device
            )
            return p, b

        # The first position is always assigned probability 1.
        first_prob = x.new_ones((B, 1))

        # Predict probabilities independently for positions 1 to L-1.
        if L > 1:
            remaining_prob = torch.sigmoid(
                self.lin(x[:, 1:, :])
            ).squeeze(-1)

            p = torch.cat(
                [first_prob, remaining_prob],
                dim=1
            )
        else:
            p = first_prob

        # Convert probabilities into binary boundary decisions.
        b = p >= 0.5

        # Suppress boundaries at masked positions.
        if mask is not None:
            b = b & mask

        return p, b

# Step 28 - arm_config
def arm_config(name, N=6):
    """
    Return the configuration for a specified experiment arm.

    Returns a dictionary containing:
        layout, widths, targets, cfg, chunker
    """

    if N <= 1:
        raise ValueError("N must be greater than 1.")

    # Shared architecture for single-stage hierarchical arms.
    hierarchical_layout = ["m2", ["T3"], "m2"]
    hierarchical_widths = [32, 96]

    if name == "dynamic":
        layout = hierarchical_layout
        widths = hierarchical_widths
        targets = [N]
        cfg = {}
        chunker = None

    elif name == "no_smoothing":
        layout = hierarchical_layout
        widths = hierarchical_widths
        targets = [N]
        cfg = {"no_smoothing": True}
        chunker = None

    elif name == "no_ste":
        layout = hierarchical_layout
        widths = hierarchical_widths
        targets = [N]
        cfg = {"no_ste": True}
        chunker = None

    elif name == "no_cosine":
        layout = hierarchical_layout
        widths = hierarchical_widths
        targets = [N]
        cfg = {"router_cls": IndependentRouter}
        chunker = None

    elif name == "space":
        layout = hierarchical_layout
        widths = hierarchical_widths
        targets = [6]
        cfg = {}
        chunker = space_boundaries

    elif name == "pool":
        layout = hierarchical_layout
        widths = hierarchical_widths
        targets = [5]
        cfg = {}
        chunker = lambda ids: pool_boundaries(ids, 5)

    elif name == "byte":
        layout = ["m5"]
        widths = [48]
        targets = []
        cfg = {}
        chunker = None

    elif name == "two_stage":
        # Correct two-stage hierarchical architecture.
        layout = [
            "m2",
            ["m2", ["T3"], "m2"],
            "m2"
        ]
        widths = [32, 32, 96]
        targets = [2, 3]
        cfg = {}
        chunker = None

    else:
        raise ValueError(f"Unknown experiment arm: {name!r}")

    return {
        "layout": layout,
        "widths": widths,
        "targets": targets,
        "cfg": cfg,
        "chunker": chunker
    }

# Step 29 - run_arm
def run_arm(
    name,
    train_data,
    val_data,
    steps,
    seed=0,
    lr=4e-3,
    N=6,
    batch_size=16,
    seq_len=128,
    log_every=0
):
    """
    Train and evaluate one experimental H-Net arm.

    Args:
        name (str): Experiment arm name.
        train_data (torch.Tensor): Training corpus as raw bytes.
        val_data (torch.Tensor): Validation corpus as raw bytes.
        steps (int): Number of training steps.
        seed (int): Random initialization and training seed.
        lr (float): Base learning rate.
        N (int): Target compression factor.
        batch_size (int): Training batch size.
        seq_len (int): Sequence length.
        log_every (int): Training statistics logging interval.

    Returns:
        dict: Experiment results containing:
            name
            seed
            bpb
            bytes_per_chunk
            kflops_per_byte
            params
            seconds
            trace
            model
    """

    # --------------------------------------------------
    # 1. Obtain the experiment configuration
    # --------------------------------------------------

    c = arm_config(name, N)

    # --------------------------------------------------
    # 2. Initialize the model reproducibly
    # --------------------------------------------------

    # Set the random seed immediately before model construction.
    torch.manual_seed(seed)

    model = HNetLM(
        c["layout"],
        c["widths"],
        c["targets"],
        c["cfg"]
    )

    # Count all trainable model parameters.
    params = sum(
        p.numel()
        for p in model.parameters()
    )

    # --------------------------------------------------
    # 3. Train the model
    # --------------------------------------------------

    train_result = train_hnet(
        model,
        train_data,
        steps,
        chunker=c["chunker"],
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        seed=seed,
        log_every=log_every
    )

    # --------------------------------------------------
    # 4. Evaluate held-out bits per byte and compression
    # --------------------------------------------------

    eval_result = evaluate_bpb(
        model,
        val_data,
        c["chunker"],
        seq_len=seq_len
    )

    bpb = eval_result["bpb"]
    bytes_per_chunk = eval_result["bytes_per_chunk"]

    # --------------------------------------------------
    # 5. Calculate computational cost per input byte
    # --------------------------------------------------

    kflops_per_byte = (
        flops_per_byte(
            c["layout"],
            c["widths"],
            bytes_per_chunk,
            seq_len=seq_len
        ) / 1000.0
    )

    # --------------------------------------------------
    # 6. Return the complete experiment results
    # --------------------------------------------------

    return {
        "name": name,
        "seed": seed,
        "bpb": bpb,
        "bytes_per_chunk": bytes_per_chunk,
        "kflops_per_byte": float(kflops_per_byte),
        "params": params,
        "seconds": train_result["seconds"],
        "trace": train_result["trace"],
        "model": model
    }

