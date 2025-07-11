import math
from typing import Tuple, Sequence

import torch


class MaskGenerator:
    r"""
    Generate random rectangular block masks for images **or** videos.

    The generator is stateless – every call produces fresh masks
    – so you can instantiate it once and reuse it inside your
    training loop.

    Returned tensors are *indices* (not binary masks):

        • ``mask_enc`` – indices of **kept** (visible) patches  
          shape: ``[batch_size, num_tokens_to_keep]``

        • ``mask_pred`` – indices of **masked** (prediction) patches  
          shape: ``[batch_size, N - num_tokens_to_keep]``

    Args
    ----
    input_size : tuple
        ``(H, W)`` for images or ``(T, H, W)`` for videos **in pixels**.
    patch_size : tuple
        ``(ph, pw)`` for images or ``(pt, ph, pw)`` for videos **in pixels**.
    num_blocks : int, default = 1
        How many masking blocks to sample per sample.
    masking_ratio : float, default = 0.5
        Target fraction of *patches* to mask (0 < r < 1).
    block_size : tuple, optional
        Size of each block in **patch units**.  
        • Images ``(bh, bw)``  
        • Videos ``(bt, bh, bw)``  
        If ``None`` the block dimensions are sampled uniformly on
        ``[1, grid_dim]`` each call.

    Notes
    -----
    * All samples in a batch see **exactly** the same number of
      kept / masked patches so the returned tensors are rectangular.
    * If the block sampling overshoots the requested ratio, extra
      patches are randomly *unmasked* (while staying within blocks)
      to hit the exact target; if it undershoots, extra patches are
      randomly *masked*.
    """

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: Sequence[int],
        num_blocks: int = 1,
        masking_ratio: float = 0.5,
        block_size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.int64,
    ):
        if not (0.0 < masking_ratio < 1.0):
            raise ValueError("masking_ratio must be in (0, 1).")

        self.img_dim = len(input_size)  # 2 → image, 3 → video
        if self.img_dim not in {2, 3}:
            raise ValueError("input_size must have 2 (H,W) or 3 (T,H,W) elements.")

        # ---- grid size in *patches* ----
        if self.img_dim == 2:
            H, W = input_size
            ph, pw = patch_size
            if H % ph or W % pw:
                raise ValueError("input_size must be divisible by patch_size.")
            self.grid = (H // ph, W // pw)
        else:
            T, H, W = input_size
            pt, ph, pw = patch_size
            if T % pt or H % ph or W % pw:
                raise ValueError("input_size must be divisible by patch_size.")
            self.grid = (T // pt, H // ph, W // pw)

        self.num_blocks = num_blocks
        self.mask_ratio = masking_ratio
        self.block_size_cfg = block_size  # in patches, may be None
        self.device = device
        self.dtype = dtype

        self.N = math.prod(self.grid)  # total tokens
        self.num_mask = int(round(self.N * masking_ratio))
        self.num_keep = self.N - self.num_mask

        # Pre-compute strides for flat indexing
        if self.img_dim == 2:
            _, Wp = self.grid
            self._coef = (Wp, 1)  # idx = h*Wp + w
        else:
            Tp, Hp, Wp = self.grid
            self._coef = (Hp * Wp, Wp, 1)  # idx = t*Hp*Wp + h*Wp + w

    # --------------------------------------------------------------------- #
    #                                helpers                                #
    # --------------------------------------------------------------------- #
    def _sample_block_dims(self) -> Tuple[int, ...]:
        """Return a tuple of block sizes in patch units."""
        if self.block_size_cfg is not None:
            # fixed size
            return self.block_size_cfg
        # else: random size on [1, dim]
        return tuple(torch.randint(1, g + 1, (1,)).item() for g in self.grid)

    def _flatten_idx(self, *coords: int) -> int:
        """Convert N-D grid coordinates to flat index."""
        return int(sum(c * k for c, k in zip(coords, self._coef)))

    # --------------------------------------------------------------------- #
    #                                __call__                               #
    # --------------------------------------------------------------------- #
    def __call__(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = torch.random.get_rng_state()  # ensures global randomness

        keep_idx = torch.empty(
            (batch_size, self.num_keep), dtype=self.dtype, device=self.device
        )
        mask_idx = torch.empty(
            (batch_size, self.num_mask), dtype=self.dtype, device=self.device
        )

        for b in range(batch_size):
            masked = set()  # flat indices
            # -- draw blocks until (>=) desired count ----------------------
            while len(masked) < self.num_mask:
                # sample block dims/pos
                b_dims = self._sample_block_dims()
                b_pos = tuple(
                    torch.randint(0, g - d + 1, (1,)).item()
                    for g, d in zip(self.grid, b_dims)
                )
                # gather coords covered by this block
                if self.img_dim == 2:
                    bh, bw = b_dims
                    h0, w0 = b_pos
                    for h in range(h0, h0 + bh):
                        for w in range(w0, w0 + bw):
                            masked.add(self._flatten_idx(h, w))
                else:
                    bt, bh, bw = b_dims
                    t0, h0, w0 = b_pos
                    for t in range(t0, t0 + bt):
                        for h in range(h0, h0 + bh):
                            for w in range(w0, w0 + bw):
                                masked.add(self._flatten_idx(t, h, w))

                # quick exit if we exploded way past target
                if len(masked) > self.num_mask * 2:
                    break

            # --------------------------------------------------------------
            # Trim / extend to hit **exact** num_mask
            # --------------------------------------------------------------
            masked = list(masked)
            if len(masked) > self.num_mask:
                # randomly *unmask* extra indices
                perm = torch.randperm(len(masked))
                masked = [masked[i] for i in perm[: self.num_mask]]
            elif len(masked) < self.num_mask:
                # randomly mask additional positions
                # (uniformly over *unmasked* tokens)
                unmasked_pool = list(set(range(self.N)).difference(masked))
                extra = torch.randperm(len(unmasked_pool))[: self.num_mask - len(masked)]
                masked.extend(unmasked_pool[i] for i in extra)

            masked.sort()
            mask_idx[b] = torch.tensor(masked, dtype=self.dtype, device=self.device)

            # Encoder indices = complement (sorted)
            keep = sorted(set(range(self.N)).difference(masked))
            keep_idx[b] = torch.tensor(keep, dtype=self.dtype, device=self.device)

        return keep_idx, mask_idx
