"""
tabicl_selective_finetuner.py
--------------------------------
Fine-grained selective fine-tuning for TabICL v2, built on top of the
existing `TabICLFinetuner` engine (imported unchanged from
run_xor_seed_sweep_tabicl.py, itself copied verbatim from
notebooks/decision_boundary_xor_tabicl.ipynb).

Only `_apply_freezing` is overridden here. Everything else -- `fit()`,
`predict()`, `predict_proba()`, `_load_pretrained()`, `_make_meta_batch()`,
`_train_epoch()`, `_validate()`, the training loop, early stopping, the
cosine warmup scheduler -- is INHERITED UNCHANGED from `TabICLFinetuner`.

TabICL v2 architecture (as inspected and reported directly, not re-verified
independently here -- see `verify_stack_lengths()` for a runtime sanity
check that fails loudly if the installed tabicl version doesn't match):

    col_embedder.tf_col.blocks     -- 3 blocks
    row_interactor.tf_row.blocks   -- 3 blocks
    icl_predictor.tf_icl.blocks    -- 12 blocks, indices 0-11

IMPORTANT: the three stacks do NOT share the same internal block layout.
  * `row_interactor.tf_row.blocks[i]` and `icl_predictor.tf_icl.blocks[i]`
    are `MultiheadAttentionBlock`s and expose `.attn`, `.linear1`,
    `.linear2`, `.norm1`, `.norm2` DIRECTLY.
  * `col_embedder.tf_col.blocks[i]` is an `InducedSelfAttentionBlock` and
    does NOT expose `.attn`/`.linear1`/`.linear2` directly. Instead each
    column block contains two inner attention sub-blocks:
        `.multihead_attn1.attn`, `.multihead_attn1.linear1`, `.multihead_attn1.linear2`
        `.multihead_attn2.attn`, `.multihead_attn2.linear1`, `.multihead_attn2.linear2`
    (an earlier version of this module incorrectly assumed `.attn`/
    `.linear1`/`.linear2` directly on column blocks too -- fixed.)

Strategies implemented
--------------------------
  attention_only
      Freeze the whole model, then unfreeze:
        - `.multihead_attn1.attn` and `.multihead_attn2.attn` in every
          col_embedder.tf_col.blocks[i] (2 attn modules per column block)
        - `.attn` in every row_interactor.tf_row.blocks[i]
        - `.attn` in every icl_predictor.tf_icl.blocks[i]
      Everything else (linear1/linear2/norm1/norm2 everywhere, including
      both column sub-blocks) stays frozen. Any attention-internal
      `ssmax_layer` is included automatically, since the complete `.attn`
      module is unfrozen as a unit.
  mlp_only
      Freeze the whole model, then unfreeze:
        - `.multihead_attn1.linear1`, `.multihead_attn1.linear2`,
          `.multihead_attn2.linear1`, `.multihead_attn2.linear2` in every
          col_embedder.tf_col.blocks[i]
        - `.linear1` and `.linear2` (direct) in every
          row_interactor.tf_row.blocks[i] and icl_predictor.tf_icl.blocks[i]
      Attention (incl. `ssmax_layer`) and normalization parameters stay
      frozen everywhere.
  layerwise_icl{0,3,6,8,11}
      Unchanged by this fix (icl_predictor blocks were already correctly
      handled as MultiheadAttentionBlocks). Freeze the whole model, then
      unfreeze the ENTIRE selected `icl_predictor.tf_icl.blocks[i]` (attn +
      linear1 + linear2 + norm1 + norm2 together, as one unit). No other
      block or top-level component is unfrozen.

Verification
---------------
`verify_freezing()` asserts, for every strategy, BEFORE any optimizer step:
  * at least one trainable parameter exists (freezing isn't total)
  * not literally every parameter is trainable (freezing did apply)
  * every trainable parameter's name starts with one of the expected
    prefixes for that strategy (nothing leaked outside the intended scope)
  * every expected prefix has at least one trainable parameter under it
    (catches a wrong attribute path silently matching nothing)
It also prints total parameter count, trainable parameter count,
percentage trainable, and the actual trainable parameter-name prefixes.
"""

from typing import Optional

import torch.nn as nn

from run_xor_seed_sweep_tabicl import TabICLFinetuner

STACK_EXPECTED_LENGTHS = {
    "col_embedder.tf_col.blocks": 3,
    "row_interactor.tf_row.blocks": 3,
    "icl_predictor.tf_icl.blocks": 12,
}
STACK_PATHS = list(STACK_EXPECTED_LENGTHS.keys())

ICL_LAYER_INDICES = [0, 3, 6, 8, 11]

STRATEGIES = ["attention_only", "mlp_only"] + [f"layerwise_icl{i}" for i in ICL_LAYER_INDICES]


def get_stack_blocks(model: nn.Module, stack_path: str):
    """Resolve a dotted attribute path (e.g. 'icl_predictor.tf_icl.blocks')
    to the actual nn.ModuleList/Sequential object on `model`."""
    obj = model
    for attr in stack_path.split("."):
        obj = getattr(obj, attr)
    return obj


def verify_stack_lengths(model: nn.Module) -> None:
    """Fail loudly if the installed tabicl version's block counts don't
    match the architecture this module was written against."""
    for stack_path, expected_len in STACK_EXPECTED_LENGTHS.items():
        blocks = get_stack_blocks(model, stack_path)
        actual_len = len(blocks)
        assert actual_len == expected_len, (
            f"Architecture mismatch: {stack_path} has {actual_len} blocks, "
            f"expected {expected_len} (per the inspected architecture this "
            f"module was written against). Check the installed tabicl "
            f"version before proceeding -- selective freezing below assumes "
            f"this exact block count."
        )


def freeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


COL_EMBEDDER_PATH = "col_embedder.tf_col.blocks"
_DIRECT_ATTN_STACK_PATHS = ["row_interactor.tf_row.blocks", "icl_predictor.tf_icl.blocks"]
_COL_SUBBLOCKS = ("multihead_attn1", "multihead_attn2")


def apply_attention_only_freezing(model: nn.Module) -> list:
    """Unfreeze the complete attention module(s) of every block.

    col_embedder blocks are InducedSelfAttentionBlocks -- attention lives
    under `.multihead_attn1.attn` and `.multihead_attn2.attn`, not `.attn`
    directly. row_interactor / icl_predictor blocks are
    MultiheadAttentionBlocks and expose `.attn` directly.
    Returns the list of expected trainable parameter-name prefixes.
    """
    expected = []

    col_blocks = get_stack_blocks(model, COL_EMBEDDER_PATH)
    for i, block in enumerate(col_blocks):
        for sub in _COL_SUBBLOCKS:
            sub_module = getattr(block, sub)
            for p in sub_module.attn.parameters():
                p.requires_grad = True
            expected.append(f"{COL_EMBEDDER_PATH}.{i}.{sub}.attn")

    for stack_path in _DIRECT_ATTN_STACK_PATHS:
        blocks = get_stack_blocks(model, stack_path)
        for i, block in enumerate(blocks):
            for p in block.attn.parameters():
                p.requires_grad = True
            expected.append(f"{stack_path}.{i}.attn")

    return expected


def apply_mlp_only_freezing(model: nn.Module) -> list:
    """Unfreeze the MLP sub-module(s) of every block.

    col_embedder blocks: `.multihead_attn1.linear1/.linear2` and
    `.multihead_attn2.linear1/.linear2` (InducedSelfAttentionBlock).
    row_interactor / icl_predictor blocks: direct `.linear1`/`.linear2`
    (MultiheadAttentionBlock).
    Returns the list of expected trainable parameter-name prefixes.
    """
    expected = []

    col_blocks = get_stack_blocks(model, COL_EMBEDDER_PATH)
    for i, block in enumerate(col_blocks):
        for sub in _COL_SUBBLOCKS:
            sub_module = getattr(block, sub)
            for p in sub_module.linear1.parameters():
                p.requires_grad = True
            for p in sub_module.linear2.parameters():
                p.requires_grad = True
            expected.append(f"{COL_EMBEDDER_PATH}.{i}.{sub}.linear1")
            expected.append(f"{COL_EMBEDDER_PATH}.{i}.{sub}.linear2")

    for stack_path in _DIRECT_ATTN_STACK_PATHS:
        blocks = get_stack_blocks(model, stack_path)
        for i, block in enumerate(blocks):
            for p in block.linear1.parameters():
                p.requires_grad = True
            for p in block.linear2.parameters():
                p.requires_grad = True
            expected.append(f"{stack_path}.{i}.linear1")
            expected.append(f"{stack_path}.{i}.linear2")

    return expected


def apply_layerwise_icl_freezing(model: nn.Module, layer_idx: int) -> list:
    """Unfreeze the ENTIRE icl_predictor.tf_icl.blocks[layer_idx] block
    (attn + linear1 + linear2 + norm1 + norm2 together). No other block or
    top-level component is unfrozen. Returns the expected prefix list.

    Note the trailing '.' in the returned prefix: without it, the string
    'icl_predictor.tf_icl.blocks.1' would also match parameter names under
    block 11, 12, etc. (a plain string-prefix collision on multi-digit
    indices). The trailing '.' forces an exact block-index boundary.
    """
    blocks = get_stack_blocks(model, "icl_predictor.tf_icl.blocks")
    assert 0 <= layer_idx < len(blocks), (
        f"layer_idx={layer_idx} out of range for icl_predictor.tf_icl.blocks "
        f"(length {len(blocks)})"
    )
    block = blocks[layer_idx]
    for p in block.parameters():
        p.requires_grad = True
    return [f"icl_predictor.tf_icl.blocks.{layer_idx}."]


def verify_freezing(model: nn.Module, strategy_name: str, expected_prefixes: list) -> dict:
    """Print + assert the freezing state of `model`. Raises AssertionError
    on any mismatch, BEFORE any optimizer step is taken."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

    print(f"[verify:{strategy_name}] total_params={total:,}  "
          f"trainable_params={trainable:,}  ({pct:.3f}% trainable)")
    print(f"[verify:{strategy_name}] expected trainable prefixes ({len(expected_prefixes)}):")
    for pre in expected_prefixes:
        print(f"    {pre}")

    assert trainable > 0, (
        f"[{strategy_name}] no trainable parameters at all -- freezing "
        f"logic is wrong (everything frozen)."
    )
    assert trainable < total, (
        f"[{strategy_name}] ALL parameters are trainable -- freeze_all_params "
        f"did not take effect before the selective unfreeze."
    )

    bad = [n for n in trainable_names if not any(n.startswith(pre) for pre in expected_prefixes)]
    assert not bad, (
        f"[{strategy_name}] {len(bad)} trainable parameter(s) fall OUTSIDE "
        f"the expected scope, e.g. {bad[:5]}"
    )

    missing = [pre for pre in expected_prefixes if not any(n.startswith(pre) for n in trainable_names)]
    assert not missing, (
        f"[{strategy_name}] expected prefix(es) with ZERO trainable "
        f"parameters under them (likely a wrong attribute path): {missing}"
    )

    unique_prefixes = sorted({
        next(pre for pre in expected_prefixes if n.startswith(pre))
        for n in trainable_names
    })
    print(f"[verify:{strategy_name}] trainable parameter-name prefixes "
          f"actually touched ({len(unique_prefixes)}):")
    for pre in unique_prefixes:
        print(f"    {pre}")

    return {
        "total_params": total,
        "trainable_params": trainable,
        "pct_trainable": pct,
        "trainable_prefixes": unique_prefixes,
    }


class TabICLSelectiveFinetuner(TabICLFinetuner):
    """TabICLFinetuner with a different `_apply_freezing` -- everything else
    (fit/predict/predict_proba/training loop/validation/early stopping) is
    inherited unchanged. See module docstring for the strategy definitions.
    """

    def __init__(self, *, strategy: str, **kwargs):
        assert strategy in STRATEGIES, (
            f"Unknown strategy '{strategy}', expected one of {STRATEGIES}"
        )
        self.strategy = strategy
        self.verification_info: Optional[dict] = None
        # The base class's freeze_col_embedder / freeze_row_interactor /
        # freeze_icl_predictor flags are irrelevant here (this subclass
        # computes freezing itself in _apply_freezing) and are intentionally
        # not accepted/forwarded.
        super().__init__(**kwargs)

    def _apply_freezing(self, model: nn.Module) -> None:
        verify_stack_lengths(model)
        freeze_all_params(model)

        if self.strategy == "attention_only":
            expected = apply_attention_only_freezing(model)
        elif self.strategy == "mlp_only":
            expected = apply_mlp_only_freezing(model)
        elif self.strategy.startswith("layerwise_icl"):
            layer_idx = int(self.strategy[len("layerwise_icl"):])
            expected = apply_layerwise_icl_freezing(model, layer_idx)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.verification_info = verify_freezing(model, self.strategy, expected)
