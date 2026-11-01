"""Semantic PW warm-start, after loading base weights and before optimizer creation."""

import torch


def _statistics(weight):
    x = weight.detach().float()
    return {"mean": x.mean().item(), "std": x.std().item(), "rms": x.square().mean().sqrt().item()}


@torch.no_grad()
def initialize_physical_words(network, tokenizer, *, source, start_token="<|im_start|>"):
    """Mean ordinary subwords from UND input and LM-head weights independently.

    ``source`` records verified files/revision or explicitly synthetic provenance.
    A resume loads PW tensors + this manifest; it must never call this initializer.
    """
    pw = network.physword
    if pw is None or pw.initialization is not None or hasattr(network, "_physword_phase_config"):
        raise ValueError("Semantic initialization requires a fresh PW module before optimizer construction")
    if not source:
        raise ValueError("Explicit pretrained/synthetic source provenance is required")
    inputs = network.language_model.model.embed_tokens.weight
    head = network.language_model.lm_head.weight
    if inputs.is_meta or head.is_meta:
        raise ValueError("Load actual UND input embedding and LM head tensors first")
    vocab = tokenizer.get_vocab()
    if start_token not in vocab:
        raise ValueError(f"Start source must be an existing pretrained token: {start_token}")
    start_id = vocab[start_token]
    tokenizations = [tokenizer.encode(word, add_special_tokens=False) for word in pw.config.vocabulary]
    for ids in [[start_id], *tokenizations]:
        if not ids or min(ids) < 0 or max(ids) >= min(inputs.shape[0], head.shape[0]):
            raise ValueError("Semantic token ID is outside pretrained embedding/head rows")
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in ids:
            raise ValueError("Unknown subword cannot initialize a Physical Word")
    before = {"embedding": _statistics(pw.embedding.weight), "word_head": _statistics(pw.word_head.weight)}
    pw.embedding.weight[0].copy_(inputs[start_id])
    rows = []
    for cls, (word, ids) in enumerate(zip(pw.config.vocabulary, tokenizations, strict=True)):
        a = inputs[torch.tensor(ids, device=inputs.device)].float().mean(0)
        b = head[torch.tensor(ids, device=head.device)].float().mean(0)
        pw.embedding.weight[cls + 1].copy_(a)
        pw.word_head.weight[cls].copy_(b)
        rows.append(
            {
                "word": word,
                "class_id": cls,
                "embedding_index": cls + 1,
                "pw_token_id": pw.config.token_id_base + cls + 1,
                "base_subword_ids": ids,
            }
        )
    pw.initialization = {
        "schema": 1,
        "pooling": "fp32 arithmetic mean of ordinary tokenizer subword rows",
        "source": source,
        "vocabulary_hash": pw.config.vocabulary_hash,
        "actual_weight_tying": inputs.data_ptr() == head.data_ptr()
        and inputs.storage_offset() == head.storage_offset(),
        "start": {"base_token": start_token, "base_token_id": start_id, "source": "UND input embedding"},
        "mapping": rows,
        "before": before,
        "after": {"embedding": _statistics(pw.embedding.weight), "word_head": _statistics(pw.word_head.weight)},
    }
    return pw.initialization
