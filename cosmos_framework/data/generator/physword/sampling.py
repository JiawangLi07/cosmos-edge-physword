"""Exact B=1 / num_workers=0 sampler resume, including epoch order and RNG."""

import torch


class StatefulWindowSampler:
    def __init__(self, size, manifest_hash, *, shuffle=True, seed=42):
        if size < 1:
            raise ValueError("Cannot sample an empty task split")
        self.size, self.manifest_hash, self.shuffle = size, manifest_hash, shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor, self.epoch = 0, -1

    def next_index(self):
        if self.cursor == self.order.numel():
            self.order = (
                torch.randperm(self.size, generator=self.generator) if self.shuffle else torch.arange(self.size)
            )
            self.cursor = 0
            self.epoch += 1
        index = int(self.order[self.cursor])
        self.cursor += 1
        return index

    def state_dict(self):
        return {
            "size": self.size,
            "manifest_hash": self.manifest_hash,
            "shuffle": self.shuffle,
            "order": self.order.clone(),
            "cursor": self.cursor,
            "epoch": self.epoch,
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        for key in ("size", "manifest_hash", "shuffle"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Dataset sampler {key} differs on resume")
        order = state["order"].cpu()
        if order.numel() and not torch.equal(order.sort().values, torch.arange(self.size)):
            raise ValueError("Invalid saved sample permutation")
        if not 0 <= state["cursor"] <= order.numel():
            raise ValueError("Invalid saved data cursor")
        self.order, self.cursor, self.epoch = order, state["cursor"], state["epoch"]
        self.generator.set_state(state["generator"])
