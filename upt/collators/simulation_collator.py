import torch
from torch.utils.data import default_collate

class SimulationCollator:
    def __init__(self, num_supernodes, deterministic):
        self.num_supernodes = num_supernodes
        self.deterministic = deterministic

    def __call__(self, batch):
        collated_batch = {}

        # inputs to sparse tensors
        # position: batch_size * (num_inputs, ndim) -> (batch_size * num_inputs, ndim)
        # features: batch_size * (num_inputs, dim) -> (batch_size * num_inputs, dim)
        input_pos = []
        input_feat = []
        input_lens = []
        for i in range(len(batch)):
            pos = batch[i]["input_pos"]
            feat = batch[i]["input_feat"]
            assert len(pos) == len(pos)
            input_pos.append(pos)
            input_feat.append(feat)
            input_lens.append(len(pos))
        collated_batch["input_pos"] = torch.concat(input_pos)
        collated_batch["input_feat"] = torch.concat(input_feat)

        # select supernodes
        supernodes_offset = 0
        supernode_idxs = []
        for i in range(len(input_lens)):
            if self.deterministic:
                rng = torch.Generator().manual_seed(batch[i]["index"])
            else:
                rng = None
            perm = torch.randperm(len(input_pos[i]), generator=rng)[:self.num_supernodes] + supernodes_offset
            supernode_idxs.append(perm)
            supernodes_offset += input_lens[i]
        collated_batch["supernode_idxs"] = torch.concat(supernode_idxs)

        # create batch_idx tensor
        batch_idx = torch.empty(sum(input_lens), dtype=torch.long)
        start = 0
        cur_batch_idx = 0
        for i in range(len(input_lens)):
            end = start + input_lens[i]
            batch_idx[start:end] = cur_batch_idx
            start = end
            cur_batch_idx += 1
        collated_batch["batch_idx"] = batch_idx

        # output_feat to sparse tensor
        output_feat = []
        for i in range(len(batch)):
            feat = batch[i]["output_feat"]
            output_feat.append(feat)
        collated_batch["output_feat"] = torch.concat(output_feat)

        # collate dense tensors
        collated_batch["output_pos"] = default_collate([batch[i]["output_pos"] for i in range(len(batch))])
        collated_batch["timestep"] = default_collate([batch[i]["timestep"] for i in range(len(batch))])

        return collated_batch
