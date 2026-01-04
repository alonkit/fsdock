
from collections import defaultdict
from itertools import cycle, islice
import random
from datasets.custom_distributed_sampler import CustomTaskDistributedSampler

import torch
class OneShotCustomTaskDistributedSampler(CustomTaskDistributedSampler):
    def __init__(self, **kwargs):
        self.aux_per_task = {}
        super().__init__(**kwargs)
        
    def set_tasks(self):
        self.aux_per_task = {}
        super().set_tasks()
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        for task in self.tasks_good:
            aux_good_examples_randomized = [self.tasks_good[task][i] for i in torch.randperm(len(self.tasks_good[task]), generator=g).tolist()]
            self.aux_per_task[task] = {idx: val for idx, val in zip(self.tasks[task], cycle(aux_good_examples_randomized))}
            
    def add_aux(self, val):
        if isinstance(val, list):
            return [self.add_aux(v) for v in val]
        task, idx = val
        aux_idx = self.aux_per_task[task][idx]
        return {"main":(task, aux_idx), "aux":(task, idx)}
    
    def __iter__(self):
        yield from map(self.add_aux, super().__iter__())
            