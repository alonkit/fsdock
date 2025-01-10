from datasets.fsmol_dock import FsDockDataset


class UntaskedFsDockDataset(FsDockDataset):
    def __init__(self, *args, **kwargs):
        kwargs['lazy_load'] = False
        super().__init__(*args, **kwargs)
        complex_graphs = []
        for task in self.complex_graphs.values():
            for graph in task['graphs']:
                complex_graphs.append(graph)
        self.complex_graphs = complex_graphs
    
    def len(self):
        return len(self.complex_graphs)
    
    def get(self, idx):
        return self.complex_graphs[idx]
    
    # def task_names(self):
    #     if hasattr(self,'_task_names'):
    #         return self._task_names
    #     assay_hist = self.tasks_df["assay_id"].value_counts()
    #     assay_hist = [f'{aid}_{count}' for aid, count in zip(assay_hist.index, assay_hist)]
        # return  assay_hist