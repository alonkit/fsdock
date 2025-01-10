from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock_grouped import GFsDockDataset


ds = GFsDockDataset('data/fsdock/smol','data/tasks_smol.csv', num_workers=1)

for t in ds:
    print(t['name'])
    print()

ds = GFsDockDataset('data/fsdock/full','../docking_cfom/csv_tasks.csv', num_workers=1)



