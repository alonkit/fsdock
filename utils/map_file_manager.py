import concurrent
import math
import numpy as np
import torch
import pickle
import zipfile
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class MapFileManager:
    def __init__(self, f_name, mode=None):
        self.mode = mode
        self.f_name = f_name
        self.zipf : zipfile.ZipFile = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def open(self):
        if self.zipf is not None:
            self.zipf.close()
        self.zipf = zipfile.ZipFile(self.f_name, self.mode)
        return self
    
    def close(self):
        self.zipf.close()
        self.zipf = None
    
    @staticmethod
    def _file_name(name:str):
        if name.endswith('.pkl'):
            return name
        return name + '.pkl'
    
    @staticmethod
    def _unfile_name(name:str):
        if name.endswith('.pkl'):
            return name[:-4]
        return name
    
    def save(self, obj, name):
        assert 'w' in self.mode , "manager must be in save mode"    
        with self.zipf.open(self._file_name(name), self.mode) as obj_f:
                torch.save(obj, obj_f)
    
    
    def load(self, name):
        assert 'r' in self.mode , "manager must be in load mode"
        with self.zipf.open(self._file_name(name), self.mode) as obj_f:
            return torch.load(obj_f)
    
    def get_all_names(self):
        names = self.zipf.namelist()
        return [self._unfile_name(name) for name in names]
    
    def load_all(self,):
        names = self.zipf.namelist()
        res_dct = {}
        for name in names:
            res_dct[self._unfile_name(name)] = self.load(name)
        return res_dct
    
            
    def __getitem__(self, key):
        return self.load(key)

    def __setitem__(self, key, value):
        self.save(value, key)
    
    def __len__(self):
        assert 'r' in self.mode , "manager must be in load mode"
        return len(self.zipf.namelist())


def _worker_load_batch(f_name, keys):
    """
    Worker function to be run in a separate thread.
    Opens its own handle to the zip file to ensure thread safety.
    """
    local_results = {}
    # Each thread must open its own MapFileManager instance
    with MapFileManager(f_name, 'r') as mf:
        for key in keys:
            try:
                local_results[key] = mf[key]
            except KeyError:
                print(f"Warning: Key {key} not found.")
                local_results[key] = None
    return local_results

def load_objects_concurrently(zip_filename, keys, max_workers=None):
    """
    Loads objects for the given keys concurrently.
    Returns a dictionary {key: object}.
    """
    # 1. Chunk the keys for the workers
    # If we have 100 keys and 4 workers, we want 4 chunks of 25.
    if keys is None:
        with MapFileManager(zip_filename, 'r') as mf:
            keys = mf.get_all_names()
    
    if max_workers is None:
        max_workers = torch.get_num_threads()
    chunk_size = math.ceil(len(keys) / max_workers)
    chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
    
    results = {}
    
    # 2. Execute in ThreadPool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # We pass the filename, not the manager object, so the worker opens a fresh handle
        future_to_chunk = {
            executor.submit(_worker_load_batch, zip_filename, chunk): chunk 
            for chunk in chunks
        }
        
        # 3. Gather results with a progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), 
                           total=len(chunks), 
                           desc="Loading Chunks"):
            try:
                batch_result = future.result()
                results.update(batch_result)
            except Exception as exc:
                print(f'Batch loading generated an exception: {exc}')

    return results

if __name__ == '__main__':
    with MapFileManager('objects.zip', 'w') as mf:
        for i in tqdm(range(10)):
            mf[f'v{i}'] = i
        
    # with MapFileManager('objects.zip', 'r') as mf:
    #     for i in tqdm(range(1000)):
    #         mf.load(f'v{i}')
    