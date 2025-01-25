import torch
import pickle
import zipfile
from tqdm import tqdm

class MapFileManager:
    def __init__(self, f_name, mode=None):
        self.mode = mode
        self.f_name = f_name
        self.zipf = None
    
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
    def _file_name(name):
        return name + '.pkl'
    
    def save(self, obj, name):
        assert 'w' in self.mode , "manager must be in save mode"    
        with self.zipf.open(self._file_name(name), self.mode) as obj_f:
                torch.save(obj, obj_f)
    
    def saves(self, obj_dct):
        for k, v in obj_dct.items():
            self.save_obj(v,k)
    
    def load(self, name):
        assert 'r' in self.mode , "manager must be in load mode"
        with self.zipf.open(self._file_name(name), self.mode) as obj_f:
            return torch.load(obj_f)
        
if __name__ == '__main__':
    with MapFileManager('objects.zip', 'w') as mf:
        for i in tqdm(range(1000)):
            mf.save(torch.arange(100000), f'v{i}')
        
    with MapFileManager('objects.zip', 'r') as mf:
        for i in tqdm(range(1000)):
            mf.load(f'v{i}')
        