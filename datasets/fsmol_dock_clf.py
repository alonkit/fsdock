import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from datasets.fsmol_dock import FsDockDataset, files_exist, osp
from datasets.process_sidechains import get_fp

class FsDockClfDataset(FsDockDataset):
    def __init__(self, 
                 min_clf_samples=300, 
                 test_fraction=0.2,
                    max_depth=2,
                    num_estimators=100,
                    min_roc_auc=0.75,
                 *args, **kwargs):
        self.min_clf_samples = min_clf_samples
        self.test_fraction = test_fraction
        self.max_depth = max_depth
        self.num_estimators = num_estimators
        self.min_roc_auc = min_roc_auc
        self.clfs_file = f'clfs_nsamples{min_clf_samples}_mdep{max_depth}_nest{num_estimators}_mroc{min_roc_auc}.pt'
        super().__init__(*args, **kwargs)

    def process(self):
        super().process()
        self.logger.info('started proccessing clf')
        self.process_clf()
        self.logger.info('finished proccessing clf')
    
    def load(self):
        super().load()
        self.load_clf()
        
    def load_clf(self):
        if not getattr(self, 'clfs', None):
            self.clfs = torch.load(osp.join(self.processed_dir, self.clfs_file))
        new_tasks = {}
        for task in self.tasks:
            if task not in self.clfs:
                continue
            else:
                new_tasks[task] = self.tasks[task]
                new_tasks[task]['clf'] = self.clfs[task]
    
    def process_clf(self):
        if files_exist([osp.join(self.processed_dir, self.clfs_file)]):
            self.clfs = torch.load(osp.join(self.processed_dir, self.clfs_file))
            return
        self.clfs = {}
        for task in self.ligands:
            if len(self.ligands[task]) < self.min_clf_samples:
                continue
            labels = self.tasks[task]['labels']
            clf, roc_auc, best_thresh = self.get_clf(self.ligands[task], labels)
            if roc_auc < self.min_roc_auc:
                self.logger.info(f'roc_auc for {task} is {roc_auc}, which is less than {self.min_roc_auc}')
                continue
            self.clfs[task] = (clf, best_thresh)
        self.load_clf()
        torch.save(self.clfs, osp.join(self.processed_dir, self.clfs_file))
                
    
    def get_clf(self, mols, labels):
        positive_fp = [get_fp(mol) for l, mol in zip(labels,mols) if l == 1]
        negative_fp = [get_fp(mol) for l, mol in zip(labels,mols) if l == 0]
        pos_ratio = len(positive_fp) / (len(negative_fp) + len(positive_fp))
        num_test = int(self.test_fraction * (len(negative_fp) + len(positive_fp)))
        pos_test_fp = positive_fp[:int(pos_ratio*num_test)]
        pos_train_fp = positive_fp[int(pos_ratio * num_test):]
        neg_test_fp = negative_fp[:int((1-pos_ratio)*num_test)]
        neg_train_fp = negative_fp[int((1-pos_ratio) * num_test):]
        X = np.stack(pos_train_fp + neg_train_fp, axis=0)
        y = np.concatenate([np.ones(len(pos_train_fp)), np.zeros(len(neg_train_fp))])
        sample_weights = [1 if cur_label == 0 else len(neg_train_fp) / len(pos_train_fp) for cur_label in y]
        clf = RandomForestClassifier(max_depth=self.max_depth, random_state=0, n_estimators=self.num_estimators)
        clf.fit(X, y, sample_weight=sample_weights)
        positives = np.ones(len(pos_test_fp))
        negatives = np.zeros(len(neg_test_fp))
        labels = np.concatenate([positives, negatives], axis=0)
        test_samples = np.concatenate([pos_test_fp, neg_test_fp], axis=0)
        probs = clf.predict_proba(test_samples)
        roc_auc = roc_auc_score(labels, probs[:, 1])
        fpr, tpr, thresholds = roc_curve(labels, probs[:, 1], pos_label=1)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        return clf, roc_auc, best_thresh

