import pandas as pd
import csv
import random
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

from sentence_transformers import SentencesDataset, SentenceTransformer
import transformers
import os
from tqdm import trange
from sentence_transformers.util import batch_to_device

import torch
from torch import nn, Tensor

import faiss
from tqdm import tqdm_notebook as tqdm


class Glove:
    def __init__(self):
        self.w2v_embs = None
        
    def load_glove(self, fname='../propaganda_task/GloVe/glove.840B.300d.txt'):
        i = 0
        self.w2v_embs = {}
        with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            for vec in f:
                i += 1
                if i > 200000:
                    break
                try:
                    line = vec.split()
                    self.w2v_embs[line[0]] = np.array(line[1:], dtype=np.float32)
                except:
                    continue

    def w2v(self, word):
        return self.w2v_embs.get(word, np.zeros(300))


def encode_text(texts, stopwords):
    glove = Glove()
    glove.load_glove()
    embs = []
    for text in texts:
        sent_emb = []
        for word in word_tokenize(text.lower()):
            if word not in stopwords and word not in string.punctuation:
                sent_emb.append(glove.w2v(word))
        if len(sent_emb) == 0:
            sent_emb.append(np.zeros(300))
        embs.append(np.mean(sent_emb, axis=0))
    return np.array(embs)


class BSCShuffler:
    def __init__(self, file_name, output_file_name, column_name, group_size, max_ind,
                 shingle_size=1, non_seq_shingle=False, sep='\t', quoting=csv.QUOTE_NONE,
                 by_clusters=False, num_clusters=100, by_kNN=False, num_neighbors=5):
        self.file_name = file_name
        self.column_name = column_name
        self.output_file_name = output_file_name
        self.group_size = group_size
        self.shingle_size = shingle_size
        self.non_seq_shingle = non_seq_shingle
        self.stopwords = set(stopwords.words('english'))
        self.sep = sep
        self.quoting = quoting
        self.max_ind = max_ind   # max column ind in data; tmp = max_ind + 1
        self.by_clusters = by_clusters
        self.by_kNN = by_kNN
        self.num_clusters = num_clusters
        self.num_neighbors = num_neighbors

    def _get_shingle(self, x):
        text = x[self.column_name]
        tokens = [word for word in word_tokenize(text.lower()) 
                  if word not in self.stopwords and word not in string.punctuation]
        if len(tokens) == 0:
            return ''
        pos = 0
        if len(tokens) > 1:
            pos = random.randint(0, len(tokens) - self.shingle_size)
        return ' '.join(sorted(tokens[pos:pos + self.shingle_size]))
    
    def _map_to_shingles(self, df):
        df['shingle'] = df.apply(lambda x: self._get_shingle(x), axis=1)
        df['rand'] = df.apply(lambda x: random.random(), axis=1)
        return df.sort_values(by=['shingle', 'rand'])
    
    def _get_shingle_groups(self, x):
        group_id = random.randint(0, 100000000)
        counter = 0
        for i, row in x.iterrows():
            if counter < self.group_size:
                counter += 1
            else:
                counter = 0
                group_id = random.randint(0, 100000000)
            x.at[i, 'group_id'] = group_id
        return x        
                
    def _reduce_group_ids(self, df):
        df = df.groupby('shingle', as_index=False).apply(lambda x: self._get_shingle_groups(x))
        return df.sort_values(by=['group_id'])
    
    def _read_to_pandas(self):
        with gzip.open(self.file_name, 'rt', encoding='utf8') if self.file_name.endswith('.gz') else open(self.file_name, encoding="utf-8") as fIn:
            data = csv.reader(fIn, delimiter=self.sep, quoting=self.quoting)
            return pd.DataFrame([row[:self.max_ind + 1] for row in data])
    
    def _find_kNN(self, df):
        texts = df[self.column_name].values
        text_embs = encode_text(texts, self.stopwords)
        logging.info("Create KDTree")
        kdt = KDTree(text_embs, leaf_size=30, metric='euclidean')
        logging.info("Done")
        nbrs = kdt.query(text_embs, k=self.num_neighbors, return_distance=False)
        df[self.max_ind + 1] = [' '.join([str(el) for el in nb]) for nb in nbrs]
        self.column_name = self.max_ind + 1
        return df
    
    def _find_clusters(self, df):
        texts = df[self.column_name].values
        text_embs = encode_text(texts, self.stopwords)
        logging.info("Training k-means")
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(text_embs)
        logging.info("Done")
        df[self.max_ind + 1] = [str(cl) for cl in kmeans.labels_]
        self.column_name = self.max_ind + 1
        return df
    
    def shuffle(self):
        df = self._read_to_pandas()
        if self.by_clusters:
            df = self._find_clusters(df)
        if self.by_kNN:
            df = self._find_kNN(df)
        df = self._map_to_shingles(df)
        df = self._reduce_group_ids(df)
        df.to_csv(self.output_file_name, sep=self.sep, index=False, header=None)

        
class ModelBSCShuffler(BSCShuffler):
    def _find_clusters(self, texts_embs):
        logging.info("Training k-means")
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(texts_embs)
        logging.info("Done")
        clusters = [str(cl) for cl in kmeans.labels_]
        self.column_name = 1
        return pd.DataFrame([np.arange(len(clusters)), clusters]).T
        
    def shuffle(self, texts_embs, texts=None):
        if self.by_clusters:
            df = self._find_clusters(texts_embs)
        if self.by_kNN:
            df = self._find_kNN(df) 
        df = self._map_to_shingles(df)
        df = self._reduce_group_ids(df)
        return df[0].values


class ModelExampleBasedShuffler:
    def __init__(self, group_size=10, allow_same=False):
        self.group_size = group_size
        self.sim_batch_size = 1000
        self.allow_same = allow_same
    
    def shuffle(self, texts_embs, texts, top_k=200):
        used_ids = set()
        shuffled_ids = []
        
        order = np.random.permutation(len(texts_embs))
        texts_embs = np.array(texts_embs)[order]
        texts = np.array(texts)[order]
        index = faiss.IndexFlatL2(texts_embs.shape[1])
        index.add(texts_embs)
        
        for i in tqdm(range(len(texts_embs))):
            if i % self.sim_batch_size == 0:
                _, sim_batch = index.search(texts_embs[i:i + self.sim_batch_size], top_k)
            
            if i not in used_ids:
                used_ids.add(i)
                shuffled_ids.append(i)
                batch = set()
                batch.add(texts[i])
                for j in sim_batch[i % self.sim_batch_size]:
                    if j not in used_ids and (self.allow_same or texts[j] not in batch):
                        shuffled_ids.append(j)
                        used_ids.add(j)
                        batch.add(texts[j])
                        if len(batch) >= self.group_size:
                            break
        
        return [order[i] for i in shuffled_ids[::-1]] 
                    

class ShuffledSentencesDataset(SentencesDataset):
    def shuffle(self, shuffler=None, model=None, col=0):
        if shuffler is None:
            return
        texts = [example.texts[col] for example in self.examples]
        if model is not None:
            texts_embs = model.encode(texts)
        else:
            texts_embs = encode_text(texts, set(stopwords.words('english')))
        self.examples = [self.examples[ind] for ind in shuffler.shuffle(texts_embs, texts)]        

        
class ShuffledSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name_or_path=None, modules=None, device=None):
        super().__init__(model_name_or_path, modules, device)
        if not hasattr(self._first_module(), 'alpha'):
            self._first_module().alpha = nn.Parameter(torch.Tensor([1]))
        
    #def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
    #    model._first_module().tau = self.model.tau
    #    super()._eval_during_training(evaluator, output_path, save_best_model, epoch, steps, callback)
    
    def fit(self,
            train_objectives,
            evaluator = None,
            epochs = 1,
            steps_per_epoch = None,
            scheduler= 'WarmupLinear',
            warmup_steps = 10000,
            optimizer_class = transformers.AdamW,
            optimizer_params = {'alpha_lr': 1e-3, 'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay = 0.01,
            evaluation_steps = 0,
            output_path = None,
            save_best_model = True,
            max_grad_norm = 1,
            use_amp = False,
            callback = None,
            output_path_ignore_not_empty = False,
            shuffler = None,
            shuffle_idxs = []
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param output_path_ignore_not_empty: deprecated, no longer used
        :param shuffle_idxs: dataloader indices for BSC shuffling
        """

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = [(n, p) for n, p in list(loss_model.named_parameters()) if 'alpha' not in n]
            alpha_param = [(n, p) for n, p in list(loss_model.named_parameters()) if 'alpha' in n]

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'alpha']
            lr = optimizer_params.pop('lr')
            alpha_lr = optimizer_params.pop('alpha_lr')
            
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay, 'lr': lr},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in alpha_param], 'weight_decay': 0.0, 'lr': alpha_lr}
            ]
            
            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)


        global_step = 0
        
        '''
        if train_idx in shuffle_idxs:
            logging.info('shuffling')
            dataset = dataloaders[train_idx].dataset
            dataset.shuffle(shuffler, self)
            dataloaders[train_idx].dataset = dataset
        '''
        
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch"):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        #logging.info("Restart data_iterator")
                        if train_idx in shuffle_idxs:
                            logging.info('shuffling')
                            dataset = dataloaders[train_idx].dataset
                            dataset.shuffle(shuffler, self)
                            dataloaders[train_idx].dataset = dataset
                            #for loss_model in loss_models:
                                #loss_model.zero_grad()
                                #loss_model.train()
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = batch_to_device(data, self._target_device)

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps)
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps)
            