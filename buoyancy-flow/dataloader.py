from random import shuffle as shuffleList
from random import choice, choices
from tensorflow.keras.utils import Sequence
import jax.numpy as jnp
from skimage.transform import resize
import imageio
import numpy as np
from copy import deepcopy
import os
import jax.random as jr
from scipy import fftpack

import h5py


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    
def listdir_nohidden(path, suffix = None):
    list_ = []
    for f in os.listdir(path):
        if (not f.startswith('.')):
            if suffix is None:
                list_.append(f)
            else: 
                if f.endswith(suffix):
                    list_.append(f)
    return list_
    
def make_generator(dl):
    def f():
        for x,y in dl:
            for n in range(x.shape[0]):
                yield x[n], y[n]
            
    return f
    
class DataLoader(Sequence):
    
    def __init__(self, files, keys, batchSize = 1, rng=jr.PRNGKey(2022), shuffle=True, name='', augmentation=None, maxTime=65):
    
        self.augmentation = augmentation
        self.name = name
        self.batchSize = batchSize
        self.rng = rng
        self.keys = deepcopy(keys)
        self.shuffle = shuffle
        self.maxTime = maxTime
    
        self.transform = []
        
        self.map_ = None
        
        self._open_h5file(files)
        
        self.numSamples = len(self.keys)
        
        self.normalize = False
        self.mean = jnp.array([0.,0.,0.])
        self.std = jnp.array([1.,1.,1.])
    
        if self.shuffle:
            shuffleList(self.keys)    
        
        self.batches = list(chunks(self.keys, self.batchSize))
        
        if len(self.batches[-1]) < self.batchSize:
            del self.batches[-1]
        
        self.numBatches = len(self.batches)
                
        print("Length: %d" % self.numSamples)
        
        
    def _open_h5file(self, paths):
        
        self.h5files = {}
        for path in paths:
            self.h5files[path] = h5py.File(path, 'r')
        
        return self.h5files
        
    def __del__(self):
        try:
            if self.h5files is not None:
                self.h5files.close()
        finally:
            self.h5files = None
            
    def load(self, key, transform = True):
        
        file, sample_id = key
        group_selector = self.h5files[file][str(sample_id)]
        
        smoke = group_selector['smoke'][:][:self.maxTime]
        mask = group_selector['mask'][:][:self.maxTime]
        velocity_x = group_selector['velocity_x'][:][:self.maxTime]
        velocity_y = group_selector['velocity_y'][:][:self.maxTime]
        
        smoke_res = int(np.asarray(group_selector['smoke_res']))
        v_res = int(np.asarray(group_selector['v_res']))
        
        bounds_data = group_selector['BOUNDS'][:]
        BOUNDS = {}
        BOUNDS['_lower'] = (bounds_data[0], bounds_data[1])
        BOUNDS['_upper'] = (bounds_data[2], bounds_data[3])
        
        inflow_data = group_selector['INFLOW'][:]
        INFLOW = {}
        INFLOW['_center'] = (inflow_data[0], inflow_data[1])
        INFLOW['_radius'] = inflow_data[2]
        
        obstacle_list = []
        obstacle_data = group_selector['obstacle_list']
        
        def parse_obstacle(o):
           
            data = {}
           
            o_split = str(o)[3:-2]
        
            o_split = o_split.split(',')
            if o_split[0] == "'SPHERE'":
                data['_center'] = (float(o_split[1]), float(o_split[2]))
                data['_radius'] = float(o_split[3])
            elif o_split[0] == "'BOX'":
                data['_lower'] = (float(o_split[1]), float(o_split[2]))
                data['_upper'] = (float(o_split[3]), float(o_split[4]))
            else:
                raise ValueError(f"UNKNOWN obstacle object {o_split[0]}")
                
            return data
        
        for o in obstacle_data:
            obstacle_list.append(parse_obstacle(o))
        
        # TODO transform
        
        data = {'smoke' : smoke, 'mask' : mask, 'vel_x' : velocity_x, 'vel_y' : velocity_y, 'BOUNDS' : BOUNDS,
                'INFLOW' : INFLOW, 'obstacle_list' : obstacle_list, 'smoke_res' : smoke_res, 'v_res' : v_res}
        
        return data
            
    def __len__(self):
        
        return self.numBatches

    def setMeanAndStd(self, mean, std):
        
        self.normalize = True
        self.mean = mean
        self.std = std
    
    def __getitem__(self, index):
        
        sample = self.batches[index]
      
        item = [self.load(s) for s in sample]
        
        return item
        
    def on_epoch_end(self):
    
        if self.shuffle:
            shuffleList(self.keys)   
            self.batches = list(chunks(self.keys, self.batchSize))
            if len(self.batches[-1]) < self.batchSize:
                del self.batches[-1]
            
    def setTransformations(self, transformations):
        
        self.transform = transformations
            
 
