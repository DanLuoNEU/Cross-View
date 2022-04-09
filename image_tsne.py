import os
import random
import numpy as np
import json
import matplotlib.pyplot
import pickle
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.manifold import TSNE

path = '/home/dan/ws/2021-CrossView/matfiles/1129/setup3/vis/0/0/'


# tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(X)
imglist = os.listdir(path)
imglist.sort()
allImages = []

# for i in range(0, len(allImages)):

