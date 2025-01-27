import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
f = h5py.File("trainset_eval_j0zwbgns.h5")
infos = pd.DataFrame({k: np.array(v).flatten() for k,v in f["info"].items()})
status = f["status"][:, :]
clip_labels = f["clip_labels"][:]

global_errors = infos[[c for c in infos.columns if c.startswith("global_error")]]
sns.boxplot(global_errors)