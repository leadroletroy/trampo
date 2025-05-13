import numpy as np
import pandas as pd
import os

path = r'results_calib\calib_0429'
file = 'WorldTCam_opt.npz'
mat = np.load(os.path.join(path, file))['arr_0']

mat2d = mat.reshape(-1, 4)

df = pd.DataFrame(mat2d)
df.to_csv(os.path.join(path, file.split('.')[0]+'.csv'), header=False, index=False)