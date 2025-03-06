import pandas as pd
from spheral import Sphere
from pathlib import Path



#%%
rlp_file = Path('data/processed/rlps_filtered.csv')
frame = pd.read_csv(rlp_file, index_col=0)


#%%
sphere = Sphere()
sphere.fit(frame)