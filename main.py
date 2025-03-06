import pandas as pd
from spheral import Sphere
from pathlib import Path



#%%
rlp_file = Path('data/raw/rlps_enschede_2023.csv')
frame = pd.read_csv(rlp_file)

#%%
frame_pivoted = frame.pivot_table(index=["BOXID", "MONTH"], columns="QUARTER", values="POWER").reset_index()
frame_pivoted.columns.name = None
frame_filtered = frame_pivoted[frame_pivoted["MONTH"]==1].copy()
frame_filtered.drop(columns=["MONTH"], inplace=True)
frame_filtered.set_index("BOXID", inplace=True)

#%%
sphere = Sphere()
sphere.fit(frame_filtered)