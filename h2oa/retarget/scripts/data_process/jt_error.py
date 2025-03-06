import joblib
from tqdm import tqdm
import torch
import os
import glob
tk_file='/cephfs_yili/shared/xuehan/H1_RL/tk8052_1110Truemc8204dn_dn_8198_h2o.pkl'
rt_file='/cephfs_yili/shared/xuehan/H1_RL/dn_8198_h2o.pkl'
rt={}
for file in glob.glob(rt_file):
    rt.update(joblib.load(file))
print(len(rt))
tk=joblib.load(tk_file)
# rt=joblib.load(rt_file)
jt_error_list=[]
for name, data in tqdm(tk.items(), total=8000):
    tk_jt = data['jt'][:,:19]
    if name not in rt:
        continue
    rt_jt = rt[name]['jt'][:,:19]
    jt_error = abs(tk_jt-rt_jt).mean()
    jt_error_list.append(jt_error)
jt_error_list=torch.tensor(jt_error_list)
jt_error=jt_error_list.mean().item()
log_path = os.path.join(os.path.dirname(tk_file), 'metrics.csv')
with open(log_path, "a") as f:
    f.write(f"{tk_file}, {jt_error:.4g}\n")