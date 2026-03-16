import sys
import math
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances

"""
重打分脚本 添加
"""

parser = argparse.ArgumentParser()
parser.add_argument('-lrp_sin', '--lrp_data_sin', default="lrp_sin.dat")
parser.add_argument('-lrp_mul', '--lrp_data_mul', default="lrp_mul.dat")
parser.add_argument('-pdb', '--pdb_file', default="file_list.dat")
parser.add_argument('-o', '--output_file', default="lpr_rescore.dat")
parser.add_argument('-s', '--raw_score', default="score_rmsd.dat")
arg_dict = parser.parse_args()

lrp_data_sin = arg_dict.lrp_data_sin
lrp_data_mul = arg_dict.lrp_data_mul
pdb_file = arg_dict.pdb_file
out_file = arg_dict.output_file
raw_score = arg_dict.raw_score

lines_sin = []
with open(lrp_data_sin, mode='r') as file:
    for line in file:
        if not line.startswith("#"):
            lines_sin.append(line.strip())
print(lines_sin)

lines_mul = []
with open(lrp_data_mul, mode='r') as file:
    for line in file:
        if not line.startswith("#"):
            lines_mul.append(line.strip())


with open(pdb_file, mode='r') as file:
    pdb_list = file.readlines()

raw_score_data = {}
with open(raw_score, mode='r') as file:
    for line in file:
        if not line.startswith("#"):
            raw_data = line.strip().split()
            raw_score_data[raw_data[-1]] = [raw_data[1], raw_data[2]]  # pdb_name rms I_sc

site_sin = []
mod_change_sin_exp = []
chain_sin = []
for line in lines_sin:
    str_ele = line.strip().split()
    site_sin.append(str_ele[0])
    mod_change_sin_exp.append((float(str_ele[2]) - float(str_ele[1])) / float(str_ele[1]))
    chain_sin.append(str_ele[-1])
mod_change_sin_exp = [x*100 for x in mod_change_sin_exp]

site_mul = []
mod_change_mul_exp = []
chain_mul = []
for line in lines_mul:
    str_ele = line.strip().split()
    site_mul.append(str_ele[0])
    mod_change_mul_exp.append((float(str_ele[2]) - float(str_ele[1])) / float(str_ele[1]))
    chain_mul.append(str_ele[-1])
mod_change_mul_exp = [x*100 for x in mod_change_mul_exp]

slope = -2.5099923750652584
intercept = 46.06167555789983
A = 1.28
B = 17
C = 0.46
D = 13.74

def extract_resi_dist(pdb_path, site, chain):
    u = mda.Universe(pdb_path)
    label_resi_atom = u.select_atoms(f'segid {chain} and resnum {int(site)} and name NZ')
    other_chain = {'A': 'B', 'B': 'A'}[chain]
    other_chain_atoms = u.select_atoms(f'segid {other_chain} and (type C or type N or type O)')

    dist = distances.distance_array(label_resi_atom, other_chain_atoms, 
                                                  result=np.zeros((len(label_resi_atom), len(other_chain_atoms))), backend='OpenMP')
    print(dist)
    min_dist = dist.min()
    return min_dist


model_penalties_sin = {}
model_penalties_mul = {}
model_penalties = {}
for pdb in pdb_list:
    model_penalty = 0
    model_penalty_sin = 0
    model_penalty_mul = 0
    modification_diff = []
    pdb = pdb.strip()
    pdb_name = pdb.split("/")[-1].split(".")[0]
    print(pdb_name)
    model_penalties_sinsites = {}
    for num in range(0, len(site_sin)):
        temp_penalty = 0
        if mod_change_sin_exp[num] >= 5:
            min_dist = extract_resi_dist(pdb, site=site_sin[num], chain=chain_sin[num])
            modification_change_pre = slope * min_dist +intercept
            mod_ch = modification_change_pre - mod_change_sin_exp[num]
            try:
                temp_penalty = (1-(1/(1+math.exp(A*(abs(mod_ch)-B)))))
                model_penalty_sin += (1-(1/(1+math.exp(A*(abs(mod_ch)-B)))))
            except OverflowError:
                print(pdb_name+": 惩罚值超出范围，跳过该数值计算，直接+1")
                model_penalty_sin += 1
                continue
            
        if -5 < mod_change_sin_exp[num] < 5 :
            min_dist = extract_resi_dist(pdb, site=site_sin[num], chain=chain_sin[num])
            temp_penalty = 1/(1+math.exp(C*(min_dist-D)))
            model_penalty_sin += 1/(1+math.exp(C*(min_dist-D)))
        
        model_penalties_sinsites[site_sin[num]] = temp_penalty
        
    model_penalties_sin[pdb_name] = model_penalty_sin

    # 多标记点位
    model_penalties_mulsites = {}
    model_penalty_mul = 0
    for num in range(0, len(site_mul)):
        temp_dist = []
        temp_penalty = 0
        site_list = site_mul[num].split('|')
        if mod_change_mul_exp[num] >= 20*len(site_list):
            for site in site_list:
                dist = extract_resi_dist(pdb, site=site, chain=chain_mul[num])
                temp_dist.append(dist)
            mean_dist = sum(temp_dist) / len(temp_dist)
            temp_penalty = (1-1/(1+math.exp(1.2*(mean_dist-11.97))))
            model_penalty_mul += (1-1/(1+math.exp(1.2*(mean_dist-11.97))))
        model_penalties_mulsites[str(site_mul[num])] = temp_penalty
    model_penalties_mul[pdb_name] = model_penalty_mul
    model_penalty = model_penalty_sin + model_penalty_mul
    print("每个单点惩罚值：")
    print(model_penalties_sinsites)
    print("每个多点惩罚：")
    print(model_penalties_mulsites)
    model_penalties[pdb_name] = model_penalty

mod_ch_max = max(model_penalties.values())
print("最大惩罚：" + str(mod_ch_max))
dimethyl_score = {}
new_score = {}
normalize_score = {}
for key, value in model_penalties.items():
    normalize_score[key] = (value / mod_ch_max)
    dimethyl_score[key] = (value / mod_ch_max) * 80
    new_score[key] = dimethyl_score[key] + float(raw_score_data[key][1])

with open(out_file, mode='+a') as file:
    file.write("pdb_name\tI_sc\tlrp_score\n")
    for pdb in pdb_list:
        pdb_name = pdb.strip().split("/")[-1].split(".")[0]
        file.write(pdb_name+"\t"+str(raw_score_data[pdb_name][1])+"\t"+str(new_score[pdb_name])+'\n')

