import os
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.SASA import ShrakeRupley

max_asa = {'ALA':  113.0,'ARG':  241.0,'ASN':  158.0,'ASP':  151.0,'CYS':  140.0,
'GLU':  183.0,'GLN':  189.0,'GLY':  85.0,'HIS':  194.0,'ILE':  182.0,'LEU':  180.0,'LYS':  211.0,
'MET':  204.0,
'PHE':	218.0 ,
'PRO':  143.0,
'SER':  122.0,
'THR':  146.0,
'TRP':  259.0,
'TYR':  229.0,
'VAL':  160.0}


def get_protein_divide(pdb_name):
    """
    pdb_name:XXXX_XX_XXXX，不包含reverse
    """
    # 确定复合物中两个蛋白质的划分
    chains = []  # 这里得到的复合物的蛋白质划分一定是准确的
    tem_data = open('/home/xin/UniBind/input/skempi_v2.csv').readlines()
    for line in tem_data:
        if line[0] == '#': continue
        name = line.split(';', 1)[0]
        if pdb_neme_toAB(name) == pdb_name.split('.')[0]:
            chains = name.split('_', 1)[1].split('_')
    return chains


def pdb_neme_toAB(i):
    """
        例如：将1KBH_A_B转化为1KBH_AB
        如果送入的为列表，则将['1CSE_E_I', 'LI45G', 'LI38G']转化为 1CSE_EI_LI38G索引1对应清理前突变，索引2对应清理后突变
    """
    mut = None
    if isinstance(i, list):
        mut = i[2]  # 清理后的突变
        i = i[0]  # pdb名称例如1KBH_A_B
    tem = i.split('_', 1)
    tem[1] = tem[1].replace('_', '')
    tem = tem[0] + '_' + tem[1]

    if mut != None:
        mut = mut.replace(',', '_')
        pdb = tem + '_' + mut
    else:
        pdb = tem
    return pdb

def cal_asa_bio(file, out_dir):

    pdb_name = file.name

    outf =out_dir / (pdb_name[:-4] + "_asa.pdb")
    p = PDBParser(QUIET=1)
    # This assumes you have a local copy of 1LCD.pdb in a directory called "PDB"
    struct = p.get_structure(None, file)
    sr = ShrakeRupley()
    sr.compute(struct, level="R")

    # 遍历结构中的每个模型、链、残基
    for model in struct:
        for chain in model:
            for residue in chain:
                # 获取残基的id
                # residue_id = residue.get_id()
                # 获取这个残基对应的B因子值
                new_b_factor = round(residue.sasa, 2)
                # 遍历残基中的每个原子，并设置B因子
                for atom in residue:
                    atom.set_bfactor(new_b_factor)
    # 创建 PDBIO 对象

    io = PDBIO()
    # 设置要输出的结构
    io.set_structure(struct)

    # 将结构保存到新的文件
    io.save(str(outf))
    return outf

# cmd = "/home/xin/anaconda3/bin/pdb_selchain -B ./1A4Y_AB_HB84A/WT_1A4Y_AB_HB84A.pdb > ./1A4Y_AB_HB84A/1A4Y_B.pdb"
# os.system(cmd)

# calc_asa_areaimol('1A4Y/1A4Y_AB_HB84A/WT_1A4Y_AB_HB84A.pdb')

def determine_region_group(row):
    if row['d_rasa'] == 0:
        if row['rasa_c'] > 0.25:
            return 'SUR'
        else:
            return 'INT'
    elif row['d_rasa'] > 0:
        if row['rasa_m'] < 0.25:
            return 'SUP'
        elif row['rasa_c'] > 0.25:
            return 'RIM'
        else:
            return 'COR'
    else:
        raise('c>m,cal error')

def analyse_asa(asa_c,asa_0 = None,asa_1=None):

    df_c = analyse_pdb(asa_c)
    df_0 = analyse_pdb(asa_0)
    df_0['protein'] = 0
    df_1 = analyse_pdb(asa_1)
    df_1['protein'] = 1
    df_m = pd.concat([df_1,df_0], ignore_index=True)
    df_m.sort_values(by='pos', inplace=True)
    df_m = df_m.reset_index(drop=True)
    df_c.sort_values(by='pos', inplace=True)
    df_c = df_c.reset_index(drop=True)
    # a = (df_c['res'] == df_m['res'])
    assert len(df_c) == len(df_m)
    # a = df_c['res'] == df_m['res']
    assert (df_c['res'] == df_m['res']).all()
    df_c['asa_m'] = df_m['asa']
    df_c['protein'] = df_m['protein']
    df_c['max_asa'] = [max_asa[key] for key in df_c['res']]
    df_c['rasa_c'] = df_c['asa'] / df_c['max_asa']
    df_c['rasa_m'] = df_c['asa_m'] / df_c['max_asa']
    df_c['d_rasa'] = df_c['rasa_m'] - df_c['rasa_c']
    df_c['region'] = df_c.apply(determine_region_group,axis=1)
    new_order = ['pos', 'res', 'chain','region','protein','rasa_c','rasa_m','d_rasa','max_asa','asa','asa_m']
    df_c = df_c[new_order]
    out_file = str(asa_c)[:-8]+'.region'
    df_c.to_csv(out_file,index =False)


def analyse_pdb(asa_pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", str(asa_pdb))
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]
    chains = list(model.get_chains())
    b_factors = []
    res_num = []
    res_type = []
    chain_name = []
    for chain in chains:
        for res in chain:
            res_num.append(res.id[1])
            res_type.append(res.resname)
            chain_name.append(chain.id)
            b_factors.append(res['CA'].bfactor)
    df = pd.DataFrame({'pos':res_num,
                       'res':res_type,
                       'chain':chain_name,
                      'asa':b_factors})
    return df




def get_region_2(name,proteinA,proteinB,out_dir):
    # parent_path = pdbfile.parent
    pdb_dir_name = name+'_'+(proteinA+proteinB)

    out_dir = out_dir / pdb_dir_name
    pdb = out_dir / (name+'_'+(proteinA+proteinB)+'.pdb')
    pdb_0 = out_dir / (name+'_'+proteinA+'.pdb')
    pdb_1 = out_dir / (name+'_'+proteinB+'.pdb')
    cmd_extract_chains_0 = "/home/xin/anaconda3/bin/pdb_selchain -" + ','.join(proteinA) +' ' + str(pdb) + ' > ' + str(pdb_0)
    cmd_extract_chains_1 = "/home/xin/anaconda3/bin/pdb_selchain -" + ','.join(proteinB) + ' ' + str(pdb) + ' > ' + str(pdb_1)
    os.system(cmd_extract_chains_0)
    os.system(cmd_extract_chains_1)
    # asa_0 = calc_asa_areaimol(pdb_0,out_dir)
    # asa_1 = calc_asa_areaimol(pdb_1,out_dir)
    # asa_c = calc_asa_areaimol(pdb,out_dir)
    asa_0 = cal_asa_bio(pdb_0,out_dir)
    asa_1 = cal_asa_bio(pdb_1,out_dir)
    asa_c = cal_asa_bio(pdb,out_dir)
    analyse_asa(asa_c,asa_0,asa_1)

def get_region_ske(name,proteinA,proteinB,out_dir):
    pdb_name = name.split('_')[1]
    # parent_path = pdbfile.parent
    # pdb_dir_name = name+'_'+(proteinA+proteinB)
    out_dir = Path(out_dir)
    # out_dir = out_dir / pdb_name
    pdb = out_dir / name
    pdb_0 = out_dir / (pdb_name+'_'+proteinA+'.pdb')
    pdb_1 = out_dir / (pdb_name+'_'+proteinB+'.pdb')
    cmd_extract_chains_0 = "pdb_selchain.exe -" + ','.join(proteinA) + ' ' + str(pdb) + ' > ' + str(pdb_0)
    cmd_extract_chains_1 = "pdb_selchain.exe -" + ','.join(proteinB) + ' ' + str(pdb) + ' > ' + str(pdb_1)
    os.system(cmd_extract_chains_0)
    os.system(cmd_extract_chains_1)
    # asa_0 = calc_asa_areaimol(pdb_0,out_dir)
    # asa_1 = calc_asa_areaimol(pdb_1,out_dir)
    # asa_c = calc_asa_areaimol(pdb,out_dir)
    asa_0 = cal_asa_bio(pdb_0,out_dir)
    asa_1 = cal_asa_bio(pdb_1,out_dir)
    asa_c = cal_asa_bio(pdb,out_dir)
    analyse_asa(asa_c,asa_0,asa_1)
