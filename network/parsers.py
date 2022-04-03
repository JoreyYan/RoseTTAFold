import numpy as np
import scipy
import scipy.spatial
import string
import os,re
import random
import util
import torchN
from ffindex import *

to1letter = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

# read A3M and convert letters into
# integers in the 0..20 range,
def parse_a3m(filename):

    msa = []
    #分别是生成 a~z, 用这26个字母作为关键字建立字典，https://www.runoob.com/python3/python3-string-maketrans.html
    #make trans 是想把用于构建要给从字符串映射到 另一个特征的映射表
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
       
    # read file line by line
    for line in open(filename,"r"):

        # skip labels
        if line[0] == '>':
            continue

        # remove right whitespaces  删除末尾空白格
        line = line.rstrip()

        # remove lowercase letters and append to MSA  去掉小写字母
        msa.append(line.translate(table))

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8) #字符转换acii格式
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i
    #A=0，R=1，N=2 这样，把msa改为 0~19
    
    
    # treat all unknown characters as gaps  未知的是个空格
    msa[msa > 20] = 20

    return msa

# parse HHsearch output
def parse_hhr(filename, ffindex, idmax=105.0):

    # labels present in the database
    label_set = set([i.name for i in ffindex])

    out = []

    with open(filename, "r") as hhr:

        # read .hhr into a list of lines
        lines = [s.rstrip() for _,s in enumerate(hhr)]

        # read list of all hits
        start = lines.index("") + 2
        stop = lines[start:].index("") + start
        hits = []
        for line in lines[start:stop]:

            # ID of the hit
            #label = re.sub('_','',line[4:10].strip())
            label = line[4:10].strip()

            # position in the query where the alignment starts
            qstart = int(line[75:84].strip().split("-")[0])-1

            # position in the template where the alignment starts
            tstart = int(line[85:94].strip().split("-")[0])-1

            hits.append([label, qstart, tstart, int(line[69:75])])

        # get line numbers where each hit starts
        start = [i for i,l in enumerate(lines) if l and l[0]==">"] # and l[1:].strip() in label_set]

        # process hits
        for idx,i in enumerate(start):

            # skip if hit is too short
            if hits[idx][3] < 10:
                continue

            # skip if template is not in the database
            if hits[idx][0] not in label_set:
                continue

            # get hit statistics
            p,e,s,_,seqid,sim,_,neff = [float(s) for s in re.sub('[=%]', ' ', lines[i+1]).split()[1::2]]

            # skip too similar hits
            if seqid > idmax:
                continue

            query = np.array(list(lines[i+4].split()[3]), dtype='|S1')
            tmplt = np.array(list(lines[i+8].split()[3]), dtype='|S1')

            simlr = np.array(list(lines[i+6][22:]), dtype='|S1').view(np.uint8)
            abc = np.array(list(" =-.+|"), dtype='|S1').view(np.uint8)
            for k in range(abc.shape[0]):
                simlr[simlr == abc[k]] = k

            confd = np.array(list(lines[i+11][22:]), dtype='|S1').view(np.uint8)
            abc = np.array(list(" 0123456789"), dtype='|S1').view(np.uint8)
            for k in range(abc.shape[0]):
                confd[confd == abc[k]] = k

            qj = np.cumsum(query!=b'-') + hits[idx][1]
            tj = np.cumsum(tmplt!=b'-') + hits[idx][2]

            # matched positions
            matches = np.array([[q-1,t-1,s-1,c-1] for q,t,s,c in zip(qj,tj,simlr,confd) if s>0])

            # skip short hits
            ncol = matches.shape[0]
            if ncol<10:
                continue

            # save hit
            #out.update({hits[idx][0] : [matches,p/100,seqid/100,neff/10]})
            out.append([hits[idx][0],matches,p/100,seqid/100,sim/10])

    return out

# read and extract xyz coords of N,Ca,C atoms
# from a PDB file
def parse_pdb(filename):

    lines = open(filename,'r').readlines()

    N  = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="N"])
    Ca = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"])
    C  = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="C"])

    xyz = np.stack([N,Ca,C], axis=0)

    # indices of residues observed in the structure
    idx = np.array([int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"])

    return xyz,idx

def parse_pdb_lines(lines):

    # indices of residues observed in the structure
    idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    return xyz,mask,np.array(idx_s)

def parse_templates(ffdb, hhr_fn, atab_fn, n_templ=10):

    # process tabulated hhsearch output to get
    # matched positions and positional scores
    infile = atab_fn 
    hits = []
    for l in open(infile, "r").readlines():
        if l[0]=='>':
            key = l[1:].split()[0]
            hits.append([key,[],[]])
        elif "score" in l or "dssp" in l:
            continue
        else:
            hi = l.split()[:5]+[0.0,0.0,0.0]  # 读取前5列，并添加三个空白位置
            hits[-1][1].append([int(hi[0]),int(hi[1])])  #最新的一个条目上，第二个[]里面是位置信息，第三个[]里面是浮点信息
            hits[-1][2].append([float(hi[2]),float(hi[3]),float(hi[4])])

    # get per-hit statistics from an .hhr file
    # (!!! assume that .hhr and .atab have the same hits !!!)
    # [Probab, E-value, Score, Aligned_cols, 
    # Identities, Similarity, Sum_probs, Template_Neff]
    lines = open(hhr_fn, "r").readlines()
    pos = [i+1 for i,l in enumerate(lines) if l[0]=='>'] #把文件名所在的行的下一行给列举出来
    for i,posi in enumerate(pos):
        hits[i].append([float(s) for s in re.sub('[=%]',' ',lines[posi]).split()[1::2]]) # 从第一项开始 每2个取一个
        
    # parse templates from FFDB   来自pdb数据库的文件
    for hi in hits:
        #if hi[0] not in ffids:
        #    continue
        entry = get_entry_by_name(hi[0], ffdb.index) # 看看pdb数据库中有没有这个搜搜的条目的名字
        if entry == None:
            continue
        data = read_entry_lines(entry, ffdb.data) #读取这个pdb数据
        hi += list(parse_pdb_lines(data))   # 添加pdb数据特征，主要是 C Ca N原子的坐标信息

    # process hits
    counter = 0
    xyz,qmap,mask,f0d,f1d,ids = [],[],[],[],[],[]
    for data in hits:
        if len(data)<7:
            continue
        
        qi,ti = np.array(data[1]).T   # 横纵坐标分开，对应矩阵xy轴
        _,sel1,sel2 = np.intersect1d(ti, data[6], return_indices=True)
        ncol = sel1.shape[0]
        if ncol < 10:
            continue
        
        ids.append(data[0])  #序列
        f0d.append(data[3])  # 蛋白质整体特征
        f1d.append(np.array(data[2])[sel1])   #残基对特征
        xyz.append(data[4][sel2])    
        mask.append(data[5][sel2])  #制作的mask，可能是对应坐标位置为1  否则为0 
        qmap.append(np.stack([qi[sel1]-1,[counter]*ncol],axis=-1))  #不清楚 ，只有跑一边才知道
        counter += 1

    xyz = np.vstack(xyz).astype(np.float32)
    qmap = np.vstack(qmap).astype(np.long)
    f0d = np.vstack(f0d).astype(np.float32)
    f1d = np.vstack(f1d).astype(np.float32)
    ids = ids
        
    return torch.from_numpy(xyz), torch.from_numpy(qmap), \
           torch.from_numpy(f0d), torch.from_numpy(f1d), ids

def read_templates(qlen, ffdb, hhr_fn, atab_fn, n_templ=10):
    xyz_t, qmap, t0d, t1d, ids = parse_templates(ffdb, hhr_fn, atab_fn)
    npick = min(n_templ, len(ids))
    sample = torch.arange(npick)
    #
    xyz = torch.full((npick, qlen, 3, 3), np.nan).float()
    f1d = torch.zeros((npick, qlen, 3)).float()
    f0d = list()
    #
    for i, nt in enumerate(sample):
        sel = torch.where(qmap[:,1] == nt)[0]
        pos = qmap[sel, 0]
        xyz[i, pos] = xyz_t[sel, :3]
        f1d[i, pos] = t1d[sel, :3]
        f0d.append(torch.stack([t0d[nt,0]/100.0, t0d[nt, 4]/100.0, t0d[nt,5]], dim=-1))
    return xyz, f1d, torch.stack(f0d, dim=0)
