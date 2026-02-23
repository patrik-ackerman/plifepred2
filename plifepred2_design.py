#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys
import pandas as pd
import joblib
import re
import tempfile

# =====================================================
# HELP TEXT
# =====================================================

short_help = """
PlifePred2 Design Module

Generates all possible single mutants and predicts peptide half-life.

Supports:
  Model 1 → Natural peptides (QSO features)
  Model 2 → Modified peptides (100 selected features)

Use --help for detailed usage.
"""

detailed_help = """
PlifePred2 Design Module
========================

Generates all single amino acid substitution mutants and predicts
their half-life using trained ML models.

Required:
  -i INPUT      Input FASTA file
  -m {1,2}      Model selection
  -o OUTPUT     Output CSV file

Optional:
  -f FLAG       Modification flags (Model 2 only)
                0=d, 1=ct, 2=nt, 3=cyc, 4=ptm
                Example: -f 1,3

  -p PROP       Physicochemical properties (comma separated)
                1 Hydrophobicity
                2 Steric hindrance
                3 Hydropathicity
                4 Amphipathicity
                5 Hydrophilicity
                6 Net Hydrogen
                7 Charge
                8 pI
                9 Molecular weight

Example:
  python design.py -i input.fasta -m 1 -o output.csv
  python design.py -i input.fasta -m 2 -f 1,3 -p 1,7,8 -o output.csv
"""

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-i", help="Input FASTA file")
parser.add_argument("-m", choices=["1","2"], help="Model type")
parser.add_argument("-f", help="Modification flags")
parser.add_argument("-p", help="Physicochemical properties")
parser.add_argument("-o", help="Output file")
parser.add_argument("-h", action="store_true")
parser.add_argument("--help", action="store_true")

args = parser.parse_args()

if args.h:
    print(short_help)
    sys.exit()

if args.help:
    print(detailed_help)
    sys.exit()

if not args.i or not args.m or not args.o:
    print(short_help)
    sys.exit(1)

INPUT = args.i
MODEL_TYPE = args.m
MOD_FLAG = args.f
PROP_SELECTION = args.p
OUTPUT = args.o

# =====================================================
# CONSTANTS
# =====================================================

PFEATURE = "./pfeature_comp"
MODEL1 = "./models/plifepred2_natural_model.sav"
MODEL2 = "./models/plifepred2_model.sav"
PROP_DIR = "./prop"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
STD_AA = list("ACDEFGHIKLMNPQRSTVWY")


# =====================================================
# FASTA FILTERING
# =====================================================

def read_fasta(file):
    valid = []
    rejected = []

    with open(file) as f:
        content = f.read().replace('\r','\n')

    records = content.split('>')[1:]

    for rec in records:
        lines = rec.split('\n')
        name = lines[0].split()[0]
        seq = ''.join(lines[1:]).upper()

        if not set(seq).issubset(VALID_AA):
            rejected.append([name, "Contains non-natural residues"])
            continue

        if len(seq) < 12 or len(seq) > 100:
            rejected.append([name, "Length not in range 12-100"])
            continue

        valid.append([name, seq])

    return valid, rejected


valid_seqs, rejected_seqs = read_fasta(INPUT)

if not valid_seqs:
    print("No valid sequences found.")
    sys.exit(1)

pd.DataFrame(rejected_seqs, columns=["ID","Reason"]).to_csv(
    "eliminated_sequences.csv", index=False
)

# =====================================================
# GENERATE SINGLE MUTANTS
# =====================================================

mutants = []

for seq_id, seq in valid_seqs:

    mutants.append([seq_id, "Original", seq])

    for i in range(len(seq)):
        for aa in STD_AA:
            if seq[i] != aa:
                mut_seq = seq[:i] + aa + seq[i+1:]
                mut_name = f"{seq[i]}{i+1}{aa}"
                mutants.append([seq_id, mut_name, mut_seq])

mutant_df = pd.DataFrame(mutants, columns=["Seq_ID","Mutant_ID","Sequence"])

print(f"[INFO] Total variants: {len(mutant_df)}")

# =====================================================
# TEMP FILE CREATION
# =====================================================

output_dir = os.path.dirname(os.path.abspath(OUTPUT)) or "."
tmp_fasta = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", dir=output_dir)

for _, row in mutant_df.iterrows():
    tmp_fasta.write(f">{row['Mutant_ID']}\n{row['Sequence']}\n".encode())

tmp_fasta.close()

feature_output = tmp_fasta.name + "_features.csv"

# =====================================================
# FEATURE GENERATION
# =====================================================

if MODEL_TYPE == "1":
    feature_type = "QSO"
    MODEL_PATH = MODEL1
else:
    feature_type = "AAC,DPC,PCP,RRI,DDR,SEP,SER,SPC,CTC,CeTD,PAAC,APAAC,QSO"
    MODEL_PATH = MODEL2

subprocess.run([PFEATURE, "-i", tmp_fasta.name, "-o", feature_output, "-j", feature_type], check=True)

X = pd.read_csv(feature_output)

# =====================================================
# FIX CeTD COLUMN BUG
# =====================================================

X.rename(columns={
    "2":"CeTD_HB2",
    "2.2":"CeTD_PO2",
    "3.2":"CeTD_PO3"
}, inplace=True)

# =====================================================
# MODEL 2 FEATURE SELECTION
# =====================================================

if MODEL_TYPE == "2":
    mod_cols = pd.DataFrame(0, index=X.index, columns=["d", "ct", "nt", "cyc", "ptm"])
    X = pd.concat([X, mod_cols], axis=1)
    mod_map = {"0":"d","1":"ct","2":"nt","3":"cyc","4":"ptm"}
    if MOD_FLAG:
    	flags = [f.strip() for f in MOD_FLAG.split(",")]
    	
    	for f in flags:
    		if f in mod_map:
    		    X[mod_map[f]] = 1
    		    
    		else:
    		    print(f"Warning: Unknown modification flag '{f}' ignored.")

    required_columns = [
    "AAC_N","DPC1_DV","DPC1_EW","DPC1_FR","DPC1_GK","DPC1_GW",
    "DPC1_IH","DPC1_KV","DPC1_LD","DPC1_LK","DPC1_LS","DPC1_QM",
    "DPC1_RK","DPC1_SD","DPC1_TC","DPC1_TV","DPC1_TY","DPC1_YT",
    "PCP_NP","PCP_AL","PCP_HX","PCP_Z5",
    "RRI_D","DDR_L","DDR_W",
    "SEP","SER_F","SER_G","SER_N",
    "SEP_AL","SEP_AR","SEP_HB","SEP_NT","SEP_SS_CO",
    "CTC_133","CTC_142","CTC_143","CTC_151","CTC_156","CTC_261",
    "CTC_363","CTC_373","CTC_426","CTC_461","CTC_523","CTC_533",
    "CTC_614","CTC_642","CTC_654","CTC_721","CTC_754",
    "CeTD_HB2","CeTD_PO2","CeTD_PO3",
    "CeTD_11_VW","CeTD_11_CH","CeTD_11_SS",
    "CeTD_12_HB","CeTD_12_PO","CeTD_12_PZ",
    "CeTD_13_HB","CeTD_13_SS",
    "CeTD_21_VW","CeTD_21_SS",
    "CeTD_22_HB","CeTD_22_VW",
    "CeTD_31_HB","CeTD_31_PO","CeTD_31_CH",
    "CeTD_32_HB","CeTD_32_VW","CeTD_32_PZ","CeTD_32_CH",
    "CeTD_33_HB",
    "CeTD_25_p_HB1","CeTD_75_p_PZ1","CeTD_25_p_CH1","CeTD_50_p_CH1",
    "CeTD_75_p_CH1","CeTD_75_p_SA1",
    "CeTD_25_p_VW2","CeTD_25_p_PO2","CeTD_25_p_PZ2","CeTD_75_p_SA2",
    "CeTD_25_p_PO3","CeTD_50_p_PO3","CeTD_50_p_PZ3","CeTD_50_p_CH3",
    "CeTD_75_p_CH3","CeTD_25_p_SS3","CeTD_25_p_SA3","CeTD_75_p_SA3",
    "PAAC1_N","APAAC1_N",
    "QSO1_SC_A","QSO1_SC_C","QSO1_SC_P",
    "QSO1_G_D","QSO1_G_N","QSO1_G_S",
    "d","ct","nt","cyc","ptm"
    ]

    missing = [c for c in required_columns if c not in X.columns]
    if missing:
        print("Missing features:", missing)
        sys.exit(1)

    X = X[required_columns]

# =====================================================
# PREDICTION
# =====================================================

model = joblib.load(MODEL_PATH)
prob = model.predict_proba(X)[:,1] if hasattr(model,"predict_proba") else model.predict(X)

mutant_df["Halflife"] = prob
mutant_df = mutant_df.sort_values(by="Halflife", ascending=False)

# =====================================================
# PHYSICOCHEMICAL PROPERTIES
# =====================================================

if PROP_SELECTION:

    def load_prop(file):
        d={}
        with open(f"{PROP_DIR}/{file}") as f:
            for line in f:
                aa,val=line.strip().split("\t")
                d[aa]=float(val)
        return d

    hydro=load_prop("hydrophobicity.txt")
    ster=load_prop("steric.txt")
    hydp=load_prop("hydrpathy.txt")
    amph=load_prop("amphipathicity.txt")
    hydph=load_prop("hydrophilicity.txt")
    netH=load_prop("nethydrogen.txt")
    charge=load_prop("charge.txt")
    mw=load_prop("mol_wt.txt")

    def calc_pI(seq):
        asp,glu,cys,tyr,his,lys,arg = seq.count('D'),seq.count('E'),seq.count('C'),seq.count('Y'),seq.count('H'),seq.count('K'),seq.count('R')
        ph=0
        while ph<=14:
            ph+=0.01
            q = (-1/(1+10**(3.55-ph))) \
                -(asp/(1+10**(4.05-ph))) \
                -(glu/(1+10**(4.45-ph))) \
                -(cys/(1+10**(9-ph))) \
                -(tyr/(1+10**(10-ph))) \
                +(his/(1+10**(ph-5.98))) \
                +(1/(1+10**(ph-8.2))) \
                +(lys/(1+10**(ph-10))) \
                +(arg/(1+10**(ph-12)))
            if q<=0: return round(ph,2)
        return 14

    prop_map={
        "1":("Hydrophobicity",hydro),
        "2":("Steric hindrance",ster),
        "3":("Hydropathicity",hydp),
        "4":("Amphipathicity",amph),
        "5":("Hydrophilicity",hydph),
        "6":("Net Hydrogen",netH),
        "7":("Charge",charge),
        "8":("pI",None),
        "9":("Mol wt",mw)
    }

    for p in PROP_SELECTION.split(","):
        if p not in prop_map: continue
        name,prop = prop_map[p]
        if p=="7":
            mutant_df[name]=mutant_df["Sequence"].apply(lambda x: sum(prop.get(a,0) for a in x))
        elif p=="8":
            mutant_df[name]=mutant_df["Sequence"].apply(calc_pI)
        elif p=="9":
            mutant_df[name]=mutant_df["Sequence"].apply(lambda x: sum(prop.get(a,0) for a in x)-(18*(len(x)-1)))
        else:
            mutant_df[name]=mutant_df["Sequence"].apply(lambda x: sum(prop.get(a,0) for a in x)/len(x))

# =====================================================
# SAVE OUTPUT
# =====================================================

mutant_df.to_csv(OUTPUT, index=False)

if os.path.exists(tmp_fasta.name): os.remove(tmp_fasta.name)
if os.path.exists(feature_output): os.remove(feature_output)

print("Design module completed successfully.")

