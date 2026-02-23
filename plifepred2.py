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
# HELP DESCRIPTION
# =====================================================

import argparse
import sys

short_help = """
PlifePred2: Peptide Half-Life Prediction Tool

Predicts peptide half-life using:
  Model 1 → Natural peptides (QSO features)
  Model 2 → Modified peptides (100 selected features)

Use --help for detailed documentation.
"""

detailed_help = """
PlifePred2 Master Standalone Pipeline
=====================================

Performs peptide half-life prediction using trained ML models.

Required Arguments:
  -i INPUT        Multi-FASTA input file
  -m {1,2}        Model selection
                    1 → Natural model (QSO features)
                    2 → Modified model (100 selected features)
  -o OUTPUT       Output CSV file

Optional Arguments:
  -f FLAG         Modification flag (Model 2 only) (comma separated numbers)
                    0=d, 1=ct, 2=nt, 3=cyc, 4=ptm

  -p PROP         Physicochemical properties (comma separated numbers)

Properties:
  1  Hydrophobicity
  2  Steric hindrance
  3  Hydropathicity
  4  Amphipathicity
  5  Hydrophilicity
  6  Net Hydrogen
  7  Charge
  8  pI
  9  Molecular weight

Examples:
  Natural model:
    python plifepred2.py -i input.fasta -m 1 -o output.csv

  Modified model:
    python plifepred2.py -i input.fasta -m 2 -f 1,3 -o output.csv

  With properties:
    python plifepred2.py -i input.fasta -m 1 -p 1,7,8 -o output.csv
    python plifepred2.py -i input.fasta -m 2 -f 2,3 -p 1,7,8 -o output.csv
"""

# Disable default help
parser = argparse.ArgumentParser(add_help=False)

parser.add_argument("-i", help="Input FASTA file")
parser.add_argument("-m", choices=["1","2"], help="Model type")
parser.add_argument("-f", help="Modification flag")
parser.add_argument("-p", help="Properties")
parser.add_argument("-o", help="Output file")

# Custom short help
parser.add_argument("-h", action="store_true", help="Show short help")

# Custom detailed help
parser.add_argument("--help", action="store_true", help="Show detailed help")

args = parser.parse_args()

# Handle help manually
if args.h:
    print(short_help)
    sys.exit()

if args.help:
    print(detailed_help)
    sys.exit()

# Enforce required arguments manually
if not args.i or not args.m or not args.o:
    print(short_help)
    sys.exit(1)

args = parser.parse_args()

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

# =====================================================
# FASTA FILTERING
# =====================================================

def read_and_filter_fasta(file):
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


valid_seqs, rejected_seqs = read_and_filter_fasta(INPUT)

if not valid_seqs:
    print("No valid sequences found.")
    sys.exit(1)

pd.DataFrame(rejected_seqs, columns=["ID","Reason"]).to_csv(
    "eliminated_sequences.csv", index=False
)

# =====================================================
# CREATE TEMP FILES IN SAME DIRECTORY AS OUTPUT
# =====================================================

output_dir = os.path.dirname(os.path.abspath(OUTPUT))

if output_dir == "":
    output_dir = "."

tmp_fasta = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".fasta",
    dir=output_dir
)

for sid, seq in valid_seqs:
    tmp_fasta.write(f">{sid}\n{seq}\n".encode())

tmp_fasta.close()

feature_output = os.path.join(
    output_dir,
    os.path.basename(tmp_fasta.name) + "_features.csv"
)


# =====================================================
# FEATURE GENERATION
# =====================================================

if MODEL_TYPE == "1":
    feature_type = "QSO"
    MODEL_PATH = MODEL1
else:
    feature_type = "AAC,DPC,PCP,RRI,DDR,SEP,SER,SPC,CTC,CeTD,PAAC,APAAC,QSO"
    MODEL_PATH = MODEL2

cmd = [
    PFEATURE,
    "-i", tmp_fasta.name,
    "-o", feature_output,
    "-j", feature_type
]

subprocess.run(cmd, check=True)

X = pd.read_csv(feature_output)

# =====================================================
# AUTO FIX CeTD COLUMN NAME BUG
# =====================================================

for col in X.columns:
    if col == "2":
        X.rename(columns={"2": "CeTD_HB2"}, inplace=True)
    elif col == "2.2":
        X.rename(columns={"2.2": "CeTD_PO2"}, inplace=True)
    elif col == "3.2":
        X.rename(columns={"3.2": "CeTD_PO3"}, inplace=True)

# =====================================================
# MODEL 2 FEATURE SELECTION (STRICT COLUMN ORDER)
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

    # Validate columns exist (important safety check)
    missing = [col for col in required_columns if col not in X.columns]
    if missing:
        print("ERROR: Missing required features:")
        print(missing)
        sys.exit(1)

    # Keep only required columns in exact order
    X = X[required_columns]


# =====================================================
# MODEL PREDICTION
# =====================================================

model = joblib.load(MODEL_PATH)
pred = model.predict(X)

if hasattr(model, "predict_proba"):
    prob = model.predict_proba(X)[:,1]
else:
    prob = pred

result_df = pd.DataFrame(valid_seqs, columns=["ID","Sequence"])
result_df["Halflife"] = prob

# =====================================================
# PHYSICOCHEMICAL PROPERTIES (FULL VERSION)
# =====================================================

if PROP_SELECTION:

    def load_prop(file):
        d = {}
        with open(f"{PROP_DIR}/{file}") as f:
            for line in f:
                aa,val = line.strip().split("\t")
                d[aa] = float(val)
        return d

    hydrophobicity = load_prop("hydrophobicity.txt")
    steric = load_prop("steric.txt")
    hydropathy = load_prop("hydrpathy.txt")
    amphipathicity = load_prop("amphipathicity.txt")
    hydrophilicity = load_prop("hydrophilicity.txt")
    nethydrogen = load_prop("nethydrogen.txt")
    charge_dict = load_prop("charge.txt")
    mol_weight = load_prop("mol_wt.txt")

    def calculate_pI(seq):
        asp = seq.count('D')
        glu = seq.count('E')
        cys = seq.count('C')
        tyr = seq.count('Y')
        his = seq.count('H')
        lys = seq.count('K')
        arg = seq.count('R')

        ph = 0.0
        while ph <= 14:
            ph += 0.01
            qn1 = -1/(1+10**(3.55-ph))
            qn2 = -asp/(1+10**(4.05-ph))
            qn3 = -glu/(1+10**(4.45-ph))
            qn4 = -cys/(1+10**(9-ph))
            qn5 = -tyr/(1+10**(10-ph))
            qp1 = his/(1+10**(ph-5.98))
            qp2 = 1/(1+10**(ph-8.2))
            qp3 = lys/(1+10**(ph-10))
            qp4 = arg/(1+10**(ph-12))
            if (qn1+qn2+qn3+qn4+qn5+qp1+qp2+qp3+qp4) <= 0:
                return round(ph,2)
        return 14

    property_map = {
        "1": ("Hydrophobicity", hydrophobicity),
        "2": ("Steric hindrance", steric),
        "3": ("Hydropathicity", hydropathy),
        "4": ("Amphipathicity", amphipathicity),
        "5": ("Hydrophilicity", hydrophilicity),
        "6": ("Net Hydrogen", nethydrogen),
        "7": ("Charge", charge_dict),
        "8": ("pI", None),
        "9": ("Mol wt", mol_weight)
    }

    selected = PROP_SELECTION.split(",")

    for p in selected:
        if p not in property_map:
            continue

        name, prop_dict = property_map[p]

        if p == "7":
            result_df[name] = result_df["Sequence"].apply(
                lambda x: sum(prop_dict.get(a,0) for a in x)
            )
        elif p == "8":
            result_df[name] = result_df["Sequence"].apply(calculate_pI)
        elif p == "9":
            result_df[name] = result_df["Sequence"].apply(
                lambda x: sum(prop_dict.get(a,0) for a in x) - (18*(len(x)-1))
            )
        else:
            result_df[name] = result_df["Sequence"].apply(
                lambda x: sum(prop_dict.get(a,0) for a in x)/len(x)
            )

# =====================================================
# SAVE OUTPUT
# =====================================================

result_df.to_csv(OUTPUT, index=False)

# ==============================
# CLEANUP
# ==============================

if os.path.exists(tmp_fasta.name):
    os.remove(tmp_fasta.name)

if os.path.exists(feature_output):
    os.remove(feature_output)

print("Pipeline completed successfully.")

