====================================
PlifePred2: Peptide Half-Life Prediction & Design Suite
====================================

PlifePred2 is a comprehensive toolkit for peptide half-life prediction and rational design using integrated machine learning, sequence-derived descriptors, and physicochemical analysis.

It supports:

• Standalone half-life prediction  
• Exhaustive single mutation scanning (design mode)

The toolkit is optimized for Linux and macOS environments and supports reproducible conda-based installation.

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/badge/license-GPLv3-green.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


Overview
========

PlifePred2 predicts peptide half-life using two complementary models:

* Model 1 — Natural peptides
* Model 2 — Modified peptides (modification flags)

It provides two workflows:

1. Prediction Mode — Estimate half-life of input peptides  
2. Design Mode — Generate all single mutants and rank them by predicted stability  


Core Functionalities
====================

* Peptide half-life prediction
* Physicochemical property calculation (pI, MW, charge, etc.)
* Exhaustive single mutation scanning
* Automatic sequence filtering and validation


Installation
============

Option 1 — Conda Environment (Recommended)

.. code-block:: bash

   conda env create -f environment.yml

Conda environment from scratch:

.. code-block:: bash

   conda create -n plifepred2 python=3.10
   conda activate plifepred2
   conda install pandas numpy scikit-learn==1.4.2 joblib

Clone repository:

.. code-block:: bash

   git clone https://github.com/raghavagps/plifepred2.git
   cd plifepred2


Usage Overview
==============

PlifePred2 provides two main scripts:

* ``plifepred2.py`` — Prediction mode
* ``design.py`` — Mutation design mode


Prediction Module
=================

General usage:

.. code-block:: bash

   python plifepred2.py -i input.fasta -m <model> -o output.csv


Arguments
---------

+------------+--------------------------------------------+
| Argument   | Description                                |
+============+============================================+
| -i         | Input multi-FASTA file                     |
+------------+--------------------------------------------+
| -m         | Model type (1 or 2)                        |
+------------+--------------------------------------------+
| -o         | Output CSV file                            |
+------------+--------------------------------------------+
| -f         | Modification flags (Model 2 only)          |
+------------+--------------------------------------------+
| -p         | Physicochemical properties (optional)      |
+------------+--------------------------------------------+


Model 1 — Natural Peptides
--------------------------

.. code-block:: bash

   python plifepred2.py -i input.fasta -m 1 -o output.csv


Model 2 — Modified Peptides
---------------------------

Modification flags:

+------+------------------------------+
| Flag | Description                  |
+======+==============================+
| 0    | D-amino acid                 |
+------+------------------------------+
| 1    | C-terminal modification      |
+------+------------------------------+
| 2    | N-terminal modification      |
+------+------------------------------+
| 3    | Cyclization                  |
+------+------------------------------+
| 4    | Post-translational mod       |
+------+------------------------------+

Example:

.. code-block:: bash

   python plifepred2.py -i input.fasta -m 2 -f 1,3 -o output.csv


Optional Physicochemical Properties
====================================

Use ``-p`` to compute properties.

Available properties:

+------+----------------------+
| Code | Property             |
+======+======================+
| 1    | Hydrophobicity       |
+------+----------------------+
| 2    | Steric hindrance     |
+------+----------------------+
| 3    | Hydropathicity       |
+------+----------------------+
| 4    | Amphipathicity       |
+------+----------------------+
| 5    | Hydrophilicity       |
+------+----------------------+
| 6    | Net hydrogen         |
+------+----------------------+
| 7    | Net charge           |
+------+----------------------+
| 8    | Isoelectric point    |
+------+----------------------+
| 9    | Molecular weight     |
+------+----------------------+

Examples:

.. code-block:: bash

   python plifepred2.py -i input.fasta -m 2 -f 1 -p 8,9 -o output.csv
   python plifepred2.py -i input.fasta -m 1 -p 8,9 -o output.csv


Output Format
=============

+-----------+--------------------------------------+
| Column    | Description                          |
+===========+======================================+
| ID        | Sequence identifier                  |
+-----------+--------------------------------------+
| Sequence  | Peptide sequence                     |
+-----------+--------------------------------------+
| Halflife  | Predicted probability                |
+-----------+--------------------------------------+
| [Props]   | Optional selected properties         |
+-----------+--------------------------------------+


Design Module
=============

The design module performs exhaustive single mutation scanning.

General usage:

.. code-block:: bash

   python plifepred2_design.py -i input.fasta -o design_output.tsv

For modified peptides:

.. code-block:: bash

   python plifepred2_design.py -i input.fasta -f 1 -o design_output.tsv


Design Output Format
====================

+------------+----------------+-----------+-------+
| Seq_ID     | Mutant_ID      | Sequence  | Score |
+============+================+===========+=======+
| Example    | A5V            | ACDVFG... | 0.87  |
+------------+----------------+-----------+-------+

Mutants are sorted from highest to lowest predicted stability.


Sequence Validation
===================

• Only standard amino acids allowed  
• Length restriction: 12–100 residues  
• Invalid sequences logged separately  


Machine Learning Model
======================

Algorithm:
   Random Forest Regressor

Training version:
   scikit-learn 1.4.2

If you encounter version mismatch warnings:

.. code-block:: bash

   pip install scikit-learn==1.4.2


Citation
========

If you use PlifePred2, please cite the corresponding publication.


Support
=======

GitHub:
   https://github.com/raghavagps/plifepred2

Email:
   raghava@iiitd.ac.in


License
=======

GPLv3 License

This software is distributed under the GNU General Public License v3.0.
