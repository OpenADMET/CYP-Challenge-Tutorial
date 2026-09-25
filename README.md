# CYP-Challenge-Tutorial
[![Logo](https://img.shields.io/badge/OSMF-OpenADMET-%23002f4a)](https://openadmet.org/)

This repo provides starter notebooks and example workflows for the [**OpenADMET CYP Inhibition Blind Challenge**](https://huggingface.co/spaces/openadmet/cyp-challenge): A community benchmark for predicting **Cytochrome P450 (CYP) inhibition** across four major human isoforms: **CYP3A4, CYP2C9, CYP2D6, and CYP1A2**.

CYP enzymes drive the metabolism of most marketed small-molecule drugs. Unexpected CYP inhibition can unexpected trigger drug–drug interactions (DDIs) by slowing the clearance of co-administered drugs. Furthermore, time-dependent (mechanism-based) inhibition (TDI) can cause DDIs that persist long after the parent compound has been cleared. This challenge asks participants to build predictive models that predicts CYP direct inhibition, TDI, and protein-ligand complex structure.

For full challenge details and background, see the [challenge announcement](https://huggingface.co/spaces/openadmet/cyp-challenge).

## ⚙️ Prerequisites & Setup

You'll need [conda](https://docs.conda.io/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/) installed to set up the Python environment.

Create and activate the environment from [`environment.yaml`](./environment.yaml):

```bash
conda env create -f environment.yaml # or mamba env create -f environment.yaml
conda activate oadmet_cyp_tutorial # or mamba activate oadmet_cyp_tutorial
```

This installs Python along with RDKit, scikit-learn, LightGBM, JupyterLab, and the other dependencies needed to run the tutorial notebooks.

## 📦 Data

Training and test data will be available on Hugging Face:
[`openadmet/cyp-challenge-train-test`](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test) 

The dataset covers a high-throughput dose-response curve (DRC) campaign across the four isoforms, including a dose-response **training set** (~1,500 DRCs per isoform), primary single-concentration data for those compounds, and a 750-compound analog-expansion **test set** assayed against all four CYPs.

## 🧪 Challenge Tracks

* **[Direct Inhibition Track](./notebooks/activity_prediction.ipynb) (Regression):** Predict direct-inhibition pIC50 for all 4 isoforms (CYP3A4, CYP2C9, CYP2D6, CYP1A2) across the 750 test-set compounds (4 continuous regression targets per compound).

  - **Primary metric:** Macro-Averaged Soft-Threshold RAE (MA-ST-RAE), where error is measured as the distance between the predicted value and the credible interval bound of the fitted dose-response curve; predictions falling anywhere inside the credible interval incur zero error.
  - **Secondary metrics** (MAE, R², Spearman ρ, Kendall's τ) are also reported with bootstrap confidence intervals.

* **[Time-Dependent Inhibition (TDI) Track](./notebooks/TDI_prediction.ipynb) (Classification):** Classify whether a compound shows significant time-dependent inhibition, evaluated on **CYP3A4 and CYP2D6 only**. 

  - **Primary metric:** Matthews Correlation Coefficient (MCC).
  - **Target Isoforms:** Evaluated on CYP3A4 and CYP2D6 only (where TDI phenomena are clinically prominent).
  - **Label Definition:** Positive means the TDI-arm IC50 shift exceeds 2-fold relative to direct inhibition, including inferred positives among low-activity compounds where a shift can't be measured directly. 

* **[Structure Prediction Track](./notebooks/structure_prediction.ipynb):** Predict the 3D structure of a **CYP3A4** protein-ligand complex given only the ligand SMILES. Our team at UCSF resolved 20 novel Cryo-EM structures of CYP3A4 complexed with a selection of assayed ligands (15 from the training set, 2 from the test set, and 3 additional compounds) — significantly expanding the chemical space beyond existing X-ray PDB structures. CYP3A4's remarkably flexible, heme-centered binding pocket, combined with the fact that similar ligands can adopt drastically distinct binding poses, makes it a notoriously hard target for computational structure prediction.

  - **Submission:** A single `.zip` archive containing all 20 predicted complexes. Co-folding models, docking software, public PDB structures, or proprietary data are all fair game — feel free to draw inspiration from workflows that succeeded in our previous [PXR challenge](https://huggingface.co/spaces/openadmet/pxr-challenge).
  - **Primary metric:** Local Distance Difference Test for Protein-Ligand Interactions (LDDT-PLI), scored automatically against the blinded ground truth via the OpenStructure pipeline. Secondary metrics include Binding-Site Superposed, Symmetry-Corrected Pose RMSD (BiSyRMSD).
  - **Physical Plausibility Checks:** Every submitted structure is also run through [PoseBusters](https://pubs.rsc.org/sc/article/15/9/3130/827511/PoseBusters-AI-based-docking-methods-fail-to) chemical/geometric sanity checks. A structure that fails basic OpenStructure checks receives an LDDT-PLI score of 0 and a 20 Å BiSyRMSD penalty; a structure that fails PoseBusters receives an LDDT-PLI score of 0.

## 💬 Community

Join the [Discord server](https://discord.com/channels/1412827471488745545/1480419832787763412) for Q&A and discussion in the `#cyp-challenge` channel.

## 📅 Key Dates

*All submission deadlines are 23:59 UTC.*

| Date | Action |
|:--- |:--- |
| **July 29, 2026** | Challenge announced |
| **August 17, 2026** | Training/Test sets released; submissions open 🚀 |
| **September 24, 2026** | Deadline for intermediate leaderboard submissions |
| **September 25, 2026** | Intermediate leaderboard released (one-time full test set reveal) |
| **November 3, 2026** | Submissions close 🏁 |
| **From November 4, 2026** | Final leaderboard released, webinars, blog posts, and wrap-up |

## 🏆 Innovation in ML Award

In addition to leaderboard rankings, the OpenADMET team will present an award for the **Most Innovative Machine Learning Approach(es)**, recognizing architectural novelty, creative data usage, and novel uncertainty quantification, regardless of final leaderboard rank.
