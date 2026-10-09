# NA-Craft: pH-Responsive Nucleic Acid Design with NA-MPNN

## Introduction

NA-Craft designs pH-responsive nucleic acids by scanning cytosine-rich sequence derivatives, folding them with AlphaFold3, and screening C·C+ geometry. Sequence compatibility is scored with NA-MPNN, a deep learning framework for nucleic acid inverse folding and protein–DNA specificity prediction. NA-MPNN extends ProteinMPNN to DNA and RNA.

This repository is the submission release of `PH-NA-MPNN` `main` (`8140f04`). The upstream NA-MPNN training and inference code is included so the design pipeline can be run from one checkout.

**Scientific Publication:**
*RNA sequence design and protein–DNA specificity prediction with NA-MPNN*
[https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2](https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2)

---

## Installation

The installation process typically takes 10-30 minutes. The inference code has been tested on Linux and Windows.

### 1. Environment Setup

We recommend using Conda to manage dependencies.

```bash
# Create and activate environment
conda create -n NA-MPNN python=3.12 -y
conda activate NA-MPNN

# Install core dependencies
conda install -c pytorch -c nvidia -c conda-forge \
    pytorch pytorch-cuda=12.4 \
    openbabel ProDy pyarrow pandas cmake numpy \
    biopython scipy -y
```

### 2. Repository Setup

```bash
git clone https://github.com/gychen-him/NA-Craft.git
cd NA-Craft
```

### 3. DSSR Installation (Optional, for Structure Analysis)

DSSR (Dissecting the Spatial Structure of RNA) is used for detailed nucleic acid structure analysis, including base pair detection and G-quadruplex annotation.

**DSSR is proprietary software** and not included in this repository. To install:

1. **Download DSSR** from [https://x3dna.org/](https://x3dna.org/)
   - Request an academic license (free for academic use)
   - Download the Linux version: `DSSR-v1.9.10-linux.zip` (or latest)

2. **Install to the project**:
```bash
# Place the downloaded zip file in the project root, then:
mkdir -p tools/dssr
unzip DSSR-*.zip -d tools/dssr/
chmod +x tools/dssr/x3dna-dssr

# Verify installation
./tools/dssr/x3dna-dssr --version
```

3. **Expected output**:
```
DSSR: an Integrated Software Tool for
Dissecting the Spatial Structure of RNA
v1.9.10-2020apr23, by xiangjun@x3dna.org
```

**Note**: The DSSR software files are excluded from git (see `.gitignore`). Each user needs to download and install DSSR separately.

---

## Basic Usage

### Sequence Design
Design a nucleic acid sequence for a given PDB structure:
```bash
python ./inference/run.py \
    --model_type "na_mpnn" \
    --mode "design" \
    --pdb_path "./inference/examples/4oqu.pdb" \
    --out_folder "./out/design"
```

### Specificity Prediction
Predict the binding specificity (PPM) of a protein-DNA complex:
```bash
python ./inference/run.py \
    --model_type "na_mpnn" \
    --mode "specificity" \
    --pdb_path "./inference/examples/1am9.pdb" \
    --out_folder "./out/specificity" \
    --output_specificity 1
```

---

## pH-Responsive Nucleic Acid Design

This repository includes a specialized pipeline for designing nucleic acids that exhibit pH-responsive behavior (e.g., pH-sensitive switches or G-quadruplexes).

### 1. Core Principle

The design targets the formation of **C·C+ base pairs** (protonated cytosine-cytosine pairs) at acidic pH. The primary objective is to **increase the C-richness** of the nucleic acid structure by introducing specific Cytosine mutations. When pH drops (e.g., pH 7.4 → 5.5), the Cytosine N3 becomes protonated, allowing the formation of stable C·C+ pairs that stabilize a specific structural state, acting as a pH-responsive switch.

### 2. Design Workflow: Three-Stage Pipeline

Our strategy employs a **physics-first, statistics-second, structure-third** approach to efficiently generate candidates through a **saturated mutation scan** of all potential C-rich derivatives.

#### Stage 1: Library (Physical & Statistical Constraints)
- **Saturated C Scan**: For each wild-type, we perform a saturated scan of all possible single (+1C) and double (+2C) mutations at non-G positions.
- **CC Distance Filter**: 
  - +1C mutations: Distance to nearest existing C ≤ 15Å
  - +2C mutations: Distance between the two mutation sites ≤ 15Å
- **NA-MPNN Probability Filter**: P(C) > 0.01 (ensures mutations are compatible with the structural backbone).
- **Rationale**: CC pairs require spatial proximity, and NA-MPNN ensures structural compatibility.

#### Stage 2: Fold (Structural Prediction)
- High-throughput structural folding using **AlphaFold3** (local deployment).
- Generates **5 diffusion samples** per design to increase structural diversity and account for AF3 variability.

#### Stage 3: Screen (Hierarchical Verification)
- **Layer 1 (Quality Control)**: **Clash Detection** ($min\_clash\_dist \ge 1.4Å$). 
  - Mandatory pre-filter to exclude AF3 artifacts (base deformation/overlap).
- **Layer 2 (G4 Detection)**: DSSR verification of G-quadruplex core (at least 2 tetrads).
- **Layer 3 (Geometry Filter)**: Mandatory geometric constraints:
  - **Planarity**: $|Stagger| < 1.0Å$ (ensures C bases are in the same plane).
  - **Distance**: $8.5Å \le C1'-C1' \le 12.5Å$ (standard base-pairing range).
- **Ranking**: Candidates are ranked by **BPO Euclidean Distance** ($\sqrt{B^2 + P^2 + O^2}$), where B=Buckle, P=Propeller, O=Opening. Lower distance indicates higher planarity and potential for C·C+ pairing.

### 3. Benchmark Results (26 G4 Wild-Types)

| Filtering Stage | Candidates | Reduction |
|:----------------|:-----------|:----------|
| **Saturated Mutations (+1C & +2C Scan)** | **2,036** | — |
| After Stage 1 (CC Dist + NA-MPNN) | 490 | −75.9% |
| **Final (Quality + G4 + Stagger + C1' Distance)** | **94** | **−95.4%** |

**Key Metrics:**
- **Positive Sample Recall**: **4/5** experimentally validated mutations passed the strictest structural filters.
- **Efficiency Gain**: **95.4%** total reduction in candidates, identifying the needle in the haystack.

#### Positive Sample Details (Final Screening)

| Sample | Type | CC Pair | Rank | BPO Dist | Status |
|:-------|:-----|:--------|:-----|:---------|:-------|
| 8PSB C17-21 | +2C | C17-C19 | **8** | 84.8 | ✓ Passed |
| 7OQT C9-11 | +1C | C3-C15 | **11** | 90.0 | ✓ Passed |
| 8JFQ C3-11 | +2C | C2-C11 | **33** | 172.9 | ✓ Passed |
| 6GH0 C10-18 | +1C | C13-C18 | **69** | 255.7 | ✓ Passed |
| 7ZEM C20-22 | +2C | — | - | — | ✗ Failed G4/Geom |

### 4. Automated Design-Fold-Verify Loop

```
WT Structure → Stage 1 (Seq Filter) → Stage 2 (AF3 Fold) → Stage 3 (Quality/G4/Geom Screen)
     ↓                ↓                     ↓                     ↓
   2036            490 (-76%)         3D Structures         94 (-95%)
```

### 5. How to Run the Pipeline

The design process is fully automated. AlphaFold3 paths are read from the environment, so the same commands work outside the original cluster:

```bash
export AF3_DIR=/path/to/alphafold3
export AF3_MODEL_DIR=/path/to/af3/models
export AF3_DB_DIR=/path/to/af3/databases
# optional: export AF3_CONDA_ENV=AF3
# optional: export CONDA_BASE="$(conda info --base)"

# Run the full pipeline for all WTs in the dataset
python ph_na_design/scripts/full_pipeline.py --stage all

# Check pipeline progress and current results status
python ph_na_design/scripts/full_pipeline.py --stage status

# Run specific stages
python ph_na_design/scripts/full_pipeline.py --stage library   # Generate candidate library
python ph_na_design/scripts/full_pipeline.py --stage fold      # AF3 structure prediction
python ph_na_design/scripts/full_pipeline.py --stage screen    # Quality + G4 + Geom Screening
```

The results are saved in `ph_na_design/benchmark/pipeline_output/03_screening_results/`, including `screening_results.csv` with detailed geometric parameters.

---

## Future Directions (TODO)

- [ ] **Policy Network Integration**: Treat NA-MPNN as a "Policy Network" for generating sequence candidates.
- [ ] **Reinforcement Learning (RL)**: Use AlphaFold3 + GCNS as a **Reward Function** to fine-tune the NA-MPNN policy. This transforms the design process into an RL loop where the model learns the hidden grammar of pH-responsive networks beyond simple sequence-structure compatibility.

---

## Directory Structure

```
NA-Craft/
├── inference/                    # Core NA-MPNN inference code
├── models/                       # Pre-trained NA-MPNN weights
├── ph_na_design/                 # pH-responsive design specialized pipeline
│   ├── run.py                    # Legacy entry point
│   ├── scripts/                  # Automated pipeline scripts
│   │   └── full_pipeline.py      # MAIN PIPELINE: Quality -> G4 -> Geom
│   ├── modules/                  # Core logic (DSSR, AF3, Geometry)
│   └── benchmark/                # Datasets & Pipeline Outputs
│       ├── ph_response_dataset.csv   # Experimental benchmark set
│       └── pipeline_output/          # Generated structures & results
```

---

## Citation

If you use NA-MPNN in your research, please cite:

```bibtex
@article{na_mpnn_2025,
  title={RNA sequence design and protein–DNA specificity prediction with NA-MPNN},
  author={...},
  journal={bioRxiv},
  year={2025},
  url={https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2}
}
```
