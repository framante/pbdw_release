# PBDW Release Repository

This repository provides the reference implementation and scripts used to reproduce the numerical results of the associated publication. It supports local environment setup, execution of the PBDW pipeline, generation of figures and tables, and visualization of solutions and input data.

The repository is designed to be portable: all bash scripts automatically detect the repository root (REPO_ROOT), so no hard-coded $HOME paths are required.

---

## Table of Contents

- Installation
- Minimal Usage
- JSON Input Files
- Scripts in mains
- Additional Utilities
- Generating Figures and Tables
- Workflow
- Data Handling
- Summary

---

## Installation

Clone the repository:

```bash
git clone git@github.com:framante/pbdw_release.git
cd pbdw_release
```

Create the Python virtual environment and install dependencies:

```bash
python setup_env.py
```

Activate the environment:

```bash
source venv/bin/activate
```

---

## Minimal Usage

```bash
JSON=<json_file> bash bash/simple_run.sh
```

Example:

```bash
JSON=json/tab1.json bash bash/simple_run.sh
```

---

## JSON Input Files

Provided examples:

- json/tab1.json
- json/tab4.json
- json/tab5.json
- json/tab7.1.json
- json/tab7.2.json
- json/tab7.3.json

IMPORTANT:

You MUST edit these JSON files and replace all paths with your own local full paths to:

- snapshot matrix
- mesh
- space matrix
- riesz representers
- output directory

All datasets are provided separately and must be referenced explicitly.

---

## Scripts in mains

Primary:

- mains/run_pbdw.py

Utilities:

- mains/plot_sv_trends.py (Figure 13: eigenvalue decay for Nsamples = 100,...,300)
- mains/visualize_solution.py (VTK solution visualization)
- mains/visualize_voxelization.py (input voxelization visualization)

---

## Additional Utilities

Singular value trends:

```bash
python mains/plot_sv_trends.py --parameters <your_parameters>
```

Solution visualization:

```bash
python mains/visualize_solution.py --parameters <your_parameters>
```

Voxelization:

```bash
python mains/visualize_voxelization.py --parameters <your_parameters>
```

---

## Generating Figures and Tables

Fill the JSON file:

- json/fig12a.json
- json/fig12b.json
- json/tab2.json

Run the corresponding bash script:

```bash
- bash bash/run_figure12a.sh
- bash bash/run_figure12b.sh
- bash bash/run_tab2.sh
```

---

## Workflow

```
Fill Data in <json>
        |
        v
JSON=<json> bash/simple_run.sh
        |
        v
    PBDW solve
        |
        +--> Figures / Tables
        |
        +--> Visualization
```

Example runs:

```bash
JSON=json/tab1.json   bash/simple_run.sh
JSON=json/tab4.json   bash/simple_run.sh
JSON=json/tab5.json   bash/simple_run.sh
JSON=json/tab7.1.json bash/simple_run.sh
JSON=json/tab7.2.json bash/simple_run.sh
JSON=json/tab7.3.json bash/simple_run.sh
```

---

## Data Handling

All data are external and will be provided separately.

Execution will fail unless JSON paths are correctly filled.

---

## Summary

1. ```bash
   python setup_env.py
   ```
2. ```bash
   source venv/bin/activate
   ```
3. Edit JSON paths
4. ```bash
   JSON=<json> bash/simple_run.sh
   ```
5. Run figures / visualization as needed

This repository is the reproducibility reference for the publication.
