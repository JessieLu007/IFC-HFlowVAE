# IFC-HFlowVAE

Official implementation of:

**IFC-HFlowVAE: A self-enhancing generative framework with structural anchoring for imbalanced clinical data augmentation**

This repository provides the implementation of IFC-HFlowVAE, a heterogeneous
tabular data generation framework designed for minority-class augmentation in
imbalanced clinical datasets.

IFC-HFlowVAE integrates heterogeneous variational autoencoder modeling,
normalizing flow enhanced latent representation learning, and iterative
feedback-based refinement to generate high-quality synthetic tabular samples
while preserving complex minority-class patterns.

The generated samples can be used for downstream classification tasks under
class imbalance scenarios.

---

## Repository Structure

```
IFC-HFlowVAE/
│
├── hflow_vae/
│   │
│   ├── notebooks/
│   │   ├── 5cvfold_run.ipynb
│   │   ├── run_hflowvae_single.ipynb
│   │   └── results_analysis.ipynb
│   │
│   ├── models/
│   ├── datasets/
│   ├── utils/
│   └── ...
│
├── diffusion_with_trees_class.py
│
├── environment.yml
├── windows_environment.yml
│
└── LICENSE
```

---

## Installation

The experiments were conducted using Python and PyTorch.

We recommend creating a conda environment:

```bash
conda env create -f environment.yml
conda activate hflowvae
```

For Windows systems:

```bash
conda env create -f windows_environment.yml
```

The provided environment files contain the required dependencies for
reproducing the experiments.

---

## Dataset Preparation

The experiments were conducted on seven publicly available clinical tabular
datasets.

Due to the distribution policies of the original datasets, the raw datasets
are not included in this repository. Users should download the datasets from
their corresponding official sources and place them according to the expected
data format before running the notebooks.

---

## Running Experiments

All main experiments can be reproduced through the notebooks located in:

```
hflow_vae/notebooks/
```

### 1. Five-fold cross-validation experiments

```
5cvfold_run.ipynb
```

This notebook performs the complete experimental pipeline, including:

- dataset loading and preprocessing;
- five-fold cross-validation splitting;
- training IFC-HFlowVAE and baseline augmentation methods;
- synthetic sample generation;
- downstream classification evaluation;
- saving generated samples and classification results.

The generated data and evaluation results are saved for subsequent analysis.

---

### 2. Running IFC-HFlowVAE individually

```
run_hflowvae_single.ipynb
```

This notebook provides an independent implementation example for directly
training and applying the proposed HFlowVAE model.

It can be used to:

- train IFC-HFlowVAE on a selected dataset;
- generate synthetic tabular samples;
- inspect generated data without running the complete benchmark pipeline.

---

### 3. Result analysis

```
results_analysis.ipynb
```

This notebook analyzes the results produced by
`5cvfold_run.ipynb`.

The analysis includes:

- aggregation of classification performance;
- comparison between augmentation methods;
- statistical analysis;
- synthetic data quality evaluation.

---

## Evaluation

The framework evaluates synthetic samples from two perspectives:

### Synthetic data fidelity

The generated samples are evaluated using statistical and structural metrics,
including:

- marginal distribution similarity;
- feature dependency preservation;
- manifold-level fidelity measurements.

### Downstream utility

The generated data are evaluated by training classifiers on augmented datasets
and measuring classification performance under class imbalance scenarios.

---

## Reproducibility

All experiments reported in the manuscript were conducted using the provided
implementation and environment configuration.

To facilitate reproducibility:

- fixed random seeds are used during experiments;
- identical data splitting strategies are applied across methods;
- generated samples and evaluation results are saved during execution.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{IFC-HFlowVAE,
  title={IFC-HFlowVAE: [Full manuscript title]},
  author={},
  journal={},
  year={}
}
```

---

## License

This project is released under the MIT License.
