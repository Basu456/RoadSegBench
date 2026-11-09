# RoadSegBench: Reproducible Benchmarking of Curvy Road Segmentation

Code, evaluation records, run-level results and analysis scripts for the paper

> B. Chougula, A. Gudnavar, and K. Naregal, "RoadSegBench: Reproducible Benchmarking of Curvy Road Segmentation".

RoadSegBench evaluates road segmentation separately on **Gentle**, **Moderate** and **Sharp**
apparent-curvature tiers of Cityscapes, BDD100K and Mapillary Vistas road masks. It reports
region accuracy (IoU) alongside boundary accuracy (BIoU) and inference efficiency for eight
architectures trained under one shared protocol with ten random seeds:
PIDNet-M, DDRNet-23, DDRNet-23+BR, BiSeNetV2, SegFormer-B2, U-Net (ResNet-34), SCTNet-B and
DeepLabV3+ (ResNet-101).

---

## Repository layout

```
RoadSegBench/
├── roadsegbench/            # Python package
│   ├── curvature.py         #   curvature extractor and tier assignment
│   ├── manifests.py         #   frozen-manifest loading and checks
│   ├── metrics.py           #   IoU / BIoU
│   ├── statistics.py        #   Friedman / Nemenyi / Wilcoxon + Holm
│   ├── audit.py             #   dataset audits
│   ├── data/                #   datasets and label decoders
│   ├── models/              #   architecture registry
│   ├── engine/              #   training loop
│   └── evaluation/          #   inference and latency measurement
├── configs/
│   ├── experiment.yaml      # shared training / evaluation protocol, seeds, curvature thresholds
│   ├── datasets/            # cityscapes.yaml, bdd100k.yaml, mapillary.yaml
│   └── models/architectures.yaml
├── tools/                   # end-to-end pipeline scripts (steps 1–8 below)
├── scripts/                 # train_matrix.py, run_analysis.py, validate_manifest.py
├── notebooks/               # Colab reproducibility notebook
├── tests/                   # unit tests
├── data/manifests/          # frozen manifest (see note in section 2)
├── records/                 # everything the paper's numbers come from:
│   ├── *_audit.csv, *_mask_stats.json, cityscapes_sharp_97_ids.csv, sharp_214_ids.csv,
│   │   reconciliation_97_vs_214.json, manifest_audit_report.json   (data definition)
│   ├── stage1_72runs_summary.csv, stage2_240runs_summary.csv,
│   │   statistical_significance_tests.json                          (run-level results)
│   ├── curvature_*.csv, curvature_probe_output.txt                  (extractor validation)
│   ├── Deliverable_1..4_*.json                                      (audit reports)
│   ├── requirements_lock_main_env.txt, third_party_lock.csv,
│   │   pretrained_sha256.csv, dataset_archives_sha256.csv, nvidia-smi*.txt
│   ├── manuscript_latex/    # generated LaTeX tables
│   ├── manuscript_figures/  # generated figures
│   └── visual_checks/       # curvature validation plot
└── requirements-colab.txt   # dependencies for the Colab notebook
```

---

## 1. Installation

Tested with Python 3.10.

Clone this repository (green **Code** button above → copy the URL), then:

```bash
git clone <repository URL>
cd RoadSegBench
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r records/requirements_lock_main_env.txt
pip install -e .
python -m pytest tests
```

GPU, driver and CUDA details for the reported runs are in `records/nvidia-smi.txt` and
`records/Deliverable_1_Environment_Report.json`.

### Third-party code

The architectures come from their original repositories. These are **not** included here.
Clone each one into `third_party/` at the commit pinned in `records/third_party_lock.csv`:

| Folder in `third_party/` | Used for |
|---|---|
| `BiSeNet` | BiSeNetV2 |
| `DDRNet` | DDRNet |
| `PIDNet` | PIDNet |
| `SCTNet` | SCTNet (including its training-only transformer teacher) |
| `SegFormer` | SegFormer |
| `cityscapesScripts` | Cityscapes label handling |
| `bdd100k` | BDD100K label handling |
| `mapillary_vistas` | Mapillary Vistas label handling |

Each repository has its own licence. Some are restricted to non-commercial use. Check the
licence before reusing them.

### Pretrained backbones

```bash
python tools/download_pretrained.py
```

Downloads ImageNet / teacher checkpoints from their original authors. The files are checked
against `records/pretrained_sha256.csv`.

---

## 2. Datasets

The dataset images are **not redistributed** because their licences forbid it. Download them
from the official providers after accepting their terms:

- Cityscapes — https://www.cityscapes-dataset.com/
- BDD100K — https://bdd-data.berkeley.edu/
- Mapillary Vistas — https://www.mapillary.com/dataset/vistas

`records/dataset_archives_sha256.csv` lists the SHA-256 of every archive we used.
Matching hashes mean you have identical data.

**Where to put them.** The scripts in `tools/` look for a `datasets` folder **next to** the
repository folder (not inside it):

```
<workspace>/
├── RoadSegBench/                      ← this repository
└── datasets/
    ├── cityscapes/
    ├── bdd100k/bdd100k_seg/bdd100k/seg/
    └── mapillary_vistas/Mapillary Vistas/
```

To keep the datasets somewhere else, set the environment variable `ROADSEGBENCH_WORKSPACE`
to the folder that contains `datasets/` (Windows: `set ROADSEGBENCH_WORKSPACE=<folder>`;
Linux/macOS: `export ROADSEGBENCH_WORKSPACE=<folder>`). Paths shown as `<REPO>` or `<WORKSPACE>` in
`records/*.json` refer to these two folders.

The experiments used archives obtained from Kaggle mirrors of these datasets.
`tools/download_kaggle_datasets.py` downloads the same archives into `datasets/_downloads/`
(it needs your own Kaggle API credentials; none are included here). The official sources above
remain authoritative; compare against `records/dataset_archives_sha256.csv` to confirm you have
the same files.

---

### Frozen manifest

All results were computed on one frozen manifest, `roadsegbench_manifest_with_curvature.csv`
(31,475 images; SHA-256 `53cbcd94881654cfe329519454d98510318742ab9113b0f0b27d0548210a6b11`).
Its partition and tier counts, and the zero-overlap checks between partitions, are recorded in
`records/manifest_audit_report.json`.

| | train | tuning | threshold validation | evaluation |
|---|---|---|---|---|
| Cityscapes | 2,600 | 275 | 100 | 500 |
| BDD100K | 6,150 | 700 | 150 | 1,000 |
| Mapillary Vistas | 16,050 | 1,800 | 150 | 2,000 |

**The manifest file itself will be added to `data/manifests/` in an upcoming release.** Until
then it can be regenerated with step 2 below; compare the SHA-256 of the result with the value
above to confirm it is identical.

---

## 3. Reproducing the benchmark

Run the steps in order from the repository root.

| Step | Command | Output |
|---|---|---|
| 1. Audit datasets | `python tools/audit_cityscapes.py`<br>`python tools/audit_bdd100k.py`<br>`python tools/audit_mapillary.py` | per-image audit CSVs and mask stats (`records/`) |
| 2. Build frozen manifest | `python tools/build_frozen_manifest.py`<br>`python tools/finalize_manifest_and_audit.py`<br>`python scripts/validate_manifest.py` | train / tuning / evaluation file lists and tier assignments; `manifest_audit_report.json` |
| 3. Validate curvature extractor | `python tools/validate_curvature_suite.py` | `records/curvature_*.csv`, `records/visual_checks/curvature_validation_suite.png` |
| 4. Pre-resize images | `python tools/build_512x1024_cache.py` | 512×1024 image / mask cache |
| 5. Check architectures | `python tools/verify_8_architectures.py` | `records/confirmed_8_architectures_table.*` |
| 6. One-seed pilot | `python tools/run_one_seed_pilot.py` | `records/Deliverable_4_Pilot_and_Protocol_Report.json` |
| 7. Full benchmark | `python tools/run_staged_benchmark.py` | Stage 1: 72 runs (8 models × 3 datasets × 3 seeds)<br>Stage 2: 240 runs (8 models × 3 datasets × 10 seeds) |
| 8. Statistics, tables and figures | `python scripts/run_analysis.py`<br>`python tools/generate_manuscript_artifacts.py` | `records/statistical_significance_tests.json`, `records/manuscript_latex/*.tex`, `records/manuscript_figures/*.png` |

To regenerate the paper's tables and figures **without retraining**, run only step 8 on the
included `records/stage2_240runs_summary.csv`.

### Paper ↔ repository map

| In the paper | File in this repository | Generated by |
|---|---|---|
| Table 1 – primary benchmark | `records/manuscript_latex/table1_primary_benchmark.tex` | step 8 |
| Table 2 – curvature tiers and discrepancies | `records/manuscript_latex/table2_curvature_and_discrepancies.tex` | step 8 |
| Table 3 – hardware efficiency | `records/manuscript_latex/table3_hardware_efficiency.tex` | step 8 |
| Table 4 – statistical significance | `records/manuscript_latex/table4_statistical_significance.tex` | step 8 |
| Fig. 1 – Nemenyi critical difference | `records/manuscript_figures/figure1_nemenyi_critical_difference.png` | step 8 |
| Fig. 2 – qualitative predictions | `records/manuscript_figures/figure2_qualitative_predictions_grid.png` | step 8 |
| Fig. 3 – Sharp-curvature failure modes | `records/manuscript_figures/figure3_sharp_curvature_failure_modes.png` | step 8 |
| Sharp subset (97 images) vs. urban Sharp control (214 images) | `records/cityscapes_sharp_97_ids.csv`, `records/sharp_214_ids.csv`, `records/reconciliation_97_vs_214.json` | step 2 |
| Curvature extractor validation | `records/curvature_*.csv`, `records/Deliverable_3_Curvature_Validation_Report.json` | step 3 |
| Raw and Holm-adjusted p-values, test statistics | `records/statistical_significance_tests.json` | step 8 |
| Per-run (model × dataset × seed) results | `records/stage2_240runs_summary.csv` | step 7 |

---

## 4. Protocol summary

- **Curvature tiers:** thresholds and extractor settings are in `configs/experiment.yaml`.
  The curvature index is measured in the normalised image plane. It is **not** physical road
  curvature.
- **Seeds:** 10 per model–dataset pair. The seed list is in `configs/experiment.yaml`.
- **Data splits:** training, tuning (loss-weight and threshold selection) and evaluation
  images are disjoint. See `data/manifests/`, `records/manifest_audit_report.json` and `records/Deliverable_2_Dataset_and_Manifest_Audit.json`.
- **Latency:** measured as described in `roadsegbench/evaluation/latency.py`. Reported
  latencies (FP32) were measured on an NVIDIA RTX A4000; see `records/nvidia-smi.txt` and
  `records/manuscript_latex/table3_hardware_efficiency.tex`.

---

## Citation

See `CITATION.cff`, or cite the paper above.

## Licence

Code in this repository: MIT (see `LICENSE`). Datasets, third-party repositories and
pretrained weights keep their original licences.

## Contact

Questions, bug reports and reproduction problems: please open an
[issue](../../issues) in this repository.

Corresponding author: Basavaraj Chougula, Research Scholar, Department of Electronics and
Communication Engineering, Visvesvaraya Technological University, Belagavi, Karnataka, India,
and Jain College of Engineering and Research, Belagavi, Karnataka, India.

## Authors

- **Basavaraj Chougula**, Visvesvaraya Technological University, Belagavi, and Jain College of
  Engineering and Research, Belagavi, Karnataka, India
- **Anand Gudnavar**, Department of Computer Science and Engineering (AIML), Jain College of
  Engineering and Research, Belagavi (Visvesvaraya Technological University, Belagavi), India
- **Keerti Naregal**, Department of Computer Science and Engineering, Graphic Era University,
  Dehradun, India
