# IoT-BIM Hazard Warning Experimental Package

This package contains the experimental code and data files used for the IoT-BIM multi-source hazard warning experiments. It is intended for experiment reproduction and archival review.

## Contents

- `config.py`: shared experiment paths, random seed, plotting style, and default settings.
- `data/raw/`: downloaded and extracted raw data files. No data crawling or downloading script is included.
- `data/processed/`: processed data files generated during the experiment workflow and required by the scripts.
- `scripts/`: executable experiment scripts.
- `src/`: implementation of the proposed method, baseline methods, data processing modules, and metric utilities.

## Data Files

- `data/raw/gas_sensor/gas_sensor.zip`: downloaded DS-1 gas sensor drift dataset archive.
- `data/raw/gas_sensor/extracted/Dataset/batch*.dat`: extracted DS-1 batch files.
- `data/processed/ds1_gas.npz`: processed DS-1 gas sensor data.
- `data/processed/ds2_shm.npz`: processed DS-2 structural health monitoring data.
- `data/processed/ds3_iot_bim.npz`: processed DS-3 IoT-BIM simulation data.

## Experiment Scripts

- `scripts/01_exp1_fusion_accuracy.py`: DS-1 fusion accuracy experiment for Table I.
- `scripts/02_exp2_robustness.py`: robustness and ablation experiments.
- `scripts/03_exp3_sensitivity.py`: DS-3 parameter sensitivity analysis for Table VII.
- `scripts/04_exp4_grade_report.py`: DS-3 warning-grade precision and recall report for Table VI.
- `scripts/05_exp5_latency_figure.py`: latency experiment and Figure 5 generation.

## Running

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run an experiment from the package root, for example:

```bash
python scripts/03_exp3_sensitivity.py
```

Generated tables, figures, and CSV files are written to `results/` during execution. The `results/` directory is intentionally not included in this package.

## Excluded

The following items are intentionally excluded:

- generated results, figures, LaTeX tables, and CSV outputs;
- data downloading, crawling, or acquisition scripts;
- Python cache folders and compiled files;
- temporary logs.
