# n-PYS: Data-Driven Variable-Exponent Analysis for Photoemission Yield Spectroscopy

This repository contains the official Python implementation of the **1/n-Scan method** and the **Integrated Residual Diagnostics Framework** for Photoemission Yield Spectroscopy (PYS) described in the associated paper (manuscript in preparation for journal submission).

Our framework enables automated, data-driven materials discovery by systematically extracting physical phenomena from PYS data obtained in ambient environments without manual intervention. It overcomes the geometric biases of conventional logarithmic transformations and introduces a self-diagnostic system (using Akaike Weights, NMAE, RMR, Durbin-Watson statistic, and ΔR²) to distinguish between hardware-induced data degradation and physical model breakdowns (e.g., the overlap of two emission components with different thresholds).

## Features
* **1/n-Scan Algorithm (`fitter.py`)**: A robust, variable-exponent fitting algorithm operating in the original signal space to prevent heteroscedasticity.
* **Automated ROI Extraction (`rk_pys_roi.py` & `robust_piecewise_segmenter.py`)**: Dynamic Programming (DP)-based piecewise regression to automatically identify signal onset, background noise, and spectral saturation limits.
* **Integrated Residual Diagnostics**: Evaluates estimation uncertainty and model validity autonomously.
* **Automated Reporting (`reportmaker.py`)**: Generates PDF/PNG/SVG diagnostic reports automatically for batch analysis.
* **Data Conversion (`datconv.py`)**: Direct parsing of Riken Keiki AC-series `.dat` files.

## Repository Structure

* `fitter.py`: Core algorithms for 1/n-Scan and 3LS optimization, including residual metrics calculation.
* `robust_piecewise_segmenter.py`: DP-based piecewise linear/quadratic regression for autonomous breakpoint detection.
* `rk_pys_roi.py`: Region of Interest (ROI) extraction module.
* `datconv.py`: Data parser and converter for PYS `.dat` files.
* `reportmaker.py`: PDF report generation module.
* `batch_analyzer.py`: Main execution script for batch processing multiple PYS datasets.
* `sample_data/Au/`: 10 real, continuously measured `.dat` files of a polycrystalline rolled Au reference sample (RIKEN KEIKI AC-5), used as the single-component benchmark in the paper (Section 4.2 / Supplementary S6).
* `sample_data/HF-Si-PL/`, `HF-Si-PH/`, `HF-Si-NH/`, `HF-Si-NL/`: the actual 60-times continuous measurement `.dat` files (HF-treated Si, in air) underlying Fig. 3, Fig. 4, and Supplementary Fig. S7.1 of the paper (4 doping types x 60 files).
* `example_usage.py`: Runnable example that analyzes the bundled Au and Si sample data, both as a single file and as a batch.

## Installation

We recommend using a virtual environment (e.g., `venv` or `conda`). Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

`npys` is a regular (non-installed) Python package: place the `npys/` folder in your project's root directory, alongside your own script, and import it as a package (do not `cd` into `npys/` and import its modules directly — they use package-relative imports).

### Quickstart

Sample data is bundled under `sample_data/Au/` (10 continuous measurements of a Au reference sample), so you can try the pipeline immediately without any external data:

```bash
python example_usage.py
```

This parses one `.dat` file and runs the 1/n-Scan method on it directly, then reproduces the paper's 10-times continuous Au benchmark via batch processing (results are written to `example_output/`).

### Batch processing your own data

Below is an example of how to configure a script (placed next to the `npys/` folder) to process a continuous measurement dataset. This uses the bundled `sample_data/HF-Si-PL/` folder (the actual 60-times continuous measurement of the heavily doped p-type Si sample from Fig. 4b of the paper), but any folder of `.dat` files can be substituted.

```python
from pathlib import Path
from npys import batch_analyzer

if __name__ == "__main__":
    # Define input and output directories
    INPUT_DIRECTORY = Path("./sample_data/HF-Si-PL")
    OUTPUT_DIRECTORY = Path("./Si_PL_output")  
    
    # Run batch processing
    batch_analyzer.process_batch(
        input_dir=INPUT_DIRECTORY, 
        output_dir=OUTPUT_DIRECTORY, 
        save_file_name="Si_PL_summary", 
        use_roi=False, # Set to True to enable the ROI extraction process
        # max_segments=3,
        file_indices=list(range(0, 60)) # Process the first 60 files
    )
```

The other three doping types can be reproduced the same way by pointing `INPUT_DIRECTORY` at `sample_data/HF-Si-NL/`, `sample_data/HF-Si-PH/`, or `sample_data/HF-Si-NH/`.

Alternatively, `batch_analyzer.py` can be run directly as a module from the directory that contains `npys/`:

```bash
python -m npys.batch_analyzer
```

## Data Availability

The primary raw data underlying the paper's main results (10-times continuous Au measurement, and the 60-times continuous HF-treated Si measurements for all four doping types) is bundled directly in this repository under `sample_data/` — see Repository Structure above. The remaining data used in the paper, together with the data bundled here, will be archived with a persistent DOI in the NIMS Materials Data Repository (MDR; https://mdr.nims.go.jp/) upon publication; the MDR deposit is the authoritative long-term archival copy, while the copy here is provided for immediate, no-download reproducibility of the code examples.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for the full text.