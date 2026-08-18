import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List, Optional

from . import fitter
from . import reportmaker
from . import datconv
from .rk_pys_roi import pys_roi


def ps_file_list(p_path: Union[str, Path], search_word: str = '*.dat', info: bool = False) -> List[Path]:
    p_paths = Path(p_path)
    p_lists = list(p_paths.rglob(search_word))
    
    if not p_lists:
        return []

    try:
        p_names_sorted = sorted(p_lists, key=lambda p: int(p.name.split('_')[-1].split('.')[0]))
    except ValueError:
        p_names_sorted = sorted(p_lists, key=lambda p: p.name)

    if info:
        print(f"Found {len(p_names_sorted)} files matching '{search_word}' in '{p_path}' and subdirectories")
        for p in p_names_sorted:
            print(f"  - {p.relative_to(p_paths)}")

    return p_names_sorted


def format_result(method_name: str, res: fitter.FitResult) -> dict:
    return {
        "Method": method_name,
        "n": round(res.n, 2) if res.n is not None else np.nan,
        "I_th": round(res.th, 3) if res.th is not None else np.nan,
        "DW": round(res.dw, 3) if res.dw is not None else np.nan,
        "AIC": round(res.aic, 1) if res.aic is not None else np.nan,
        "R2_RMSE": round(res.r2_rmse, 3) if getattr(res, 'r2_rmse', None) is not None else np.nan,
        "R2_MAE": round(res.r2_mae, 3) if res.r2_mae is not None else np.nan,
        "dR2": round(res.d_r2, 4) if getattr(res, 'd_r2', None) is not None else np.nan,
        "RMR": round(res.rmr, 4) if res.rmr is not None else np.nan,
        "NMAE": round(getattr(res, 'nmae', np.nan), 4) if getattr(res, 'nmae', None) is not None else np.nan,
        "Variance": round(res.variance_n, 4) if res.variance_n is not None else np.nan,
        "H": round(getattr(res, 'H', np.nan), 3) if getattr(res, 'H', None) is not None else np.nan,
        "P": round(getattr(res, 'P', np.nan), 3) if getattr(res, 'P', None) is not None else np.nan,
        "step_n": getattr(res, 'd_n', np.nan)
    }


def plot_summary_heatmaps(df_summary: pd.DataFrame, output_path: Union[str, Path]):
    samples = df_summary["FileName"].unique()
    methods = ["1/n-Scan (Best)", "Fixed n=2.0", "Fixed n=3.0"]
    
    metrics = {
        "AIC": {"cmap": "magma_r", "vmin": None, "vmax": None},
        "n": {"cmap": "plasma", "vmin": 0.5, "vmax": 4.0},
        "I_th": {"cmap": "coolwarm", "vmin": None, "vmax": None}, 
        "DW": {"cmap": "RdYlGn", "vmin": 0, "vmax": 3},
        "dR2": {"cmap": "viridis", "vmin": None, "vmax": None}, 
        "RMR": {"cmap": "YlGnBu", "vmin": 0.9, "vmax": 1.5},
        "n_err": {"cmap": "YlOrRd", "vmin": 0.0, "vmax": None}
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), max(4, len(samples) * 0.6)))
    fig.suptitle("PYS Batch Analysis Summary (All Methods)", fontsize=18, fontweight="bold", y=1.02)

    for i, (metric, config) in enumerate(metrics.items()):
        ax = axes[i]
        
        pivot_df = df_summary.pivot(index="FileName", columns="Method", values=metric)
        pivot_df = pivot_df.reindex(columns=methods)
        data_matrix = pivot_df.values

        im = ax.imshow(data_matrix, cmap=config["cmap"], aspect="auto", 
                       vmin=config["vmin"], vmax=config["vmax"])
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric, fontsize=12)

        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(samples)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
        
        if i == 0:
            ax.set_yticklabels(samples, fontsize=9)
        else:
            ax.set_yticklabels([]) 
            
        ax.set_title(f"{metric}", fontsize=14, fontweight='bold')

        for row in range(len(samples)):
            for col in range(len(methods)):
                val = data_matrix[row, col]
                if not np.isnan(val):
                    if metric in ["n", "Variance", "n_err"] and col > 0:
                        continue
                        
                    if metric == "AIC": text_str = f"{val:.1f}"
                    elif metric in ["Variance", "n_err", "dR2"]: text_str = f"{val:.3f}"
                    else: text_str = f"{val:.2f}"
                        
                    ax.text(col, row, text_str, ha="center", va="center", color="black", 
                            fontsize=10, fontweight="bold",
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Success] Main summary heatmap saved to: {output_path}")


def plot_1n_heatmaps(df_summary: pd.DataFrame, output_path: Union[str, Path]):
    df_1n = df_summary[df_summary["Method"] == "1/n-Scan (Best)"]
    if df_1n.empty: return

    samples = df_1n["FileName"].unique()
    method = "1/n-Scan (Best)"

    metrics = {
        "n": {"cmap": "plasma", "vmin": 0.5, "vmax": 4.0},
        "I_th": {"cmap": "coolwarm", "vmin": None, "vmax": None},
        "DW": {"cmap": "RdYlGn", "vmin": 0, "vmax": 3},
        "AIC": {"cmap": "magma_r", "vmin": None, "vmax": None},
        "dR2": {"cmap": "viridis", "vmin": None, "vmax": None},
        "RMR": {"cmap": "YlGnBu", "vmin": 0.9, "vmax": 1.5},
        "n_err": {"cmap": "YlOrRd", "vmin": 0.0, "vmax": None}
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(2.5 * len(metrics), max(4, len(samples) * 0.6)))
    fig.suptitle("1/n-Scan Only Summary Metrics", fontsize=18, fontweight="bold", y=1.02)

    for i, (metric, config) in enumerate(metrics.items()):
        ax = axes[i]
        
        pivot_df = df_1n.pivot(index="FileName", columns="Method", values=metric)
        data_matrix = pivot_df.values

        im = ax.imshow(data_matrix, cmap=config["cmap"], aspect="auto",
                       vmin=config["vmin"], vmax=config["vmax"])
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric, fontsize=12)

        ax.set_xticks([0])
        ax.set_yticks(np.arange(len(samples)))
        ax.set_xticklabels([method], rotation=30, ha="right", fontsize=10)
        
        if i == 0:
            ax.set_yticklabels(samples, fontsize=9)
        else:
            ax.set_yticklabels([]) 
            
        ax.set_title(f"{metric}", fontsize=14, fontweight='bold')

        for row in range(len(samples)):
            val = data_matrix[row, 0]
            if not np.isnan(val):
                if metric == "AIC": text_str = f"{val:.1f}"
                elif metric in ["n_err", "dR2"]: text_str = f"{val:.3f}"
                else: text_str = f"{val:.2f}"
                    
                ax.text(0, row, text_str, ha="center", va="center", color="black", 
                        fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Success] 1/n-Scan only heatmap saved to: {output_path}")


def process_batch(input_dir: Union[str, Path], output_dir: Union[str, Path], save_file_name: str = "batch_summary", file_indices: Union[List[int], range, None] = None, use_roi: bool = True):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    report_dir = output_path / "individual_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    dat_files = ps_file_list(input_path, search_word="*.dat", info=False)
    
    if not dat_files:
        print(f"[Error] No .dat files found in {input_path}")
        return

    if file_indices is not None:
        dat_files = [dat_files[i] for i in file_indices if i < len(dat_files)]
        if not dat_files:
            print(f"[Error] No files found for specified indices")
            return
        print(f"Found {len(dat_files)} files (filtered by indices). Starting batch processing...")
    else:
        print(f"Found {len(dat_files)} files. Starting batch processing...")
    
    summary_records = []

    for filepath in dat_files:
        file_name_full = filepath.name
        file_name = filepath.stem
        print(f"\nProcessing: {file_name_full} ...")

        try:
     
            ac_converter = datconv.AdvAcConv(str(filepath))
            ac_converter.convert()

            sample_name = getattr(ac_converter, 'sampleName', np.nan)
            x_raw = ac_converter.calcdata.uvEnergy
            y_raw = ac_converter.calcdata.ydata
            if len(x_raw) < 10:
                print(f"  [Skip] Not enough data points in {file_name_full}")
                continue
            
            p_num = getattr(ac_converter, 'powerNumber', np.nan)
            th_eng = getattr(ac_converter.metadata, 'thresholdEnergy', np.nan) if ac_converter.metadata else np.nan

            fig_roi = None
            th0, bg0 = None, None

            try:
                # dw_threshold=0.0 disables the DW-based segment-rejection safeguard
                # (see Supplementary S10) for batch runs, since the stricter default
                # (dw_threshold=1.0) can reject valid segments on some samples/instruments.
                # Raise this value if stronger rejection of autocorrelated segments is desired.
                x_fit, y_fit, _, _, onset_x, bg_intercept, fig_roi = pys_roi(
                    x_raw, y_raw, plot=True, rollback_bps=0,
                    dw_threshold=0.0, spike_threshold_z=10.0
                )
                th0 = onset_x
                bg0 = max(0, bg_intercept)
            except ValueError as ve:
                print(f"  [Warning] ROI Extraction failed for {file_name_full}. Falling back to Full Data.")
                x_fit, y_fit = x_raw, y_raw

            if use_roi:
                print("  -> Running ROI Extraction...")
                x_fit, y_fit = x_fit, y_fit

            else:
                fig_roi=None
                x_fit, y_fit = x_raw, y_raw

            analyzer = fitter.OneOverNScanFitter(x_fit, y_fit)
            
            best_res, all_res = analyzer.fit(th0=th0, bg0=bg0)
            res_n2 = analyzer.oneshot_fit(n=2.0, th0=th0, bg0=bg0)
            res_n3 = analyzer.oneshot_fit(n=3.0, th0=th0, bg0=bg0)

            res_list = [
                format_result("1/n-Scan (Best)", best_res),
                format_result("Fixed n=2.0", res_n2),
                format_result("Fixed n=3.0", res_n3)
            ]
            df_results = pd.DataFrame(res_list)

            df_pdf = df_results.rename(columns={
                "R2_RMSE": "R2_RMS",
                "Variance": "Var",
                "step_n": "dn"
            })
            df_pdf["Method"] = df_pdf["Method"].replace({"1/n-Scan (Best)": "1/n-Scan"})

            for record in res_list:
                record["SampleName"] = sample_name
                record["FileName"] = file_name
                record["powerNumber"] = p_num
                record["thresholdEnergy"] = th_eng
                summary_records.append(record)

            builder = reportmaker.ReportBuilder(dpi=150)
           
            intro_text = (f"SampleName: {sample_name}\n"
                          f"FileName: {file_name}\n"
                          f"n: {best_res.n:.2f}, $I_{{th}}$: {best_res.th:.2f}, DW: {best_res.dw:.2f}, RMR: {best_res.rmr:.3f}\n"
                          f"$\Delta R^2$: {best_res.d_r2:.4f}, P: {best_res.P:.2f}")
            builder.add_text("header", intro_text, title="PYS Measurement Report")
            
            report_items = ["header"]
            
            # ROI
            if fig_roi is not None:
                builder.add_graph("roi_plot", fig_roi)
                report_items.append("roi_plot")

            fig_fit = analyzer.plot_result_ext2(best_res, all_res, show_plot=False)
            builder.add_graph("fit_plot", fig_fit)
            report_items.append("fit_plot")
            
            builder.add_dataframe("table", df_pdf, max_cols=18, fontsize=6)
            report_items.append("table")

            pdf_path = report_dir / f"{file_name}_report.pdf"
            builder.generate_report(report_items, title=sample_name, filename=str(pdf_path), report_format="pdf")
            print(f"  [Saved] Report: {pdf_path}")

        except Exception as e:
            print(f"  [Error] Failed to process {file_name_full}: {e}")

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        
        df_summary["n_err"] = df_summary["P"] * df_summary["step_n"]
        
        cols = ["SampleName", "FileName", "Method", "powerNumber", "thresholdEnergy", "n", "I_th", "DW", "AIC", 
                "R2_RMSE", "R2_MAE", "dR2", "RMR", "NMAE", 
                "Variance", "H", "P", "step_n", "n_err"]
        
        df_summary = df_summary[[c for c in cols if c in df_summary.columns]]

        csv_path = output_path / f"{save_file_name}.csv"
        df_summary.to_csv(csv_path, index=False)
        print(f"\n[Success] Batch summary CSV saved to: {csv_path}")

        df_best_only = df_summary[df_summary["Method"] == "1/n-Scan (Best)"]
        csv_path_best = output_path / f"{save_file_name}_1n_only.csv"
        df_best_only.to_csv(csv_path_best, index=False)
        print(f"[Success] 1/n-Scan only summary CSV saved to: {csv_path_best}")

        heatmap_path = output_path / f"{save_file_name}_heatmap.png"
        plot_summary_heatmaps(df_summary, heatmap_path)

        heatmap_1n_path = output_path / f"{save_file_name}_1n_heatmap.png"
        plot_1n_heatmaps(df_summary, heatmap_1n_path)


if __name__ == "__main__":
    
    INPUT_DIRECTORY = Path("./data")     
    OUTPUT_DIRECTORY = Path("./Si_PL")  
    
    # use_roi=True enables the ROI extraction process
    process_batch(
        input_dir=INPUT_DIRECTORY, 
        output_dir=OUTPUT_DIRECTORY, 
        save_file_name="Si_PL_summary", 
        use_roi=False,
        # max_segments=3,
        file_indices=list(range(0, 60))
    )