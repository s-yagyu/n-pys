"""Example: analyze the sample Au data bundled in sample_data/Au/.

The 10 files in sample_data/Au/ are a real 10-times continuous PYS
measurement of a polycrystalline rolled Au reference sample (RIKEN KEIKI
AC-5, 4.0-6.2 eV, 0.05 eV step), used in the associated paper as the
single-component benchmark (Section 4.2 / Supplementary S6). Running the
batch analysis below should reproduce values close to: optimal n ~ 1.8-1.9,
NMAE ~ 4.5%, RMR ~ 1.62, DW ~ 2.15, P*dn ~ 0.27.

Run this script from the directory that contains the `npys/` folder:
    python example_usage.py
"""
from pathlib import Path

from npys import datconv, batch_analyzer, OneOverNScanFitter, pys_roi

SAMPLE_DIR = Path(__file__).parent / "sample_data" / "Au"
OUTPUT_DIR = Path(__file__).parent / "example_output"


def analyze_single_file(use_roi: bool = False):
    """Parse one .dat file and run the 1/n-Scan method on it directly.

    use_roi: toggles the DP-based piecewise-regression ROI extractor
        (`pys_roi`, see rk_pys_roi.py / robust_piecewise_segmenter.py).
        When True, it is used to automatically detect the background level
        and signal onset, and to trim any high-energy saturation/bump
        region before fitting; the detected onset/background are also
        passed to the fitter as initial guesses (th0, bg0). This changes
        both the fitted energy range and the initial guesses, so the final
        n, I_th, and residual metrics (NMAE, RMR, DW, P*dn) can differ from
        the use_roi=False case -- compare the two printed results below.
    """
    dat_file = SAMPLE_DIR / "Au 240229034207_1.dat"

    converter = datconv.AdvAcConv(str(dat_file))
    converter.convert()

    x_raw, y_raw = converter.calcdata.uvEnergy, converter.calcdata.ydata

    x_fit, y_fit = x_raw, y_raw
    th0, bg0 = None, None

    if use_roi:
        try:
            # Same call pattern as batch_analyzer.process_batch: x_full/y_full
            # keep the background region (needed by the fitter's own BG term),
            # only the high-energy tail past the detected peak/shoulder is cut.
            x_full, y_full, _, _, onset_x, bg_intercept, fig_roi = pys_roi(
                x_raw, y_raw, plot=True, rollback_bps=0,
                dw_threshold=0.0, spike_threshold_z=10.0,
            )
            th0 = onset_x
            bg0 = max(0.0, bg_intercept)
            x_fit, y_fit = x_full, y_full
            if fig_roi is not None:
                # Note: pys_roi() always builds this figure with show_plot=False
                # internally, which closes it (plt.close(fig)) before returning
                # it -- so fig_roi.show() would raise "Figure.show works only
                # for figures managed by pyplot". Save it to disk instead.
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                roi_fig_path = OUTPUT_DIR / "single_file_roi_diagnostic.png"
                fig_roi.savefig(roi_fig_path, dpi=150, bbox_inches="tight")
                print(f"[Single file] ROI diagnostic figure saved to: {roi_fig_path}")
        except ValueError as exc:
            print(f"[Single file] ROI extraction failed ({exc}); using the full spectrum instead.")

    fitter = OneOverNScanFitter(x_fit, y_fit)
    best_result, all_results = fitter.fit(th0=th0, bg0=bg0)

    p_dn = best_result.P * best_result.d_n
    print(f"[Single file] (use_roi={use_roi}) n={best_result.n:.2f}, I_th={best_result.th:.2f} eV, "
          f"NMAE={best_result.nmae * 100:.2f}%, RMR={best_result.rmr:.2f}, "
          f"DW={best_result.dw:.2f}, P*dn={p_dn:.3f}")

    # Opens the diagnostic figure (Fig. 2-style panel layout in the paper).
    fitter.plot_result_ext2(best_result, all_results)


def analyze_batch():
    """Reproduce the 10-times continuous Au benchmark (paper Section 4.2 / Supp. S6)."""
    batch_analyzer.process_batch(
        input_dir=SAMPLE_DIR,
        output_dir=OUTPUT_DIR,
        save_file_name="Au_summary",
        use_roi=False,
        file_indices=list(range(0, 10)),
    )
    print(f"[Batch] Summary CSV and per-file PDF reports written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("--- Single-file analysis (Au 240229034207_1.dat), use_roi=False ---")
    analyze_single_file(use_roi=False)

    print("\n--- Single-file analysis (Au 240229034207_1.dat), use_roi=True ---")
    analyze_single_file(use_roi=True)

    print("\n--- Batch analysis (all 10 continuous Au measurements) ---")
    analyze_batch()
