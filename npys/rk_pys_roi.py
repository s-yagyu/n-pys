
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .robust_piecewise_segmenter import PiecewiseLinearRegression
                                
@dataclass
class PeakInfo:
    type: str 
    x: float
    val_before: Optional[float] = None 
    val_after: Optional[float] = None

class PeakEstimator:
    """
    Estimate peak positions from piecewise regression results
    (slope changes or quadratic vertex points) in a noise-robust way.
    """
    def __init__(self, fit_model):
        self.model = fit_model
        self.best_model_info = fit_model.models[fit_model.best_k]
        self.segments_info = self.best_model_info.segments_info
        self.breakpoints = self.best_model_info.segment_positions
        self.peaks: List[PeakInfo] = []

    def estimate_peaks(self, slope_threshold=1e-3) -> List[PeakInfo]:
        self.peaks = []
        if not self.segments_info: 
            return self.peaks

        if hasattr(self.segments_info[0], 'slope'):
            for i in range(len(self.segments_info) - 1):
                m1 = self.segments_info[i].slope
                m2 = self.segments_info[i+1].slope
                bp_x = self.breakpoints[i]
                if m1 > slope_threshold and m2 < -slope_threshold:
                    self.peaks.append(PeakInfo('Clear Peak', bp_x, m1, m2))
                    
        elif hasattr(self.segments_info[0], 'c2'):
            for seg in self.segments_info:
                if seg.c2 < -1e-6: 
                    x_vertex_local = -seg.c1 / (2 * seg.c2)
                    x_vertex_global = x_vertex_local + seg.start_x
                    if seg.start_x <= x_vertex_global <= seg.end_x:
                        self.peaks.append(PeakInfo('Clear Peak', x_vertex_global, seg.d2, seg.d2))

        self.peaks.sort(key=lambda p: p.x)
        return self.peaks


class SpectrumROIExtractor:
    """
    Remove background (BG) from data such as PYS spectra and extract the
    power-law region of interest (ROI) from signal onset to the pure ROI.
    """
    def __init__(self, fit_model, peak_estimator=None):
        self.model = fit_model
        self.peak_estimator = peak_estimator
        self.bg_slope = 0.0
        self.bg_intercept = 0.0
        self.onset_x = None
        self.peak_x = None
        self.bg_line = None
        self.y_bg_removed = None

    def process_and_extract(self, slope_drop_tolerance=0.98, rollback_bps=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform BG estimation, fully automatic onset detection, and ROI extraction
        while removing shoulders.

        Args:
            slope_drop_tolerance: Threshold for considering a segment as a shoulder
                when its slope drops below this fraction of the previous segment.
                Default is 0.98.
            rollback_bps: Number of breakpoints to roll back from the detected
                shoulder before cutting the ROI. Default is 0.

        Returns:
            roi_x: X coordinates of the ROI region.
            roi_y_orig: Original ROI data including background.
            roi_y_bg_removed: ROI data after background removal.
        """
        model_info = self.model.models[self.model.best_k]
        
     
        bg_seg = model_info.segments_info[0]
        mask_bg = (self.model.x >= bg_seg.start_x) & (self.model.x <= bg_seg.end_x)
        
        # Fail-safe: if the BG region is too narrow (< 10% of total)
        if np.sum(mask_bg) < 3:
            cutoff = self.model.x[0] + (self.model.x[-1] - self.model.x[0]) * 0.1
            mask_bg = self.model.x <= cutoff

        coeffs = np.polyfit(self.model.x[mask_bg], self.model.y_orig[mask_bg], 1)
        self.bg_slope = coeffs[0]
        self.bg_intercept = coeffs[1]
        
        self.bg_line = self.bg_slope * self.model.x + self.bg_intercept
        self.y_bg_removed = self.model.y_orig - self.bg_line


        self.onset_x = self.model.x[-1]
        
        for i in range(1, len(model_info.segments_info)):
            seg = model_info.segments_info[i]
            if hasattr(seg, 'slope'):
                if seg.slope > self.bg_slope:
                    self.onset_x = seg.start_x
                    print(f"[Onset Detection] Onset automatically found at X={self.onset_x:.2f} (Model detected signal rise)")
                    break
            elif hasattr(seg, 'c2'): # Fallback for quadratic or similar models
                slope_start = 2 * seg.c2 * 0 + seg.c1 
                if slope_start > self.bg_slope:
                    self.onset_x = seg.start_x
                    print(f"[Onset Detection] Onset automatically found at X={self.onset_x:.2f} (Model detected signal rise)")
                    break


        self.peak_x = self.model.x[-1]
        segments = model_info.segments_info
        
        for i in range(1, len(segments)):
            prev_seg = segments[i-1]
            curr_seg = segments[i]
            
            if curr_seg.start_x > self.onset_x:
                if hasattr(prev_seg, 'slope') and hasattr(curr_seg, 'slope'):
                    m_prev = prev_seg.slope
                    m_curr = curr_seg.slope
                    
                    if m_curr < m_prev * slope_drop_tolerance:
                        if rollback_bps == 1:
                            self.peak_x = prev_seg.start_x
                            print(f"[ROI Extraction] Shoulder at X={curr_seg.start_x:.2f} (Slope {m_prev:.1f} -> {m_curr:.1f})")
                            print(f"                 -> Rolled back 1 segment. Cutoff set to X={self.peak_x:.2f}")
                        else:
                            self.peak_x = curr_seg.start_x
                            print(f"[ROI Extraction] Shoulder at X={self.peak_x:.2f} (Slope {m_prev:.1f} -> {m_curr:.1f})")
                        break

        # Set 1: Data including background (from start to peak)
        mask_full_to_peak = (self.model.x <= self.peak_x)
        data_with_bg = (self.model.x[mask_full_to_peak], self.model.y_orig[mask_full_to_peak])
        
        # Set 2: Data without background (from onset to peak)
        mask_roi_only = (self.model.x >= self.onset_x) & (self.model.x <= self.peak_x)
        data_no_bg = (self.model.x[mask_roi_only], self.y_bg_removed[mask_roi_only])
        
        return data_with_bg, data_no_bg
    
  
    def plot_results2(self, show_plot=True):
        """
        Combine the four plot panels from plot_results into two rows:
        1. Model fit with highlighted BG/Onset/Peak and ROI with BG.
        2. 1st derivative (left axis) and breakpoint jumps (right axis).
        """
       
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
        
        # ==========================================
        # First plot (axs[0]): Fit results and highlighted ROI (with BG)
        # ==========================================
        self.model.plot(ax=axs[0])
        
        axs[0].plot(self.model.x, self.bg_line, '--', color='blue', label='Estimated BG', alpha=0.8)
        axs[0].axvline(x=self.onset_x, color='green', linestyle=':', linewidth=2, label=f'Onset ({self.onset_x:.2f})')
        axs[0].axvline(x=self.peak_x, color='purple', linestyle=':', linewidth=2, label=f'End/Cutoff ({self.peak_x:.2f})')
        
        roi_mask = (self.model.x >= self.onset_x) & (self.model.x <= self.peak_x)
        axs[0].plot(self.model.x[roi_mask], self.model.y_orig[roi_mask], 's', 
                    color='tab:blue', markersize=6, alpha=0.6, label='ROI (with BG)')
        
        axs[0].legend(loc='upper left')
        axs[0].set_title("1. Model Fit & ROI Detection (with BG)")
        axs[0].set_ylabel('Intensity')
        
        # ==========================================
        # Second plot (axs[1]): 1st derivative (left axis) and jumps (right axis)
        # ==========================================
        ax1 = axs[1]
        ax2 = ax1.twinx() 
        
        model_info = self.model.models[self.model.best_k]
        segments = model_info.segments_info
        breakpoints = model_info.segment_positions
        
        for seg in segments:
            slope = getattr(seg, 'slope', getattr(seg, 'c1', 0))
            ax1.hlines(y=slope, xmin=seg.start_x, xmax=seg.end_x, 
                       color='tab:orange', linewidth=2, alpha=0.8)
            
        ax1.set_ylabel('1st Derivative (Slope)', color='tab:orange')
        ax1.tick_params(axis='y', labelcolor='tab:orange')
        
        jumps = []
        bp_xs = []
        for i in range(len(segments) - 1):
            m1 = getattr(segments[i], 'slope', getattr(segments[i], 'c1', 0))
            m2 = getattr(segments[i+1], 'slope', getattr(segments[i+1], 'c1', 0))
            jumps.append(m2 - m1)
            bp_xs.append(breakpoints[i])
            
        if bp_xs:
            ax2.bar(bp_xs, jumps, width=(self.model.x[-1] - self.model.x[0]) * 0.01, 
                    color='tab:cyan', alpha=0.4, label='Jumps (\u0394 Slope)')
            ax2.plot(bp_xs, jumps, 'D', color='tab:cyan', markersize=6)
            
        ax2.set_ylabel('Jumps (\u0394 Slope)', color='tab:cyan')
        ax2.tick_params(axis='y', labelcolor='tab:cyan')
        
        ax1.axvline(x=self.onset_x, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
        ax1.axvline(x=self.peak_x, color='purple', linestyle=':', linewidth=1.5, alpha=0.5)
        
        ax1.set_title("2. Analytical 1st Derivative & Breakpoint Jumps")
        ax1.set_xlabel('X')
        ax1.grid(True, linestyle=':', alpha=0.4)
        
        custom_lines = [Line2D([0], [0], color='tab:orange', lw=2)]
        custom_labels = ['1st Derivative']
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(custom_lines + lines2, custom_labels + labels2, loc='upper left')

        plt.tight_layout()
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        return fig
        
    def plot_results(self):
        """Plotting section unchanged."""
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [1.5, 1.2, 1, 1]})
        
        self.model.plot(ax=axs[0])
        axs[0].plot(self.model.x, self.bg_line, '--', color='blue', label='Estimated BG', alpha=0.8)
        axs[0].axvline(x=self.onset_x, color='green', linestyle=':', linewidth=2, label=f'Onset ({self.onset_x:.2f})')
        axs[0].axvline(x=self.peak_x, color='purple', linestyle=':', linewidth=2, label=f'End/Cutoff ({self.peak_x:.2f})')
        axs[0].legend(loc='upper left')
        axs[0].set_title("1. Model Fit & Detection (BG/Onset/Peak)")

        roi_mask = (self.model.x >= self.onset_x) & (self.model.x <= self.peak_x)
        axs[1].plot(self.model.x, self.y_bg_removed, 'o', color='lightgray', label='BG Removed Data', alpha=0.5)
        axs[1].plot(self.model.x[roi_mask], self.y_bg_removed[roi_mask], 'o', color='tab:orange', markersize=6, label='Extracted ROI')
        axs[1].axhline(0, color='blue', linestyle='--', alpha=0.5)
        axs[1].axvline(x=self.onset_x, color='green', linestyle=':', linewidth=2)
        axs[1].axvline(x=self.peak_x, color='purple', linestyle=':', linewidth=2)
        axs[1].set_ylabel('Intensity (BG Removed)')
        axs[1].set_title('2. Extracted ROI (Ready for Fowler Fit)')
        axs[1].legend(loc='upper left')
        axs[1].grid(True, linestyle=':', alpha=0.6)

        self.model.plot_derivatives(axes=(axs[2], axs[3]))
        axs[2].set_title('3. Analytical 1st Derivative (Step-wise Slope)')
        axs[3].set_title('4. Analytical 2nd Derivative & Jumps (Detection Basis)')
        
        for ax in [axs[2], axs[3]]:
            ax.axvline(x=self.onset_x, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
            ax.axvline(x=self.peak_x, color='purple', linestyle=':', linewidth=1.5, alpha=0.5)

        plt.tight_layout()
        plt.show()


def pys_roi(x, y, plot=False,
            rollback_bps=0,
            select_k_method='bic_min',
            threshold=6.0,
            max_segments=None,
            dw_threshold=0.5,           
            spike_threshold_z=10.0):
    """Run DP-based piecewise regression to detect BG/onset/peak and extract the ROI.

    Returns (x_full, y_full, x_roi, y_roi, onset_x, bg_intercept, fig):
    x_full/y_full include the background region up to the detected peak/
    shoulder cutoff; x_roi/y_roi are background-removed, onset-to-peak only.

    Note: when plot=True, the returned `fig` is built with show_plot=False
    internally, which closes it (plt.close(fig)) before returning it. Use
    `fig.savefig(...)` to export it (this works on a closed figure); calling
    `fig.show()` will raise "Figure.show works only for figures managed by
    pyplot" since plt.close() deregisters it from pyplot's figure manager.
    """
    model_lin = PiecewiseLinearRegression(
        x, y, 
        select_k_method=select_k_method, 
        threshold=threshold,
        max_segments=max_segments,
        dw_threshold=dw_threshold,         
        spike_threshold_z=spike_threshold_z 
    )
    
    peak_est = PeakEstimator(model_lin)
    peak_est.estimate_peaks()

    extractor = SpectrumROIExtractor(model_lin, peak_estimator=peak_est)
    data_with_bg, data_no_bg = extractor.process_and_extract(rollback_bps=rollback_bps)

    x_full, y_full = data_with_bg  
    x_roi, y_roi = data_no_bg      
    
    fig = None
    if plot:
        print(f"points with BG : {len(x_full)} points (Start X: {x_full[0]:.2f})")
        print(f"points in ROI : {len(x_roi)} points (Start X: {x_roi[0]:.2f})")
        fig = extractor.plot_results2(show_plot=False)

    return x_full, y_full, x_roi, y_roi, extractor.onset_x, extractor.bg_intercept, fig