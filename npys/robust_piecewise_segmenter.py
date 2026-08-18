import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.interpolate import LSQUnivariateSpline
from dataclasses import dataclass
from typing import List, Union, Optional
from abc import ABC, abstractmethod

# ==========================================
# 1. Data Structures (Dataclasses)
# ==========================================
@dataclass
class SegmentInfo:
    segment_idx: int
    start_x: float
    end_x: float
    slope: float

@dataclass
class QuadSegmentInfo:
    segment_idx: int
    start_x: float
    end_x: float
    c2: float
    c1: float
    c0: float
    d2: float

@dataclass
class FitResult:
    num_breakpoints: int
    segment_positions: List[float]
    segments_info: List[Union[SegmentInfo, QuadSegmentInfo]]
    fitted_y: np.ndarray
    sse: float
    bic: float
    aic: float
    hqic: float = 0.0
    gcv: float = 0.0
    d1_y: Optional[np.ndarray] = None
    d2_y: Optional[np.ndarray] = None
    fitted_y_cont: Optional[np.ndarray] = None
    bp_d1_jumps: Optional[List[float]] = None
    bp_d2_jumps: Optional[List[float]] = None

# ==========================================
# 2. Base Class
# ==========================================
class BasePiecewiseRegression(ABC):
    def __init__(self, x: np.ndarray, y: np.ndarray, max_segments: int = None,
                 apply_median_filter: bool = False, spike_threshold_z: float = 5.0,
                 spike_dynamic_factor: float = 0.05):

        sort_idx = np.argsort(x)
        self.x = np.asarray(x)[sort_idx]
        self.y_orig = np.asarray(y)[sort_idx]

        N = len(self.x)
        self.max_segments = max_segments if max_segments is not None else max(1, min(N // 5, 20))

        self.models = {}
        self.best_k = 1
        self.selection_method = 'Not Selected'

        self.spike_threshold_z = spike_threshold_z
        self.spike_dynamic_factor = spike_dynamic_factor

        self.apply_median_filter = apply_median_filter
        self.spikes = self._detect_spikes(self.y_orig)
        self.y_fit = np.copy(self.y_orig)

        if self.apply_median_filter:
            self.y_fit = medfilt(self.y_fit, kernel_size=3)

        self.y_fit[self.spikes] = np.nan

    def _detect_spikes(self, y: np.ndarray) -> np.ndarray:
        y_med = medfilt(y, kernel_size=5)
        dev = np.abs(y - y_med)

        dy = np.abs(np.diff(y))
        base_noise = np.median(dy)
        if base_noise == 0:
            base_noise = 1e-8

        dynamic_threshold = (self.spike_threshold_z * base_noise) + (self.spike_dynamic_factor * np.abs(y_med))
        return dev > dynamic_threshold

    @abstractmethod
    def _calc_cost_matrix(self) -> np.ndarray: pass

    @abstractmethod
    def _get_num_params(self, k: int) -> int: pass

    @abstractmethod
    def _fit_segment(self, xi: np.ndarray, yi: np.ndarray, start_x: float, end_x: float, seg_idx: int, start_idx: int = None, end_idx: int = None): pass

    def _get_model_params(self, k: int, indices: List[int]) -> int:
        """Hook method to calculate total parameters, overrideable for Hybrid models."""
        return self._get_num_params(k)

    def _run_fit(self):
        N = len(self.x)
        C = self._calc_cost_matrix()
        dp = np.full((self.max_segments + 1, N + 1), np.inf)
        split = np.zeros((self.max_segments + 1, N + 1), dtype=int)
        dp[1, 1:] = C[0, 1:]

        for k in range(2, self.max_segments + 1):
            prev_dp = dp[k-1, :N].reshape(N, 1)
            total_costs = prev_dp + C
            min_costs = np.min(total_costs, axis=0)
            best_splits = np.argmin(total_costs, axis=0)
            dp[k, k:] = min_costs[k:]
            split[k, k:] = best_splits[k:]

        for k in range(1, self.max_segments + 1):
            if np.isinf(dp[k, N]): continue

            breakpoints = []
            current_end = N
            for m in range(k, 1, -1):
                bp = split[m, current_end]
                breakpoints.append(bp)
                current_end = bp

            segment_indices = breakpoints[::-1]
            segment_positions = [self.x[idx] for idx in segment_indices]
            indices = [0] + segment_indices + [N]

            fitted_y = np.zeros_like(self.x)
            d1_y = np.zeros_like(self.x)
            d2_y = np.zeros_like(self.x)
            segments_info = []

            for i in range(len(indices) - 1):
                start, end = indices[i], indices[i+1]
                xi, yi = self.x[start:end], self.y_fit[start:end]
                st_x = self.x[start]
                en_x = self.x[end - 1] if end > start else self.x[start]

                # Pass start and end indices to support Hybrid models
                seg_fit_y, seg_info, seg_d1, seg_d2 = self._fit_segment(xi, yi, st_x, en_x, i + 1, start, end)

                fitted_y[start:end] = seg_fit_y
                if seg_d1 is not None: d1_y[start:end] = seg_d1
                if seg_d2 is not None: d2_y[start:end] = seg_d2
                if seg_info is not None: segments_info.append(seg_info)

            sse = dp[k, N]
            num_params = self._get_model_params(k, indices)

            if sse > 0:
                bic = N * np.log(sse / N) + num_params * np.log(N)
                aic = N * np.log(sse / N) + 2 * num_params
                hqic = N * np.log(sse / N) + 2 * num_params * np.log(np.log(N)) if N > 2 else -np.inf
            else:
                bic = aic = hqic = -np.inf

            gcv = (sse / N) / ((1.0 - num_params / N) ** 2) if num_params < N else np.inf

            bp_d1_jumps, bp_d2_jumps = [], []
            for bp_idx in segment_indices:
                if 0 < bp_idx < len(self.x):
                    bp_d1_jumps.append(d1_y[bp_idx] - d1_y[bp_idx - 1])
                    bp_d2_jumps.append(d2_y[bp_idx] - d2_y[bp_idx - 1])

            d1_final = d1_y if np.any(d1_y) else np.zeros_like(self.x)
            d2_final = d2_y if np.any(d2_y) else np.zeros_like(self.x)

            self.models[k] = FitResult(
                num_breakpoints=k - 1, segment_positions=segment_positions,
                segments_info=segments_info, fitted_y=fitted_y, sse=sse,
                bic=bic, aic=aic, hqic=hqic, gcv=gcv,
                d1_y=d1_final, d2_y=d2_final,
                bp_d1_jumps=bp_d1_jumps, bp_d2_jumps=bp_d2_jumps
            )

    def get_continuous_fit(self, k: int = None, degree: int = 2) -> np.ndarray:
        if k is None: k = self.best_k
        model = self.models[k]
        internal_bps = model.segment_positions[:]

        valid_mask = ~np.isnan(self.y_fit)
        valid_x = self.x[valid_mask]
        valid_y = self.y_fit[valid_mask]

        try:
            valid_bps = [bp for bp in internal_bps if valid_x[0] < bp < valid_x[-1]]
            if len(valid_x) <= len(valid_bps) + degree:
                return model.fitted_y

            spline = LSQUnivariateSpline(valid_x, valid_y, t=valid_bps, k=degree)
            return spline(self.x)
        except ValueError:
            return model.fitted_y

    def get_smooth_derivatives(self, k: int = None, degree: int = 3) -> tuple:
        if k is None: k = self.best_k
        model = self.models[k]
        internal_bps = model.segment_positions[:]

        valid_mask = ~np.isnan(self.y_fit)
        valid_x = self.x[valid_mask]
        valid_y = self.y_fit[valid_mask]

        try:
            valid_bps = [bp for bp in internal_bps if valid_x[0] < bp < valid_x[-1]]
            if len(valid_x) <= len(valid_bps) + degree:
                return np.zeros_like(self.x), np.zeros_like(self.x)

            spline = LSQUnivariateSpline(valid_x, valid_y, t=valid_bps, k=degree)
            return spline.derivative(1)(self.x), spline.derivative(2)(self.x)
        except ValueError:
            return np.zeros_like(self.x), np.zeros_like(self.x)

    def select_best_k(self, method='gcv_min', threshold=5.0):
        self.selection_method = method
        k_values = sorted(self.models.keys())

        if method == 'bic_min': self.best_k = min(k_values, key=lambda k: self.models[k].bic)
        elif method == 'aic_min': self.best_k = min(k_values, key=lambda k: self.models[k].aic)
        elif method == 'hqic_min': self.best_k = min(k_values, key=lambda k: self.models[k].hqic)
        elif method == 'gcv_min': self.best_k = min(k_values, key=lambda k: self.models[k].gcv)
        elif method == 'bic_threshold':
            best = k_values[0]
            for i in range(1, len(k_values)):
                if self.models[k_values[i-1]].bic - self.models[k_values[i]].bic < threshold:
                    best = k_values[i-1]
                    break
                best = k_values[i]
            self.best_k = best
        return self.best_k

    def plot_model_selection(self, ax=None):
        k_values = list(self.models.keys())
        bp_values = [self.models[k].num_breakpoints for k in k_values]
        sses = [self.models[k].sse for k in k_values]
        gcvs = [self.models[k].gcv for k in k_values]
        bics = [self.models[k].bic for k in k_values]
        aics = [self.models[k].aic for k in k_values]
        hqics = [self.models[k].hqic for k in k_values]

        if ax is None:
            fig, ax1 = plt.subplots(figsize=(9, 5))
            show_plot = True
        else:
            ax1 = ax
            show_plot = False

        ax1.set_xlabel('Number of Breakpoints (BP)')
        ax1.set_ylabel('Error / GCV Score (Log Scale)', color='black')
        ax1.plot(bp_values, sses, marker='o', color='lightgray', linewidth=2, label='SSE', alpha=0.7)

        valid_idx = [i for i, g in enumerate(gcvs) if not np.isinf(g)]
        ax1.plot([bp_values[i] for i in valid_idx], [gcvs[i] for i in valid_idx],
                 marker='D', color='tab:green', linewidth=2, label='GCV')

        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='black')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Information Criteria (AIC/BIC/HQIC)', color='black')
        ax2.plot(bp_values, bics, marker='s', color='tab:red', linestyle='--', label='BIC', alpha=0.7)
        ax2.plot(bp_values, aics, marker='v', color='tab:orange', linestyle='-.', label='AIC', alpha=0.7)
        ax2.plot(bp_values, hqics, marker='*', color='tab:purple', linestyle=':', label='HQIC', alpha=0.7)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

        best_bp = self.models[self.best_k].num_breakpoints

        ax1.axvline(x=best_bp, color='red', linestyle=':', linewidth=2, alpha=0.7)
        min_gcv = min([gcvs[i] for i in valid_idx]) if valid_idx else 1.0
        ax1.text(best_bp, min_gcv, f' Selected [{self.selection_method}]\n (BP={best_bp})', color='red', va='bottom')

        ax1.set_title('Model Selection Metrics')
        ax1.grid(True, linestyle=':', alpha=0.6)

        if show_plot:
            fig.tight_layout()
            plt.show()

    def plot(self, smooth: bool = False, ax=None):
        k = self.best_k
        model = self.models[k]

        if ax is None:
            fig, ax_main = plt.subplots(figsize=(10, 5))
            show_plot = True
        else:
            ax_main = ax
            show_plot = False

        ax_main.plot(self.x, self.y_orig, 'o', markersize=3, color='lightgray', label='Original Data')

        if smooth:
            y_plot = self.get_continuous_fit(k)
            label = 'Fitted Segments (Continuous/Smooth)'
            color = 'green'
        else:
            y_plot = model.fitted_y
            label = 'Fitted Segments (Independent)'
            color = 'red'

        ax_main.plot(self.x, y_plot, '-', color=color, linewidth=2.5, label=label)

        y_min, y_max = ax_main.get_ylim()
        for bp_x in model.segment_positions:
            ax_main.axvline(x=bp_x, color='black', linestyle='--', alpha=0.5)
            ax_main.text(bp_x, y_min, f' BP: {bp_x:.2f} ', rotation=90, verticalalignment='bottom',
                         horizontalalignment='right', color='black', fontsize=9, alpha=0.7)

        suffix = " (Smooth)" if smooth else " (Independent)"
        ax_main.set_title(f"{self.__class__.__name__}{suffix} | BP: {model.num_breakpoints}")
        ax_main.set_xlabel('X')
        ax_main.set_ylabel('Y')
        ax_main.legend()
        ax_main.grid(True, linestyle=':', alpha=0.6)

        if show_plot:
            plt.tight_layout()
            plt.show()

    def plot_derivatives(self, axes=None):
        k = self.best_k
        model = self.models[k]

        if axes is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            show_plot = True
        else:
            ax1, ax2 = axes
            show_plot = False

        for i, seg in enumerate(model.segments_info):
            x_seg = np.linspace(seg.start_x, seg.end_x, 50)

            if hasattr(seg, 'slope'):
                d1_seg = np.full_like(x_seg, seg.slope)
                d2_seg = np.zeros_like(x_seg)
            elif hasattr(seg, 'c2'):
                x_shift = x_seg - seg.start_x
                d1_seg = 2 * seg.c2 * x_shift + seg.c1
                d2_seg = np.full_like(x_seg, seg.d2)
            else:
                continue

            label1 = '1st Derivative' if i == 0 else ""
            label2 = '2nd Derivative' if i == 0 else ""

            ax1.plot(x_seg, d1_seg, '-', color='tab:blue', linewidth=2, label=label1)
            ax2.plot(x_seg, d2_seg, '-', color='tab:purple', linewidth=2, label=label2)

        if model.bp_d1_jumps is not None:
            for bp_x, jump_d1, jump_d2 in zip(model.segment_positions, model.bp_d1_jumps, model.bp_d2_jumps):
                ax1.axvline(x=bp_x, color='gray', linestyle='--', alpha=0.3)
                ax2.axvline(x=bp_x, color='gray', linestyle='--', alpha=0.3)

                if abs(jump_d1) > 1e-10:
                    ax2.plot([bp_x, bp_x], [0, jump_d1], color='tab:red', linewidth=2)
                    label_jump = 'd1 Jump' if bp_x == model.segment_positions[0] else ""
                    ax2.plot(bp_x, jump_d1, 'ro', markersize=6, label=label_jump)

                if abs(jump_d2) > 1e-10:
                    label_jump_d2 = 'd2 Jump' if bp_x == model.segment_positions[0] else ""
                    ax2.plot(bp_x, jump_d2, 's', color='tab:orange', markersize=6, label=label_jump_d2)

        ax1.set_title(f'Derivatives Analysis | Model BP: {model.num_breakpoints}')
        ax1.set_ylabel('1st Derivative')

        ax2.set_yscale('symlog', linthresh=1e-1)
        ax2.set_ylabel('2nd Derivative\n(SymLog Scale)')
        ax2.set_xlabel('X')

        ax1.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax2.axhline(0, color='black', linewidth=1, alpha=0.5)

        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax2.grid(True, linestyle=':', alpha=0.6)

        if show_plot:
            plt.tight_layout()
            plt.show()

    def plot_smooth_derivatives(self, ax=None):
        k = self.best_k
        model = self.models[k]

        if ax is None:
            fig, ax1 = plt.subplots(figsize=(10, 4))
            show_plot = True
        else:
            ax1 = ax
            show_plot = False

        smooth_d1, smooth_d2 = self.get_smooth_derivatives()

        ax1.plot(self.x, smooth_d1, label="Smooth 1st Derivative (Spline)", color="blue", linewidth=2)
        ax1.set_ylabel('1st Derivative', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(self.x, smooth_d2, label="Smooth 2nd Derivative (Spline)", color="purple", linewidth=2)
        ax2.set_ylabel('2nd Derivative', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

        for bp in model.segment_positions:
            ax1.axvline(bp, color='gray', linestyle=':', alpha=0.5)

        ax1.axhline(0, color='black', linestyle='--', alpha=0.5)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        ax1.set_title(f"Spline-Smoothed Derivatives | Model BP: {model.num_breakpoints}")
        ax1.set_xlabel('X')
        ax1.grid(True, linestyle=':', alpha=0.6)

        if show_plot:
            plt.tight_layout()
            plt.show()

# ==========================================
# 3. Child Classes (Linear, Quadratic, Hybrid)
# ==========================================

class PiecewiseLinearRegression(BasePiecewiseRegression):
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 max_segments: int = None, apply_median_filter: bool = False,
                 select_k_method: str = 'bic_min', threshold: float = 6,
                 min_points_per_seg: int = None, spike_threshold_z: float = 5.0,
                 spike_dynamic_factor: float = 0.05, dw_threshold: float = 1.0):

        N = len(x)
        super().__init__(x, y, max_segments, apply_median_filter, spike_threshold_z, spike_dynamic_factor)

        self.min_points_per_seg = min_points_per_seg if min_points_per_seg is not None else max(3, N // 20)
        self.dw_threshold = dw_threshold

        self._run_fit()
        self.select_best_k(method=select_k_method, threshold=threshold)

    def _get_num_params(self, k: int) -> int:
        return 2 * k + (k - 1)

    def _calc_cost_matrix(self) -> np.ndarray:
        N = len(self.x)
        C = np.full((N, N + 1), np.inf)
        valid_mask = ~np.isnan(self.y_fit)

        x_safe = np.where(valid_mask, self.x, 0.0)
        y_safe = np.where(valid_mask, self.y_fit, 0.0)
        count_safe = np.where(valid_mask, 1.0, 0.0)

        sum_x = np.append(0, np.cumsum(x_safe))
        sum_y = np.append(0, np.cumsum(y_safe))
        sum_xx = np.append(0, np.cumsum(x_safe**2))
        sum_yy = np.append(0, np.cumsum(y_safe**2))
        sum_xy = np.append(0, np.cumsum(x_safe * y_safe))
        sum_n = np.append(0, np.cumsum(count_safe))

        for i in range(N):
            n = sum_n[i+1:] - sum_n[i]
            sx = sum_x[i+1:] - sum_x[i]
            sy = sum_y[i+1:] - sum_y[i]
            sxx = sum_xx[i+1:] - sum_xx[i]
            syy = sum_yy[i+1:] - sum_yy[i]
            sxy = sum_xy[i+1:] - sum_xy[i]

            valid_idx = n > 2

            S_xx = np.zeros_like(n)
            S_yy = np.zeros_like(n)
            S_xy = np.zeros_like(n)

            S_xx[valid_idx] = sxx[valid_idx] - (sx[valid_idx]**2) / n[valid_idx]
            S_yy[valid_idx] = syy[valid_idx] - (sy[valid_idx]**2) / n[valid_idx]
            S_xy[valid_idx] = sxy[valid_idx] - (sx[valid_idx] * sy[valid_idx]) / n[valid_idx]

            S_xx_safe = np.where(S_xx > 1e-10, S_xx, np.inf)
            sse_raw = np.full_like(n, np.inf)
            sse_raw[valid_idx] = S_yy[valid_idx] - (S_xy[valid_idx]**2) / S_xx_safe[valid_idx]

            valid_j_indices = np.arange(i+1, N+1)
            C[i, valid_j_indices] = np.maximum(0.0, sse_raw)

            a_arr = np.zeros_like(n)
            b_arr = np.zeros_like(n)
            a_arr[valid_idx] = S_xy[valid_idx] / S_xx_safe[valid_idx]
            b_arr[valid_idx] = (sy[valid_idx] - a_arr[valid_idx] * sx[valid_idx]) / n[valid_idx]

            for idx, j in enumerate(valid_j_indices):
                if n[idx] < self.min_points_per_seg:
                    C[i, j] = np.inf
                    continue

                if n[idx] >= 10:
                    xv = self.x[i:j][valid_mask[i:j]]
                    yv = self.y_fit[i:j][valid_mask[i:j]]
                    res = yv - (a_arr[idx] * xv + b_arr[idx])

                    sum_res_sq = np.sum(res**2)
                    if sum_res_sq > 1e-10:
                        dw = np.sum(np.diff(res)**2) / sum_res_sq
                        if dw < self.dw_threshold:
                            C[i, j] = np.inf
        return C

    def _fit_segment(self, xi, yi, start_x, end_x, seg_idx, start_idx=None, end_idx=None):
        valid = ~np.isnan(yi)
        if np.sum(valid) > 1:
            coeffs = np.polyfit(xi[valid], yi[valid], 1)
            slope = coeffs[0]
            fitted = np.polyval(coeffs, xi)
        else:
            slope, fitted = 0.0, np.zeros_like(xi)

        info = SegmentInfo(seg_idx, start_x, end_x, slope)
        return fitted, info, np.full_like(xi, slope), np.zeros_like(xi)


class PiecewiseQuadraticRegression(BasePiecewiseRegression):
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 max_segments: int = None, apply_median_filter: bool = False,
                 select_k_method: str = 'bic_min', threshold: float = 6,
                 max_points_per_seg: int = 500, min_points_per_seg: int = None,
                 spike_threshold_z: float = 5.0, spike_dynamic_factor: float = 0.05,
                 dw_threshold: float = 1.0):

        N = len(x)
        super().__init__(x, y, max_segments, apply_median_filter, spike_threshold_z, spike_dynamic_factor)

        self.max_points_per_seg = max_points_per_seg
        self.min_points_per_seg = min_points_per_seg if min_points_per_seg is not None else max(4, N // 20)
        self.dw_threshold = dw_threshold

        self._run_fit()
        self.select_best_k(method=select_k_method, threshold=threshold)

    def _get_num_params(self, k: int) -> int:
        return 3 * k + (k - 1)

    def _calc_cost_matrix(self) -> np.ndarray:
        N = len(self.x)
        C = np.full((N, N + 1), np.inf)
        valid_mask = ~np.isnan(self.y_fit)

        min_pts = max(4, self.min_points_per_seg)

        for i in range(N):
            j_max = min(N + 1, i + self.max_points_per_seg)
            for j in range(i + self.min_points_per_seg, j_max):
                mask_ij = valid_mask[i:j]
                n_valid = np.sum(mask_ij)

                if n_valid < min_pts: continue

                xi = self.x[i:j][mask_ij]
                yi = self.y_fit[i:j][mask_ij]
                xi_shift = xi - self.x[i]

                X_mat = np.vstack([xi_shift**2, xi_shift, np.ones(n_valid)]).T

                try:
                    XTX = X_mat.T @ X_mat
                    XTy = X_mat.T @ yi
                    coeffs = np.linalg.solve(XTX, XTy)
                    sse = np.sum((yi - (X_mat @ coeffs))**2)
                except np.linalg.LinAlgError:
                    coeffs, residuals, _, _ = np.linalg.lstsq(X_mat, yi, rcond=None)
                    sse = residuals[0] if len(residuals) > 0 else np.sum((yi - (X_mat @ coeffs))**2)

                if n_valid >= 12:
                    res_array = yi - (X_mat @ coeffs)
                    sum_res_sq = np.sum(res_array**2)
                    if sum_res_sq > 1e-10:
                        dw = np.sum(np.diff(res_array)**2) / sum_res_sq
                        if dw < self.dw_threshold: sse = np.inf

                C[i, j] = max(0.0, sse)
        return C

    def _fit_segment(self, xi, yi, start_x, end_x, seg_idx, start_idx=None, end_idx=None):
        valid = ~np.isnan(yi)
        n_valid = np.sum(valid)

        if n_valid >= self.min_points_per_seg:
            xi_shift_valid = xi[valid] - start_x
            X_mat = np.vstack([xi_shift_valid**2, xi_shift_valid, np.ones(n_valid)]).T
            coeffs, _, _, _ = np.linalg.lstsq(X_mat, yi[valid], rcond=None)
            c2, c1, c0 = coeffs[0], coeffs[1], coeffs[2]

            xi_shift_all = xi - start_x
            fitted = c2 * xi_shift_all**2 + c1 * xi_shift_all + c0

            info = QuadSegmentInfo(seg_idx, start_x, end_x, c2, c1, c0, 2 * c2)
            return fitted, info, 2 * c2 * xi_shift_all + c1, np.full_like(xi, 2 * c2)
        else:
            return np.zeros_like(xi), None, np.zeros_like(xi), np.zeros_like(xi)

class PiecewiseHybridRegression(BasePiecewiseRegression):
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 max_segments: int = None, apply_median_filter: bool = False,
                 select_k_method: str = 'bic_min', threshold: float = 6,
                 max_points_per_seg: int = 500, min_points_per_seg: int = None,
                 spike_threshold_z: float = 5.0, spike_dynamic_factor: float = 0.05,
                 dw_threshold: float = 1.0, hybrid_gcv_ratio: float = 1.2):

        N = len(x)
        super().__init__(x, y, max_segments, apply_median_filter, spike_threshold_z, spike_dynamic_factor)

        self.max_points_per_seg = max_points_per_seg
        self.min_points_per_seg = min_points_per_seg if min_points_per_seg is not None else max(4, N // 20)
        self.dw_threshold = dw_threshold
        self.hybrid_gcv_ratio = hybrid_gcv_ratio

        self.degree_matrix = np.zeros((N, N + 1), dtype=int)

        self._run_fit()
        self.select_best_k(method=select_k_method, threshold=threshold)

    def _calc_cost_matrix(self) -> np.ndarray:
        N = len(self.x)
        C = np.full((N, N + 1), np.inf)
        valid_mask = ~np.isnan(self.y_fit)

        min_pts = max(4, self.min_points_per_seg)

        for i in range(N):
            j_max = min(N + 1, i + self.max_points_per_seg)

            for j in range(i + self.min_points_per_seg, j_max):
                mask_ij = valid_mask[i:j]
                n_valid = np.sum(mask_ij)
                if n_valid < min_pts: continue

                xi, yi = self.x[i:j][mask_ij], self.y_fit[i:j][mask_ij]

                # === 1. Linear ===
                coeffs_lin = np.polyfit(xi, yi, 1)
                res_lin = yi - np.polyval(coeffs_lin, xi)
                sse_lin = np.sum(res_lin**2)

                if n_valid >= 10 and sse_lin > 1e-10:
                    if (np.sum(np.diff(res_lin)**2) / sse_lin) < self.dw_threshold: sse_lin = np.inf

                gcv_lin = (sse_lin / n_valid) / ((1.0 - 2 / n_valid) ** 2) if n_valid > 2 else np.inf

                # === 2. Quadratic ===
                xi_shift = xi - self.x[i]
                X_mat = np.vstack([xi_shift**2, xi_shift, np.ones(n_valid)]).T

                try:
                    XTX = X_mat.T @ X_mat
                    XTy = X_mat.T @ yi
                    coeffs_quad = np.linalg.solve(XTX, XTy)
                    sse_quad = np.sum((yi - (X_mat @ coeffs_quad))**2)
                except np.linalg.LinAlgError:
                    coeffs_quad, residuals, _, _ = np.linalg.lstsq(X_mat, yi, rcond=None)
                    sse_quad = residuals[0] if len(residuals) > 0 else np.sum((yi - (X_mat @ coeffs_quad))**2)

                if n_valid >= 12 and sse_quad > 1e-10:
                    res_quad = yi - (X_mat @ coeffs_quad)
                    if (np.sum(np.diff(res_quad)**2) / sse_quad) < self.dw_threshold: sse_quad = np.inf

                gcv_quad = (sse_quad / n_valid) / ((1.0 - 3 / n_valid) ** 2) if n_valid > 3 else np.inf

                # === 3. Competition ===
                if gcv_lin <= gcv_quad * self.hybrid_gcv_ratio:
                    C[i, j], self.degree_matrix[i, j] = sse_lin, 1
                else:
                    C[i, j], self.degree_matrix[i, j] = sse_quad, 2
        return C

    def _get_num_params(self, k: int) -> int:
        return 0 # Dummy implementation, logic is handled by _get_model_params below

    def _get_model_params(self, k: int, indices: List[int]) -> int:
        """Override to compute exact parameters depending on actual degrees of segments."""
        actual_num_params = 0
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i+1]
            deg = self.degree_matrix[start, end]
            if deg == 0: deg = 1
            actual_num_params += (deg + 1)
        return actual_num_params + (k - 1)

    def _fit_segment(self, xi, yi, start_x, end_x, seg_idx, start_idx=None, end_idx=None):
        deg = 1
        if start_idx is not None and end_idx is not None:
            deg = self.degree_matrix[start_idx, end_idx]
            if deg == 0: deg = 1

        valid = ~np.isnan(yi)
        n_valid = np.sum(valid)

        if deg == 1:
            if n_valid > 1:
                coeffs = np.polyfit(xi[valid], yi[valid], 1)
                slope = coeffs[0]
                fitted = np.polyval(coeffs, xi)
            else:
                slope, fitted = 0.0, np.zeros_like(xi)

            info = SegmentInfo(seg_idx, start_x, end_x, slope)
            return fitted, info, np.full_like(xi, slope), np.zeros_like(xi)

        else: # deg == 2
            if n_valid > 2:
                xi_shift_valid = xi[valid] - start_x
                X_mat = np.vstack([xi_shift_valid**2, xi_shift_valid, np.ones(n_valid)]).T
                coeffs, _, _, _ = np.linalg.lstsq(X_mat, yi[valid], rcond=None)
                c2, c1, c0 = coeffs[0], coeffs[1], coeffs[2]

                xi_shift_all = xi - start_x
                fitted = c2 * xi_shift_all**2 + c1 * xi_shift_all + c0
                d1_y = 2 * c2 * xi_shift_all + c1
                d2_y = np.full_like(xi, 2 * c2)
            else:
                fitted, c2, c1, c0 = np.zeros_like(xi), 0.0, 0.0, 0.0
                d1_y, d2_y = np.zeros_like(xi), np.zeros_like(xi)

            info = QuadSegmentInfo(seg_idx, start_x, end_x, c2, c1, c0, 2 * c2)
            return fitted, info, d1_y, d2_y

# ==========================================
# 4. Utilities
# ==========================================
def plot_comprehensive_results(model: BasePiecewiseRegression, dataset_name: str = ''):
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [1, 1.5, 1, 1]})

    model.plot_model_selection(ax=axs[0])
    model.plot(ax=axs[1])
    model.plot_derivatives(axes=(axs[2], axs[3]))

    fig.suptitle(f"[{model.__class__.__name__}] Applied to: {dataset_name}",
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_comprehensive_results5(model: BasePiecewiseRegression, dataset_name: str = ''):
    fig, axs = plt.subplots(5, 1, figsize=(12, 20), gridspec_kw={'height_ratios': [1, 1.5, 1, 1, 1.2]})

    model.plot_model_selection(ax=axs[0])
    model.plot(ax=axs[1])
    model.plot_derivatives(axes=(axs[2], axs[3]))
    model.plot_smooth_derivatives(ax=axs[4])

    fig.suptitle(f"[{model.__class__.__name__}] Applied to: {dataset_name}",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
