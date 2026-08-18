import os
import datetime
import warnings
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats, optimize
from scipy.interpolate import interp1d
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ==========================================================================
# --- 1. Common math/statistics and noise extraction modules ---
# ==========================================================================

def exrelu(x: np.ndarray, th: float, slope: float, bg: float) -> np.ndarray:
    """Extended ReLU function."""
    # return np.where(x < th, bg, slope * (x - th) + bg)
    return np.maximum(bg, slope * (x - th) + bg)

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))

def calculate_nmae(y_true: np.ndarray, mae: float) -> float:
    """Calculate normalized MAE (NMAE)."""
    y_mean = float(np.mean(y_true))
    return float(mae / y_mean) if y_mean > 1e-9 else 0.0

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate root mean squared error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calculate_r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R2 based on RMSE (squared error)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - float(ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

def calculate_r2_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R2 based on MAE (absolute error)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    med = float(np.median(y_true))
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - med))
    return 1.0 - float(num / den) if den > 1e-9 else 0.0

def calculate_rmr(rmse: float, mae: float) -> float:
    """Calculate RMR (RMSE / MAE)."""
    return float((rmse / mae) / np.sqrt(np.pi / 2)) if mae > 1e-9 else np.inf

def calculate_rsi(e_std: np.ndarray) -> float:
    """Calculate Pearson's second skewness coefficient (RSI: asymmetry)."""
    std_res = np.std(e_std) + 1e-9
    return float(3 * (np.mean(e_std) - np.median(e_std)) / std_res)

def calculate_durbin_watson(residuals: np.ndarray) -> float:
    """Calculate the Durbin-Watson statistic."""
    numerator = np.sum(np.diff(residuals) ** 2)
    denominator = np.sum(residuals ** 2)
    return float(numerator / denominator) if denominator != 0 else 0.0

def calculate_aic_mae(n_data: int, mae: float, n_params: int) -> float:
    """Calculate MAE-based AIC assuming a Laplace distribution."""
    if mae <= 1e-9: return np.inf
    return 2 * n_data * np.log(mae) + 2 * n_data * (1 + np.log(2)) + 2 * n_params

def calculate_akaike_weights(aics: np.ndarray) -> np.ndarray:
    """Calculate Akaike weights (probabilities) from AIC values."""
    min_aic = np.nanmin(aics)
    delta_aic = aics - min_aic
    delta_aic = np.clip(delta_aic, 0, 100) 
    likelihoods = np.exp(-0.5 * delta_aic)
    likelihoods[np.isnan(likelihoods)] = 0
    total_likelihood = np.nansum(likelihoods)
    if total_likelihood == 0:
        return np.zeros_like(aics)
    return likelihoods / total_likelihood

def calculate_akaike_metrics(ns: np.ndarray, aics: np.ndarray) -> Tuple[float, float, float]:
    """Compute expected n, variance, and Shannon entropy from 1D AIC scans."""
    weights = calculate_akaike_weights(aics)
    if np.sum(weights) == 0:
        return np.nan, np.nan, np.nan
    n_expected = np.sum(weights * ns)
    n_variance = np.sum(weights * (ns - n_expected)**2)
    safe_weights = weights[weights > 0]
    shannon_entropy = -np.sum(safe_weights * np.log2(safe_weights))
    return float(n_expected), float(n_variance), float(shannon_entropy)

def calculate_akaike_metrics_2d(grid_n: np.ndarray, grid_aic: np.ndarray) -> Tuple[float, float, float]:
    """Compute expected n, variance, and Shannon entropy from 2D AIC grids."""
    weights = calculate_akaike_weights(grid_aic)
    valid_mask = ~np.isnan(weights) & ~np.isnan(grid_n)
    if not np.any(valid_mask):
        return np.nan, np.nan, np.nan
    w_valid = weights[valid_mask]
    n_valid = grid_n[valid_mask]
    w_sum = np.sum(w_valid)
    if w_sum <= 0: return np.nan, np.nan, np.nan
    w_norm = w_valid / w_sum
    n_expected = np.sum(w_norm * n_valid)
    n_variance = np.sum(w_norm * (n_valid - n_expected)**2)
    safe_weights = w_norm[w_norm > 0]
    shannon_entropy = -np.sum(safe_weights * np.log2(safe_weights))
    return float(n_expected), float(n_variance), float(shannon_entropy)

def _save_figure(fig: plt.Figure, save_folder: Optional[str], base_filename: str):
    if save_folder is None: return
    try:
        os.makedirs(save_folder, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_filename}_{timestamp}.png"
        filepath = os.path.join(save_folder, filename)
        fig.savefig(filepath, dpi=600, bbox_inches='tight')
    except Exception as e:
        warnings.warn(f"Failed to save plot: {e}", UserWarning)

@dataclass
class FitResult:
    n: float
    aic: float
    th: float
    bg: float
    a: Optional[float] = None
    slope: Optional[float] = None
    adj_r2: Optional[float] = None
    r2_rmse: Optional[float] = None
    r2_mae: Optional[float] = None
    d_r2: Optional[float] = None     
    
    nmae: Optional[float] = None
  
    rmr: Optional[float] = None
    dw: Optional[float] = None
    rsi: Optional[float] = None
    is_linear: Optional[bool] = None

    # --- Akaike weights analysis ---
    expected_n: Optional[float] = None
    variance_n: Optional[float] = None
    H: Optional[float] = None        
    P: Optional[float] = None        
    d_n: Optional[float] = None      

# ==========================================================================
# --- 2. Analysis classes (Fitters) ---
# ==========================================================================

class BaseYieldFitter(ABC):
    """Abstract base class for PYS spectrum analysis."""
    def __init__(self, x: np.ndarray, y: np.ndarray, spike_level: str = 'low'):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.n_data = len(self.x)
        self.spike_level = spike_level

    @abstractmethod
    def oneshot_fit(self, *args, **kwargs) -> FitResult:
        pass

    @abstractmethod
    def fit(self, *args, **kwargs) -> Tuple[FitResult, Any]:
        pass

class OneOverNScanFitter(BaseYieldFitter):
    """1/n-scan method fitter."""
    def __init__(self, x: np.ndarray, y: np.ndarray, spike_level: str = 'low'):
        super().__init__(x, y, spike_level)
    
    def _initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        try:
            dy_dx = np.gradient(y, x)
            thr0 = float(x[np.nanargmax(np.abs(dy_dx))])
        except Exception:
            thr0 = 0.5 * (x_min + x_max)
        mask_above = x > thr0
        if np.sum(mask_above) > 2:
            try: slope0 = float(np.polyfit(x[mask_above], y[mask_above], 1)[0])
            except Exception: slope0 = (y_max - y_min) / max((x_max - x_min), 1e-8)
        else:
            slope0 = (y_max - y_min) / max((x_max - x_min), 1e-8)
        mask_below = x < thr0
        if np.sum(mask_below) > 0: bg0 = float(np.median(y[mask_below]))
        else: bg0 = max(0.0, y_min)
        return [thr0, max(0.0, slope0), max(0.0, bg0)]

    def oneshot_fit(self, n: float, 
                    th0: Optional[float] = None, 
                    slope0: Optional[float] = None, 
                    bg0: Optional[float] = None) -> FitResult:
        
        # Perform fitting in 1/n space for numerical stability
        y_tr = np.power(np.maximum(self.y, 1e-9), 1.0 / n)
        
        def objective(params):
            th, slope, bg = params
            return np.mean(np.abs(y_tr - exrelu(self.x, th, slope, bg)))

        auto_guess = self._initial_guess(self.x, y_tr)
        final_th0 = th0 if th0 is not None else auto_guess[0]
        final_slope0 = slope0 if slope0 is not None else auto_guess[1]

   
        if bg0 is not None:
            final_bg0 = np.power(max(bg0, 1e-9), 1.0 / n)
        else:
            final_bg0 = auto_guess[2]

        initial_guess = [final_th0, final_slope0, final_bg0]
        upper_slope = initial_guess[1] * 6 if initial_guess[1] > 0 else 1e6
        bounds = [(float(self.x.min()), float(self.x.max())), (1e-6, upper_slope), (1e-6, 2 + initial_guess[2] * 6)]

        opt_result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        if opt_result.success: best_params = opt_result.x
        else:
            de_result = optimize.differential_evolution(objective, bounds=bounds, maxiter=500, popsize=10, tol=0.01)
            best_params = de_result.x

        th_opt, slope_opt, bg_opt = best_params
        
    
        y_pred_tr = exrelu(self.x, th_opt, slope_opt, bg_opt)
        y_fit = np.power(y_pred_tr, n)
        
    
        e_raw = self.y - y_fit

        mae_raw = calculate_mae(self.y, y_fit)
        rmse_raw = calculate_rmse(self.y, y_fit)
        nmae = calculate_nmae(self.y, mae_raw)
        aic = calculate_aic_mae(self.n_data, mae_raw, 3) 
        rmr = calculate_rmr(rmse_raw, mae_raw)
        dw = calculate_durbin_watson(e_raw)
        rsi = calculate_rsi(e_raw)

        r2_rmse = calculate_r2_rmse(self.y, y_fit)
        r2_mae = calculate_r2_mae(self.y, y_fit)
        d_r2 = abs(r2_rmse - r2_mae)

        return FitResult(n=n, aic=aic, th=th_opt, bg=bg_opt, slope=slope_opt,
                         adj_r2=r2_rmse, r2_rmse=r2_rmse, r2_mae=r2_mae, d_r2=d_r2,
                         nmae=nmae, rmr=rmr, dw=dw, rsi=rsi)

    def fit(self, n_min: float = 0.5, 
            n_max: float = 4.0, 
            n_step: float = 0.01,
            th0: Optional[float] = None, 
            slope0: Optional[float] = None,
              bg0: Optional[float] = None) -> Tuple[FitResult, List[FitResult]]:
        
        n_search = np.arange(n_min, n_max + n_step/2, n_step)
        all_results = [self.oneshot_fit(n, th0=th0, slope0=slope0, bg0=bg0) for n in n_search]
        best_res = min(all_results, key=lambda r: r.aic)
        ns_array = np.array([r.n for r in all_results])
        aics_array = np.array([r.aic for r in all_results])
        n_exp, n_var, entropy = calculate_akaike_metrics(ns_array, aics_array)

        best_res.expected_n = n_exp
        best_res.variance_n = n_var
        best_res.H = entropy
        best_res.P = 2 ** entropy if entropy is not None and not np.isnan(entropy) else np.nan
        best_res.d_n = n_step
        
        for r in all_results: r.d_n = n_step
        return best_res, all_results

    def _add_stats_textbox(self, ax, best_res: FitResult):
        if best_res.H is not None and not np.isnan(best_res.H):
            p_val = best_res.P * best_res.d_n if best_res.P is not None and best_res.d_n is not None else np.nan
            stats_text = (f"$n$: {best_res.expected_n:.2f}\n"
                          f"$\\sigma^2$: {best_res.variance_n:.3f}\n"
                          f"$H$: {best_res.H:.3f}\n"
                          f"$P \\times \\Delta n$: {p_val:.3f}")
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                     fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    def plot_result_ext2(self, best_res: FitResult, all_results: List[FitResult],
                            title_suffix: str = "", save_folder: Optional[str] = None, show_plot: bool = True):
        """Plot the 11-panel diagnostic figure (Fig. 2-style layout) and return it.

        Note: with show_plot=False, the returned figure is closed
        (plt.close(fig)) before being returned, so it is only safe to
        export via fig.savefig(...); calling fig.show() on it will raise
        "Figure.show works only for figures managed by pyplot".
        """
        ns = np.array([r.n for r in all_results])
        aics = np.array([r.aic for r in all_results])
        ths = np.array([r.th for r in all_results])
        r2_rmses = np.array([r.r2_rmse for r in all_results])
        r2_maes = np.array([r.r2_mae for r in all_results])
        nmaes = np.array([r.nmae for r in all_results]) * 100
        rmrs = np.array([r.rmr for r in all_results])
        dws = np.array([r.dw for r in all_results])
        weights = calculate_akaike_weights(aics)
        
        fig = plt.figure(figsize=(15, 17))
        gs = gridspec.GridSpec(14, 2, width_ratios=[1, 1.2], hspace=1.2) 

        # --- Left column (diagnostic metrics behavior) ---
        ax1 = fig.add_subplot(gs[0:2, 0])      
        ax2 = fig.add_subplot(gs[2:4, 0], sharex=ax1)  
        ax3 = fig.add_subplot(gs[4:6, 0], sharex=ax1)  
        ax4 = fig.add_subplot(gs[6:8, 0], sharex=ax1)  
        ax5 = fig.add_subplot(gs[8:10, 0], sharex=ax1) 
        ax6 = fig.add_subplot(gs[10:12, 0], sharex=ax1)
        ax7 = fig.add_subplot(gs[12:14, 0], sharex=ax1)
        
        # --- Right column (fitting and residuals) ---
        ax_fit1 = fig.add_subplot(gs[0:4, 1])                 # 1/n Fit
        ax_res1 = fig.add_subplot(gs[4:6, 1], sharex=ax_fit1) # 1/n Resid
        ax_fit2 = fig.add_subplot(gs[6:11, 1])                # Orig Fit
        ax_res2 = fig.add_subplot(gs[11:14, 1], sharex=ax_fit2) # Orig Resid

        # (a) AIC
        ax1.plot(ns, aics, color='tab:blue', linestyle='-')
        ax1.plot(best_res.n, best_res.aic, 'ro', label=f'n: {best_res.n:.2f}')
        ax1.set_ylabel('AIC')
        ax1.set_title('(a) 1/n-Scan')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.bar(ns, weights, width=(ns[1]-ns[0])*0.8, color='green', alpha=0.6, label='Weights')
        ax2.set_ylabel('Probability')
        ax2.set_title('(b) Akaike Weights')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        self._add_stats_textbox(ax2, best_res) 
        
        ax3.plot(ns, ths, color='darkorange')
        ax3.plot(best_res.n, best_res.th, 'ro', label=f'$I_{{th}}$:{best_res.th:.2f}')
        ax3.set_ylabel('$I_{th}$ [eV]')
        ax3.set_title('(c) $I_{th}$')
        ax3.grid(True, alpha=0.3); ax3.legend()
        
        ax4.plot(ns, r2_rmses, color='tab:blue', linestyle='-', label=f'$R^2_{{RMSE}}$:{best_res.r2_rmse:.2f}')
        ax4.plot(ns, r2_maes, color='tab:green', linestyle='--', label=f'$R^2_{{MAE}}$:{best_res.r2_mae:.2f}')
        ax4.plot(best_res.n, best_res.r2_rmse, 'ro', 
                    label=f'$\\Delta R^2$:{best_res.d_r2:.4f}')
        ax4.set_ylabel('$R^2$')
        ax4.set_title('(d) $R^2$')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='lower right')
        
        ax5.plot(ns, nmaes, color='darkviolet', linestyle='-', linewidth=2, )
        ax5.plot(best_res.n, best_res.nmae * 100, 'ro', alpha=0.8, label=f'NMAE:{best_res.nmae*100:.2f}%')
        ax5.set_ylabel('NMAE [%]')
        ax5.set_title('(e) NMAE (Normalized MAE)')
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='upper right')

        # (f) RMR 
        ax6.plot(ns, rmrs, color='tab:olive', linestyle='-')
        ax6.plot(best_res.n, best_res.rmr, 'ro', label=f'{best_res.rmr:.3f}')
        ax6.axhline(y=1, color='tab:blue', linestyle='--', linewidth=1, alpha=0.7)
        ax6.axhline(y=1.35, color='tab:red', linestyle='--', linewidth=1, alpha=0.7)
        ax6.set_ylabel('RMR')
        ax6.set_title('(f) RMR (RMSE/MAE)')
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='upper right')
        
        # (g) DW 
        ax7.plot(ns, dws, color='tab:orange', )
        ax7.plot(best_res.n, best_res.dw, 'ro', label=f'DW:{best_res.dw:.2f}' )
        ax7.axhline(y=2, color='tab:blue', linestyle='--', linewidth=1, alpha=0.7)
        ax7.set_ylabel('DW')
        ax7.set_xlabel('n')
        ax7.set_title('(g) DW (Durbin-Watson)')
        ax7.grid(True, alpha=0.3); ax7.legend()

        # --- Right column plotting operations ---
        n_opt = best_res.n
        y_1n = np.power(np.maximum(self.y, 1e-9), 1.0/n_opt)
        y_1n_pred = exrelu(self.x, best_res.th, best_res.slope, best_res.bg)
        y_pred_orig = np.power(y_1n_pred, n_opt)
        
        residuals_1n = y_1n - y_1n_pred
        residuals_orig = self.y - y_pred_orig
        
        # (h) 1/n Space Best Fit
        ax_fit1.scatter(self.x, y_1n, c='gray', alpha=0.5, label='Data (1/n space)')
        fit_label = f'Fit (n={n_opt:.2f}, $I_{{th}}$={best_res.th:.2f},\nSlope={best_res.slope:.1f}, Bg={best_res.bg:.2f})'
        ax_fit1.plot(self.x, y_1n_pred, 'b-', lw=2, label=fit_label)
        ax_fit1.set_ylabel(f'Yield$^{{(1/{n_opt:.2f})}}$')
        ax_fit1.set_title('(h) 1/n Space Best Fit')
        ax_fit1.legend(title=title_suffix) 
        ax_fit1.grid(True, alpha=0.3)
        plt.setp(ax_fit1.get_xticklabels(), visible=False)
        
        # (i) 1/n Space Residuals
        ax_res1.axhline(0, color='black', lw=1, ls='--')
        ax_res1.scatter(self.x, residuals_1n,  c='red', alpha=0.6)
        ax_res1.plot(self.x, residuals_1n, 'r-', alpha=0.3)
        ax_res1.set_ylabel('Residuals')
        ax_res1.set_title('(i) Residuals (1/n Space)')
        ax_res1.grid(True, alpha=0.3)
        plt.setp(ax_res1.get_xticklabels(), visible=False)
        
        # (j) Original Scale Best Fit
        ax_fit2.scatter(self.x, self.y,  c='gray', alpha=0.5, label='Data')
        ax_fit2.plot(self.x, y_pred_orig, 'b-', lw=2, label=fit_label)
        ax_fit2.set_title('(j) Original Scale Best Fit')
        ax_fit2.set_ylabel('Yield')
        ax_fit2.legend()
        ax_fit2.grid(True, alpha=0.3)
        plt.setp(ax_fit2.get_xticklabels(), visible=False)
        
        # (k) Original Scale Residuals 
        ax_res2.axhline(0, color='black', lw=1, ls='--')
        ax_res2.plot(self.x, residuals_orig, color='gray', alpha=0.5)
        ax_res2.scatter(self.x, residuals_orig, c='red', alpha=0.6)
        ax_res2.set_ylabel('Residuals')
        ax_res2.set_title('(k) Residuals (Original Scale)')
        ax_res2.set_xlabel('Energy [eV]')
        # ax_res2.legend(loc='upper left')
        ax_res2.grid(True, alpha=0.3)

        plt.tight_layout()
        _save_figure(fig, save_folder, "exrelu_fit_ext2")
        if show_plot: plt.show()
        else: plt.close(fig)
        return fig


class ThreeLSFitter(BaseYieldFitter):
    """3LS method fitter (Standard / Weighted / MAE selectable)."""
    def __init__(self, x: np.ndarray, y: np.ndarray, mode: str = 'standard', spike_level: str = 'low'):
        super().__init__(x, y, spike_level)
        self.mode = mode.lower()

    def _estimate_search_grid(self, th_grid_num: int, bg_grid_num: int)-> Tuple[np.ndarray, np.ndarray]:
        th_vals = np.linspace(self.x.min(), np.percentile(self.x, 75), th_grid_num)
        bg_region_mask = self.x < np.percentile(self.x, 30)
        y_bg_guess = self.y[bg_region_mask]
        
        if len(y_bg_guess) > 0:
            bg_median = np.median(y_bg_guess)
            bg_mad = np.median(np.abs(y_bg_guess - bg_median))
            bg_std_est = bg_mad * 1.4826 
            bg_min_limit = max(0.0, bg_median - 3 * bg_std_est) 
            bg_max_limit = bg_median + 3 * bg_std_est
            if bg_max_limit == bg_min_limit:
                bg_max_limit = bg_min_limit + 1e-5
        else:
            bg_min_limit = 0.0
            bg_max_limit = 1.0

        bg_vals = np.linspace(bg_min_limit, bg_max_limit, bg_grid_num)
        return th_vals, bg_vals

    def oneshot_fit(self, th: float, bg: float, min_data_points: int = 8) -> Optional[FitResult]:
        mask = (self.x > th) & (self.y > bg)
        if np.sum(mask) < min_data_points:
            return None

        # Perform fitting in log-log space for numerical stability
        x_valid, y_valid = self.x[mask], self.y[mask]
        log_x, log_y = np.log(x_valid - th), np.log(y_valid - bg)
        weights = (y_valid - bg)**2 if self.mode == 'weighted' else None
        
        try:
            coeffs, _ = np.polyfit(log_x, log_y, 1, w=weights, cov=True)
            n_est, log_a = coeffs[0], coeffs[1]
        except np.linalg.LinAlgError:
            return None
            
        if n_est <= 0: return None

        # Bring predictions back to the original scale
        a_est = np.exp(log_a)
        if self.mode == 'mae':
            def mae_objective(params):
                n_trial, log_a_trial = params
                if n_trial <= 0:
                    return 1e12
                a_trial = np.exp(log_a_trial)
                y_trial = a_trial * np.power(np.maximum(0, x_valid - th), n_trial) + bg
                return np.mean(np.abs(y_valid - y_trial))

            initial_params = np.array([n_est, log_a], dtype=float)
            log_a_upper = np.log(np.maximum(np.max(y_valid), 1e-9) * 10.0 + 1e-9)
            try:
                res_opt = optimize.minimize(
                    mae_objective,
                    initial_params,
                    bounds=[(1e-6, 20.0), (np.log(1e-9), log_a_upper)],
                    method='L-BFGS-B'
                )
                if res_opt.success:
                    n_est = float(res_opt.x[0])
                    a_est = float(np.exp(res_opt.x[1]))
            except Exception:
                pass

        y_pred_orig = a_est * np.power(np.maximum(0, self.x - th), n_est) + bg
        
        e_raw = self.y - y_pred_orig
       
        mae_raw = calculate_mae(self.y, y_pred_orig)
        nmae = calculate_nmae(self.y, mae_raw)
        rmse_raw = calculate_rmse(self.y, y_pred_orig)
        aic = calculate_aic_mae(self.n_data, mae_raw, 4)
        rmr = calculate_rmr(rmse_raw, mae_raw)
        dw_stat = calculate_durbin_watson(e_raw)
        rsi = calculate_rsi(e_raw)

        is_linear = 1.5 < dw_stat < 2.5
        
        r2_rmse = calculate_r2_rmse(self.y, y_pred_orig)
        r2_mae = calculate_r2_mae(self.y, y_pred_orig)
        adj_r2_mae = 1 - (1 - r2_mae) * (self.n_data - 1) / (self.n_data - 4 - 1)
        primary_r2 = adj_r2_mae if self.mode == 'mae' else r2_rmse

        return FitResult(n=n_est, aic=aic, th=th, bg=bg, a=a_est,
                         adj_r2=primary_r2, r2_rmse=r2_rmse, r2_mae=r2_mae, d_r2=abs(r2_rmse - r2_mae),
                         dw=dw_stat, is_linear=is_linear, rmr=rmr, nmae=nmae, rsi=rsi)

    def fit(self, th_grid_num: int = 30, bg_grid_num: int = 30, 
            min_data_points: int = 8, r2_threshold_ratio: float = 0.977) -> Tuple[FitResult, Dict]:
        th_vals, bg_vals = self._estimate_search_grid(th_grid_num, bg_grid_num)
        
        grid_r2 = np.full((bg_grid_num, th_grid_num), np.nan)
        grid_aic = np.full((bg_grid_num, th_grid_num), np.nan)
        grid_n = np.full((bg_grid_num, th_grid_num), np.nan)
        grid_rmr = np.full((bg_grid_num, th_grid_num), np.nan)
        grid_dw = np.full((bg_grid_num, th_grid_num), np.nan)
        grid_r2_mae = np.full((bg_grid_num, th_grid_num), np.nan)
        all_solutions = []

        for i, bg in enumerate(bg_vals):
            for j, th in enumerate(th_vals):
                res = self.oneshot_fit(th, bg, min_data_points)
                if res:
                    grid_r2[i, j] = res.adj_r2
                    grid_aic[i, j] = res.aic
                    grid_n[i, j] = res.n
                    grid_rmr[i, j] = res.rmr
                    grid_dw[i, j] = res.dw
                    grid_r2_mae[i, j] = res.r2_mae
                    all_solutions.append(res)

        if not all_solutions:
            best = FitResult(n=np.nan, aic=np.inf, th=np.nan, bg=np.nan, adj_r2=-np.inf)
        else:
            max_r2 = max(s.adj_r2 for s in all_solutions)
            if self.mode == 'mae':
                candidate_solutions = all_solutions
            else:
                r2_threshold_val = max_r2 * r2_threshold_ratio if max_r2 > 0 else -1.0
                candidate_solutions = [s for s in all_solutions if s.adj_r2 >= r2_threshold_val]
                if not candidate_solutions: candidate_solutions = all_solutions
                
            best = min(candidate_solutions, key=lambda x: x.aic)

        n_exp, n_var, entropy = calculate_akaike_metrics_2d(grid_n, grid_aic)
        best.expected_n = n_exp
        best.variance_n = n_var
        best.H = entropy

        grid_data = {
            'th_vals': th_vals, 'bg_vals': bg_vals, 'grid_r2': grid_r2, 
            'grid_aic': grid_aic, 'grid_n': grid_n, 'grid_rmr': grid_rmr,
            'grid_dw': grid_dw, 'grid_r2_mae': grid_r2_mae
        }
        return best, grid_data

    def _add_stats_textbox(self, ax, best_res: FitResult):
        if best_res.H is not None and not np.isnan(best_res.H):
            stats_text = (f"Exp. $n$: {best_res.expected_n:.2f}\n"
                          f"Var $\\sigma^2$: {best_res.variance_n:.3f}\n"
                          f"Entropy $H$: {best_res.H:.3f}")
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                     fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    def plot_result2(self, best_res: FitResult, grid_data: Dict, title_suffix: str = "",
                    save_folder: Optional[str] = None, show_plot: bool = True):
        """Plot the 3LS diagnostic figure and return it.

        Note: with show_plot=False, the returned figure is closed
        (plt.close(fig)) before being returned, so it is only safe to
        export via fig.savefig(...); calling fig.show() on it will raise
        "Figure.show works only for figures managed by pyplot".
        """
        th_vals, bg_vals = grid_data['th_vals'], grid_data['bg_vals']
        grid_aic, grid_n = grid_data['grid_aic'], grid_data['grid_n']
        grid_rmr, grid_dw, grid_r2_mae = grid_data['grid_rmr'], grid_data['grid_dw'], grid_data['grid_r2_mae']
        grid_r2 = grid_data['grid_r2']

        grid_weights = calculate_akaike_weights(grid_aic)
        if np.all(np.isnan(grid_aic)):
            print("Warning: All AIC values are NaN. Plotting skipped.")
            return None

        min_idx = np.unravel_index(np.nanargmin(grid_aic), grid_aic.shape)
        max_r2_idx = np.unravel_index(np.nanargmax(grid_r2), grid_r2.shape)

        fig = plt.figure(figsize=(18, 15))
        gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.25)

        gs_log = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 0], hspace=0.1)
        ax1_fit = fig.add_subplot(gs_log[0:3, 0])
        ax1_res = fig.add_subplot(gs_log[3, 0], sharex=ax1_fit)
        
        gs_orig = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1], hspace=0.1)
        ax2_fit = fig.add_subplot(gs_orig[0:3, 0])
        ax2_res = fig.add_subplot(gs_orig[3, 0], sharex=ax2_fit)
        
        ax3 = fig.add_subplot(gs[0, 2]) 
        ax4 = fig.add_subplot(gs[1, 0]) 
        ax5 = fig.add_subplot(gs[1, 1]) 
        ax6 = fig.add_subplot(gs[1, 2]) 
        ax7 = fig.add_subplot(gs[2, 0]) 
        ax8 = fig.add_subplot(gs[2, 1]) 
        ax9 = fig.add_subplot(gs[2, 2]) 

        mask = (self.x > best_res.th) & (self.y > best_res.bg)
        if np.sum(mask) > 0 and best_res.a is not None:
            log_x = np.log(self.x[mask] - best_res.th)
            log_y = np.log(self.y[mask] - best_res.bg)
            y_pred_log = best_res.n * log_x + np.log(best_res.a)
            residuals_log = log_y - y_pred_log
            
            ax1_fit.scatter(log_x, log_y, s=20, c='blue', alpha=0.6, label='Data')
            ax1_fit.plot(log_x, y_pred_log, 'r-', lw=2, label=f'Fit n={best_res.n:.2f}')
            ax1_fit.set_title(f'(a) 3LS Log-Log Fit & Residuals')
            ax1_fit.set_xlabel(f"log(Energy - {best_res.th:.3f})")
            ax1_fit.set_ylabel(f"log(Yield - {best_res.bg:.3f})")
            ax1_fit.legend()
            ax1_fit.grid(True, alpha=0.3)
            plt.setp(ax1_fit.get_xticklabels(), visible=False)
            
            ax1_res.axhline(0, color='black', lw=1, ls='--')
            ax1_res.scatter(log_x, residuals_log, s=10, c='red', alpha=0.6)
            ax1_res.plot(log_x, residuals_log, 'r-', alpha=0.3)
            ax1_res.set_xlabel(f"log(Energy - {best_res.th:.3f})")
            ax1_res.set_ylabel('Resid')
            ax1_res.grid(True, alpha=0.3)

        if best_res.a is not None:
            y_pred = best_res.a * np.power(np.maximum(0, self.x - best_res.th), best_res.n) + best_res.bg
            residuals = self.y - y_pred
            
            ax2_fit.scatter(self.x, self.y, s=15, c='gray', alpha=0.5, label='Data')
            fit_label = f'Fit (n={best_res.n:.2f}, $I_{{th}}$={best_res.th:.2f}, $Y_{{bg}}$={best_res.bg:.2f})'
            ax2_fit.plot(self.x, y_pred, 'b-', lw=2, label=fit_label)
            ax2_fit.set_title('(b) Original Scale Fit & Residuals')
            ax2_fit.set_ylabel('Yield')
            ax2_fit.legend(); ax2_fit.grid(True, alpha=0.3)
            plt.setp(ax2_fit.get_xticklabels(), visible=False) 
            
            ax2_res.axhline(0, color='black', lw=1, ls='--')
            ax2_res.scatter(self.x, residuals, s=10, c='red', alpha=0.6)
            ax2_res.plot(self.x, residuals, 'r-', alpha=0.3)
            ax2_res.set_xlabel('Energy [eV]'); ax2_res.set_ylabel('Raw Resid')
            ax2_res.grid(True, alpha=0.3)
            
        def plot_min_marker(ax, z_grid=None, format_str="Min AIC\nVal: {:.3f}"):
            if z_grid is not None and not np.isnan(z_grid[min_idx]): label = format_str.format(z_grid[min_idx])
            else: label = 'Min AIC'
            ax.plot(th_vals[min_idx[1]], bg_vals[min_idx[0]], 'r*', ms=14, label=label)
            ax.legend(loc='lower left', fontsize=9)
            
        def plot_min_marker2(ax, z_grid=None, format_str="Min AIC: {:.3f}"):
            if z_grid is not None and not np.isnan(z_grid[min_idx]): label = format_str.format(z_grid[min_idx])
            else: label = 'Min AIC'
            ax.plot(th_vals[min_idx[1]], bg_vals[min_idx[0]], 'r*', ms=14, label=label)
            ax.plot(th_vals[max_r2_idx[1]], bg_vals[max_r2_idx[0]], 'bX', ms=10, label='Max Adj. $R^2$')
            ax.legend(loc='lower left', fontsize=9)

        im3 = ax3.contourf(th_vals, bg_vals, grid_weights, levels=20, cmap='Greens')
        fig.colorbar(im3, ax=ax3, label='Probability')
        plot_min_marker2(ax3, grid_weights, "Min AIC: {:.3f}")
        ax3.set_title('(c) Akaike Weights')
        ax3.set_xlabel('$I_{th}$'); ax3.set_ylabel('$Y_{bg}$')
        self._add_stats_textbox(ax3, best_res) 

        im4 = ax4.contourf(th_vals, bg_vals, grid_aic, levels=20, cmap='viridis_r')
        fig.colorbar(im4, ax=ax4, label='AIC')
        plot_min_marker2(ax4, grid_aic, "Min AIC Val: {:.1f}")
        ax4.set_title('(d) AIC Map (Standardized)')
        ax4.set_xlabel('$I_{th}$'); ax4.set_ylabel('$Y_{bg}$')

        im5 = ax5.contourf(th_vals, bg_vals, grid_n, levels=20, cmap='plasma')
        fig.colorbar(im5, ax=ax5, label='$n$')
        plot_min_marker(ax5, grid_n, "Min AIC n: {:.2f}")
        ax5.set_title('(e) $n$ Value')
        ax5.set_xlabel('$I_{th}$'); ax5.set_ylabel('$Y_{bg}$')

        th_grid_mesh, _ = np.meshgrid(th_vals, bg_vals)
        im6 = ax6.contourf(th_vals, bg_vals, th_grid_mesh, levels=20, cmap='coolwarm')
        fig.colorbar(im6, ax=ax6, label='$I_{th}$ Value')
        plot_min_marker(ax6, th_grid_mesh, "Min AIC Ith: {:.2f}")
        ax6.set_title('(f) $I_{th}$ Map')
        ax6.set_xlabel('$I_{th}$'); ax6.set_ylabel('$Y_{bg}$')

        im7 = ax7.contourf(th_vals, bg_vals, grid_rmr, levels=20, cmap='YlGn')
        fig.colorbar(im7, ax=ax7, label='RMR')
        plot_min_marker(ax7, grid_rmr, "Min AIC RMR: {:.3f}")
        ax7.set_title('(g) RMR (Std Resid)')
        ax7.set_xlabel('$I_{th}$'); ax7.set_ylabel('$Y_{bg}$')

        im8 = ax8.contourf(th_vals, bg_vals, grid_dw, levels=20, cmap='RdBu')
        fig.colorbar(im8, ax=ax8, label='Durbin-Watson')
        plot_min_marker(ax8, grid_dw, "Min AIC DW: {:.2f}")
        ax8.set_title('(h) Durbin-Watson (Std Resid)')
        ax8.set_xlabel('$I_{th}$'); ax8.set_ylabel('$Y_{bg}$')

        im9 = ax9.contourf(th_vals, bg_vals, grid_r2_mae, levels=20, cmap='magma')
        fig.colorbar(im9, ax=ax9, label='$R^2_{MAE}$')
        plot_min_marker(ax9, grid_r2_mae, "Min AIC $R^2$: {:.4f}")
        ax9.set_title('(i) $R^2_{MAE}$')
        ax9.set_xlabel('$I_{th}$'); ax9.set_ylabel('$Y_{bg}$')

        _save_figure(fig, save_folder, "ThreeLS_fit2")

        if show_plot: 
            plt.show()
        else: 
            plt.close(fig)
        return fig

# ==========================================================================
# --- 3. Simulation module ---
# ==========================================================================

class DataGenerator:
    """Generate test spectrum data and add noise."""
    @staticmethod
    def get_true_spectrum(x: np.ndarray, p: dict) -> np.ndarray:
        y_true = p['a'] * np.power(np.maximum(0, x - p['th']), p['n']) + p['bg']
        return np.maximum(0, y_true)

    @staticmethod
    def calculate_beta(relative_noise: float, y_max: float, gamma: float) -> float:
        if relative_noise * y_max < gamma: return 0.0
        numerator = (relative_noise * y_max)**2 - gamma**2
        return np.sqrt(np.maximum(0, numerator / y_max))

    @staticmethod
    def generate_noisy_data(y_true: np.ndarray, beta: float, gamma: float, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sigma_shot = beta * np.sqrt(np.maximum(1e-9, y_true))
        noise = rng.normal(0, sigma_shot) + rng.normal(0, gamma)
        return np.clip(y_true + noise, 0.0, None)


# ==========================================================================
# --- 4. Main execution block ---
# ==========================================================================

if __name__ == '__main__':
    # Common parameter setup
    x_dat = np.linspace(4.2, 6.2, 80)
    p_true = {'a': 2600.0, 'th': 5.3, 'n': 2.5, 'bg': 5.0}
    gamma = 2.0
    
    print(f"Running Simulations...")
    print(f"Max Signal Count: {np.max(DataGenerator.get_true_spectrum(x_dat, p_true)):.0f}")

    # --- (1) Demo: Single Analysis (Noise 5%) ---
    print("\n--- (1) Demo: Single Analysis (Noise 5%) ---")
    y_true = DataGenerator.get_true_spectrum(x_dat, p_true)
    beta = DataGenerator.calculate_beta(0.05, np.max(y_true), gamma)
    y_demo = DataGenerator.generate_noisy_data(y_true, beta, gamma, seed=123)
    
    # 1/n-scan analysis
    fitter1 = OneOverNScanFitter(x_dat, y_demo)
    best_1n, all_1n = fitter1.fit() 
    print(f"[1/n-Scan] Best n: {best_1n.n:.2f}, Ith: {best_1n.th:.2f}, NMAE: {best_1n.nmae*100:.2f}%, RMR: {best_1n.rmr:.2f}, DW: {best_1n.dw:.2f}")
    fitter1.plot_result_ext2(best_1n, all_1n)
    
    # 3LS analysis
    fitter3 = ThreeLSFitter(x_dat, y_demo, mode='standard')
    best_3ls, grid_3ls = fitter3.fit(th_grid_num=40, bg_grid_num=40)
    print(f"[3LS (MAE)] Best n: {best_3ls.n:.2f}, Ith: {best_3ls.th:.2f}, NMAE: {best_3ls.nmae*100:.2f}%, RMR: {best_3ls.rmr:.2f}, DW: {best_3ls.dw:.2f}")
    fitter3.plot_result2(best_3ls, grid_3ls)
