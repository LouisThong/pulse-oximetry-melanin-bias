"""
three_wavelength_decomposition.py
---------------------------------
Stage 5 of the PPG pipeline: a candidate *melanin-decoupled* SpO2 estimator.

Approach
========
Standard pulse oximetry uses the ratio of AC/DC at two wavelengths (red and
near-infrared) and a single empirical calibration curve.  The well-documented
failure mode of that method is that melanin -- the dominant chromophore in
epidermal skin pigmentation -- attenuates the red wavelength far more than
the IR wavelength, so darker Monk Skin Tones bias the ratio-of-ratios and,
therefore, the reported SpO2 (Sjoding et al., NEJM 2020; Fawzy et al.,
JAMA 2022).

If the pulsatile absorbance is measured at THREE wavelengths, we can solve a
linear inverse problem that treats melanin as a third chromophore rather
than ignoring it.  The Beer-Lambert linearisation for small pulsatile
signals gives:

    A_lambda = epsilon_HbO2(lambda) * C_HbO2
             + epsilon_Hb  (lambda) * C_Hb
             + epsilon_mel (lambda) * C_mel

For three wavelengths this is a 3-equation, 3-unknown system:

    |A_527|   |e_HbO2,527  e_Hb,527  e_mel,527| |C_HbO2|
    |A_660| = |e_HbO2,660  e_Hb,660  e_mel,660| |C_Hb  |
    |A_940|   |e_HbO2,940  e_Hb,940  e_mel,940| |C_mel |

We solve C = inv(E) @ A with ``numpy.linalg.inv`` and then form the
melanin-corrected oxygen saturation from only the two hemoglobin
components:

    SpO2_corrected = C_HbO2 / (C_HbO2 + C_Hb) * 100

Extinction-coefficient values
=============================
Hemoglobin: molar extinction coefficients (cm^-1 / (moles/L)) from Scott
Prahl's compilation of the Gratzer / Kollias / Takatani-Graham tables at
  https://omlc.org/spectra/hemoglobin/summary.html
Values at 527 nm are linearly interpolated between the tabulated 526 nm and
528 nm entries.

    lambda = 527 nm :  HbO2 =  43,660   Hb = 42,744
    lambda = 660 nm :  HbO2 =    319.6  Hb =  3,226.56
    lambda = 940 nm :  HbO2 =  1,214    Hb =    693.44

Melanin: bulk absorption coefficient (cm^-1) for a melanosome interior,
from Steven L. Jacques, "Optical properties of biological tissues: a
review", Phys. Med. Biol. 58 (2013) R37-R61, equation for eumelanin
absorption

    mu_a,mel(lambda) = 1.70 x 10^12 * lambda^(-3.48)     [lambda in nm, mu_a in cm^-1]

Evaluated at the three wavelengths:

    lambda = 527 nm :  mu_a,mel = 573.5  cm^-1
    lambda = 660 nm :  mu_a,mel = 262.1  cm^-1
    lambda = 940 nm :  mu_a,mel =  76.6  cm^-1

Because the hemoglobin column is in cm^-1/M and the melanin column is in
cm^-1, the recovered C_mel is in arbitrary (volume-fraction-like) units.
This is fine for our purpose: SpO2_corrected depends only on the ratio
C_HbO2 / C_Hb, which shares consistent units.

Pulsatile absorbance input
==========================
For each participant we build A = [AC/DC]_527, [AC/DC]_660, [AC/DC]_940
from the green, red, and IR channels of summary.csv respectively.

Outputs
=======
  results/summary.csv             : overwritten with 4 new columns
                                    (HbO2_conc, Hb_conc, melanin_conc,
                                    SpO2_corrected).
  plots/05_model_comparison/
      scatter_both_spo2_vs_MST.png
  model_evaluation.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE      = Path("/home/claude/oximeter")
SUMMARY   = BASE / "results" / "summary.csv"
PLOT_DIR  = BASE / "plots" / "05_model_comparison"
REPORT    = BASE / "model_evaluation.txt"


# ---------------------------------------------------------------------------
# Extinction-coefficient matrix (3 wavelengths x 3 chromophores)
# ---------------------------------------------------------------------------
# Rows: 527 nm (green), 660 nm (red), 940 nm (IR)
# Cols: HbO2, Hb, melanin
#
# HbO2 / Hb: cm^-1 / (moles/L),   source = Prahl/Gratzer/Kollias (omlc.org)
# Melanin  : cm^-1,               source = Jacques (2013), 1.7e12 * lambda^-3.48

E = np.array([
    [43660.0, 42744.0, 573.5],   # 527 nm
    [  319.6,  3226.56, 262.1],  # 660 nm
    [ 1214.0,   693.44,  76.6],  # 940 nm
])

E_INV = np.linalg.inv(E)


# ---------------------------------------------------------------------------
# Per-participant decomposition
# ---------------------------------------------------------------------------

def decompose_row(row: pd.Series) -> dict[str, float]:
    """Return HbO2/Hb/melanin concentrations and corrected SpO2 for one row."""
    # Pulsatile absorbance vector A = AC/DC at [527, 660, 940] nm.
    A = np.array([
        row["AC_green"] / row["DC_green"],   # 527 nm
        row["AC_red"]   / row["DC_red"],     # 660 nm
        row["AC_ir"]    / row["DC_ir"],      # 940 nm
    ], dtype=float)

    C = E_INV @ A                            # [C_HbO2, C_Hb, C_mel]
    c_hbo2, c_hb, c_mel = C.tolist()

    # SpO2 uses only the two hemoglobin species.  Because the 3x3 inversion
    # is mildly ill-conditioned, C_Hb can come out negative on participants
    # whose red-channel AC/DC sits near the noise floor; in that case the
    # ratio is unphysical.  We still persist the raw value to the CSV (per
    # spec), and flag it as non-physiological downstream.
    denom = c_hbo2 + c_hb
    if denom == 0 or not np.isfinite(denom):
        spo2 = float("nan")
    else:
        spo2 = c_hbo2 / denom * 100.0

    return {
        "HbO2_conc":      c_hbo2,
        "Hb_conc":        c_hb,
        "melanin_conc":   c_mel,
        "SpO2_corrected": spo2,
    }


def run_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    extra = df.apply(decompose_row, axis=1, result_type="expand")
    return pd.concat([df, extra], axis=1)


# ---------------------------------------------------------------------------
# Regression & plot
# ---------------------------------------------------------------------------

def regress(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """OLS simple linear regression with two-sided t-test on the slope.

    Implemented without scipy so the script runs in a minimal environment.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    xm = x.mean()
    ym = y.mean()
    sxx = float(np.sum((x - xm) ** 2))
    sxy = float(np.sum((x - xm) * (y - ym)))

    if sxx == 0 or n < 3:
        return {"slope": float("nan"), "intercept": float("nan"),
                "rvalue": float("nan"), "r2": float("nan"),
                "pvalue": float("nan"), "stderr": float("nan"), "n": n}

    slope = sxy / sxx
    intercept = ym - slope * xm
    yhat = intercept + slope * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - ym) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.sign(slope)) * float(np.sqrt(max(r2, 0.0)))

    # SE of slope, two-sided t-test, df = n-2.  p-value computed from the
    # t -> beta -> incomplete-beta relation without scipy.
    df = n - 2
    if df <= 0 or ss_res <= 0:
        p = float("nan")
        stderr = float("nan")
    else:
        sigma2 = ss_res / df
        stderr = float(np.sqrt(sigma2 / sxx))
        t = slope / stderr if stderr > 0 else float("inf")
        # Two-sided p-value from Student-t via the regularised incomplete beta
        # function.  p = I_x(df/2, 1/2) with x = df / (df + t^2).
        x_ib = df / (df + t * t)
        p = _betainc(df / 2.0, 0.5, x_ib)
    return {"slope": float(slope), "intercept": float(intercept),
            "rvalue": float(r), "r2": float(r2),
            "pvalue": float(p), "stderr": float(stderr), "n": n}


def _betainc(a: float, b: float, x: float, max_iter: int = 400,
             eps: float = 1e-12) -> float:
    """Regularised incomplete beta function I_x(a, b) via continued fraction.

    Based on Numerical Recipes, sufficient for our t-test p-values.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    import math
    # log of the prefactor
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x, max_iter, eps) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x, max_iter, eps) / b


def _betacf(a: float, b: float, x: float, max_iter: int, eps: float) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def make_comparison_plot(df: pd.DataFrame,
                         fit_dual: dict[str, float],
                         fit_corr: dict[str, float],
                         valid_mask: np.ndarray) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / "scatter_both_spo2_vs_MST.png"

    x       = df["MST"].to_numpy(dtype=float)
    y_dual  = df["SpO2_dual_wavelength"].to_numpy(dtype=float)
    y_corr  = df["SpO2_corrected"].to_numpy(dtype=float)

    # Fit lines span MST 1-10 so the slope difference is easy to eyeball.
    xx = np.linspace(1, 10, 50)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

    # ---- Left panel: dual-wavelength ---------------------------------------
    ax = axes[0]
    ax.scatter(x, y_dual, s=60, color="tab:blue", edgecolor="k",
               linewidth=0.5, zorder=3, label="participants")
    if np.isfinite(fit_dual["slope"]):
        yy = fit_dual["intercept"] + fit_dual["slope"] * xx
        ax.plot(xx, yy, color="tab:blue", lw=2.0,
                label=f"slope = {fit_dual['slope']:+.2f} %/MST")
    ax.set_title("SpO2_dual_wavelength")
    _decorate_axis(ax, fit_dual)
    ax.set_ylabel("SpO2 estimate (%)")

    # ---- Right panel: 3-wavelength corrected -------------------------------
    ax = axes[1]
    # Valid (physiological) points get the normal marker; invalid points are
    # shown at the top of the panel with an 'x' so the reader can see that
    # four of six participants produced an unphysical corrected estimate.
    ax.scatter(x[valid_mask], y_corr[valid_mask], s=60, color="tab:orange",
               edgecolor="k", linewidth=0.5, zorder=3,
               label=f"valid (n = {int(valid_mask.sum())})")
    if (~valid_mask).any():
        ax.scatter(x[~valid_mask],
                   np.full((~valid_mask).sum(), 108),
                   marker="x", s=80, color="firebrick", linewidth=2,
                   zorder=3,
                   label=f"unphysical (n = {int((~valid_mask).sum())})")
    if np.isfinite(fit_corr["slope"]):
        yy = fit_corr["intercept"] + fit_corr["slope"] * xx
        ax.plot(xx, yy, color="tab:orange", lw=2.0,
                label=f"slope = {fit_corr['slope']:+.2f} %/MST (valid only)")
    ax.set_title("SpO2_corrected (3-wavelength)")
    _decorate_axis(ax, fit_corr)

    fig.suptitle("Two- vs three-wavelength SpO2 estimates across the MST scale",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _decorate_axis(ax, fit: dict[str, float]) -> None:
    ax.axhline(100, color="grey", lw=0.6, ls=":")
    ax.axhline(70,  color="grey", lw=0.6, ls=":")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(30, 115)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Monk Skin Tone (MST)")
    if np.isfinite(fit["slope"]):
        ax.text(0.04, 0.06,
                f"R$^2$ = {fit['r2']:.3f}\np = {fit['pvalue']:.3f}"
                f"  (n = {fit['n']})",
                transform=ax.transAxes, fontsize=9,
                va="bottom", ha="left",
                bbox=dict(facecolor="white", edgecolor="grey", alpha=0.8))
    else:
        ax.text(0.04, 0.06,
                "too few valid points to regress",
                transform=ax.transAxes, fontsize=9,
                va="bottom", ha="left",
                bbox=dict(facecolor="white", edgecolor="grey", alpha=0.8))
    ax.legend(loc="upper right", fontsize=9)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

REPORT_TEMPLATE = """\
==============================================================================
MODEL EVALUATION - 3-Wavelength Spectroscopic Decomposition for SpO2
Generated: 2026-04-20
Source:    results/summary.csv (N = {n_total} participants, MST 2-3)
==============================================================================

1. HOW THE CORRECTED ESTIMATE WAS COMPUTED
------------------------------------------------------------------------------
A 3x3 extinction-coefficient matrix was constructed for the three
chromophores (HbO2, Hb, melanin) at the three measurement wavelengths
(527 nm green, 660 nm red, 940 nm IR):

                HbO2 [cm^-1/M]   Hb [cm^-1/M]   Melanin [cm^-1]
    527 nm      {e0[0]:>10.1f}       {e0[1]:>10.1f}    {e0[2]:>10.1f}
    660 nm      {e1[0]:>10.1f}       {e1[1]:>10.1f}    {e1[2]:>10.1f}
    940 nm      {e2[0]:>10.1f}       {e2[1]:>10.1f}    {e2[2]:>10.1f}

    Sources:
      Hemoglobin : Prahl, "Tabulated molar extinction coefficient for
                   hemoglobin in water", Oregon Medical Laser Center
                   (omlc.org/spectra/hemoglobin/summary.html), compiled
                   from Gratzer (Med. Res. Council Labs), Kollias (Wellman
                   Labs) and Takatani-Graham. Values at 527 nm are linearly
                   interpolated between tabulated 526 nm and 528 nm rows.
      Melanin    : Jacques, S. L. (2013). "Optical properties of biological
                   tissues: a review." Phys. Med. Biol. 58, R37-R61.
                   mu_a,mel(lambda) = 1.70e12 * lambda^-3.48  (lambda in nm).

For each participant the pulsatile-absorbance vector
A = [AC/DC at 527, AC/DC at 660, AC/DC at 940] was formed from the green,
red, and IR channels of summary.csv.  The system C = inv(E) * A was solved
with numpy.linalg.inv.  SpO2_corrected = HbO2 / (HbO2 + Hb) * 100.

Matrix condition number = {cond:.2e}.  In absolute terms this is only
mildly ill-conditioned (worst-case a few-hundred-fold amplification of
input error), but when combined with the small signal-to-noise ratio of
the AC/DC measurements the inversion is still fragile in practice.

Per-participant decomposition:

{decomp_block}

IMPORTANT: the hemoglobin extinction columns are ~10,000x larger than the
melanin column at 527 nm, so the recovered concentrations are small
numbers (order 1e-7) that sit close to the noise level of the
measurement.  {n_invalid} of {n_total} participants returned a NEGATIVE
C_Hb or a denominator close to zero, which pushed SpO2_corrected outside
the physiological range.  Those rows are kept in summary.csv verbatim
but excluded from the regression and the scatter below.


2. BIAS COMPARISON vs MST  (valid values only)
------------------------------------------------------------------------------
                               slope (%/MST)     R^2       p        n
    SpO2_dual_wavelength       {fd[slope]:>+8.3f}         {fd[r2]:>6.3f}   {fd[pvalue]:>5.3f}   {n_total}
    SpO2_corrected (3-wave)    {corr_slope_str:>8}         {corr_r2_str:>6}   {corr_p_str:>5}   {n_valid}

    {reduction_line}


3. PLAIN-ENGLISH VERDICT
------------------------------------------------------------------------------
{verdict_block}


4. LIMITATIONS
------------------------------------------------------------------------------
(a) MST coverage of the current cohort is only 2-3.  The central claim of
    a 3-wavelength melanin correction is that it flattens the bias across
    the FULL MST range (1-10).  On this dataset we can only demonstrate
    the method on the lighter end of the scale, where dual-wavelength
    oximeters already behave acceptably.  Any slope comparison on
    MST 2-3 is dominated by intra-group noise in the ratio-of-ratios,
    not by a true skin-tone gradient.  Definitive evaluation requires
    participants in the Medium (MST 4-6) and Dark (MST 7-10) bands.

(b) At MST 9-10 the red (660 nm) pulsatile signal is expected to approach
    the noise floor of the MAX30101 (as flagged in the project report,
    section on signal quality across skin tones).  In that regime the
    AC_red / DC_red entry of the absorbance vector is noise-dominated,
    which injects noise straight into the inverse problem.  Because the
    3x3 matrix is mildly ill-conditioned (condition number ~ {cond:.1e}),
    a noisy A vector produces a VERY noisy C vector and hence a very
    noisy SpO2_corrected estimate.  Increasing the LED drive current,
    averaging more pulses, or switching to a detector with a lower
    dark-current floor would help, but cannot fully solve this - at
    some point the tissue simply does not return enough red photons.

(c) The 527 nm green channel is the only wavelength where melanin is a
    sizable fraction of total absorbance; it is therefore doing most of
    the melanin-disambiguation work.  But the green channel also has
    the largest contribution from superficial capillaries rather than
    arterial blood, so the "pulsatile absorbance" at 527 nm is not a
    clean arterial measurement.  The decomposition silently attributes
    all of that non-arterial green pulsatility to the three
    chromophores, inflating the melanin term in particular.

(d) The Beer-Lambert linearisation assumes single-path attenuation in a
    homogeneous medium.  Human skin is layered and highly scattering;
    effective path length is wavelength-dependent.  A more faithful
    model would use wavelength-specific differential path-length
    factors (DPFs) or a full Monte-Carlo forward model.

(e) The corrected estimate uses arbitrary-unit concentrations; it is
    NOT calibrated against a gold-standard co-oximeter reading.  It
    should be read as a relative, bias-reduction indicator, not as a
    drop-in replacement for a clinically calibrated SpO2 value.


5. FUTURE HARDWARE IMPROVEMENTS
------------------------------------------------------------------------------
(i)   Higher-power red LEDs (or a separate 660 nm laser diode) to push
      the red AC signal above the noise floor at MST 7-10.
(ii)  A detector with lower dark current and higher red-wavelength
      responsivity (e.g. a silicon photodiode optimised for 600-700 nm,
      or an APD) to improve SNR without heating the skin.
(iii) A fourth wavelength around 810 nm (the Hb/HbO2 isosbestic point)
      to overdetermine the inverse problem and regularise the matrix
      inversion - a 4x3 least-squares solve is far better conditioned
      than the 3x3 inversion used here.
(iv)  A reflective geometry with multiple source-detector separations to
      give a crude depth-resolved measurement: shallow separations
      carry more melanin signal, deep separations carry more arterial
      signal.  This decouples melanin from arterial SpO2 geometrically
      rather than spectrally, which is much more robust.
(v)   Active ambient-light rejection (modulated LEDs with lock-in
      detection) to keep the noise floor low under realistic use.


6. CONCLUSION
------------------------------------------------------------------------------
The 3-wavelength spectroscopic decomposition is theoretically the right
way to remove the melanin term from the pulse-oximetry forward model,
and the mathematics of the method - Beer-Lambert linearisation, a 3x3
chromophore matrix, a single matrix inversion per participant - was
reproduced here against published extinction coefficients.  On this
cohort, though, the method does NOT deliver a demonstrable reduction in
skin-tone bias: the participants all sit at MST 2-3, and on {n_invalid}
of {n_total} subjects the inversion returned a negative Hb
concentration, which places SpO2_corrected outside the physiological
range and disqualifies it from the bias regression.  The honest conclusion is that the present
data cannot either confirm or refute the 3-wavelength correction - the
method needs a skin-tone-balanced cohort AND a cleaner red channel
before it can be meaningfully evaluated.  The three highest-value next
steps are (i) recruiting MST 4-10 participants, (ii) lifting the red-
channel noise floor with a brighter 660 nm LED and a more sensitive
detector, and (iii) adding a fourth wavelength (e.g. 810 nm isosbestic)
so the inverse problem becomes overdetermined and can be solved by
regularised least squares rather than a bare matrix inversion.

==============================================================================
Plots written to: plots/05_model_comparison/
  - scatter_both_spo2_vs_MST.png
==============================================================================
"""


def build_verdict(fit_dual: dict[str, float],
                  fit_corr: dict[str, float],
                  n_total: int, n_valid: int) -> str:
    """Return the verdict paragraph for the report."""
    sd = fit_dual["slope"]
    sc = fit_corr["slope"]
    absd = abs(sd) if np.isfinite(sd) else float("nan")

    if not np.isfinite(sc):
        return (
            f"The 3-wavelength decomposition produced physiologically "
            f"valid SpO2_corrected values for only {n_valid} of {n_total} "
            "participants, which is not enough to fit a meaningful "
            "regression against MST.  On the remaining participants the "
            "recovered C_Hb came out negative (or so close to zero that "
            "SpO2 blew up to >1000 %), which is the classic failure mode "
            "of a bare 3x3 inversion in the presence of measurement "
            "noise.  In plain English: on this cohort the 3-wavelength "
            "model does NOT successfully reduce the skin-tone bias - it "
            "fails to produce usable SpO2 estimates at all for most "
            "subjects.  The result is INCONCLUSIVE rather than negative: "
            "we cannot tell from MST 2-3 data alone whether the method "
            "would work on darker skin tones where the correction "
            "actually matters."
        )

    absc = abs(sc)
    flatter = absc < absd
    if flatter:
        pct = (1 - absc / absd) * 100.0 if absd > 0 else 0.0
        return (
            f"The magnitude of the MST slope falls from {absd:.2f} %/MST "
            f"(dual-wavelength, n = {n_total}) to {absc:.2f} %/MST "
            f"(3-wavelength, n = {n_valid} valid points), a reduction "
            f"of {pct:.0f} % in the apparent skin-tone bias.  The "
            "correction therefore points in the right direction on the "
            "light end of the MST scale, but the n is too small and the "
            "MST range (2-3) too narrow to call this a statistically "
            "meaningful bias reduction.  A skin-tone-balanced cohort is "
            "required before the claim can be made with any confidence."
        )

    pct = (absc / absd - 1) * 100.0 if absd > 0 else 0.0
    return (
        f"The magnitude of the MST slope rises from {absd:.2f} %/MST "
        f"(dual-wavelength, n = {n_total}) to {absc:.2f} %/MST "
        f"(3-wavelength, n = {n_valid}), i.e. the bias is NOT reduced "
        "in this sample.  This is almost certainly an artefact of "
        "applying a 3x3 matrix inversion to a cohort with almost no "
        "melanin variation: the inversion amplifies within-group noise "
        "and produces a LARGER apparent slope than the dual-wavelength "
        "baseline.  The method will have to be re-evaluated on a cohort "
        "that actually spans MST 1-10 before any verdict on bias "
        "reduction can be defended."
    )


def write_report(fit_dual: dict[str, float],
                 fit_corr: dict[str, float],
                 cond_number: float,
                 n_total: int,
                 n_valid: int,
                 valid_mask: np.ndarray,
                 df: pd.DataFrame) -> Path:
    n_invalid = n_total - n_valid
    verdict_block = build_verdict(fit_dual, fit_corr, n_total, n_valid)

    # Per-participant decomposition table for the "how it was computed" block.
    lines = [
        "    pid       MST   C_HbO2        C_Hb          C_mel         "
        "SpO2_corr    valid?",
        "    -------   ---   -----------   -----------   -----------   "
        "----------   ------",
    ]
    for i, r in df.iterrows():
        mark = "yes" if valid_mask[i] else "NO"
        sc = r["SpO2_corrected"]
        sc_str = f"{sc:>10.2f}" if np.isfinite(sc) else "       nan"
        lines.append(
            f"    {r['participant_id']:<7}   {int(r['MST']):>3}   "
            f"{r['HbO2_conc']:+.3e}   {r['Hb_conc']:+.3e}   "
            f"{r['melanin_conc']:+.3e}   {sc_str}   {mark}"
        )
    decomp_block = "\n".join(lines)

    if np.isfinite(fit_corr["slope"]):
        corr_slope_str = f"{fit_corr['slope']:+.3f}"
        corr_r2_str = f"{fit_corr['r2']:.3f}"
        corr_p_str = f"{fit_corr['pvalue']:.3f}"
        absd, absc = abs(fit_dual["slope"]), abs(fit_corr["slope"])
        red_pct = (1 - absc / absd) * 100.0 if absd > 0 else 0.0
        reduction_line = (
            f"|slope| change: {red_pct:+.1f} %  "
            "(positive = flatter = less bias)"
        )
    else:
        corr_slope_str = "N/A"
        corr_r2_str = "N/A"
        corr_p_str = "N/A"
        reduction_line = (
            "|slope| change: N/A  "
            "(too few physiologically valid SpO2_corrected values to fit)"
        )

    body = REPORT_TEMPLATE.format(
        e0=E[0], e1=E[1], e2=E[2],
        cond=cond_number,
        fd=fit_dual,
        corr_slope_str=corr_slope_str,
        corr_r2_str=corr_r2_str,
        corr_p_str=corr_p_str,
        reduction_line=reduction_line,
        n_total=n_total,
        n_valid=n_valid,
        n_invalid=n_invalid,
        decomp_block=decomp_block,
        verdict_block=verdict_block,
    )
    REPORT.write_text(body)
    return REPORT


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    df = pd.read_csv(SUMMARY)
    # Idempotent: if a previous run appended decomposition columns, drop
    # them before re-computing so we don't end up with duplicate column
    # names (which would silently turn df['SpO2_corrected'] into a
    # 2-column DataFrame).
    for col in ("HbO2_conc", "Hb_conc", "melanin_conc", "SpO2_corrected"):
        if col in df.columns:
            df = df.drop(columns=[col])
    df = run_decomposition(df)

    # Persist extended summary (overwriting with the new columns appended).
    # We save BEFORE filtering so the raw decomposition values are on disk.
    # Using %.6g so the existing 4-5 sig-fig columns stay readable AND the
    # new concentration columns (which are ~1e-7) are not truncated to 0.
    df.to_csv(SUMMARY, index=False, float_format="%.6g")

    # Regression of both SpO2 estimates vs continuous MST.
    x = df["MST"].to_numpy(dtype=float)
    fit_dual = regress(x, df["SpO2_dual_wavelength"].to_numpy(dtype=float))

    # For the corrected estimate we keep only physiologically plausible
    # values.  The raw method can return hundreds or thousands of percent
    # when C_Hb comes out near zero; including those in a linear regression
    # would be meaningless.
    spo2_corr = df["SpO2_corrected"].to_numpy(dtype=float)
    valid = np.isfinite(spo2_corr) & (spo2_corr >= 0.0) & (spo2_corr <= 100.0)
    if valid.sum() >= 3:
        fit_corr = regress(x[valid], spo2_corr[valid])
    else:
        fit_corr = {"slope": float("nan"), "intercept": float("nan"),
                    "rvalue": float("nan"), "r2": float("nan"),
                    "pvalue": float("nan"), "stderr": float("nan"),
                    "n": int(valid.sum())}

    cond_number = float(np.linalg.cond(E))

    plot_path = make_comparison_plot(df, fit_dual, fit_corr, valid)
    report_path = write_report(fit_dual, fit_corr, cond_number,
                               n_total=len(df), n_valid=int(valid.sum()),
                               valid_mask=valid, df=df)

    # Console summary.
    print("=" * 70)
    print("3-WAVELENGTH DECOMPOSITION -- RESULTS")
    print("=" * 70)
    print("Extinction matrix (rows = 527/660/940 nm, cols = HbO2/Hb/Mel):")
    print(E)
    print(f"Condition number: {cond_number:.3e}")
    print()
    print(df[[
        "participant_id", "MST",
        "HbO2_conc", "Hb_conc", "melanin_conc",
        "SpO2_dual_wavelength", "SpO2_corrected",
    ]].to_string(index=False))
    print()
    print(f"Slope  dual  : {fit_dual['slope']:+8.3f} %/MST   "
          f"R^2 = {fit_dual['r2']:.3f}  p = {fit_dual['pvalue']:.3f}")
    print(f"Slope  corr  : {fit_corr['slope']:+8.3f} %/MST   "
          f"R^2 = {fit_corr['r2']:.3f}  p = {fit_corr['pvalue']:.3f}")
    print()
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
