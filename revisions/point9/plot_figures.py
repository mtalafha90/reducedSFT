#!/usr/bin/env python3
"""Render the frozen Point 9 data as vector figures in the Solar Physics artwork style.

The style follows Springer's artwork rules, which Solar Physics uses
(README.md lists the sources):

* Helvetica-type sans-serif lettering. FreeSans is a TrueType Helvetica
  design; it is embedded in every PDF.
* One lettering size, 9 pt, for axis labels, tick labels and keys
  (Springer: 8-12 pt, with minimal variation). Subscripts and superscripts
  are naturally smaller.
* The spr-sola text width, 347 pt = 122 mm (Springer's full width for
  book-sized journals). Every figure is lower than the 198 mm maximum.
* Lines of at least 0.5 pt, RGB colour, vector output, no titles.
* Each key sits inside the plotting frame, in a region that holds no data;
  save() checks this. Units are given in parentheses.
* Symbols and line styles, not colour alone, separate the curves.

The default output is the revision copy: all lettering is bold, the revision
markup used throughout main_revised.tex. ``--final`` renders the same figures
with regular-weight lettering for the accepted version.

``--manuscript DIR`` writes the PDFs into the figures/ folder of a checkout of
Reduced_SP_manuscript and records the layout next to this script; ``--out DIR``
writes both to DIR instead. Importing this module only defines the figures.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = np.load(HERE / "figure_data.npz", allow_pickle=False)
CHECKS = json.loads((HERE / "numerical_checks.json").read_text())
WIDTH = 347. / 72.27          # spr-sola \textwidth: 347 TeX points = 122 mm
FONT = "FreeSans"
SIZE = 9.                     # every piece of lettering, in points
WEIGHTS = {"bold": 600, "regular": 400}   # FreeSansBold.ttf is registered as 600

CASES = ["linear", "tilt", "latitude", "inflow", "combined"]
LABELS = ["Linear", "Tilt quenching", "Latitude quenching", "Inflow quenching", "Combined"]
MARKERS = ["o", "s", "^", "D", "v"]
T = DATA["T"]
XLABEL = r"Normalised cycle amplitude $T$"
# Degree sign for mathtext. FreeSans gives it 0.15 em side bearings,
# Helvetica about 0.05 em; the negative spaces restore Helvetica's spacing.
DEG = r"\hspace{-0.1}°\hspace{-0.1}"

# Key position for each figure. headroom extends the vertical axis by that
# fraction of its range to make an empty strip for the key; it changes no data.
PLACEMENT = {
    "fig1_main_signed_axial_dipole_response_raw": dict(loc="upper left"),
    "fig2_normalized_response_vs_theory": dict(loc="upper left"),
    "fig3_normalized_nonlinear_response": dict(loc="upper right", ncol=2),
    "figF4_closure_ladder": dict(loc="lower right"),
    "figV2_shift_validation": dict(loc="lower left"),
    "figV4_inflow_validation": dict(loc="lower right"),
    "figM3_bifurcation_comparison": dict(loc="upper left"),
    "figP2_lambdaR_scaling": dict(loc="upper left"),
}


def style(weight: str) -> None:
    w = WEIGHTS[weight]
    upright = FONT if weight == "regular" else f"{FONT}:semibold"
    italic = f"{FONT}:italic" if weight == "regular" else f"{FONT}:semibold:italic"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT],
        "font.size": SIZE, "font.weight": w,
        "axes.labelsize": SIZE, "axes.labelweight": w, "axes.linewidth": .8,
        "xtick.labelsize": SIZE, "ytick.labelsize": SIZE,
        "xtick.major.width": .8, "ytick.major.width": .8,
        "xtick.major.size": 3., "ytick.major.size": 3.,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.fontsize": SIZE, "legend.title_fontsize": SIZE,
        "lines.linewidth": 1.5, "lines.markersize": 4.5,
        "mathtext.fontset": "custom", "mathtext.rm": upright, "mathtext.it": italic,
        "mathtext.bf": f"{FONT}:semibold", "mathtext.bfit": f"{FONT}:semibold:italic",
        # "normal": letters italic; digits, brackets, operators and capital
        # Greek upright, as in typeset mathematics.
        "mathtext.sf": upright, "mathtext.cal": upright, "mathtext.default": "normal",
        "mathtext.fallback": "stixsans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def axes(ylabel, height):
    fig, ax = plt.subplots(figsize=(WIDTH, height), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.055, h_pad=.055)
    ax.set_xlabel(XLABEL, labelpad=5.)
    ax.set_ylabel(ylabel, labelpad=5.)
    return fig, ax


def reference(ax, y=None, x=None, **kw):
    """Grey guide line (y = 0 or 1) or analytic marker; a key may pass over it."""
    kw = {"color": "0.45", "lw": .8, "zorder": 0, **kw}
    line = ax.axhline(y, **kw) if x is None else ax.axvline(x, **kw)
    line.set_gid("reference")
    return line


def key(ax, loc, ncol=1, headroom=0., title=None):
    if headroom:
        y0, y1 = ax.get_ylim()
        if loc.startswith("upper"):
            ax.set_ylim(y0, y1 + headroom * (y1 - y0))
        elif loc.startswith("lower"):
            ax.set_ylim(y0 - headroom * (y1 - y0), y1)
    lg = ax.legend(loc=loc, ncol=ncol, title=title, frameon=True, fancybox=False,
                   framealpha=1., edgecolor="0.6", borderaxespad=.6, borderpad=.45,
                   handlelength=2.0, handletextpad=.6, columnspacing=1.1, labelspacing=.35)
    lg.get_frame().set_linewidth(.5)
    if title:
        lg.get_title().set_fontweight(plt.rcParams["font.weight"])
    return lg


def covered(fig, ax, lg) -> int:
    """Count data symbols and curve samples under the key (guide lines excluded)."""
    fig.canvas.draw()
    box = lg.get_window_extent(fig.canvas.get_renderer()).padded(1.5 * fig.dpi / 72.)
    hits = 0
    for line in ax.get_lines():
        if line.get_gid() == "reference":
            continue
        xy = line.get_transform().transform(np.asarray(line.get_xydata(), float))
        xy = xy[np.isfinite(xy).all(axis=1)]
        if line.get_marker() not in (None, "None", "", " "):
            r = .5 * line.get_markersize() * fig.dpi / 72.
            hits += int(np.sum((xy[:, 0] + r > box.x0) & (xy[:, 0] - r < box.x1)
                               & (xy[:, 1] + r > box.y0) & (xy[:, 1] - r < box.y1)))
        if line.get_linestyle() not in ("None", "none", "", " ") and len(xy) > 1:
            for a, b in zip(xy[:-1], xy[1:]):
                s = a + np.linspace(0., 1., max(2, int(np.hypot(*(b - a))) + 1))[:, None] * (b - a)
                hits += int(np.sum((s[:, 0] > box.x0) & (s[:, 0] < box.x1)
                                   & (s[:, 1] > box.y0) & (s[:, 1] < box.y1)))
    return hits


def fig1(place):
    fig, ax = axes("Final signed axial dipole\n" + r"$P(T)$ (arb. units)", 4.1)
    for i, case in enumerate(CASES):
        ax.plot(T, DATA[f"{case}_P_signed_main"], marker=MARKERS[i], color=f"C{i}", label=LABELS[i])
    reference(ax, y=0.)
    key(ax, **place)
    return fig, ax


def fig2(place):
    fig, ax = axes("Normalised response\n" + r"$\hat{P}(T)=P(T)/P(1)$", 4.3)
    for i, case in enumerate(CASES):
        ax.plot(DATA["Tfine"], DATA[f"{case}_theory"], "--", color=f"C{i}", zorder=2)
        ax.plot(T, DATA[f"{case}_P_hat"], ls="none", marker=MARKERS[i], color=f"C{i}",
                markerfacecolor="white", markeredgewidth=1.1, label=LABELS[i], zorder=3)
    reference(ax, y=1.)
    key(ax, title="Symbols: SFT; dashed curves: theory", **place)
    return fig, ax


def fig3(place):
    fig, ax = axes(r"$\mathrm{d}\hat{P}/\mathrm{d}T$", 4.1)
    for i, case in enumerate(CASES):
        ax.plot(T, DATA[f"{case}_R_hat"], marker=MARKERS[i], color=f"C{i}", label=LABELS[i])
    reference(ax, y=0.)
    key(ax, **place)
    return fig, ax


def fig4(place):
    fig, ax = axes(r"Normalised response $\hat{P}(T)$", 4.35)
    ax.plot(T, DATA["latitude_P_hat"], "o", color="C0", ls="none", label="SFT latitude case", zorder=3)
    for k, ls, col, label in (
            ("exact", "-", "C1", "exact kernel average"),
            ("yield_weighted", "--", "C2", "yield-weighted Gaussian form"),
            ("envelope", "-.", "C3", "Gaussian-envelope form"),
            ("static", ":", "C4", "static form")):
        ax.plot(T, DATA["ladder_" + k], ls=ls, color=col, label=label)
    reference(ax, y=1.)
    key(ax, **place)
    return fig, ax


def fig5(place):
    fig, ax = axes("Latitude-quenching efficiency\n" + r"$Q(T)=D/(aT)$", 4.35)
    ax.plot(T, DATA["shift_Q"], "o", ls="none", color="C0", label="SFT, physical shift mode", zorder=3)
    ax.plot(T, DATA["shift_exact_Q"], "-", color="C1", label="exact kernel prediction")
    ax.plot(T, DATA["shift_derived_Q"], "--", color="C2",
            label=r"derived Gaussian form ($\lambda_R^{\mathrm{pair}}=12.1" + DEG + "$)")
    ax.plot(T, DATA["shift_prescribed_Q"], ":", color="C3",
            label=r"prescribed form ($\lambda_R^{\mathrm{cl}}=20" + DEG + "$)")
    reference(ax, y=0.)
    key(ax, **place)
    return fig, ax


def fig6(place):
    fig, ax = axes(r"$T\,Q_{\mathrm{IQ}}(T)$ (dipole in units of $a$)", 4.35)
    for i, speed in enumerate((2, 5)):
        ax.plot(T, DATA[f"inflow{speed}_D_over_a"], ls="none", marker=("o", "s")[i], color=f"C{i}",
                label=fr"physical inflow {speed} m s$^{{-1}}$: $D/a$", zorder=3)
    for i, speed in enumerate((2, 5)):
        b = CHECKS[f"inflow{speed}_b_fit_leading"]
        ax.plot(T, DATA[f"inflow{speed}_rational"], ls=("--", ":")[i], color=f"C{i+2}",
                label=fr"rational form, leading-order fit $b_{{\mathrm{{IQ}}}}={b:.2f}$")
    reference(ax, y=0.)
    key(ax, **place)
    return fig, ax


def fig7(place):
    fig, ax = axes(r"Attractor of $T_n$", 4.35)
    gains = DATA["bifurcation_gains"]
    ax.plot(np.repeat(gains, 60), DATA["bifurcation_map"].ravel(), ".", color="0.4", markersize=2.4,
            label="memory-corrected map attractor", zorder=2)
    ax.plot(np.repeat(gains, 16), DATA["bifurcation_sft"].ravel(), "o", color="C0", markersize=3.,
            markerfacecolor="none", markeredgewidth=.7, label="multi-cycle SFT attractor", zorder=3)
    for k, ls, col, label in (
            ("D0_period_doubling_memory", "--", "C3", "memory"),
            ("D0_period_doubling_memoryless", ":", "C2", "memoryless")):
        d0 = CHECKS[k]
        reference(ax, x=d0, ls=ls, color=col, lw=1.3, zorder=2,
                  label=fr"analytic period doubling ({label}), $D_0={d0:.2f}$")
    ax.set_xlabel(r"Linear gain $D_0=k\,a$")
    key(ax, **place)
    return fig, ax


def fig8(place):
    fig, ax = axes(r"Measured $\lambda_R^{\mathrm{pair}}$ (deg)", 4.1)
    ax.plot(DATA["transport_x"], DATA["transport_y"], "o", color="C0", markersize=4.2,
            label="measured (adjoint kernel)", zorder=3)
    ax.plot(DATA["transport_curve_x"], DATA["transport_curve_y"], "--", color="C1",
            label="empirical fit:\n"
            + r"$(\lambda_R^{\mathrm{pair}})^2\simeq(1.19)^2\lambda_{\mathrm{tr}}^2+(3.5" + DEG + ")^2$")
    ax.plot(DATA["transport_curve_x"], DATA["transport_curve_x"], ":", color="0.4", lw=1.2,
            label="one-to-one reference")
    ax.set_xlabel(r"Transport scale $\lambda_{\mathrm{tr}}$ (deg)")
    key(ax, **place)
    return fig, ax


FIGURES = [
    ("fig1_main_signed_axial_dipole_response_raw", fig1),
    ("fig2_normalized_response_vs_theory", fig2),
    ("fig3_normalized_nonlinear_response", fig3),
    ("figF4_closure_ladder", fig4),
    ("figV2_shift_validation", fig5),
    ("figV4_inflow_validation", fig6),
    ("figM3_bifurcation_comparison", fig7),
    ("figP2_lambdaR_scaling", fig8),
]


def check_and_save(fig, ax, name, out, weight, place):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lg = ax.get_legend()
    # Labels, ticks and key must fit on the page; saving uncropped at the exact
    # text width keeps every font at its stated size.
    ticks = [t for v, t in zip(ax.get_xticks(), ax.get_xticklabels())
             if ax.get_xlim()[0] <= v <= ax.get_xlim()[1]]
    ticks += [t for v, t in zip(ax.get_yticks(), ax.get_yticklabels())
              if ax.get_ylim()[0] <= v <= ax.get_ylim()[1]]
    for artist in [ax.xaxis.label, ax.yaxis.label, lg, *ticks]:
        box = artist.get_window_extent(renderer)
        if box.width and box.height:
            assert box.x0 >= -.5 and box.y0 >= -.5, (name, str(artist))
            assert box.x1 <= fig.bbox.width + .5 and box.y1 <= fig.bbox.height + .5, (name, str(artist))
    frame, lbox = ax.get_window_extent(renderer), lg.get_window_extent(renderer)
    assert (lbox.x0 >= frame.x0 and lbox.y0 >= frame.y0
            and lbox.x1 <= frame.x1 and lbox.y1 <= frame.y1), (name, "key outside the frame")
    hits = covered(fig, ax, lg)
    assert hits == 0, (name, f"key covers {hits} data samples")
    fig.savefig(out / (name + ".pdf"), metadata={
        "Title": name, "Author": "Mohammed H. Talafha",
        "Subject": "Solar Physics artwork style; "
                   + ("lettering in bold (revision copy)" if weight == "bold" else "regular lettering"),
        "CreationDate": None, "ModDate": None})
    entry = {"file": f"figures/{name}.pdf",
             "width_inches": float(fig.get_figwidth()),
             "height_inches": float(fig.get_figheight()),
             "width_mm": round(25.4 * fig.get_figwidth(), 1),
             "height_mm": round(25.4 * fig.get_figheight(), 1),
             "key_location": place["loc"], "key_columns": place.get("ncol", 1),
             "key_headroom": place.get("headroom", 0.),
             "key_covers_data_samples": hits,
             "x_label": ax.get_xlabel(), "y_label": ax.get_ylabel(),
             "legend": [s.get_text() for s in lg.get_texts()]}
    plt.close(fig)
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", action="store_true",
                        help="regular-weight lettering for the accepted version "
                             "(default: bold lettering, the revision copy)")
    parser.add_argument("--manuscript", type=Path,
                        help="checkout of Reduced_SP_manuscript: write the PDFs to its "
                             "figures/ folder and record the layout next to this script")
    parser.add_argument("--out", type=Path,
                        help="write the PDFs and the layout record to this folder instead")
    args = parser.parse_args(argv)
    if (args.manuscript is None) == (args.out is None):
        parser.error("give exactly one of --manuscript or --out")
    if args.manuscript is not None:
        if not (args.manuscript / "main_revised.tex").is_file():
            parser.error(f"{args.manuscript} is not a checkout of Reduced_SP_manuscript")
        out, layout_dir = args.manuscript / "figures", HERE
    else:
        out, layout_dir = args.out, args.out
    # Matplotlib asks for weight 700; FreeSansBold is registered as 600.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    weight = "regular" if args.final else "bold"
    # Stop rather than fall back silently to another typeface.
    try:
        font_file, _ = (font_manager.findfont(font_manager.FontProperties(
            family=FONT, weight=WEIGHTS[weight], style=s), fallback_to_default=False)
            for s in ("normal", "italic"))
    except ValueError:
        raise SystemExit(f"{FONT} is not installed (GNU FreeFont; Debian package fonts-freefont-ttf)")
    style(weight)
    out.mkdir(parents=True, exist_ok=True)
    manifest = [check_and_save(*make(PLACEMENT[name]), name, out, weight, PLACEMENT[name])
                for name, make in FIGURES]
    layout = layout_dir / "figure_layout.json"
    layout.write_text(json.dumps({
        "matplotlib_version": matplotlib.__version__, "style": "Solar Physics (Springer) artwork rules",
        "font": FONT, "font_file": Path(font_file).name, "lettering_pt": SIZE,
        "text_weight": weight, "pdf_fonttype": 42, "figures": manifest}, indent=2) + "\n")
    print(f"Saved {len(manifest)} vector PDFs to {out} and {layout}")


if __name__ == "__main__":
    main()
