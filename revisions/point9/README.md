# Point 9: publication figures

The revised manuscript uses eight vector PDFs at the class's full text width
(347 TeX points = 122 mm). Since PA03 they follow the Solar Physics (Springer)
artwork rules listed below. The PDFs live in the `figures/` folder of the
manuscript repository (`mtalafha90/Reduced_SP_manuscript`); this folder holds
their data and the script that draws them. The original PNGs and `main.tex`
are preserved.
No equation, model parameter, caption or scientific claim is changed. In the
revision copy all figure lettering is bold, the revision markup used in
`main_revised.tex`. The `--final` option gives the same figures with
regular-weight lettering for the accepted version.

## Journal style

| Rule | Source | How the figures meet it |
|---|---|---|
| Standard fonts such as Times Roman, Courier and Helvetica | Solar Physics guidelines | FreeSans, a TrueType Helvetica design, embedded in every PDF |
| Lettering in Helvetica or Arial, consistently 2–3 mm (8–12 pt), with minimal variation in size | Springer figure guide | All lettering 9 pt; only subscripts and superscripts are smaller |
| No titles or captions inside the figure | Springer figure guide | None; captions stay in the TeX file |
| A key may sit inside the plot frame if it fits without crowding; otherwise it goes in the caption | Springer figure guide | Every key is inside its frame, in a region without data; the script checks that no symbol or curve lies under it |
| Units in parentheses | Springer figure guide | (arb. units), (deg) |
| Lines at least 0.1 mm (0.3 pt) wide | Springer artwork guidelines | Thinnest line 0.5 pt (key frame) |
| Book-sized journals: 80 or 122 mm wide, at most 198 mm high | Springer artwork guidelines; spr-sola text block 122 × 198 mm | 122 mm wide, 78–83 mm high (104–110 mm until PA12 made each figure 25% shorter) |
| Vector graphics with embedded fonts; colour in RGB | Springer artwork guidelines | Vector PDF, no raster images, TrueType fonts embedded, RGB |
| Patterns as well as colours | Springer accessibility rules | Each series has its own symbol or line style |

Placing the key, choosing the colours and the frame are left to authors; the
choices are recorded in `plot_figures.py`. Mathematics follows typeset
convention: italic letters, upright digits, brackets and operators.

Sources: the Solar Physics submission guidelines
(<https://link.springer.com/journal/11207/submission-guidelines>, which
require a sign-in for automated access), Springer's artwork guidelines
(<https://www.springer.com/br/authors-editors/journal-author/journal-author-helpdesk/preparation/1276>)
and Springer's figure guide
(<https://media.springer.com/full/springer-instructions-for-authors-assets/pdf/41356%20Figure.pdf>).

## Files

- `plot_figures.py`: presentation only; reads the frozen data and produces
  the eight PDFs in the manuscript's `figures/` folder, plus
  `figure_layout.json` here. Importing it writes nothing.
- `figure_data.npz`: all plotted coordinates, including the recovered
  bifurcation scan; contains no pickled objects.
- `recover_figure_data.py`: recovers archived data and regenerates missing
  coordinates using the original scientific routines.
- `numerical_checks.json`: source hashes, recovered numerical values and
  checks against the original results.
- `figure_layout.json`: exact labels, keys, key positions, sizes and font
  settings.
- `pdf_checks.json`: PDF checks and layout review of the Point 9 build.
- `pdf_checks_pa03.json`: PDF checks for both lettering weights, key
  clearances and the manuscript comparison for the PA03 restyle.

## Replot without running simulations

From the root of this repository, with NumPy, Matplotlib and the FreeSans
font (GNU FreeFont; Debian package `fonts-freefont-ttf`) installed, and a
checkout of the manuscript repository:

```bash
python3 revisions/point9/plot_figures.py --manuscript /path/to/Reduced_SP_manuscript           # revision copy, bold lettering
python3 revisions/point9/plot_figures.py --manuscript /path/to/Reduced_SP_manuscript --final   # accepted version, regular lettering
```

Both commands write the PDFs to the manuscript's `figures/` folder and record
the layout in `figure_layout.json` here; `--out DIR` writes both to `DIR`
instead. The script stops if
FreeSans is missing, if a label leaves the page, or if a key leaves its frame
or covers data. The Matplotlib version is recorded in `figure_layout.json`.
Plot dimensions are fixed rather than tightly cropped, so inclusion at
`width=\linewidth` keeps all lettering at 9 pt. Curves, symbols and text
remain vector objects.

## Recover the data

Make a checkout of the pinned solver commit
`41640e936c5d777a168ff5c0753491b840263c49` next to this repository (see
`../README.md`). Install NumPy, Pandas and Matplotlib, then run from the root
of this repository:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 revisions/point9/recover_figure_data.py --solver ../reducedSFT-41640e9
python3 revisions/point9/plot_figures.py --manuscript /path/to/Reduced_SP_manuscript
```

Single-threaded BLAS avoids overhead in the many small matrix products.
The recovery run took approximately 38 seconds in the revision environment.

Data provenance, by manuscript figure:

1. Signed dipoles read directly from the solver checkout's
   `figures_nonlinear_response_v2/nonlinear_response_results_v2.csv`. Before
   this folder moved to the code repository, the script read the copy in the
   manuscript's `figures/` folder; the two files are byte-identical (SHA-256
   `2fa2dade…`). A rerun therefore records this input under its solver path
   in `numerical_checks.json`.
2. Normalised responses read from the same CSV; the original
   `theory_normalized` routine supplies 500 theory samples per case.
3. Numerical slopes read directly from the CSV's `R_hat` column. They are
   not recalculated or smoothed.
4. SFT latitude responses read from that CSV; the four approximation
   levels are recovered with the original adjoint and closure routines.
5. Physical-shift SFT values are the independently checked Point 5
   baseline (`nmu=181`, `dt_days=1`, `protocol=legacy`, column `D`). The
   original adjoint routine, 40-year kernel extension, pair-yield fit and
   Gaussian closure routines recover the comparison curves.
6. Physical-inflow values use the same Point 5 baseline. The leading-order
   rational fits use the original positive-Q, `T<=1.2` least-squares rule.
7. The original multi-cycle routines regenerate the 34 gains, 46 SFT cycles
   per gain (last 16 retained), and 400 map iterations (last 60 retained).
   The initial amplitude, source construction and sign update are unchanged.
   The original raw coordinates were not archived: this is a rerun, not a
   digitisation or a claim of bitwise recovery. Individual trajectories in
   the modulated regime can depend on floating-point rounding. The memory
   coefficient and both analytic thresholds reproduce the archived values.
8. All 63 widths come from `figures_parameter_space/transport_constants.csv`
   (four decimal places). The curve uses the separately archived original
   quadrature coefficients `C=1.189298`, `w0=3.469989` degrees. Refitting the
   rounded widths agrees within their propagated rounding bounds. No new
   fit or transport model is introduced.

Recovery preserves the original nominal-cycle loop and trapezoidal dipole
diagnostic. It does not switch to Point 5's exact-duration or full-cell
quadrature alternatives. The known resolution caveats therefore remain.
The original source/data files were checked against their Git blobs at the
pinned solver commit and the manuscript's Point 8 commit
`fcc08285354a417d5c7de241659b98afa7258efc`.

See `revision_log.tex` in the manuscript repository, entries R9-01 to R9-04,
PA03 and PA05, for every visual change, the response to the referee and the
move of this folder.
