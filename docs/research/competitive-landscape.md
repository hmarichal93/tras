# Competitive landscape: dendrochronology image-analysis software

Checked 2026-10-11 against vendor pages, arXiv, publishers, Zenodo, CRAN and the GitHub API.
Every row links its source. *(unverified)* marks facts from secondary sources or inference.

**Bottom line.**

- **The market leaders are semi-automatic, Windows-only tools.** These are CooRecorder/CDendro
  (US$68 per install) and WinDENDRO. Both measure along a path the user draws. Neither
  publishes a detection accuracy.
- **TRAS is the only free GUI found that detects closed rings on whole discs automatically and
  also measures cores.** It is also the only one with a reproducible public benchmark.
- **To be "best", TRAS must match the incumbents' lab workflow, not only out-detect them.** That
  means crossdating in the GUI, earlywood/latewood widths, Heidelberg export, pith offset for
  cores, blue intensity, and installing without a terminal.
- **Two CC BY 4.0 core datasets with ring annotations are public**
  ([Poláček 2023](https://zenodo.org/records/7981462),
  [Zambrano-Suárez 2026](https://zenodo.org/records/22915111)). TRAS lists them as related
  work and does not benchmark on them (PI decision 2026-10-11: TRAS is a GUI product, not a
  new algorithm). Our docs used to say no such dataset exists; fixed in T11.
- **The TRAS paper is published** in *Forestry* 99(4), cpag046,
  [doi:10.1093/forestry/cpag046](https://doi.org/10.1093/forestry/cpag046), online
  2026-07-09. `CITATION.cff` and the README cite it since T11.

## What "best on the market" has to mean

Our target users are students and lab staff who are not programmers. For them, "best" means
all of the following. The rest of this note scores TRAS against each point.

1. **Accuracy proven on public data.** Error rates on shared datasets, with a protocol anyone can
   rerun. No competitor publishes this today.
2. **Less clicking than CooRecorder for the same result.** Automatic detection, plus fast review
   and correction.
3. **Output a lab can use unchanged.** Correct `.pos`, Tucson `.rwl` and Heidelberg `.fh` files,
   calendar dates checked by crossdating, earlywood/latewood (EW/LW) widths, and pith offset.
4. **Installs and runs on a lab Windows PC with no terminal.** CooRecorder ships a Windows
   installer ([Cybis](https://www.cdendro.se/forfun/dendro/buy/index.htm)).
5. **Free and open.** This is already an advantage over every commercial tool.

## Who we compete with

### Commercial measurement software

No vendor publishes a ring-detection accuracy figure.

| Tool (vendor) | Type, platform | Price / licence | Samples | Automation | Key features | Latest release |
|---|---|---|---|---|---|---|
| [CooRecorder + CDendro](https://www.cdendro.se/forfun/dendro/lbarcden.htm) (Cybis, Sweden) | GUI, **Windows only** (.NET 3.5; Mac only under emulated Windows) | **US$68 per install**; school licence US$1,360 = 20 seats; 30-day trial capped at 80 points ([buy page](https://www.cdendro.se/forfun/dendro/buy/index.htm)) | Core and disc scans; X-ray density | **Semi-automatic.** The user clicks a path; [AutoPlace](https://www.cdendro.se/forfun/dendro/helpcoorecorder7/autoplace/index.htm) places borders from EW/LW shade patterns, and the user checks them | `.pos` coordinates; [blue intensity](https://www.cdendro.se/forfun/dendro/helpcoorecorder7/bluechannel80/index.htm); EW/LW points; [pith distance and years to pith](https://www.cdendro.se/forfun/dendro/helpcoorecorder7/distanceToPith/index.htm); CDendro crossdating, Tucson and Heidelberg export | 9.8.4, 2024-05-22 |
| [WinDENDRO](https://www.regentinstruments.com/assets/windendro_about.html) (Regent, Canada) | GUI (optionally with a scanner), Windows | Commercial; price lists on request ([ordering](https://www.regentinstruments.com/assets/orderinginformation.html)) | Discs, cores, X-ray films | **Automatic plus interactive.** Basic has 1 detection method; Reg and Density have 4 | EW/LW widths; density and blue reflectance; multi-segment paths; disc area and perimeter; batch reanalysis; Tucson; "elementary" crossdating | [2025a, 2025-01-10](https://www.regentinstruments.com/assets/windendro_mostrecent.html) |
| [LINTAB + TSAP-Win](https://rinntech.info/products/tsap-win/) (Rinntech, Germany) | Measuring stage + GUI, Windows *(unverified)* | Commercial, price on quote | Discs, cores | **Manual** (stage at 1/100 mm) | COFECHA-like segment checks; Heidelberg `.fh`, Tucson, CATRAS | Not stated |
| [LignoVision](https://rinntech.info/products/lignovision/) (Rinntech) | GUI | Commercial, price on quote | Discs, cores; optical or X-ray | "Automatic and manual" detection (method not described) | Width, density, colour profiles; `.fh` | Not stated |
| [MeasureJ2X](https://www.voortech.com/projectj2x/) (VoorTech, with Velmex stages) | Stage + GUI | Subscription, price on quote | Cores, discs | Manual | Records stage measurements | Not stated |
| [WinCELL](https://www.regentinstruments.com/assets/wincell_about.html) (Regent) | GUI | Commercial | Microsections | Automatic plus interactive | Cell anatomy per ring | Not stated |
| [ROXAS](https://wsl.ch/en/services-produkte/roxas-analysis-tool-for-xylem-anatomy) (WSL) | Plug-in for Image-Pro Plus (commercial) | Licence not found | Anatomical thin sections | Highly automatic; [deep-learning version 2025](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1516635/pdf) | Cell anatomy; >100 rings per sample | Not stated |

Two adjacent products are not image-analysis competitors. PAST4 (SCIEM) is a statistics and
crossdating package ([2009 catalogue](https://silva.nancy.hub.inrae.fr/Media/Files/francois-lebourgeois/dendrochronologie/catalog2009_system_mesure.pdf);
current status *unverified*). [LignoStation](https://rinntech.info/products/lignostation/) is a
milling, density and scanning machine.

### Free and open-source tools

| Tool | Type, licence | Samples | Automation | Features | Activity (2026-10-11) |
|---|---|---|---|---|---|
| **TRAS** (this project) | PyQt5 GUI + CLI + API, GPL-3.0 | Discs and cores | **Fully automatic** on discs (CS-TRD, DeepCS-TRD, INBD) and on core paths; manual correction | `.pos`, `.rwl`, CSV, PDF; ring area, perimeter, eccentricity; QC flags; batch | [v2.2.0, 2026-06-20](https://github.com/hmarichal93/tras/releases); 9 stars |
| [MtreeRing](https://docs.ropensci.org/MtreeRing/) (rOpenSci) | R + Shiny GUI, GPL-3 | Cores and discs, along a path | Automatic (watershed, Canny) plus manual | `.rwl` via dplR; tilted-ring correction | **[Removed from CRAN 2025-06-13](https://cran.r-project.org/package=MtreeRing)**; last push 2024-07 |
| [measuRing](https://cran.r-project.org/package=measuRing) | R library, GPL-3 | Scans along a path | Automatic grey-profile borders plus manual edits | Ring widths | CRAN 0.5.2, 2025-11-03 |
| [xRing](https://rdrr.io/cran/xRing) (Campelo) | R + Tcl/Tk GUI | X-ray density profiles | Automatic borders, EW/LW (`detectEwLw`), manual correction | Density, EW/LW | CRAN 0.1.2, 2025-11-11 |
| [WIAD](https://www.biorxiv.org/content/10.1101/2020.12.16.423133v1) | R Shiny web app, AGPL-3 | Cores, thin sections, X-ray | Manual | Widths, EW/LW, density, image archive | Last push 2024-07 |
| [DendroElevator](https://dendro.elevator.umn.edu) (Univ. Minnesota) | Web service, open source | Gigapixel core and disc scans | Manual in the browser | Measuring, dating, annotation; >10,000 images (2022, *unverified*) | Toolkit last push 2026-02 |
| [Fritts](https://pypi.org/project/fritts/) | PyQt6 desktop app, MIT | Measured series; image import | Projection-profile "auto-detect rings" | Crossdating, COFECHA-style QC, detrending, chronologies; Tucson, TRiDaS, Heidelberg | **New**: 0.1.1 (2026-06-05) to 0.3.0 (2026-07-01) |
| [Tree Ring Analyzer / napari-tree-rings](https://github.com/MontpellierRessourcesImagerie/tree-ring-analyzer) (MRI Montpellier) | Python + napari plugin, MIT | Discs | Automatic bark, pith and rings | Ring properties | Last push 2025-12; no published accuracy |
| [TRG-ImageProcessing](https://github.com/Gregor-Mendel-Institute/TRG-ImageProcessing) (Poláček et al.) | Python CLI / Docker, MIT | Conifer cores | **Fully automatic** (Mask R-CNN) | Widths; writes `.pos` | v1.0.2 (2023-05); last push 2026-08 |
| [Tellervo](https://github.com/ltrr-arizona-edu/tellervo) (LTRR) | Java desktop + database, GPL-3 | Stage measurements | Manual | Lab database, crossdating, TRiDaS | v2.1.8, 2026-09-24 |
| [dplR](https://cran.r-project.org/package=dplR) / [xDateR](https://github.com/OpenDendro/xDateR) / [dplPy](https://pypi.org/project/dplpy/) | R / Shiny / Python | Ring-width series only | — | Crossdating, detrending, chronologies; the de facto analysis stack | dplR 1.8.0, 2026-10-01 |

[TRiCYCLE](https://www.sourceforge.net/projects/tridas) only converts between dendro file formats
([Brewer et al. 2011](https://doi.org/10.3959/2010-12.1)). [CaptuRING](https://github.com/CambiumRG/CaptuRING)
is a DIY imaging rig. Neither is a measurement competitor.

### Names in the brief that we could not find

Treat these as non-existent until someone shows a source: CybisElevator, DendroTool (only
the unrelated R package dendroTools), Dendro2D, iDendro, Kartal, CoreProcessor, TreeRingCV,
TreeRingNet, RingIT, QuantaCell, PAST5, DendroScan. The ImageJ
[ObjectJ "TreeRings" example](https://sils.fnwi.uva.nl/bcb/objectj/examples/TreeRings/TreeRings-9.htm)
exists, but it is a marking project for otoliths and rings, not a detector.

## State of the art in automatic detection

The numbers below use **different metrics**, and a number is only meaningful with its
protocol. Read "Comparing TRAS numbers" before quoting any of them next to ours.

### Cross-sections (discs)

| Method | Data | Metric | Best reported | Code |
|---|---|---|---|---|
| Kennel et al. 2015 ([HAL](https://hal-lirmm.ccsd.cnrs.fr/lirmm-01275680)) | *Abies alba* slices | F vs expert | F 0.91 | Not found |
| CS-TRD ([IPOL 2025](https://www.ipol.im/pub/art/2025/485/), [arXiv 2305.10809](https://arxiv.org/abs/2305.10809)) | UruDendro (64), Kennel | Influence-zone P/R/F, **≥ 60 %** of nodes | F 0.89 (arXiv) / 0.91 (IPOL) on UruDendro; 0.97 Kennel | MIT |
| INBD ([CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Gillert_Iterative_Next_Boundary_Detection_for_Instance_Segmentation_of_Tree_Rings_CVPR_2023_paper.html), [arXiv 2212.03022](https://arxiv.org/abs/2212.03022)) | Shrub microscopy, 213 images | mAR (IoU .5:.95), ARAND | mAR 0.553–0.738 per species | MPL-2.0; data CC BY-NC-SA |
| DeepCS-TRD ([ICIAP 2025](https://doi.org/10.1007/978-3-032-10185-3_3), [arXiv 2504.16242](https://arxiv.org/abs/2504.16242)) | UruDendro 1/2/3a, Salix | mAR, ARAND, ground-truth pith | mAR 0.884 on UruDendro1 *(from the paper tables, read by a sub-agent)* | MIT |
| UruDendro4 baseline ([arXiv 2511.20935](https://arxiv.org/abs/2511.20935)) | 102 *P. taeda* discs | mAP/mAR (IoU .5–1), ARAND, APD pith | DeepCS-TRD mAP 0.838, mAR 0.782, ARAND 0.084 | Data CC BY 4.0 |
| Wu et al. 2024 ([doi:10.1016/j.inpa.2023.10.002](https://doi.org/10.1016/j.inpa.2023.10.002)) | 136 hardwood cookies, 11 species | Ring F1, overlap > 25 % of width, 5-fold CV | F1 0.735 | MIT |
| Kim et al. 2023 ([*Dendrochronologia* 79:126088](https://research.knu.ac.kr/en/publications/method-for-detecting-tree-ring-boundary-in-conifers-and-broadleaf/)) | 300 images, 4 species; scanner and smartphone | Boundary detection rate | 88–93 % scanner, 79–86 % smartphone (disc or core *unverified*) | Not found |
| TRAS paper ([*Forestry* 2026](https://doi.org/10.1093/forestry/cpag046)) | 18 *P. taeda* discs, APD pith | Influence-zone P/R/F, **≥ 90 %** of nodes | DeepCS-TRD F 0.810 (P 0.864) | GPL-3.0 |

### Increment cores

| Method | Data | Metric | Best reported | Code / data |
|---|---|---|---|---|
| Fabijańska et al. 2017 ([doi:10.1016/j.compag.2017.06.006](https://doi.org/10.1016/j.compag.2017.06.006)) | Cores, 12 species | Boundary detection rate | ~100 % conifers, ~85 % diffuse-porous | Not released |
| DeepDendro, Fabijańska & Danek 2018 ([doi:10.1016/j.compag.2018.05.005](https://doi.org/10.1016/j.compag.2018.05.005)) | Ring-porous cores | P/R | P 0.97, R 0.96 *(unverified: quoted via [CS-TRD](https://arxiv.org/abs/2305.10809))* | Not released |
| Poláček et al. 2023 ([MEE 14:2233](https://doi.org/10.1111/2041-210X.14183)) | 34 *P. abies* cores, 4,355 rings, plus 3 other conifers | Ring count vs COFECHA-dated series | **R 0.987, P 0.948 (strict) / 0.969 (permissive)** | MIT; data CC BY 4.0 |
| Zambrano-Suárez et al. 2026 ([PLOS ONE 21:e0321841](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0321841)) | 64 *P. sylvestris* cores from disturbed trees, 11,947 rings | Pixel Dice; instance F1 at IoU > 0.5 | Dice 0.772; recall drops to 0.59 for rings 100–200 µm | Code and data CC BY 4.0 |
| TRAS core detector ([benchmarks](../benchmarks.md)) | 16 CooRecorder radii on 4 *Pinus* discs | One-to-one match within 6 px | P 0.965, R 0.896, F1 0.929 | GPL-3.0 |

### Public datasets with ring annotations

| Dataset | Content | Licence | Link |
|---|---|---|---|
| UruDendro | 64 *P. taeda* discs, 1,221 rings, pith | CC BY 4.0 | [Zenodo 15110647](https://zenodo.org/records/15110647) |
| UruDendro2 / 3a | 53 *P. taeda*; 9 *Gleditsia* discs | CC BY 4.0 | [15652452](https://zenodo.org/records/15652452), [15652668](https://zenodo.org/records/15652668) |
| UruDendro4 | 102 *P. taeda* discs at several stem heights | CC BY 4.0 | [Zenodo 15653340](https://zenodo.org/records/15653340) |
| INBD MiSCS | 213 shrub microscopy images | **CC BY-NC-SA 4.0** (non-commercial) | [INBD repo](https://github.com/alexander-g/INBD) |
| Wu et al. hardwood | 136 cookies, 19,898 rings | Not stated | [growth-ring-detection](https://github.com/wufanyou/growth-ring-detection) |
| **Poláček cores** | *P. abies* core patches with >8,000 hand-annotated boundaries; full cores of 4 conifers with measurement tables (no per-ring coordinates *unverified*) | CC BY 4.0 | [Zenodo 7981462](https://zenodo.org/records/7981462) |
| **Zambrano-Suárez cores** | *P. sylvestris* cores with binary ring masks, 7.7 GB | CC BY 4.0 (open since 2026-09-23) | [Zenodo 22915111](https://zenodo.org/records/22915111) |

### Comparing TRAS numbers with published ones

| TRAS number | Published number | Comparable? | Why |
|---|---|---|---|
| `tras_benchmark` CS-TRD F1 0.904, UruDendro test split (32 discs), ≥ 60 % of 360 rays | CS-TRD F 0.89–0.91, all 64 UruDendro discs, ≥ 60 % of nodes | **Roughly.** Same data source and threshold | Half the discs; rays instead of chain nodes. A good sanity check, not a claim |
| `tras_benchmark` DeepCS-TRD F1 0.947 (same split) | TRAS paper F 0.810, 18 discs | **No** | The paper needs ≥ 90 % of nodes in the zone, we need ≥ 60 %. Other images, APD pith vs ground-truth pith |
| Same F1 0.947 | DeepCS-TRD mAR 0.884; UruDendro4 mAR 0.782 | **No** | Region IoU vs boundary zone. The released generic weights were trained on the DeepCS-TRD train/validation discs only, not its test set (H. Marichal, 2026-10-11). Our even/odd split is a different partition, so some of our test discs may be in that training data |
| Core F1 0.929 (6 px, 16 radii) | Poláček P 0.948–0.969, R 0.987 | **No** | Count-based against dated series vs position-matched; spruce vs pine; 34 cores vs 16 radii |

**Do not write "TRAS beats Poláček" or "DeepCS-TRD reaches 95 %" anywhere without these
qualifiers.** A fair head-to-head would need the TRAS core detector run on the Poláček and
Zambrano-Suárez data (recommendation 2, declined by the PI on 2026-10-11).

## Gap analysis: what users expect versus what TRAS has

Status comes from the code on branch `claude-full-autonomus` (2026-10-11) and
[the roadmap](../product-roadmap.md). Priority is for the goal "best for non-programmers".

| Capability | Who offers it | TRAS status | Priority | Reasoning |
|---|---|---|---|---|
| Automatic closed rings on discs | WinDENDRO (path-based), napari plugin (no accuracy) | **Has**: 3 detectors, benchmarked | Keep | Our clearest lead |
| Automatic detection on cores | CooRecorder AutoPlace, WinDENDRO, TRG-ImageProcessing | **Has** (`tras/cores/`, `tras_core`) | Keep | P 0.965 / R 0.896 on CooRecorder radii |
| Correct, undo, review | All GUIs | **Has** (canvas edits, undo); core edit export in T2 | P1 | T2 finishes it |
| Crossdating (r, t, GLK, COFECHA-style segments) | CDendro, TSAP-Win, Fritts, xDateR | **Partial**: `tras/analysis/dating.py` exists but no CLI or GUI (T3) | **P1** | Dating is why labs measure. Without it TRAS output still goes through CDendro or dplR |
| Tucson `.rwl` | All | **Has** (read and write) | — | |
| CooRecorder `.pos` | CooRecorder, TRG-ImageProcessing | **Has** (double points, gap markers read) | — | |
| Heidelberg `.fh` | TSAP-Win, LignoVision, CDendro, Fritts | **Missing** | **P1** | The standard in German/European labs; cheap to add in `tras/io/` |
| TRiDaS / CATRAS | Tellervo, TSAP-Win, Fritts | Missing | P3 | dplR and TRiCYCLE convert from `.rwl` |
| EW/LW widths and areas | CooRecorder, WinDENDRO, xRing, WIAD | **Partial**: CS-TRD can find EW/LW borders on the [`early_late_wood` branch](https://github.com/hmarichal93/cstrd/tree/early_late_wood) (last commit 2024-10-05), not in TRAS; `.pos` reader parses `#W#` points | **P1** | Standard output for climate work; the *Forestry* paper already claims EW/LW areas |
| Blue intensity / density | CooRecorder, WinDENDRO Density, xRing | Missing | P2 | CooRecorder is the reference BI tool ([Rydval et al. 2014](https://doi.org/10.1016/j.dendro.2014.04.003)). Needs calibration and a domain review |
| Missing / locally absent rings, cracks | CooRecorder, TSAP-Win | **Partial** (`.pos` gap reading; T6 next) | P2 | Common in damaged cores |
| Incomplete outer ring | CooRecorder | Missing (T5) | P2 | Wrong dating if counted |
| Pith offset for cores (distance and years to pith) | CooRecorder, modendro | **Partial**: read from `.pos`, not estimated | P2 | Needed for age and RCS standardisation |
| Multiple radii per sample | WinDENDRO, CooRecorder | **Has** on discs (compass radii, custom paths); several core paths per image | — | |
| Ring area, perimeter, eccentricity (2D) | WinDENDRO (area/perimeter) | **Has** | Keep | Unique among free tools |
| QC flags | CDendro/COFECHA (statistical) | **Has** in reports; canvas display T4 | P2 | |
| Batch | WinDENDRO, TRG-ImageProcessing | **Has** (discs GUI and CLI; `tras_core` folders) | — | |
| Metadata (sample code, harvest year) | All | **Has** | — | |
| PDF report | WinDENDRO (reports *unverified*) | **Has** | — | |
| Install without a terminal | CooRecorder, WinDENDRO | **Missing** (T9) | **P1** | Today Windows users need Anaconda Prompt (README). This is the largest barrier for students |
| User manual for novices | CooRecorder help pages | **Missing** (T8; `docs/manual/` holds screenshots only) | **P1** | |
| Very large scans (gigapixel) | DendroElevator, CooRecorder | Unknown *(untested)* | P2 | Core scans at 2,400+ dpi are large |
| Chronology building, detrending | dplR, Fritts, CDendro | Missing | P3 | Leave to dplR; export clean `.rwl` |
| Anatomy (cells) | ROXAS, WinCELL | Missing | Out of scope | Different user group |

## Top 10 recommendations

Ranked by user value divided by effort. Effort: S = days, M = 1–3 weeks, L = more than a month.

| # | Recommendation | Value | Effort | Evidence |
|---|---|---|---|---|
| 1 | **Fix the record**: cite the *Forestry* version of record in `CITATION.cff` and README; settle the licence (paper says MIT, repo is GPL-3.0); correct the "no public core dataset" claim in `benchmarks.md` and `core-detection-sota.md` | Credibility with reviewers and users | S | [OUP page](https://academic.oup.com/forestry/article-abstract/99/4/cpag046/8729338); [paper §3.1, "MIT license"](https://arxiv.org/abs/2605.08025); repo `LICENSE`; Zenodo [7981462](https://zenodo.org/records/7981462), [22915111](https://zenodo.org/records/22915111) |
| 2 | *Declined by the PI 2026-10-11 (TRAS is a GUI product, not a new algorithm).* **Benchmark the core detector on the two public core datasets** (Zambrano-Suárez masks → boundary positions; Poláček patches and dated measurement tables). Add them to `tras_benchmark` | First public, comparable core numbers; a fair test against Mask R-CNN | M | Both CC BY 4.0; no competitor GUI reports core accuracy |
| 3 | **Windows installer with no terminal** (T9) | Unblocks the target user | M | CooRecorder installs from an `.exe`; README requires Anaconda Prompt |
| 4 | **Crossdating in the GUI and CLI** (T3), reusing `tras/analysis/dating.py` | Dated, checked series inside TRAS; removes the trip to CDendro | M | CDendro, TSAP-Win, Fritts and xDateR all offer it |
| 5 | **Heidelberg `.fh` reader and writer** in `tras/io/` | European labs can import TRAS output directly | S | `.fh` is the native format of [TSAP-Win](https://rinntech.info/products/tsap-win/) and [LignoVision](https://rinntech.info/products/lignovision/) |
| 6 | **EW/LW boundaries and widths**: port the CS-TRD [`early_late_wood` branch](https://github.com/hmarichal93/cstrd/tree/early_late_wood); add EW/LW to cores (`#W#` points in `.pos`) and to the `.rwl` export (separate EW and LW files, as dplR expects) | Climate studies need EW/LW; makes the paper's claim true in code | M | CooRecorder, WinDENDRO, xRing all provide it; the [paper](https://doi.org/10.1093/forestry/cpag046) claims EW/LW areas |
| 7 | **User manual for novices** (T8) | Adoption without help | M | Competitors ship help pages; ours has screenshots only |
| 8 | **Core-lab completeness**: incomplete bark ring (T5), missing rings and cracks (T6), pith-offset estimate (distance and years to pith) | Correct dating and age estimates on real cores | M | CooRecorder [pith help](https://www.cdendro.se/forfun/dendro/helpcoorecorder7/distanceToPith/index.htm); modendro `yrs_to_pith` |
| 9 | **Report the cross-section benchmark also under the paper's 90 % rule and on UruDendro4** (mAR/ARAND too), and state the DeepCS-TRD training overlap | One table reviewers can check against our own papers | S | `tras/evaluation/cross_section.py` uses 0.6; the paper uses 0.9; DeepCS-TRD weights trained on its train/val split, a different partition from ours (stated in `docs/benchmarks.md` since T11) |
| 10 | **Blue intensity along core paths** (mean of the darkest latewood pixels, CooRecorder-style settings), after a domain review | Matches the reference BI tool for free | L | [CooRecorder BI](https://www.cdendro.se/forfun/dendro/helpcoorecorder7/bluechannel80/index.htm); [Rydval 2014](https://doi.org/10.1016/j.dendro.2014.04.003) |

Not recommended now: chronology building (dplR does it well), anatomy (ROXAS/WinCELL serve
another community), TRiDaS.

## Errors found in current TRAS material

| Statement | Where | Status | Source |
|---|---|---|---|
| "No public increment-core dataset with ring annotations exists (2026-10)" | `docs/benchmarks.md`, `docs/research/core-detection-sota.md` | **Fixed** (T11) | [Zenodo 7981462](https://zenodo.org/records/7981462), [22915111](https://zenodo.org/records/22915111) |
| Paper cited as arXiv only, "accepted" | `CITATION.cff`, `README.md` | **Fixed** (T11): published 2026-07-09 | [doi:10.1093/forestry/cpag046](https://doi.org/10.1093/forestry/cpag046) |
| "Open source under the MIT license" | *Forestry* paper §3.1 | **Conflicts** with GPL-3.0 `LICENSE` | [arXiv v2](https://arxiv.org/abs/2605.08025) |
| TRAS computes EW and LW areas | *Forestry* paper abstract, §4.3 | **Not in the code** on this branch or `main` | `git grep` 2026-10-11 |
| "Zambrano-Suárez et al. 2025, PLOS ONE"; "Zenodo access restricted" | `core-detection-sota.md` | **Fixed** (T11): published 2026-06-18; data open since 2026-09-23 | [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0321841), [Zenodo](https://zenodo.org/records/22915111) |
| Poláček "P 0.95–0.97, R 0.99"; "data not released" | `core-detection-sota.md` | Numbers fine (R 0.987); **data is released** (CC BY 4.0); **fixed** (T11) | [Zenodo 7981462](https://zenodo.org/records/7981462) |

## Open questions for the authors

1. Licence: is TRAS GPL-3.0 (repo) or MIT (paper)? Should the paper get an erratum, or the repo
   change licence? The vendored code also mixes licences (INBD MPL-2.0, CS-TRD MIT, and an AGPL
   notice in the `cstrd` `early_late_wood` README).
2. *Answered 2026-10-11 (H. Marichal):* the released DeepCS-TRD generic weights were trained on
   the DeepCS-TRD train/validation split only. Follow-up: which discs are in that split? With the
   list, the UruDendro benchmark could report only discs the weights never saw.
3. Which code produced the EW/LW areas in the paper (Figures 5 and 7)? Can it be merged?
4. Is the Kennel *Abies alba* dataset public?
5. Do the Poláček real-world cores include per-ring coordinates (`.pos`)? If not, use ring
   counts and dated widths as ground truth.

## Method and limits of this survey

Three sub-agents searched vendor pages, GitHub, CRAN, Zenodo and arXiv. The author re-checked
the key claims: CooRecorder price and platform, WinDENDRO version, MtreeRing's CRAN removal,
Tellervo's release, Fritts on PyPI, both core datasets on Zenodo, the UruDendro4 abstract, the
TRAS paper text and its *Forestry* record. Not re-checked: DeepCS-TRD per-dataset mAR values,
DeepDendro's P/R (the publisher page was blocked), CDendro's exact crossdating statistics
(cybis.se is behind a bot check), and citation counts (not collected).
