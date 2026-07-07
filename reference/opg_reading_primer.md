# OPG reading primer — in-context reference for §5.5 / E11 (provenance)

**Live artifact:** [`opg_primer.txt`](opg_primer.txt) — the exact text prepended to each question.
**Versioned snapshots:** [`versions/`](versions/) — immutable copies (`opg_primer_v1.txt`, …). Bump on each rebuild.
**Current version: v1** (2026-07-07).

The primer is a faithful paraphrase of the sources below; no source prose is copied
verbatim. Facts and the systematic-reading checklist are used as ideas and cited in the
methods. The full source PDFs are archived in the **private** research repo under
`reference_papers/` (see `dental_research/CONTEXT_ARTIFACTS.md`).

## Sources used in v1

| Tag | Citation | License | Contributes |
|-----|----------|---------|-------------|
| **[1]** | Różyło-Kalinowska I. *Panoramic radiography in dentistry.* Clin Dent Rev 2021;5:26. doi:10.1007/s41894-021-00111-4 | **CC BY 4.0** ✓ (verified in PDF) | OPG scope, focal trough, ghost/double shadows, artefacts, distortion, limitations (calibration) |
| **[2]** | Kazimierczak W, et al. *Periapical Lesions in Panoramic Radiography and CBCT Imaging.* J Clin Med 2024;13(9):2709. doi:10.3390/jcm13092709 | **CC BY** (MDPI) ✓ | periapical lesion appearance, differential, OPG detection limits |
| **[A]** | Huettig F, Axmann D. *Reporting of dental status from full-arch radiographs.* World J Clin Cases 2014;2(10):552–564. doi:10.12998/wjcc.v2.i10.552 | freely available; PDF says "© Baishideng, all rights reserved" — **used as factual checklist only, paraphrased + cited** | systematic reporting checklist: teeth/implants/caries/fillings/crowns/RCF/apical health (PAI)/alveolar bone level (thirds) |
| **[B]** | FDI World Dental Federation two-digit notation (ISO-3950) | public standard | quadrants 1–4, tooth positions 1–8 |

**Dropped:** Choi et al. BMC Med Educ 2025 (s12909-025-07829-w) — an education-assessment study with no radiographic descriptions (names only); not useful.

## What v1 covers vs. the benchmark
Teeth (FDI + count/impacted), HisT (implants/fillings/crowns/RCF), Patho (caries + periapical),
Jaw (bone level + sinus/canal scope), plus calibration (focal trough, ghosts, artefacts).
**Known thin spots (targets for v2):** Jaw visual detail (mandibular canal, maxillary sinus,
bone-loss patterns) and caries/abscess *appearance* (v1 has criteria, not the look).

## v1 result (E11 screen — go/no-go)
Enriched paired screen, gemini-3.5-flash, shuffled key, 130 items (100 misses + 30 controls):
no-context 23.1% → +primer 40.8%; **rescued 26 / broke 3 (McNemar p<0.001)**; rescue rate 26%
of misses vs break rate 10% of controls (signal +16 pts). Concentrates in Teeth/HisT/Patho,
weak on Jaw. Canary (FDI/count) rescued 4/20 — benefit is diffuse systematic reading, not
answer-lookup. Verdict: real signal, worth a rigorous round 2 (placebo control, full set,
break-rate pinning, more models). Generator: `paper_analysis/knowledge_context.py`.
