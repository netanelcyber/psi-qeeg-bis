# PR #3: verified methodology corrections

**13 October 2026.** This update addresses the nine active, non-outdated review threads at parent commit `bea3d0a631be1e1da8c34b1c0e63afcbde4a9f96`. Earlier outdated threads remain historical; they are not automatically treated as current defects. No abnormal-behavior model was fitted on real event data.

## Corrected behavior and regressions

| Review concern | Enforced behavior | Regression evidence |
| --- | --- | --- |
| Animal electrodes/reference | Preserve one canonical physical acquisition contract through epochs, preparation, fitting and inference; reject changed sites/reference | `test_animal_workflow.py`: metadata retention, reference/site mismatch and epoch-duration checks |
| Human acquisition mismatch | Every epoch/prepared window/manifest carries sites, verified reference, normalized units, sample rate, scope, duration, preprocessing and QC; missing/mixed/mismatched contracts fail | `test_events.py`: known-waveform receipt, mismatches, model reload and legacy rejection |
| Forbidden animal outcome spelling | Strip and case-fold reserved human/genotype behavior names before checking | `test_animal_workflow.py`: `Meltdown`, `GENOTYPE` and padded mixed-case variants |
| Unequal human recording duration | Scaler mass 1 per participant; SVC weights also balance classes by participant counts, computed inside every training fold | `test_ml.py`: equal person/class mass, all LOSO/refit weighting calls, exact epoch-replication invariance |
| EEG/LFP pooling or transfer | Exactly one signal scope per animal model; persist it and compare during inference | `test_animal_workflow.py`: mixed-scope preparation and LFP-to-EEG inference rejection |
| Looser inference QC | Persist numerical QC method/thresholds and require a fitted-policy match for human and animal inference | `test_events.py`, `test_animal_workflow.py`: changed amplitude/flat thresholds and incomplete legacy contracts |
| Wrong surrogate count in report | Use the actual configured count in computation, protocol, PAC rows and report; CLI supports `--surrogates` | `test_ibl_runners.py`: complete mocked runner at nondefault count 19 |
| Inclusive one-second shifts | Candidates strictly exceed one second in either circular direction; save chosen offsets/seeds for audit | `test_coupling.py`: forced endpoint sampling, distinct-shift limit, seeded reproducibility and known modulated/uncoupled signals |
| Zero-duration wheel annotation | Require end > start and ordered start times before export | `test_ibl_runners.py`: zero/negative-duration, missing and unordered intervals rejected |

Human event dataset manifests are version 2. Animal manifests are version 3, with acquisition contracts version 2. Legacy tables/models without the complete contract must be regenerated and refitted from documented inputs; missing metadata is not guessed. Contracts enforce agreement of supplied records and do not independently certify electrode placement or expert artifact review.

## Real recomputation and independent evidence

The public IBL source was downloaded again. The 240-second, 2.5-kHz four-channel LFP and video movement annotations retain their exact prior SHA-256 hashes. The corrected alignment receipt differs only in the source-validator fingerprint. All **143 unique windows, 199 surrogates per relationship and 2,288 PAC tests** were rerun, with 429 saved window/mode offset receipts. PLV/coherence and observed MI do not depend on the changed shift pool; empirical p/BH q values were recomputed. Rejection counts remain **273 original / 190 notch / 8 bipolar**, while first-onset bipolar local minimum q changed to **0.241350 before / 0.138069 after**. The 199-surrogate plus-one p resolution is 0.005. Overlapping tests remain dependent and these BH-adjusted results are exploratory.

The independent verifier checks all saved modulation indices using relative entropy, all BH values against SciPy, and one real PAC/null against independent binned means and the actual saved offsets. It also checks independent real coherence, strict exclusion distances, distinct offsets, window/cutoff boundaries, unique multiplicity membership and report fingerprints. [Current coupling receipt](ibl_coupling_summary.json) and [verification](ibl_coupling_verification.json) replace the [historical inclusive-shift receipt](ibl_coupling_summary_inclusive_history.json).

All 28 public RepOD EDFs were verified and analyzed again: **7,205 complete / 6,870 accepted epochs, 28 independent people**. Corrected subject balanced accuracies are **78.57% full scalp / 50% Fp1/F7 / 25% Fp2/F8**. Equal person/class weight totals, weighted scaler means, train/test subject separation, saved-model reloads and subject metrics/confusion matrices passed independent checks. See [current human results](EXPANDED_HUMAN_COHORT.md), [current receipt](expanded_human_cohort.json) and [historical unweighted receipt](expanded_human_cohort_unweighted_history.json). The participant-level weight normalization also changes SVM regularization relative to epoch-pooled training; no improvement or pure duration effect is claimed.

The complete local suite passes **159 tests, zero failures/errors/skips** (15 more than the reviewed 144-test suite), including ten estimator regressions. The real VitalDB example, Python compilation and diff checks pass. [methodology_corrections_verification.json](methodology_corrections_verification.json) records the result checks and exact implementation fingerprints.

```bash
python -m pip install -e '.[edf,ml]' ibl-neuropixel
python scripts/validate_public_ibl_video_lfp.py
python scripts/analyze_ibl_coupling.py --surrogates 199 \
  --movement-intervals data/model_organism/ibl_sh023/alf/_ibl_wheelMoves.intervals.d8a0c780-a2b9-4b5b-9c41-d4665f0333db.npy \
  --out outputs/ibl_coupling_corrected
python scripts/validate_public_schizophrenia_cohort.py --window-s 0 --epoch-s 4 \
  --discard-waveform-csv --out outputs/repod_subject_balanced
python scripts/verify_methodology_corrections.py
python -m unittest discover -s tests -v
python scripts/verify_example.py
```

IBL extraction also needs system `ffmpeg`/`ffprobe`. Raw downloads, waveform/video exports and fitted models remain reproducible run artifacts; only aggregate evidence is committed.

## What each result supports

| Result | Supported interpretation | Evidence still needed for further claims |
| --- | --- | --- |
| PLV, coherence, PAC | Descriptive synchrony/coupling in one LFP excerpt; reference/filter sensitivity and a timing-null test | Independent animals, verified reference, artifact adjudication, replication and suitable dependency/null controls |
| RepOD participant-held-out SVM | Exploratory schizophrenia/control diagnosis-group differences within one cohort | Participant-level uncertainty and locked independent-site/prospective validation |
| Encoder/video-aligned windows | Signal descriptors associated with measured task wheel movements; some pre-bout windows overlap prior movement | Independently adjudicated abnormal-event definitions/onsets, body-motion/artifact review and separate episodes/animals |
| Human/animal future-event code | Engineering support for past-only windows, independently labelled targets and compatible acquisition | Genuine event/follow-up labels, external/prospective event-level sensitivity, false alerts/hour, lead time and calibration |

No psychosis-attack or autistic-meltdown forecasting capability, minute-ahead warning threshold, clinical probability, SOI/RDI validation or animal-to-human transfer is established by these data. Synthetic fitting regressions test code behavior only.
