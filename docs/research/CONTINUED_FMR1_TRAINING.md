# Continued Fmr1 qEEG SVM training — 13 October 2026

The continuation uses the same **43 independent Fmr1-KO2/WT mice**, not additional animals or longitudinal behavioral data. It fits and saves both a selected SVM and a fixed-C=1 reference model on all animals. **The nested parameter search did not improve animal-held-out accuracy over the fixed reference.** No behavioral-onset model is trained, and LSTM/Transformer remains unimplemented.

## Data and selection protocol

The two [OSF as7cz](https://osf.io/as7cz/) source files are still pinned by SHA-256 and size. Experiment 1 supplies 20 animals; experiment 2 supplies 23, using its saline condition only. Both files are processed power tables, not continuous pre-event EEG with behavioral annotations. There are 22 KO and 21 WT animals, each represented by 198 deposited-power values: two source locations and 99 source frequency labels. Native units/frequency-label mappings are not guessed.

The grid was fixed for this continuation: linear balanced SVC with `C={0.01,0.1,1,10}`, using the full vector or PCA with 5/10 components. StandardScaler and PCA are fitted inside each inner training fold. Selection uses three shuffled folds stratified jointly by source experiment and genotype, with seed `20261013`. A candidate that cannot fit the available inner-training sample is excluded explicitly.

Outer leave-one-animal-out evaluation holds each entire animal outside parameter selection and preprocessing. The fixed-C=1/full-vector reference uses the same outer splits. Two additional evaluations hold out whole source experiments and select parameters within the other experiment only. **Both cohorts and earlier fixed-C results were already inspected; these are exploratory reuses, not new locked external tests.** No outer score is used to choose the final refit's parameters.

## Results, including unsuccessful improvement

| Evaluation | Fixed C=1, full vector | Nested selection | Independent animal membership |
| --- | ---: | ---: | --- |
| Leave one animal out | 69.91% | 67.42% | 43 predictions, each animal excluded from its own training/selection |
| Train experiment 1 → test experiment 2 saline | 57.95% | 53.79% | 20 training / 23 test animals |
| Train experiment 2 saline → test experiment 1 | 71.72% | 77.27% | 23 training / 20 test animals |

All entries are animal balanced accuracy. The fixed cross-experiment numbers are the [earlier recorded benchmark](fmr1_qeeg_genotype_summary.json). The new animal-held-out fixed reference has KO/WT confusion matrix `[[14,8],[5,16]]`; nested selection has `[[15,7],[7,14]]`. The inconsistent directions do not establish a general improvement. These small within-laboratory point estimates have no uncertainty interval, significance test or clinical/prospective validation.

For a KO/WT confusion matrix, `BA = 0.5*(correct_KO/all_KO + correct_WT/all_WT)`. Thus the nested animal score is `0.5*(15/22 + 14/21) = 67.42%`; the reference is `0.5*(14/22 + 16/21) = 69.91%`.

## Final refits and calculation

Selection on the full development set chose **C=0.01 with five PCA components**. This model and the prespecified **C=1 without PCA** reference are both refitted on all 43 animals and saved. Neither final refit is an independent test, and its training predictions are not advertised as performance.

For deposited power `Pow_j`, define `u_j=log10(Pow_j)` and `z_j=(u_j-mu_train,j)/s_train,j`. With PCA, compute `v=components*(z-PCA_mean)`. The linear SVM margin is `w·v+b`; without PCA it is `w·z+b`. The exported coefficient table and intercept also reconstruct this margin directly from `u`. Positive margins correspond to source WT, negative margins to source KO; they are not calibrated probabilities. **A genotype coefficient is not a wave abnormality threshold or a sign of an impending behavioral event.**

```bash
python -m pip install -e '.[ml]'
python scripts/continue_fmr1_qeeg_training.py
python -m unittest discover -s tests -v
```

For explicit offline reproduction after the sources have been cached, pass `--source-provenance retained_source_provenance.json`. The retained snapshot must name exactly the two pinned files, and each cached SHA-256 and byte count is rechecked. The completed final run used this mode after an OSF metadata timeout; the earlier successful source verification is retained in the execution record.

Outputs include `nested_held_out_animals.csv`, both experiment-held-out prediction/model pairs, the two final model bundles, coefficient/intercept files, `training_animal_vectors.csv` and the full split/candidate report. The compact [continued-training summary](fmr1_continued_training_summary.json) and [verification record](fmr1_continued_training_verification.json) are retained here; fitted models and full result artifacts accompany the delivered results archive.

## Behavioral workflow corrections and remaining input gap

Animal power epochs now retain their electrode locations, reference, sample rate, units and declared acquisition filters in a canonical acquisition contract. Preparation, fitting and inference require a compatible contract; inference also requires the fitted epoch duration. Old animal epoch tables missing this contract must be regenerated. Human event preparation/training rejects annotation timing/outcome fields as predictors and mixed real/synthetic scopes. Synthetic models cannot be used on real observations.

The 137-test suite includes nested split isolation, changing the outer test animal's features/label without changing its inner selection, saved model/linear-rule checks, electrode/reference/epoch compatibility and event-label leakage regressions. These software tests remain distinct from biological prediction evidence. Real behavioral fitting still requires continuous animal EEG/LFP plus independently timed behavioral events and complete synchronized follow-up. Human transfer remains unvalidated.
