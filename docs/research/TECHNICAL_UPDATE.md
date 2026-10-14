# PSI: human EEG and event-prediction status

**Netanel Stern | Updated 14 October 2026 | Technical research update**

PSI runs an offline FFT-based qEEG pipeline and optional linear SVM research workflows. The full human EEG experiment below predicts diagnosis groups. A separate event workflow requires independently timed outcomes; it has engineering tests but no real event-trained model. **No LSTM/Transformer is implemented or trained, and no imminent-attack threshold is validated.** Reviewable changes: [PR #3](https://github.com/netanelcyber/psi-qeeg-bis/pull/3).

## Implemented signal processing

EEG is represented in microvolts with an explicit sample clock. Default epochs are four seconds without overlap. Welch PSD uses two-second Hann windows, 50% overlap, constant detrending and a 0.5 Hz frequency grid. Exact-edge integration gives delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30 and gamma 30–40 Hz powers and relative powers. Delta/alpha ratio, entropy and SEF95 are computed. Missing/flat/extreme-amplitude epochs are rejected; EMG contamination is flagged rather than removed.

Experimental single-channel auto-bicoherence screens the complete past 60 seconds as one window; constituent epochs are not individually required to pass QC; verified-montage alpha PLV is also implemented. Both are absent from this cohort's SVM features. Raw analysis keeps classification indeterminate and SOI/RDI unavailable without all components and a valid personal baseline. FFT describes the waveform; its output is not a psychiatric diagnosis.

## Clinical and scalp measurement correction

The [clinical requirements](CLINICAL_REQUIREMENTS.md) and [new perspective/proposed protocol](PSI_CLINICAL_PERSPECTIVE.md) now make clinical assessment the reference standard and require evaluation of added EEG value against a clinical-only comparator. Context and duration support assessment but do not supply a sole diagnostic cutoff; uncertain and mixed presentations must remain possible. No numerical SOI/RDI output establishes reality testing.

Scalp gamma and scalp theta–gamma PAC are excluded from the proposed primary human feature protocol, pending independent EMG/EOG/movement measurements and expert review for any secondary analysis. Existing gamma output is a waveform descriptor, not a verified neural marker. The diagnosis SVM does not explicitly select gamma power, but its entropy/SEF95 inputs include the high-frequency passband and are not automatically artifact resistant. Intracranial LFP results remain separate from scalp EEG claims. This is an interpretation/protocol correction; previous benchmark data have not been silently refiltered.

## Expanded human-data experiment

Public RepOD v1.0 (CC0) contains 14 schizophrenia and 14 control participants, 19 channels, 250 Hz. All 28 source checksums/sizes were verified. Full recordings provide 28,863 seconds, 7,205 four-second epochs and 6,870 accepted epochs, compared with 840/821 previously. There are still 28 independent people.

| Coverage | Corrected entire-recording subject balanced accuracy |
| --- | ---: |
| Full scalp | 78.57% |
| Fp1/F7 | 50.00% |
| Fp2/F8 | 25.00% |

These are subject balanced accuracies from corrected leave-one-subject-out evaluation. Each training fold fits a StandardScaler with total weight 1 per participant, then linear SVC (`C=1`, participant-weighted and classes balanced by participant counts). Four descriptors per channel are used: alpha power, delta/alpha ratio, SEF95 and entropy. Epoch majority voting yields subject predictions. Coverage sets share accepted epochs; final model refits are not test results. The earlier epoch-pooled regional scores of 67.86%/39.29% are historical; the normalization also changes regularization, so this is a corrected training protocol rather than a pure duration comparison. These small, single-site estimates lack uncertainty intervals and external/prospective validation. See [current results and prior-run boundaries](EXPANDED_HUMAN_COHORT.md).

## Literature candidates with explicit publication status

| Candidate | Evidence boundary | Current code |
| --- | --- | --- |
| Shorter alpha bursts / lower time occupancy [1] | ASD resting-state association; preprint, no future meltdown test | Alpha power available; burst estimator absent |
| Posterior gamma increase with eyes open [2] | Visual-condition comparison; conference abstract | 30–40 Hz only versus study 30–100 Hz; no condition contrast |
| Beta/gamma source patterns [3] | Later psychosis transition over at least three years | Scalp power available; source CSD absent |
| Reduced duration MMN [4] | Future conversion, median follow-up 32 months | Human stimulus-locked MMN absent; not an FFT band |

The first two rows are preliminary sources, kept separate from full journal clinical evidence. These findings identify candidates to investigate, not an abnormal-wave cutoff indicating an attack in the next few minutes. Feature annotations mark estimator mismatch and unavailable individual thresholds. No EEG deviation is used to manufacture an event label.

## Event SVM and future LSTM requirements

The separate event workflow computes mean, SD and slope from complete past-only histories and trains two binary linear SVMs for future meltdown and psychosis onset. Independent onset/offset annotations and complete reviewed follow-up are required; psychosis outcomes need clinical adjudication. Current events, insufficient lead and missing follow-up are censored. Scaling/evaluation are subject-separated. The 60-second history/300-second horizon defaults are illustrative, not demonstrated warning times. Timing, leakage, reload and CLI tests use labelled synthetic fixtures only. RepOD lacks onset/follow-up labels, so no real event model or probability forecast was produced.

LSTM development requires synchronized longitudinal EEG/context, independently adjudicated events, a prespecified sequence/horizon/missingness contract, implemented architecture/training/inference and versioned weights. Compare with temporal/SVM baselines on locked external/prospective data, reporting event sensitivity, false alerts/hour, lead time, calibration and uncertainty. Longer resting recordings alone do not satisfy this requirement.

## Preclinical route clarified 13 October

The intended next prediction stage uses continuous model-animal iEEG/LFP and independently annotated animal behavioral-event onsets. The animal qEEG/annotation/SVM adapter is now implemented and engineering-tested; no real animal behavioral-onset model is trained. A separate real Fmr1 qEEG genotype benchmark uses 43 animals (57.95% primary and 71.72% reverse-direction balanced accuracy). Existing ERP and deposited-power data lack behavioral-onset annotations. The human profile ends at 40 Hz; the separate animal profiles cover 1–80/1–100 Hz. Custom 60–100 Hz high gamma and 150–200 Hz HFO can be integrated in the standalone example, but the animal CLI does not estimate HFO or reproduce the post-dose MK-801 study's full estimator. See [implemented animal model and results](AUTISM_ANIMAL_MODEL.md), [animal onset protocol](ANIMAL_IEEG_ONSET.md) and [qEEG calculation methods/example](ANIMAL_QEEG_METHODS.md).

## Suggested clarification for the next correspondence

“I would like to clarify the current PSI implementation: it uses FFT-based qEEG processing and optional linear SVM research models. LSTM/Transformer is a proposed future component, not an implemented or trained model. The expanded public human-data experiment evaluates schizophrenia-versus-control group discrimination; it does not validate impending meltdown or psychotic-episode prediction. We can review the existing code, exploratory results and the event-annotation protocol together.”

**References:** RepOD [10.18150/repod.0107441](https://doi.org/10.18150/repod.0107441). [1] Vanneau et al., bioRxiv v1, [10.64898/2026.04.03.716324](https://www.biorxiv.org/content/10.64898/2026.04.03.716324v1). [2] Kakuszi et al., European Psychiatry abstract, [10.1192/j.eurpsy.2024.653](https://doi.org/10.1192/j.eurpsy.2024.653). [3] Ramyead et al. (2016), [10.3109/15622975.2015.1083614](https://doi.org/10.3109/15622975.2015.1083614). [4] Bodatsch et al. (2011), [10.1016/j.biopsych.2010.09.057](https://doi.org/10.1016/j.biopsych.2010.09.057). Detailed reproduction and evidence tables accompany this update.
