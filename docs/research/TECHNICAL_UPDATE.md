# PSI: human EEG and event-prediction status

**Netanel Stern | 12 October 2026 UTC | Technical update for Felix Benninger and colleagues**

PSI runs an offline FFT-based qEEG pipeline and optional linear SVM research workflows. The full human EEG experiment below predicts diagnosis groups. A separate event workflow requires independently timed outcomes; it has engineering tests but no real event-trained model. **No LSTM/Transformer is implemented or trained, and no imminent-attack threshold is validated.** Reviewable changes: [PR #3](https://github.com/netanelcyber/psi-qeeg-bis/pull/3).

## Implemented signal processing

EEG is represented in microvolts with an explicit sample clock. Default epochs are four seconds without overlap. Welch PSD uses two-second Hann windows, 50% overlap, constant detrending and a 0.5 Hz frequency grid. Exact-edge integration gives delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30 and gamma 30–40 Hz powers and relative powers. Delta/alpha ratio, entropy and SEF95 are computed. Missing/flat/extreme-amplitude epochs are rejected; EMG contamination is flagged rather than removed.

Experimental single-channel auto-bicoherence uses 60 seconds of accepted past EEG; verified-montage alpha PLV is also implemented. Both are absent from this cohort's SVM features. Raw analysis keeps classification indeterminate and SOI/RDI unavailable without all components and a valid personal baseline. FFT describes the waveform; its output is not a psychiatric diagnosis.

## Expanded human-data experiment

Public RepOD v1.0 (CC0) contains 14 schizophrenia and 14 control participants, 19 channels, 250 Hz. All 28 source checksums/sizes were verified. Full recordings provide 28,863 seconds, 7,205 four-second epochs and 6,870 accepted epochs, compared with 840/821 previously. There are still 28 independent people.

| Coverage | Prior 120 s | Entire recording |
| --- | ---: | ---: |
| Full scalp | 64.29% | 78.57% |
| Fp1/F7 | 46.43% | 67.86% |
| Fp2/F8 | 60.71% | 39.29% |

These are subject balanced accuracies from leave-one-subject-out evaluation. Each training fold fits StandardScaler then linear SVC (`C=1`, balanced class weights). Four descriptors per channel are used: alpha power, delta/alpha ratio, SEF95 and entropy. Epoch majority voting yields subject predictions. Coverage sets share accepted epochs; final model refits are not test results. The right result declined. These small, single-site estimates lack uncertainty intervals and external/prospective validation.

## Literature candidates and implementation gaps

| Candidate | Evidence boundary | Current code |
| --- | --- | --- |
| Shorter alpha bursts / lower time occupancy [1] | ASD resting-state association; preprint, no future meltdown test | Alpha power available; burst estimator absent |
| Posterior gamma increase with eyes open [2] | Visual-condition comparison; conference abstract | 30–40 Hz only versus study 30–100 Hz; no condition contrast |
| Beta/gamma source patterns [3] | Later psychosis transition over at least three years | Scalp power available; source CSD absent |
| Reduced duration MMN [4] | Future conversion, median follow-up 32 months | Human stimulus-locked MMN absent; not an FFT band |

These findings identify candidates to investigate, not an abnormal-wave cutoff indicating an attack in the next few minutes. Feature annotations mark estimator mismatch and unavailable individual thresholds. No EEG deviation is used to manufacture an event label.

## Event SVM and future LSTM requirements

The separate event workflow computes mean, SD and slope from complete past-only histories and trains two binary linear SVMs for future meltdown and psychosis onset. Independent onset/offset annotations and complete reviewed follow-up are required; psychosis outcomes need clinical adjudication. Current events, insufficient lead and missing follow-up are censored. Scaling/evaluation are subject-separated. The 60-second history/300-second horizon defaults are illustrative, not demonstrated warning times. Timing, leakage, reload and CLI tests use labelled synthetic fixtures only. RepOD lacks onset/follow-up labels, so no real event model or probability forecast was produced.

LSTM development requires synchronized longitudinal EEG/context, independently adjudicated events, a prespecified sequence/horizon/missingness contract, implemented architecture/training/inference and versioned weights. Compare with temporal/SVM baselines on locked external/prospective data, reporting event sensitivity, false alerts/hour, lead time, calibration and uncertainty. Longer resting recordings alone do not satisfy this requirement.

## Suggested clarification for the next correspondence

“I would like to clarify the current PSI implementation: it uses FFT-based qEEG processing and optional linear SVM research models. LSTM/Transformer is a proposed future component, not an implemented or trained model. The expanded public human-data experiment evaluates schizophrenia-versus-control group discrimination; it does not validate impending meltdown or psychotic-episode prediction. We can review the existing code, exploratory results and the event-annotation protocol together.”

**References:** RepOD [10.18150/repod.0107441](https://doi.org/10.18150/repod.0107441). [1] Vanneau et al., bioRxiv v1, [10.64898/2026.04.03.716324](https://www.biorxiv.org/content/10.64898/2026.04.03.716324v1). [2] Kakuszi et al., European Psychiatry abstract, [10.1192/j.eurpsy.2024.653](https://doi.org/10.1192/j.eurpsy.2024.653). [3] Ramyead et al. (2016), [10.3109/15622975.2015.1083614](https://doi.org/10.3109/15622975.2015.1083614). [4] Bodatsch et al. (2011), [10.1016/j.biopsych.2010.09.057](https://doi.org/10.1016/j.biopsych.2010.09.057). Detailed reproduction and evidence tables accompany this update.
