# EEG candidates: association is not an imminent-event threshold

Reviewed 12 October 2026 UTC (13 October in Israel). **No validated single-wave threshold for an imminent meltdown or psychotic attack was established by the sources inspected here.** The table marks research candidates and estimator gaps, not abnormal participants, individual risk probabilities or event labels. A resting-state diagnosis comparison cannot demonstrate that an episode will begin minutes later.

| Candidate | Primary study and finding | Time/outcome studied | Match to current PSI code |
| --- | --- | --- | --- |
| Alpha bursts, approximately 7–13 Hz | Vanneau et al., 2026 preprint v1: autistic children had shorter bursts and lower alpha time occupancy; individual burst amplitude did not differ between groups | Resting recordings, 52 autistic children, 39 non-autistic children and 26 siblings; no impending-meltdown outcome | PSI estimates 8–13 Hz power. It does not estimate the study's burst duration/occupancy. Four-second power SD is a different statistic |
| Posterior gamma response, 30–100 Hz | Kakuszi et al., 2024 conference abstract: gamma increased with eyes open relative to eyes closed in temporal/occipital sites in ASD (23 ASD, 24 controls) | Sensory/visual condition comparison; no subsequent meltdown labels | PSI gamma is 30–40 Hz, a partial frequency match. The RepOD run has no EO/EC contrast or ASD group; muscle contamination remains a possible confound |
| Beta/gamma spatial patterns | Ramyead et al., 2016 (online 2015): a multivariate current-source-density model distinguished later psychosis transition in 53 high-risk participants, of whom 18 transitioned | Follow-up of at least three years; transition risk, not a minutes-ahead attack detector | Scalp beta/gamma power is available, but the study's source CSD estimator is absent. An SVM on diagnosis groups is not this prediction task |
| Reduced duration mismatch negativity (MMN) | Bodatsch et al., 2011: lower duration-MMN amplitude in future converters among 62 high-risk participants | Median 32-month follow-up; auditory oddball ERP at baseline | MMN is a stimulus-locked ERP, not a resting FFT band. PSI has no human MMN estimator or synchronized stimulus markers in RepOD |

These selected studies concern different populations and outcomes. None supplies a rule such as "gamma above X means meltdown in five minutes". The preprint and conference abstract also carry different evidence status from full journal articles. Reported group associations must not become pseudo-labels for training an event classifier.

## What is marked in the human-data output

[signal_evidence.json](signal_evidence.json) is an evidence register. The committed [candidate_feature_annotations.csv](candidate_feature_annotations.csv), a frozen aggregate from the full 28-person RepOD run, maps actual output columns to candidate IDs, the implemented frequency range, estimator match, inclusion in the diagnosis SVM and finite accepted-epoch count. An empty candidate set means no direct mapping to these selected studies. `individual_abnormality_threshold` is unavailable and `imminent_event_validation` is not established for every row. No participant or epoch is labelled pre-meltdown/pre-psychotic from these papers.

For a future event study, measure the specified burst/ERP/CSD features, obtain personal baselines and independent onset/follow-up annotations, then lock the horizon and evaluate participant-held-out and prospective predictions. Delta/alpha ratio, SEF95, entropy, bicoherence and PLV remain research descriptors; their availability alone establishes no acute warning threshold. See [EVENT_PREDICTION.md](EVENT_PREDICTION.md).

## Primary references

1. Vanneau et al., 2026, alpha oscillatory-states study. [bioRxiv v1, DOI 10.64898/2026.04.03.716324](https://www.biorxiv.org/content/10.64898/2026.04.03.716324v1). The inspected version is a preprint, not a peer-reviewed event-prediction result.
2. Kakuszi B et al. *Gamma activity in Autism Spectrum Disorder: Enhanced response to visual input*. [European Psychiatry conference abstract, DOI 10.1192/j.eurpsy.2024.653](https://doi.org/10.1192/j.eurpsy.2024.653).
3. Ramyead A et al. *Prediction of psychosis using neural oscillations and machine learning in neuroleptic-naïve at-risk patients*. [World Journal of Biological Psychiatry, DOI 10.3109/15622975.2015.1083614](https://doi.org/10.3109/15622975.2015.1083614); [author abstract](https://pubmed.ncbi.nlm.nih.gov/26453061/).
4. Bodatsch M et al. *Prediction of Psychosis by Mismatch Negativity*. [Biological Psychiatry, DOI 10.1016/j.biopsych.2010.09.057](https://doi.org/10.1016/j.biopsych.2010.09.057); [author abstract](https://pubmed.ncbi.nlm.nih.gov/21167475/).
