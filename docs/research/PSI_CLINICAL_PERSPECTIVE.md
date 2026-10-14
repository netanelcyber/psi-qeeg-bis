# Reframing the Psychiatric State Index Through Clinical Assessment and Measurement Validation

Netanel Stern

Perspective and proposed validation protocol

14 October 2026

Manuscript for scientific discussion. Not peer reviewed. Psychiatric review of this proposed protocol is pending.

## Abstract

The Psychiatric State Index (PSI) project explores quantitative electrophysiology and computational models in psychiatric research. Its present implementation does not establish a diagnostic distinction between autistic meltdown and psychosis, or predict either event prospectively. This perspective reframes the project around a clinical reference standard and a staged measurement-validation protocol. Agitation and movement are nonspecific observations; event adjudication must consider baseline functioning, thought content, perceptual experiences, context and clinical course. Scalp gamma is particularly vulnerable to muscle and ocular contamination, making an apparent association during distress difficult to interpret. We therefore propose excluding scalp gamma and scalp theta–gamma phase-amplitude coupling from primary human analyses, with any secondary use contingent on independent artifact measurements and expert review. Intracranial local field potential results are treated as a separate measurement domain. Software tests and exploratory diagnosis-group benchmarks are distinguished from clinical validation. A prospective study would first establish reproducible psychiatrist-adjudicated labels, measure acquisition and artifact constraints, and compare clinical-only models with models that add prespecified EEG features. Participant-level evaluation, event-level uncertainty, false alerts, calibration and measurement coverage would be reported before any clinical deployment is considered. Published journal evidence, clinical guidance and preliminary reports are identified separately. The proposed endpoint is evidence about incremental clinical value, not a presumption that a numerical index measures reality testing or supplies an imminent-event warning.

Keywords: quantitative EEG; psychosis; autism; electromyography; clinical adjudication; phase-amplitude coupling; prediction validation

## Rationale and scope

A physiological measurement is useful only when its interpretation and intended decision are explicit. In PSI, several questions had been discussed together: whether recorded signals show coupling, whether features differ between diagnosis groups, whether signals change around movement, and whether a future psychiatric event can be predicted. These are distinct hypotheses with different reference standards. Success on one does not establish another.

This article is a narrative perspective and proposed protocol, not a systematic review or a report of a new clinical cohort. Its purpose is to define what a defensible next study would need to measure. Selected published sources support the measurement and clinical framing; recommendations about PSI feature selection, annotation and evaluation are prospective design proposals. No human event model has been validated, and no clinician endorsement is asserted.

The motivating question is whether an electrophysiological adjunct provides information that improves a specified clinical decision after ordinary clinical assessment. If a competent clinician already distinguishes the relevant presentations adequately, the burden of demonstrating useful additional information lies with the proposed adjunct. An attractive index name or a sophisticated model cannot substitute for this comparison.

## Clinical distinctions before signal classification

Psychosis and autistic distress should not be operationalized as interchangeable forms of agitation. NICE recommends a comprehensive assessment for people presenting with psychotic symptoms, rather than reliance on an isolated physiological observation [1]. Autism can complicate the interpretation of unusual sensory experiences, communication and changes in functioning; a review and a small case series illustrate potential differential-diagnostic difficulties [2,3]. These sources do not validate an EEG rule for distinguishing meltdown from psychosis.

For this proposed study, meltdown would be a descriptive event label whose definition is agreed with psychiatrists, autistic people and other relevant professionals. The National Autistic Society describes meltdown in relation to overwhelming distress; this is practice information rather than a peer-reviewed diagnostic study [4]. The label must not imply psychotic thought content merely because a person moves intensely, shouts or cannot communicate during an episode. Conversely, an apparent environmental trigger is not sufficient to exclude a psychotic presentation.

Adjudicators would document usual functioning, the observed change, assessable thought content and reality testing, perceptual experiences, medical or substance explanations, collateral observations and evolution over time. Trigger and duration would be recorded as context, without a universal minute cutoff. Recovery may help interpretation, but incomplete assessment during distress must remain uncertainty rather than presumed intact or impaired reality testing.

The clinical ontology would permit overlapping phenomena. A person could experience overwhelming distress and psychotic symptoms in the same observation period. Separate dimensions for observed distress and adjudicated psychotic symptoms are preferable to forcing every event into an exclusive binary category. The final definitions, assessment instruments and adjudication rules require specialist review before prospective data collection.

## Scalp gamma and the problem of source attribution

Recorded high-frequency power does not identify its biological source. Experimental work using neuromuscular blockade, scalp EMG characterization and simultaneous eye-movement assessment demonstrates important muscle and ocular contributions to high-frequency scalp recordings [5–7]. These experiments address particular conditions, not a universal prohibition on studying neural gamma. A published review describes the overlap between muscle spectra and neural high-frequency activity and the limits of artifact-removal approaches [8].

This limitation is especially consequential for a study involving movement, facial tension or distress. A rise in scalp gamma around an event could reflect muscle recruitment, an ocular transient, movement-induced electrode disturbance, neural activity, or a mixture. A notch filter addresses selected line frequencies; a finite spectral estimate or accepted quality score does not resolve source attribution. Removing components by ICA likewise requires independent justification and sensitivity analysis.

As a conservative protocol decision, scalp gamma power and scalp theta–gamma phase-amplitude coupling would be excluded from the primary human feature set. Their secondary investigation would require synchronized EMG, EOG and movement measurements, prespecified review criteria and expert examination of raw data. Quiet reference recordings and repeated within-person comparisons would help characterize contamination, but would not guarantee that an active distress period remains interpretable. If source attribution is unsupported, the output would be described as recorded high-frequency power rather than a neural psychiatric marker.

Other descriptors are not automatically protected by omitting a feature named gamma. Spectral entropy and spectral edge frequency can include the same contaminated high-frequency samples. The study would specify each estimator's passband and assess sensitivity to artifact handling. Literal amplifier saturation would be assessed separately using hardware dynamic-range information and clipping checks; muscle dominance and electronic saturation are different measurement failures.

## Coupling measures and measurement domains

Phase-locking value, spectral coherence and phase-amplitude coupling summarize relationships within recorded signals. Tort's modulation index supplies one defined PAC estimator [9]; its computation does not by itself establish interaction between neural generators or a psychiatric mechanism. Cross-frequency coupling can have alternative explanations, including waveform structure and nonstationarity [10]. Common reference, common sources and shared artifacts must also be considered when interpreting interchannel similarity.

The existing PSI coupling experiment uses public intracranial LFP and synchronized task video. Such measurements differ from scalp EEG in geometry, tissue filtering, reference and artifact exposure. They can test numerical methods within that domain, but cannot validate a scalp biomarker by transfer of terminology. Animal and human measurements would retain explicit signal-domain metadata and separately justified feature definitions.

The corrected software excludes both one-second circular-shift boundaries in its PAC surrogate analysis and records the configured surrogate count. With 199 surrogates, the smallest attainable plus-one Monte Carlo p value is 1/200. Surrogate validity still depends on the selected null and the temporal structure of the recording. Multiple-comparison correction controls its specified family under stated assumptions; it is not a clinical validation procedure. Overlapping windows also do not create independent biological replicates.

## What the present computational evidence establishes

The public repository contains tested spectral and coupling code, explicit acquisition contracts, and exploratory benchmarks [11]. The corrected IBL analysis examines 143 distinct 16-second windows from a 240-second segment. The 2,288 PAC tests yield 273, 190 and 8 Benjamini–Hochberg rejections in original, notch-filtered and bipolar modes, respectively. These counts depend on preprocessing and the chosen test family. The annotations describe task movements; no psychotic or meltdown event reference standard is available.

The corrected public human EEG benchmark uses 28 participants and evaluates schizophrenia-versus-control diagnosis groups. Participant-balanced leave-one-subject-out accuracies are 78.57% for the full scalp, 50.00% for Fp1/F7 and 25.00% for Fp2/F8. The corrected training protocol gives equal total weight to each participant, with additional class balancing by participant counts. These are exploratory single-dataset estimates, without external validation or reported uncertainty intervals, and must not be read as event-warning performance. The underlying cohort contains neither prospective meltdown onsets nor acute psychosis-onset follow-up.

Synthetic regression tests establish behavior for known signals and specified numerical cases. They do not establish artifact resistance in hypermotor recordings or psychiatric diagnostic accuracy. The project's SOI and RDI arithmetic remains an unvalidated mathematical construction; the name “Reality Degradation Index” does not make a waveform estimate a measure of reality testing. No present result justifies an individual clinical cutoff, probability of an attack or treatment decision.

| Research claim | Required observation | Present evidence boundary |
| --- | --- | --- |
| Signal coupling or synchrony | Defined estimator, acquisition and suitable null | Numerical and exploratory LFP analyses |
| Diagnosis-group difference | Independent participants and a valid group comparison | Small public human EEG benchmark |
| Movement-associated change | Synchronized movement and artifact measurements | Task movements, with neural source uncertain |
| Future clinical event prediction | Independent onset labels and prospective follow-up | Not established by the available datasets |

## Evidence hierarchy and temporal relevance

Publication format and outcome must both be inspected. A DOI or PubMed record is not sufficient to establish a full peer-reviewed article. The project's separate evidence register identifies the inspected alpha-burst report as a preprint and the visual-gamma report as a conference abstract [11]. They may generate hypotheses, but are not primary clinical support for this protocol. A conference abstract may itself undergo selection or review; that does not supply the methods and assessment of a full journal article.

Published longitudinal studies have examined later psychosis transition using EEG-related measures [12,13]. Their prediction targets and follow-up differ from a warning that an episode will begin in minutes. Replication would require the specified estimators, population and temporal endpoint. Resting alpha power is not an alpha-burst-duration estimator, and a resting FFT spectrum is not stimulus-locked mismatch negativity. Substituting available descriptors for the published constructs changes the hypothesis.

## Proposed prospective validation protocol

### Stage one Clinical definitions and feasibility

A psychiatric lead would establish a manual for event ascertainment before data collection. The manual would specify eligible presentations, consent procedures, clinical assessment timing, collateral information, how uncertainty is recorded and whether mixed states can receive more than one label. Two independent adjudicators would review clinical material without viewing the investigational EEG score; disagreements would be retained and subsequently adjudicated by a prespecified process. Inter-rater agreement and the frequency of unassessable events would be reported.

The feasibility phase would assess whether tolerable recording is possible during the relevant periods and whether sufficiently precise event timing can be obtained. It would not assume that a psychotic process has a sharply identifiable second of onset. Where only an interval is defensible, onset would remain interval-censored. A minute-scale prediction task would proceed only if its clinical endpoint and timing are credible. No distress would be deliberately induced merely to obtain training examples.

### Stage two Acquisition and artifact characterization

Every recording would carry electrode identities and locations, reference and ground, units, sample rate, hardware filters, dynamic range, preprocessing version, segment duration and quality-control policy. Training and inference contracts would reject incompatible inputs; different montages or signal domains would require a separately designed validation rather than silent substitution. Time synchronization would be measured, not inferred from filenames.

The human protocol would obtain synchronized EOG, appropriate EMG and movement information wherever feasible, and prespecify artifact review. Reviewers would assess raw signals independently of event predictions. Rejection rates and usable coverage would be reported by participant and behavioral context. An algorithm that loses usable input during distress cannot be evaluated only on the remaining quiet segments while its missing-input periods disappear from the performance denominator.

### Stage three Models and comparators

The primary clinical question would compare a prespecified clinical/context model against the same model with eligible EEG features. Additional comparators would include a transparent temporal baseline and a movement/artifact-only model. The latter would help test whether an apparent gain is explained by movement rather than interpretable cerebral information. Feature definitions, history length, lead interval, target horizon and alert policy would be fixed before held-out evaluation; current software defaults would not be treated as demonstrated warning times.

The primary human feature set would exclude scalp gamma and scalp theta–gamma PAC. Other features would enter only after feasibility work defines acquisition, artifact sensitivity and the meaning of their baseline normalization. Participant-level training weights would limit domination by long recordings. All scaler fitting, feature selection, threshold selection and hyperparameter tuning would occur within training participants; overlapping segments from one person would not be split across training and evaluation.

### Stage four Locked prospective evaluation

An external or later prospectively collected cohort would be evaluated with the model and decision policy locked. Reviewed follow-up would determine negative examples; absence of an annotation would not automatically mean absence of an event. Current events, gaps in observation and onset uncertainty would be handled under prespecified censoring rules. Clinical management would remain independent of investigational alerts during initial evaluation.

| Evaluation target | Prespecified reporting |
| --- | --- |
| Clinical labels | Definitions, agreement, mixed states, unknowns and timing uncertainty |
| Acquisition | Montage, reference, units, synchronization and compatibility failures |
| Measurement availability | Valid coverage and rejected/missing time by person and context |
| Event prediction | Event sensitivity, false alerts per observed hour and lead-time distribution |
| Probability estimates | Calibration, prediction horizon and participant-level uncertainty |
| Added value | Clinical-only versus clinical-plus-EEG results and artifact-only sensitivity analysis |

## Statistical analysis and reporting

The unit of generalization would be the participant, with events nested within people. Sample planning would account for independent participants, event counts, repeated observations and the precision of prespecified error rates. Thousands of windows cannot compensate for few independently observed people. Cluster-aware uncertainty would be reported, and per-participant summaries would accompany pooled time-based metrics. Group balancing in model fitting would not justify imposing an artificial event prevalence on a real-world evaluation.

Multiplicity correction would be tied to an explicitly defined family of exploratory tests. Confirmatory hypotheses, model comparisons and alert thresholds would be prespecified separately. Diagnosis-group accuracy would not be reported as prospective clinical sensitivity. If probabilities are produced, their calibration would be tested on the intended population; a raw SVM margin or normalized composite would not be labelled a calibrated probability.

TRIPOD+AI provides a reporting framework for clinical prediction models, STARD addresses diagnostic accuracy studies, and DECIDE-AI addresses early clinical evaluation of AI decision support [14–16]. Their applicability depends on the actual study design. Following a checklist does not establish methodological quality, clinical benefit or regulatory readiness, but makes the intended design and its limitations assessable.

## Ethical and clinical governance

Prospective research would require an appropriate institutional and ethics-review pathway before recruitment. Consent capacity, assent where appropriate, supported communication, video privacy and withdrawal during distress would be planned with clinical and lived-experience input. Research staff would follow an agreed clinical escalation pathway for concerning presentations; care would not depend on the investigational index. Data access would be limited, and identifiable recordings would not be released with a public software repository.

Psychiatric consultation is a prerequisite for the clinical protocol, not an endorsement obtained from publication or software availability. Contributions would be documented after they occur. Authorship would require actual scholarly contributions and agreement; receiving a manuscript would not make a clinician a coauthor. Funding and competing-interest declarations would likewise be confirmed rather than inferred.

## Limitations and conclusion

This perspective selects relevant evidence rather than systematically reviewing every candidate biomarker. Its proposed adjudication and recording procedures have not been prospectively tested, and the existing computational datasets cannot resolve the central clinical question. Scalp measurement may remain least available during the very episodes of interest, which could make the proposed adjunct impractical even if numerical performance on quieter data is favorable.

PSI should proceed only as a testable research framework. The next credible milestone is a clinician-reviewed protocol and a feasibility study that establishes labels and measurement availability. Clinical usefulness would require demonstrated added value over competent clinical assessment, with acceptable uncertainty and error costs. Signal synchrony, group discrimination and task-movement associations should remain explicitly separate from claims about predicting future psychiatric events.

## Data and software availability

The computational examples and their limitations are documented in the public repository, including the 13 October 2026 corrections at commit 427d3122e3dfb8c7d58e78034f45999ef311a0da [11]. This manuscript introduces no new participant data and reports no new prospective clinical results. Its protocol, feature exclusions and clinical evaluation proposals have not been implemented as a validated clinical service.

## References

1. National Institute for Health and Care Excellence. Psychosis and schizophrenia in adults: prevention and management. Clinical guideline CG178. Recommendations, section 1.3.3.1. https://www.nice.org.uk/guidance/cg178/chapter/recommendations. Clinical guidance; accessed 14 October 2026.
2. Ribolsi M, et al. Recognizing Psychosis in Autism Spectrum Disorder. Front Psychiatry. 2022;13:768586. https://doi.org/10.3389/fpsyt.2022.768586. Full journal review.
3. Dossetor DR. “All That Glitters Is Not Gold”: Misdiagnosis of Psychosis in Pervasive Developmental Disorders—A Case Series. Clin Child Psychol Psychiatry. 2007;12:537–548. https://doi.org/10.1177/1359104507078476. Full journal case series.
4. National Autistic Society. Meltdowns. https://www.autism.org.uk/advice-and-guidance/behaviour/meltdowns/all-audiences. Practice information; accessed 14 October 2026. Not a journal study.
5. Whitham EM, et al. Scalp electrical recording during paralysis: quantitative evidence that EEG frequencies above 20 Hz are contaminated by EMG. Clin Neurophysiol. 2007;118:1877–1888. https://doi.org/10.1016/j.clinph.2007.04.027. Full journal experimental article.
6. Goncharova II, McFarland DJ, Vaughan TM, Wolpaw JR. EMG contamination of EEG: spectral and topographical characteristics. Clin Neurophysiol. 2003;114:1580–1593. https://doi.org/10.1016/S1388-2457(03)00093-2. Full journal experimental article.
7. Yuval-Greenberg S, et al. Transient induced gamma-band response in EEG as a manifestation of miniature saccades. Neuron. 2008;58:429–441. https://doi.org/10.1016/j.neuron.2008.03.027. Full journal experimental article.
8. Muthukumaraswamy SD. High-frequency brain activity and muscle artifacts in MEG/EEG: a review and recommendations. Front Hum Neurosci. 2013;7:138. https://doi.org/10.3389/fnhum.2013.00138. Full journal review.
9. Tort ABL, et al. Measuring phase-amplitude coupling between neuronal oscillations of different frequencies. J Neurophysiol. 2010;104:1195–1210. https://doi.org/10.1152/jn.00106.2010. Full journal methods article.
10. Aru J, et al. Untangling cross-frequency coupling in neuroscience. Curr Opin Neurobiol. 2015;31:51–61. https://doi.org/10.1016/j.conb.2014.08.002. Full journal review.
11. Stern N. PSI qEEG / BIS repository. Methodology corrections, coupling analysis and expanded human cohort documentation. https://github.com/netanelcyber/psi-qeeg-bis/pull/3. Computational project documentation; not peer reviewed.
12. Ramyead A, et al. Prediction of psychosis using neural oscillations and machine learning in neuroleptic-naïve at-risk patients. World J Biol Psychiatry. 2016; first published online 2015. https://doi.org/10.3109/15622975.2015.1083614. Full journal longitudinal article.
13. Bodatsch M, et al. Prediction of Psychosis by Mismatch Negativity. Biol Psychiatry. 2011;69:959–966. https://doi.org/10.1016/j.biopsych.2010.09.057. Full journal longitudinal article.
14. Collins GS, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378. Full journal reporting guideline.
15. Bossuyt PM, et al. STARD 2015: an updated list of essential items for reporting diagnostic accuracy studies. BMJ. 2015;351:h5527. https://doi.org/10.1136/bmj.h5527. Full journal reporting guideline.
16. Vasey B, et al. Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. Nat Med. 2022;28:924–933. https://doi.org/10.1038/s41591-022-01772-9. Full journal reporting guideline; publisher correction available.
