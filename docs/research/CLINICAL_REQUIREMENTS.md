# Clinical assessment and scalp EEG measurement requirements

Updated 14 October 2026. These requirements constrain the interpretation of PSI research outputs and propose a future clinical protocol. They are not a validated diagnostic rule or a completed psychiatric review.

## Clinical assessment is the reference standard

Meltdown is a descriptive term for an episode of overwhelming distress; it is not interchangeable with psychosis. Agitation, vocalization or movement alone cannot establish either outcome. Psychiatric adjudication should consider the person's usual communication and functioning, thought content, perceptual experiences, reality testing when assessable, medical/substance explanations, collateral information and the course over time. NICE CG178 recommends comprehensive assessment; autism-specific differential problems are discussed by Ribolsi et al. and Dossetor.

Context, an apparent trigger and duration are useful observations, not sufficient diagnostic cutoffs. Do not require every psychotic presentation to be hypermotor, infer psychosis from missing speech during distress, or assume that an external trigger excludes psychosis. Preserve co-occurring and uncertain states. A future protocol requires psychiatrist-approved definitions and adjudication before recruitment or label generation. Clinical-only assessment must be a comparator so that any incremental benefit from EEG can be tested.

## Gamma is not a primary human endpoint

Scalp high-frequency power can be dominated by muscle and ocular activity. Whitham et al., Goncharova et al., Yuval-Greenberg et al. and Muthukumaraswamy provide full journal evidence for these measurement limitations. This supports a conservative design decision: **exclude scalp gamma and scalp theta–gamma PAC from the proposed primary human feature set**. Secondary analyses require prespecified synchronized EOG, EMG and movement measurements, expert artifact review and sensitivity analyses. This exclusion is a proposed clinical-study requirement, not a new filter silently applied to past benchmarks.

The existing 30–40 Hz output remains a waveform descriptor. A finite value, accepted epoch, notch filter, ICA component removal or numerical quality score does not establish a neural source. SEF95 and spectral entropy also use high-frequency data and are not automatically immune to EMG contamination. Missing or contaminated segments must be reported, including their relation to movement and distress; silently dropping the hardest periods could bias a warning study.

Muscle contamination and amplifier saturation are different problems. Literal clipping needs hardware range/ADC information and appropriate detection; amplitude heuristics alone do not verify its absence. An intracranial LFP gamma/PAC result cannot validate a scalp EEG psychiatric marker. Training and inference must match signal scope, electrode locations, reference, units, sample rate, filters, segment duration and QC contract.

## Evidence and conclusions

Full journal papers, clinical guidance, preprints, conference abstracts and repository tests must be identified separately. A PubMed entry or DOI does not establish full journal peer review. The Vanneau alpha-burst source inspected here is a bioRxiv preprint; the Kakuszi visual-gamma source is a conference abstract. Neither is primary support for a clinical event-prediction claim. Ramyead and Bodatsch are full journal longitudinal transition studies, not validation of minute-scale warnings.

Keep four claims separate: signal coupling/synchrony, diagnosis-group discrimination, movement-associated measurements and prediction of an independently adjudicated future event. The current IBL data have task-movement annotations; RepOD has diagnosis groups. Neither supplies future meltdown/psychosis onset labels. The original SOI/RDI names and arithmetic are unvalidated constructs: no EEG calculation establishes reality testing or provides a clinical probability.

## Questions requiring prospective psychiatric review

1. Are the descriptive event definitions appropriate, including mixed/uncertain states and delayed assessment after communication becomes possible?
2. What clinical information and follow-up can support a reproducible reference standard, and how should disagreement between adjudicators be retained?
3. Does an EEG adjunct address a real unmet decision after a competent clinical-only assessment? What benefit would justify its burden and error rates?
4. Which medical, medication, sleep and environmental covariates need prespecified assessment, and which participants/settings should be excluded?
5. What privacy, consent, distress-management and independent ethics-review procedures are needed before collecting prospective data?

No clinician's authorship, endorsement or completed consultation is implied by this document. See the [new perspective and proposed protocol](PSI_CLINICAL_PERSPECTIVE.md).

## Sources by publication category

**Full journal articles and reviews:** Whitham et al. (2007), [10.1016/j.clinph.2007.04.027](https://doi.org/10.1016/j.clinph.2007.04.027); Goncharova et al. (2003), [10.1016/S1388-2457(03)00093-2](https://doi.org/10.1016/S1388-2457(03)00093-2); Yuval-Greenberg et al. (2008), [10.1016/j.neuron.2008.03.027](https://doi.org/10.1016/j.neuron.2008.03.027); Muthukumaraswamy (2013), [10.3389/fnhum.2013.00138](https://doi.org/10.3389/fnhum.2013.00138); Ribolsi et al. (2022), [10.3389/fpsyt.2022.768586](https://doi.org/10.3389/fpsyt.2022.768586); Dossetor (2007), [10.1177/1359104507078476](https://doi.org/10.1177/1359104507078476).

**Clinical guidance:** [NICE CG178 recommendations](https://www.nice.org.uk/guidance/cg178/chapter/recommendations). This is clinical guidance, not a journal experiment.

The preliminary candidate sources and their explicit status are maintained in [PRE_EVENT_SIGNAL_EVIDENCE.md](PRE_EVENT_SIGNAL_EVIDENCE.md) and [signal_evidence.json](signal_evidence.json).
