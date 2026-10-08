# Article-to-code alignment

Source: Netanel Stern, *Toward a Psychiatric State Index: Differentiating Autistic Meltdown from Psychotic Episode Onset Using Real-Time EEG and AI*, 2026, supplied PDF `PSI_Stern_2026(1).pdf`. The exact file's SHA-256 and page count are in `article_source.json`. No public DOI or publication status is invented. The supplied version is authoritative for this implementation; it is not assumed to be the later corrected Google Docs version.

| Article section | Implementation | Boundary |
| --- | --- | --- |
| 2: BIS analogy and raw EEG | BIS raw waveform importer, separate recorded monitor numerics, electrode documentation | No proprietary BIS recreation or inference of psychiatric state from BIS |
| 6.1: electrodes | Quatro contact specification, unverified VitalDB mapping, explicit expanded scalp specification | Ground/reference are contacts, not signal channels |
| 6.2: SOI/RDI components | Ten named component schema and strict missing-component handling | Event gating, posterior features, CSD and autonomic inputs are not invented from BIS |
| 6.3: personal baseline | Individual mean/SD fitting from labelled calm epochs; 24-hour duration requirement | No normative population diagnosis thresholds |
| 6.4: clinical alert states | Documented as unvalidated proposals | No real-data diagnosis or medication/sedation recommendations are emitted |
| 6.5: normalization | Exact directional clipped SOI mapping and symmetric inverted RDI mapping | Missing values are not filled or silently reweighted |
| 7.1: feature extraction | Spectral, experimental bispectral and verified-site PLV descriptors | Microstate, event gating and CSD methods require separate specifications |
| 7.2: AI | Optional labelled-cohort linear SVM with leave-one-subject-out validation | No pretrained psychosis/meltdown classifier; no LSTM without longitudinal labels |
| 8–10: dataset and validation gap | Data card and research validation protocol | Surgical dataset provides engineering evidence, not psychiatric validation |

## Exact normalization

For an SOI component whose increase is hypothesized to represent overload:

`s_i = clip(100 * (x_i - mean_i) / (3 * sd_i), 0, 100)`

The direction is reversed for gating efficiency and frontoparietal PLV, matching the worked table: `mean_i - x_i`. RDI uses the article's written formula:

`r_i = clip(100 * (1 - abs(x_i - mean_i) / (3 * sd_i)), 0, 100)`

This makes deviations in **either direction** lower the RDI component. The paper's prose about “deviations below baseline” is narrower than the formula. The implementation follows the formula and documents the distinction.

Both composites use five weights, default 0.20, summing to one. Positive, finite personal calibration SDs are required. Zero SD is an error; no epsilon creates invented sensitivity. Missing components return a missing composite and explicit coverage. A numeric result is a mathematical research composite, not a calibrated probability or proof of reality contact.

## Worked-example inconsistencies

The supplied PDF's five SOI component values all clip at 100, giving SOI 100. It also lists a trajectory beginning at SOI 18 for `t=0`. Those cannot be the same epoch under the written formula. The repository reproduces **SOI 100** and describes the trajectory as illustrative rather than reconstructing it from the component table.

The alert table specifies ORANGE for RDI 40–60, while the worked trajectory describes an ORANGE transition near 70 and an ORANGE value of 64. Several threshold boundaries and rate definitions are unspecified. Automatic alert logic is therefore deferred pending an explicit corrected research specification and clinical validation. No medical actions from these tables are encoded.

## Inputs requiring scientific specification

The article gives feature concepts, but it does not completely specify a burst threshold, event-locking protocol, CSD geometry/estimator, bicoherence convention, posterior channel aggregation, autonomic combination or calibration-quality protocol. Generic available band powers are not relabelled as those complete components. The `score` command accepts independently produced, defined values; their validity and unit matching are part of the research protocol.

Medication profile identifiers are checked between current features and baseline. This is context matching, not a pharmacological correction model. Learning medication effects and treatment-related EEG changes requires labelled data and external validation.

