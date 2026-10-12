# Preclinical onset prediction from model-animal iEEG/LFP

**Research direction clarified 13 October 2026, Asia/Jerusalem.** The intended preclinical task is to forecast a defined abnormal behavioral event in a model animal from continuous intracranial recordings available before that event. It is separate from the completed human diagnosis-group benchmark. **No model-animal onset predictor has been trained or evaluated in PSI.** The user selected a behavioral event as the target. Its precise observable definition and prediction horizon still require specification.

Depth-electrode LFP is one intracranial signal modality. Neuropixels spike trains, skull EEG, processed ERPs and continuous LFP are different inputs. Their measurement location, reference, units and bandwidth must be recorded, rather than all being renamed iEEG.

## Evidence and data availability

| Source inspected | Actual measurement/task | Relevance and remaining gap |
| --- | --- | --- |
| [Fmr1-KO2 sensory dataset](https://doi.org/10.17605/OSF.IO/CVEFK), already used by PSI | Public processed five-channel ERPs and stimulus-gating ratios from 22 animals | Supports sensory-response/genotype analysis. The source README requires author requests for raw EDF; the analyzed files lack continuous pre-event signals and behavioral-crisis onset labels |
| [Cui et al., 2022, MK-801](https://doi.org/10.3389/fnins.2022.1001869) | mPFC/CA1 depth LFP at 1 kHz, 30-minute pre-dose baseline and 120 minutes after administration | High-gamma 60–100 Hz and HFO 150–200 Hz changes are mechanistic candidates. The article/supplementary data statement does not identify a public continuous raw-LFP archive; it reports post-dose effects, not an onset-prediction evaluation |
| [IBL 2025 autism release](https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_autism_noel.html) | Neuropixels electrophysiology, task behavior and synchronized video across Fmr1, Cntnap2, Shank3 and wildtype mice | Useful task/neural data. The documented download example provides sorted spikes/trials, not proof of available continuous LFP or annotated crisis onset. A wrong task choice is not automatically the requested abnormal event |
| [Ghanaei/Erfanian rat iEEG v2](https://doi.org/10.17632/kn3k9f5vph.2) | Five rats; three implanted recording sites; healthy/epileptic trial MAT files in a PTZ epilepsy model | Excluded from the selected behavioral-onset task; a separately named epilepsy benchmark would be a different task. Sampling/units, independently timed onsets and continuous event-free coverage must be checked in the files. Folder-level condition labels alone do not establish forecasting, and epilepsy labels do not validate psychiatric events |

This is a bounded source review, not a claim that suitable animal data do not exist. No raw recordings from the new candidates were downloaded or model fitting performed. No author was contacted. The existing 22-animal ERP comparison remains the only completed animal analysis in this repository.

## Contract before fitting an onset model

1. Specify the animal, genetic/pharmacological model and actual event: for example, an independently reviewed behavioral transition. Use an animal-specific outcome name such as `animal_behavioral_event_onset`. This is a proposed namespace, **not a currently supported CLI target**. Human `meltdown` and `psychosis_onset` labels are not assigned to animals.
2. Obtain continuous recordings containing baseline, pre-event history, confirmed event intervals and completely observed event-free follow-up. Record animal/session IDs, anatomical sites, reference, sample clock, units, source/version and artifact annotations. Repeated sessions in the same animal do not increase the independent animal count.
3. Synchronize independent behavior/video or expert event annotations to the recording clock. Keep injection/stimulus time as experimental context. It cannot stand in for the measured onset of an abnormal behavioral event. Genotype, post-dose group membership and EEG band deviations are not crisis labels.
4. At prediction cutoff `t`, use signal samples at or before `t`, and require a later confirmed onset inside a prespecified horizon with a positive minimum lead. Exclude active/recent events and unreviewed future time. Filtering, baseline fitting and scaling must not use future samples, future episodes or held-out animals.
5. Fit a simple temporal/linear-SVM baseline before considering LSTM. Evaluate by held-out animal and, where appropriate, later sessions, preserving genotype, intervention and acquisition context. Report event sensitivity, false alerts per observed hour, lead-time distributions and animal-level uncertainty. Separate trial condition detection after dosing from forecasting before event onset.

The current `events.py` provides tested human-target history and annotation infrastructure, but its fixed human target/provenance contract is **not an animal onset-training implementation**. An animal adapter, outcome namespace and independent annotated dataset must be implemented and checked before invoking an animal predictor. No fabricated labels or synthetic scores supply that empirical evidence.

## Frequency requirements and transfer to humans

The current scalp feature profile ends at 40 Hz. An animal LFP profile spanning 60–100 Hz high gamma and 150–200 Hz HFO would require a sampling rate and acquisition filter that preserve those bands, source-specific artifact controls and anatomical/reference metadata. Sampling must exceed 400 Hz to cover 200 Hz with margin; the cited MK-801 acquisition used 1 kHz. Extending a band constant alone would not reproduce the study or establish a warning signal. These higher-band estimators remain unimplemented in PSI.

The MK-801 results concern effects after administration, including changes in CA1 high gamma and mPFC HFO. They are candidate measurements for a prospective preclinical analysis, not proven pre-event signs. Human scalp gamma 30–40 Hz, animal high gamma and HFO are not interchangeable. The 24-hour human PSI baseline and SOI/RDI definitions are not transplanted to mice.

Even a successful animal onset forecast would establish performance for that species, model and operational event. Transfer to autistic meltdown or psychosis onset in humans requires a separate annotated human validation study. LSTM/Transformer is still a future proposal, without an implemented architecture, trained weights or inference path.

## Wording suitable for correspondence

“The proposed preclinical prediction stage would use continuous model-animal iEEG/LFP and independently annotated event-onset times. This stage has not yet been trained or evaluated in PSI. The existing animal analysis uses processed sensory-response measurements, while the human EEG experiment evaluates diagnosis-group discrimination. Evidence of post-intervention oscillatory changes motivates candidate features; it does not establish impending-event prediction or clinical transfer.”
