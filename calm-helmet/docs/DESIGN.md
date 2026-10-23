# calm-helmet: design document

Status: draft, October 2026. Nothing here has been tested on people.

## 1. Goal and target users

The goal is to warn the wearer, a caregiver or staff **minutes before** a meltdown or an episode of challenging behaviour (aggression, self-injury, property destruction). Warning that early gives time for low-intensity help: a break, a quieter place, a familiar regulation strategy, or a calm conversation.

**First target group:** autistic people, children or adults, who already wear a **soft protective helmet** because of head-directed self-injury. For them a helmet is not a new burden. They also tend to be the people with the highest need and the fewest ways to say "I'm getting overwhelmed."

**Later:** a lighter cap or headband version for people who don't wear a helmet, and a psychiatric-ward profile for psychotic agitation. That profile is the `psychosis` profile in psi-qeeg-bis.

## 2. Why a helmet, and why several signals

| Evidence level | What we know |
|---|---|
| Strong | Wrist signals (HR, EDA, motion) rise before aggression in autistic youth. Forecasts reach about **1–3 minutes** ahead (Goodwin et al., *Autism Research* 2019; *JAMA Network Open* 2023). In adult psychiatry, HR and skin conductance rise over about 20–30 minutes before aggressive incidents (de Looff et al., *Psychophysiology* 2019). |
| Hypothesis | Cortical signals (EEG; prefrontal haemodynamics measured with fNIRS) change **earlier**, as regulation is strained, and could extend the warning window. **Untested.** |
| Known confound | Autonomic signals also rise with exercise, heat and excitement, and movement corrupts most sensors. This is the main source of false alarms in wrist-only systems. |

A helmet can carry brain, muscle, autonomic and movement sensors at the same time. Each one covers the weak spots of the others.

## 3. Sensor suite (the everyday helmet)

| Layer | Sensor | Placement | Output feature (1 Hz) | Notes |
|---|---|---|---|---|
| Cortical, electrical | Dry EEG, 8–16 channels | Fp1/Fp2, F3/F4, C3/Cz/C4, T7/T8, P3/P4, O1/O2 subset | β/α log-ratio, relative α, frontal α asymmetry | Spring-loaded flexible pins. Unlike a Muse headband, gives central coverage (Cz) |
| Cortical, haemodynamic | **fNIRS / HD-DOT** with **short-separation channels** | Prefrontal (forehead, hair-free), optionally temporal | HbO↑ / HbR↓ over PFC | See §4 |
| Muscle | Bipolar EMG | Temporalis (both sides) | RMS 20–250 Hz | Jaw clenching and muscle tension get their own clean channel instead of being "guessed" from EEG 30–45 Hz |
| Autonomic | Reflective PPG | Forehead | HR, HRV (RMSSD) | |
| Autonomic | EDA | **Wrist** (companion band) | tonic + phasic EDA | The forehead is a poor site for EDA, so it stays on the wrist |
| Behaviour | 6-axis IMU | Helmet shell | Activity counts, rocking frequency, head-impact detection | Used as **context** (movement gating), plus impact logging for self-injury |

**Mechanical and sensory comfort.** These matter as much as the electronics.
- Soft, washable liner. No point pressure, and sensor pods recessed into the foam.
- Total added mass below about 250 g, with the battery at the back to balance the weight.
- No lights, sounds or vibration on the helmet itself unless the wearer chooses them. Alerts go to a phone or watch.
- The wearer, or their family, can see and stop monitoring at any time.

## 4. Near-infrared brain imaging (fNIRS / HD-DOT)

**Principle.** Near-infrared light (about 700–900 nm) passes through scalp and skull. Some of it scatters back from the outer 1.5–2 cm of cortex. Using two or more wavelengths, the modified Beer–Lambert law turns changes in optical density into changes in **oxy-haemoglobin (HbO)** and **deoxy-haemoglobin (HbR)**. Neurovascular coupling makes these a slow (seconds) index of local cortical activity.

**HD-DOT** (high-density diffuse optical tomography) uses many overlapping source-detector distances and reconstructs a **3-D map** of cortical haemodynamics instead of single channels. Commercial wearable systems exist; check the current specs before choosing one.

**Why it fits this problem**
- Prefrontal cortex supports emotion regulation. One hypothesis is that PFC HbO rises while the wearer is "holding it together," then falls when regulation fails. This is **untested**.
- The signal is slow by nature. That suits a minutes-scale build-up and costs nothing for a 10-minute horizon.
- It is much less sensitive to muscle artefacts than EEG.
- A map can show **where** on the cortex the change starts. This is a partial answer to the "onset map" limitation of the EEG-only prototype.

**Pitfalls and how the design handles them**

| Pitfall | Handling |
|---|---|
| **Scalp blood flow rises with arousal** and mimics brain activity | **Short-separation channels** (~8 mm) on every optode cluster. Their signal is regressed out of the long channels. The coupling is fitted on **calm** calibration data, because a running fit during a build-up would also remove the cortical signal (`calm_helmet/fnirs.py`). Synthetic benchmark: false alarms in quiet hours drop from 8 to 3. |
| Hair blocks light | Start with the hair-free forehead. Use comb-tipped optodes for temporal sites. Choose the wavelength to suit skin tone. |
| Motion | Accelerometer-referenced motion correction, plus signal-quality indices. Bad windows are dropped. Fusion renormalises the weights. |
| Depth | Only superficial cortex. Amygdala and insula are out of reach (see §7, lab track). |
| Haemodynamic delay (~5–6 s) | Irrelevant at a minutes horizon. |

**Time-domain fNIRS** (picosecond photon timing with SPAD detectors) separates early, shallow photons from late, deep ones. That improves depth sensitivity and scalp rejection. It is a candidate upgrade.

## 5. Signal processing and alert logic

This is implemented in `calm_helmet/`. The README has the diagram.

1. **Per-device adapters** produce 1 Hz features. Not written yet; the first targets are an OpenBCI/LSL EEG adapter and an fNIRS adapter (SNIRF files or LSL).
2. **Calibration.** 3 calm minutes at the start of each session set the median and MAD for every feature.
3. **Robust z-scores, sign-aligned** so that "up" always means more arousal, and clipped at ±6.
4. **Movement gating.** HR, HRV and EDA are removed while head-IMU activity is above 3 z, and for 120 s after.
5. **Weighted fusion.** Default weights: EEG 0.20, fNIRS 0.20, EMG 0.15, autonomic 0.35, IMU 0 (context only). Weights are renormalised over the features that are valid that second.
6. **Detector.** One-sided CUSUM with a rising-trend gate produces the **early warning**. A smoothed index of at least 2.5 z for 10 s produces the **imminent** alert. Hysteresis and refractory periods apply. No alerts are sent with fewer than 2 valid modalities.
7. **Clinical context.** Planned, ported from psi-qeeg-bis: the Brøset Violence Checklist on wards, and caregiver-entered "setting events" such as poor sleep, pain, illness or a change of routine. These tune the detector's sensitivity for the day.

**Personalisation path.** The current index is rule-based on purpose: it is transparent and needs no training data. Once labelled data exist, the next steps are per-person weight learning (regularised logistic regression on pre-event windows), then a population model with personal fine-tuning. These will be evaluated leave-one-person-out.

## 6. Labelling: video as ground truth

Video-EEG units already record many autistic children. The literature shows that most "seizure-like" events captured there are **not epileptic**, even in children whose EEG shows epileptiform discharges (e.g. Kim et al., *Epilepsia* 2006). This gives two things:

- **A labelling protocol.** Synchronised video, the helmet and the wrist band. Trained coders mark onset, severity and **precursor behaviours** (functional-analysis style) from the video, blind to the sensor data.
- **A possible retrospective data source.** Existing video-EEG recordings of behavioural events in autistic patients, used with ethics approval.

**Epileptiform activity must be handled.** The helmet should include, or at least flag, epileptiform discharges and seizures as a separate class, so that a seizure is never treated as "high arousal," or the other way round.

## 7. Lab track: mechanistic studies (not wearable in daily life)

The question here is what happens in the brain in the minutes before a meltdown, and which helmet features truly reflect it.

| Method | What it adds | Constraint |
|---|---|---|
| **OPM-MEG + EEG + fNIRS** (simultaneous systems have been published) | Source localisation of electrical activity, magnetic (little muscle or skull distortion), haemodynamics at the same moment | Needs a magnetically shielded room, or at least light shielding with active field-nulling coils. Sensitive to head movement |
| **Wearable / helmet PET** | Metabolism and neurotransmitter systems while the person can move | Radiation dose. Short lab sessions only |
| **fMRI** | Deep structures (amygdala, insula) | Not tolerable during distress. Rest and task only |

**Running MEG without a full shielded room.** This would make lab sessions far more tolerable. It needs several layers together:
- light passive shielding;
- active nulling coils driven by reference magnetometers (closed loop);
- hardware or synthetic gradiometry;
- reference-sensor regression;
- homogeneous field correction (HFC) and SSS/tSSS;
- precise head-motion tracking, because in a residual field a small sensor rotation produces artefacts far larger than brain signals.

Total-field OPM gradiometers that operate in Earth's field are an emerging option. Daily-life MEG is **not** feasible today.

**Ethics.** Meltdowns are never provoked. Only mild, graded paradigms are allowed (for example, sensory load or a short mildly stressful task), with an immediate stop on request and a trusted person present.

## 8. Next-generation sensing (candidate upgrades)

| Technology | Physical principle | Maturity | Why it is interesting here |
|---|---|---|---|
| **SCOS / DCS** | Laser speckle fluctuations → cerebral **blood flow** | DCS: clinical research. SCOS: early human studies | Flow and oxygenation from the **same optodes**. Faster signal |
| **TD-fNIRS (SPAD)** | Photon time of flight | Research-grade products | Better depth, better rejection of scalp signal |
| **EIT through the EEG electrodes** | Small injected currents → impedance map | Clinical for lungs, research for brain | A second modality from the **same electrodes**. Cheap |
| **Facial thermography** (small IR camera in the visor) | Passive thermal emission. The nose tip cools and the periorbital region changes under stress | Research | Contact-free autonomic signal |
| **Acousto-optic / ultrasound-modulated optics** | Ultrasound "tags" a point deep in tissue for the light | Early | Possible depth beyond fNIRS |
| **Functional ultrasound / photoacoustics** | Blood-flow Doppler / laser-induced sound | Strong in infants and through skull windows. Weak through the adult skull | Long-term |
| **mmWave radar** (room-mounted) | Chest and skin micro-motion → HR, respiration | Research, early products | Monitoring with no wearable at all, for people who reject any device |

**Integrated concept (idea, untested).** Shared optodes give fNIRS plus SCOS (oxygenation plus flow). Shared electrodes give EEG plus EIT. Add a visor thermal camera, EMG, PPG and IMU. That covers four physical mechanisms (electrical, haemodynamic, impedance, thermal) on largely shared hardware.

## 9. Clinical status of the everyday sensor suite

Every sensor in §3 exists today, and several are in clinical or consumer use: EEG, PPG, EMG, IMU, wrist EDA, and NIRS cerebral oximetry. Functional fNIRS is mainly a research tool. **What is new is the combination and the algorithm, not the hardware.** That makes an ethics-approved pilot realistic.

## 10. Validation plan

1. **Bench tests.** Synchronisation across devices (target under 10 ms between EEG and fNIRS, under 100 ms across all streams). Signal quality during walking and rocking. Effectiveness of short-separation regression.
2. **Tolerance and feasibility.** Daily wear by consenting participants who already use protective helmets. Measure wear time, comfort ratings and the share of usable data.
3. **Labelled observational study.** Helmet, wrist band and video (§6). Caregiver app for setting events.
4. **Offline evaluation.** Leave-one-person-out and leave-one-session-out. Report lead-time distribution, sensitivity, false alarms per hour, and the gain from each modality (ablation, as in the synthetic benchmark).
5. **Prospective pilot.** Live alerts to caregivers. Outcomes: alert burden, usefulness rated by families and staff, change in incident frequency and severity, and the wearer's own view wherever it can be obtained.

## 11. Ethics and safe use

- **Purpose.** Earlier, gentler support. The system must never be used to justify restraint, seclusion, forced medication or punishment.
- **Consent and control.** Informed consent, or the appropriate legal framework when capacity is limited. Involve the person as much as possible. Anyone involved can see what is measured, who receives alerts, and how to stop.
- **Human in the loop.** False alarms and missed events will happen. Alerts are prompts for people who know the wearer.
- **Privacy.** Physiological and video data are sensitive health data. Process on the device where possible, store as little as possible, encrypt, and comply with GDPR, the Israeli Privacy Protection Law, or HIPAA.
- **Stigma.** Report results at group level. Never present a risk score as a judgement of the person.

## 12. Limitations

1. **No real data yet.** The synthetic benchmark encodes assumptions about the build-up. It tests the software, not the idea.
2. The cortical-precursor hypothesis may be wrong. Cortical change may come later than autonomic change, or vary too much between people.
3. Movement during agitation corrupts most channels exactly when they matter. Gating keeps false alarms down but also removes information.
4. Closely spaced episodes are hard: in the benchmark, the one missed event is a second episode that starts before the first has calmed down.
5. A helmet suits only some users. The approach must also work, in a reduced form, with a headband or a wrist band alone.

## References

- Goodwin MS, Mazefsky CA, Ioannidis S, Erdogmus D, Siegel M. Predicting aggression to others in youth with autism using a wearable biosensor. *Autism Research* 2019.
- Goodwin MS et al. Wearable biosensing to predict imminent aggressive behavior in psychiatric inpatient youths with autism. *JAMA Network Open* 2023.
- de Looff P et al. Changes in heart rate and skin conductance in the 30 min preceding aggressive behavior. *Psychophysiology* 2019.
- Kim HL, Donnelly JH, Tournay AE, Book TM, Filipek P. Absence of seizures despite high prevalence of epileptiform EEG abnormalities in children with autism monitored in a tertiary care center. *Epilepsia* 2006.
- Nicotera AG et al. EEG abnormalities as a neurophysiological biomarker of severity in autism spectrum disorder: a pilot cohort study. *J Autism Dev Disord* 2019.
- Almvik R, Woods P, Rasmussen K. The Brøset Violence Checklist. *J Interpersonal Violence* 2000.
