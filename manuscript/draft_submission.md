# How much of "99% accuracy" survives a change of machine? A three-rig audit of bearing-fault benchmarks

Jiarui Liu^1
^1 Qingdao University of Technology, Linyi, China
Corresponding author: Jiarui Liu, ages257@outlook.com; ORCID: 0009-0002-3035-9295

## Abstract

Deep-learning bearing-fault classifiers routinely exceed 99% accuracy, yet almost all such numbers come from a single test rig. We harmonised three public vibration datasets into one three-class problem and evaluated three models under random-window, run-level and cross-rig protocols. Within a dataset accuracy saturates at 0.999–1.000 and cannot rank methods. Between rigs balanced accuracy spans 0.04–0.61 and follows the descriptor-space distance between the machines (ρ = −0.96), not the architecture: the cue, not the capacity, changes. Twenty labels per class recover most of the loss, so deployment risk is estimable before installation.

**Keywords:** benchmarking; bearings; diagnostics; reproducibility; transferability

## Introduction

Machine-learning papers on bearing fault diagnosis share one property: the numbers are near-perfect. Studies built on the Case Western Reserve University (CWRU) dataset, the de facto standard of the field, report accuracies of 99% and above^1,2, and improvements are argued in tenths of a percentage point. To a reader outside mechanical engineering this looks like a solved problem. To anyone who has maintained a machine, it does not.

That gap has a specific origin. The standard evaluation divides the recordings into training and test sets by randomly assigning short, overlapping windows of vibration signal^3. Consecutive windows share the same bearing, load, mounting and often the same instant of time. A classifier therefore needs only to recognise the recording it came from, and a window-level split hands it that recognition task directly. The resulting number measures memorisation within one experimental campaign.

None of this is peculiar to bearings. Applied machine learning in other fields, from structural health monitoring through medical imaging to materials informatics, relies on the same convention of a random split inside one curated dataset^4. Whenever the split unit is finer than the unit that varies in deployment, reported performance becomes an upper bound that no practitioner reproduces. Bearing diagnosis makes an unusually clean case study because the quantity that changes between machines, the fault characteristic frequency, follows from bearing geometry^5, so the failure can be measured rather than speculated about.

Two questions have been asked of this benchmark before: whether the split convention inflates accuracy, and whether transfer learning closes the gap^6. A third has not. If diagnostic skill survives a change of machine to different degrees, what property of the two machines decides how much? Answering it requires more than the usual pair of datasets, because a single pair can only show that collapse happens, not what governs its depth. We therefore assembled three rigs whose pairwise distances differ, and tested whether a scalar summary of that distance predicts the transfer result.

The contribution is a measurement, not an architecture: three standard models, three protocols, three datasets, one question, all code and result files released. The answer is a quantified gap, a predictor of its size, a demonstration that a few dozen labelled samples close most of it, and a checklist general readers and referees can apply to any data-driven applied paper.

## Data and methods

### Datasets

Three public datasets, recorded on three rigs by three groups. All are third-party; we generated no data.

*CWRU.* Drive-end accelerometer, 12 kHz sampling, four motor loads (0–3 hp), artificial single-point defects of three diameters on the inner race and on the outer race at the 6 o'clock position, plus a healthy baseline^7. Harmonised: 32 runs, 4,942 windows.

*MFPT.* The Machinery Failure Prevention Technology Society challenge data: a different rig, bearing and shaft speed, with sampling at 97,656 Hz for baseline records and 48,828 Hz for fault records. Harmonised: 20 runs, 890 windows.

*PU.* The Paderborn University dataset: a 32-bearing testbed with accelerated lifetime damage^8. We use three bearings at one operating point (1,500 rpm, 0.7 N m, 400 N): one healthy (K001) and two with single damage on the inner race (KA04) and outer race (KI04). Vibration channel 1 at 64 kHz^9. Harmonised: 60 runs, 2,709 windows.

All records were resampled to 12 kHz and cut into windows of 2,048 samples (171 ms) at 50% overlap, then assigned to healthy, inner-race and outer-race classes. Rolling-element faults are excluded because MFPT does not contain them, which keeps the label space identical across datasets. That identity is the precondition for measuring transfer. One run corresponds to one bearing condition under one load, and for PU to one recording of one bearing. Runs, not windows, are the unit that varies between machines, and we treat them that way throughout.

The three rigs were not chosen to be representative. They were chosen so that the pairwise dissimilarity spans a range: two laboratory rigs with machined faults on different bearings (CWRU, MFPT), and one accelerated-lifetime testbed whose damage was produced by fatigue rather than by a cutting tool (PU).

### Models

Three models span current practice. A gradient-boosted tree on 21 hand-crafted features (eight time-domain statistics, eight spectral band energies, spectral centroid and entropy, three envelope-spectrum descriptors), hereafter the 21-feature model, represents the classical approach^10. A shallow one-dimensional CNN in the WDCNN style^11 uses a wide first convolutional layer and three further convolutional blocks (0.1 M parameters). A one-dimensional ResNet with six residual blocks (1.3 M parameters)^12 represents current deep practice.

### Evaluation protocols (Fig. 1)

*P1, random window split.* Windows are shuffled and divided 70/30. This is the published convention.

*P2, split by run.* Every window of a file goes to the same side, so the test set contains only recordings the model has not seen. Window-level leakage disappears; machine-level similarity remains, since runs from one rig still share bearing geometry, sensor and mounting.

*P3, cross-dataset transfer.* Train on all runs of one dataset, test on runs of another. The target set is split by run into a calibration pool and a held-out test pool, so labels used for adaptation never enter the evaluation.

### Normalisation, adaptation and noise

Two normalisation schemes, because they turn out to matter. Per-window z-scoring removes the window mean and divides by the window standard deviation, discarding absolute amplitude. Global normalisation removes the window mean but divides by one scale estimated on the training set, preserving relative energy between windows. Unless stated otherwise, cross-dataset results use global normalisation, the scheme that keeps the amplitude cue available.

Adaptation is simulated by fine-tuning the trained model on k windows per class from the target calibration pool (k = 5, 10, 20), or, for the feature-based model, refitting on the union of source features and those k samples. Each configuration runs over three random seeds, two of them over four; we report mean ± s.d. The complete per-model tables for all three protocols are given in the Supplementary Material (Tables S1–S3).

Additive white Gaussian noise at 8, 4, 0 and −4 dB SNR is applied to the test set to check whether the transfer results depend on clean signals.

### A scalar for the distance between rigs

To ask what governs the depth of the collapse, we need a number for "how different are these two machines". We compute the 21 features for every window, standardise each feature across the pooled windows of all three datasets, average within each dataset and class, and take the root-mean-square Euclidean distance between the two sets of three class means. The result is a single number in units of pooled standard deviations, dominated by whatever properties actually separate the rigs: amplitude, spectral shape, impulsiveness. It is a descriptive statistic, not a metric with an interpretation in physics, and we use it only to rank the three pairs.

## Results

Fig. 2 shows what the classifier actually sees: one window from each class in CWRU, its envelope spectrum and its time-frequency map. The classes differ in an impulse train and in the band that train excites, and in little else that is visible to the eye.

### Within a dataset, the benchmark is saturated

On CWRU all three models reach 0.999–1.000 accuracy under both the random and the run-level split and under both normalisation schemes (Fig. 3). The 21-feature model, the shallow CNN and the residual network are indistinguishable. Changing the split changes nothing.

PU behaves the same way: 1.000 under both splits. Its 60 recordings come from three bearings, each run recorded once, so a run-level split removes whole recordings from training and the task is still solved perfectly.

MFPT is the exception, and the exception is informative. Under a random split all models again reach 0.996–1.000. Under a run-level split with global normalisation, accuracy falls to 0.803 for the 21-feature model, 0.861 for the 1D-CNN and 0.814 for the ResNet, a loss of 13 to 20 points produced by changing only the unit of the split. CWRU's runs are near-duplicates of one another: same bearing, same mounting, load varied in four steps, and windows from different loads share almost everything. MFPT's runs vary more, and the leakage becomes visible.

Two conclusions follow. On the most widely used dataset in the field, accuracy cannot discriminate between methods, because the task is saturated. And the size of the inflation is a property of the dataset rather than of the model: invisible on CWRU and PU, large on MFPT.

### Across datasets, the collapse is real but not uniform

Without adaptation, cross-dataset balanced accuracy spans 0.039–0.612 across the six directions and three models, against a chance level of 0.333 (Fig. 4a).

The worst case is total. A ResNet trained on CWRU and tested on MFPT reaches 0.043 balanced accuracy, less than a seventh of the chance level; averaged over the three models the direction gives 0.050. The confusion matrix (Fig. 5b) shows why this is not a partial failure: healthy windows are all labelled inner race, and the two fault classes are exchanged with 94% probability in each direction. The model has not lost the signal, it has applied a decision boundary calibrated on a different scale and then mapped impulsiveness onto the wrong label.

Two directions sit near chance (MFPT → CWRU 0.282, MFPT → PU 0.324). Three are clearly above it (CWRU → PU 0.418, PU → CWRU 0.425, PU → MFPT 0.412). No direction reaches a level that a maintenance engineer could use, but the spread is systematic rather than random: the same pair of rigs gives similar results in both directions, and the best direction reaches eight times the worst (0.425 against 0.050, averaged over models).

### The depth of the collapse is predicted by the distance between the rigs

The three pairwise distances are 7.87 (CWRU–MFPT), 6.87 (MFPT–PU) and 5.42 (CWRU–PU) pooled standard deviations. Sorted by that distance, the six transfer accuracies, one per direction and averaged over the three model families, fall into three pairs that do not overlap: 0.050 and 0.282 at 7.87, 0.324 and 0.412 at 6.87, 0.418 and 0.425 at 5.42 (Fig. S1c, Table 1). The rank correlation is Spearman ρ = −0.956 (p = 0.003, n = 6). A straight-line fit is weaker, r = −0.763 (p = 0.077), because both directions involving the closest pair, together with PU → MFPT from the middle pair, land inside a band 0.013 wide (0.412 to 0.425) and flatten the right-hand end of the scatter. Six points is a small sample, and the relation we claim is an ordering rather than a calibration curve: we do not read the slope as a conversion factor between distance and accuracy.

One objection is easy to raise: the distance is computed in the feature space of the 21-feature model, so it may be predicting itself. The data say otherwise. The distance predicts the two networks, which are trained on raw windows and never see those descriptors, more strongly than it predicts the model that does: ρ = −0.91 (p = 0.012) for the ResNet and −0.84 (p = 0.038) for the CNN, against −0.48 (p = 0.34) for the 21-feature model. What the statistic captures is a property of the two machines rather than a shared representation; the weaker correlation for the feature model reflects its own irregular behaviour at the closest pair, where its accuracy falls below that of its neighbours.

The ordering is not a restatement of "different datasets are different". It gives the practitioner's question, will this model work on my machine, a measurable proxy computable from unlabelled recordings on both sides.

Fig. 5a shows the same structure in the feature space: the three datasets occupy distinct regions of a t-SNE projection, but CWRU and PU interleave, and those are the two rigs whose transfer works best.

### Why the cue changes

The amplitude statistics explain the failure mode. In CWRU the median RMS of healthy windows is 0.065 against 0.332 for inner-race and 0.558 for outer-race windows, a factor of 5.1 to 8.6. In PU the same ratio is 1.4 to 1.9; in MFPT, 1.4 to 3.4 (Fig. S1a). A threshold on energy that separates the classes in one rig is misplaced in every other, and the direction of the error depends on which rig supplied the training data.

Impulsiveness tells a similar story with a different ordering. The median excess kurtosis of inner-race windows is 3.0 in CWRU, 36.9 in MFPT and 1.1 in PU (Fig. S1b): the same nominal fault is impulsive in one rig, mildly impulsive in the second and nearly sinusoidal in the third. The feature that identifies inner-race damage in MFPT resembles severe outer-race damage in CWRU, which is exactly the swap seen in the confusion matrix. In PU the ordering is reversed outright — healthy windows have higher kurtosis (2.1) than inner-race windows (1.1) — so a rule learned on either of the other two rigs not only fails but inverts.

Normalisation changes the magnitude without changing the picture. Per-window z-scoring removes the amplitude cue and moves CWRU → MFPT slightly upward (0.129 against 0.068 for the CNN) while moving MFPT → CWRU down (0.205 against 0.318). Neither scheme rescues transfer alone. The choice alone moves the result by a factor of two in either direction, which is why it belongs in the methods section of every paper in this area.

### Twenty labels per class recover most of the loss

Balanced accuracy rises with k in every direction (Fig. S2). At k = 20 the deep models reach 0.405–0.998 (1D-CNN) and 0.645–1.000 (1D-ResNet), the upper end being the CWRU → PU direction, where 20 windows per class from the target rig are enough for a balanced accuracy of 1.000 and 0.998. Where the rigs are far apart, the same budget buys much less: 0.586 (CNN) and 0.777 (ResNet) for CWRU → MFPT.

The 21-feature model is the more useful at this budget. Refitted on 20 target windows per class it reaches 0.785 for CWRU → MFPT and 0.763 for MFPT → CWRU, against 0.405–0.777 for the two deep models on the same two directions. It is best or joint-best in all six, with a mean of 0.887 across directions against 0.780 for the ResNet and 0.709 for the CNN. Most of its gain arrives early: at k = 5 it already averages 0.849, where the deep models are at 0.540 and 0.607. Stability is a separate matter and the advantage is not uniform: the ResNet's seed-to-seed s.d. reaches 0.20 in the PU → CWRU direction and its CWRU → MFPT result ranges from 0.667 to 0.928 across three seeds, while the 21-feature model's own s.d. reaches 0.22 in the opposite direction. Both are unstable, in different places; the difference is that the instability costs the 21-feature model less accuracy on average.

The practical reading: recovery needs 60 labelled windows, a few seconds of vibration data per class, collectable during one maintenance stop, and the model that best uses them is not the one that wins the saturated benchmark.

### Noise degrades the benchmark, not the conclusion

Adding white Gaussian noise degrades within-dataset performance, and the three rigs lose it at different rates: on CWRU the 1D-CNN falls from 0.998 balanced accuracy at 8 dB to 0.890 at 0 dB and 0.508 at −4 dB, on PU from 1.000 to 0.521 and 0.333, while MFPT barely moves (0.870 at 8 dB, 0.856 at −4 dB). In the CWRU–MFPT directions, cross-dataset accuracy stays between 0.06 and 0.32 at every SNR tested and the two directions never cross (Fig. S3). The transfer failure is therefore neither caused by clean signals nor repaired by noisy ones. That the two saturated rigs differ so much in noise tolerance is a reminder that "1.000" hides a second axis of variation, which the standard benchmark never reports.

## Discussion

In this literature, the number that is optimised, reported and compared measures how saturated a single-rig dataset has become. On CWRU the saturation is complete: every model we tried reaches essentially 1.000, and only cross-machine evaluation moves it. The ranking of methods that fills most published tables cannot be a ranking of diagnostic capability.

The field does not notice because the failure is invisible from inside a single dataset. A model that has learned an amplitude threshold reports 100% on any split of the rig where the threshold was fitted and 0% on any other. Accuracy, F1, confusion matrices and ablations all look healthy in the first case. The only diagnostic that separates the two situations is a test on an unseen machine, and that is precisely what the standard workflow omits.

Adding the third rig changes the claim. With two datasets one can only say that transfer fails; the failure reads as a property of deep learning, which invites the reply that better architectures or better domain adaptation would fix it. With three, the failure has a magnitude that varies systematically with a property of the machines, and the best direction reaches 0.425 without adaptation and 1.000 with 60 labels. Transfer is not broken; it is conditional, and the condition is measurable before deployment.

The recovery experiment carries the practical message. A calibration set of 60 windows closes most of the gap, and the 21-feature model closes more of it than either deep network (mean 0.887 against 0.780 and 0.709), needing only five per class to reach 0.849. It also relocates the bottleneck: not architecture, but access to a small, well-labelled calibration set from the deployment machine. That is an organisational problem, and it does not appear in benchmark tables.

The study covers three laboratory rigs, three classes and one architecture family per model class. Gearbox data, rolling-element faults and open-set deployment (where the target classes are not known in advance) are outside it, as is transfer from laboratory to plant. Our adaptation sampling assumes balanced presence of all three classes in the calibration set, which a planned calibration satisfies and an unplanned one may not. The distance statistic is descriptive: it ranks these three pairs correctly, and whether it ranks a fourth pair correctly is a prediction we cannot test with three rigs.

## Conclusion

The number this literature optimises does not survive the first change of machine. On CWRU and PU every model we tried saturates at 0.999–1.000 whatever the split; the one rig whose runs genuinely differ, MFPT, loses 13–20 points once the split unit is corrected. Across rigs, balanced accuracy lands between 0.04 and 0.61, and where it lands follows the descriptor distance between the two machines (ρ = −0.96 over six directions) rather than the choice of architecture.

What changes across machines is the scale of the cue, not the capacity of the model: healthy-to-faulty amplitude ratios of 5–8× in one rig against 1.4–1.9× in another, inner-race kurtosis of 3.0, 36.9 and 1.1. Sixty labelled windows, about two seconds of signal per class, take the 21-feature model to a mean of 0.887, above both deep models, and they can be collected during one maintenance stop.

The reporting we would ask for is five items: the unit of the train/test split, per-class recall alongside aggregate accuracy, the chance level of the label set, the normalisation scheme, and one result on a dataset recorded elsewhere. The last cannot be satisfied by relabelling an existing benchmark, and it decides whether a reported accuracy means anything outside the laboratory.

## Acknowledgements

We thank the Case Western Reserve University Bearing Data Center, the Machinery Failure Prevention Technology Society and the Paderborn University Bearing Data Center for making their recordings publicly available.

## Declarations

**Data availability.** All three datasets are public: CWRU^7, MFPT via the Machinery Failure Prevention Technology Society, and PU^9. We generated no data.
**Code availability.** Code, cached window indices and all result files: https://github.com/liu-4564/bearing-transfer-audit.
**Funding.** No external funding was received for this work.
**Conflict of interest.** The author declares no competing interests.
**Author contributions.** J.L. conceived the study, wrote the code, ran the experiments, analysed the results and wrote the manuscript.

## References

1. Smith, W. A. and Randall, R. B., Rolling element bearing diagnostics using the Case Western Reserve University data: a benchmark study. *Mech. Syst. Signal Process.*, 2015, **64–65**, 100–131.
2. Lei, Y., Yang, B., Jiang, X., Jia, F., Li, N. and Nandi, A. K., Applications of machine learning to machine fault diagnosis: a review and roadmap. *Mech. Syst. Signal Process.*, 2020, **138**, 106587.
3. Hendriks, J., Dumond, P. and Knox, D. A., Towards better benchmarking using the CWRU bearing fault dataset. *Mech. Syst. Signal Process.*, 2022, **169**, 108732.
4. Kapoor, S. and Narayanan, A., Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 2023, **4**(9), 100804.
5. Randall, R. B. and Antoni, J., Rolling element bearing diagnostics — a tutorial. *Mech. Syst. Signal Process.*, 2011, **25**(2), 485–520.
6. Zhao, Z., Zhang, Q., Yu, X., Sun, C., Wang, S., Yan, R. and Chen, X., Applications of unsupervised deep transfer learning to intelligent fault diagnosis: a survey and comparative study. *IEEE Trans. Instrum. Meas.*, 2021, **70**, 1–28.
7. Case Western Reserve University Bearing Data Center, Seeded fault test data, 2013; https://engineering.case.edu/bearingdatacenter
8. Lessmeier, C., Kimotho, J. K., Zimmer, D. and Sextro, W., Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: a benchmark data set for data-driven classification. In *Proceedings of the European Conference of the Prognostics and Health Management Society*, 2016.
9. Paderborn University Bearing Data Center, Bearing damage data set (64 kHz vibration), 2016; https://groups.uni-paderborn.de/kat/BearingDataCenter
10. Chen, T. and Guestrin, C., XGBoost: a scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 785–794.
11. Zhang, W., Peng, G., Li, C., Chen, Y. and Zhang, Z., A new deep learning model for fault diagnosis with good anti-noise and domain adaptation ability on raw vibration signals. *Sensors*, 2017, **17**(2), 425.
12. He, K., Zhang, X., Ren, S. and Sun, J., Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016, pp. 770–778.
