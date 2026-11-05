# Figure captions — Current Science submission

> Each caption stands alone, so the figure is readable without the text; colours are
> colour-blind safe (Okabe–Ito), PDF is vector and PNG is 600 dpi.
> The numbers match `results/tables.md`, which is the authority if the experiments are
> ever re-run; anything changed there must be changed here too.

---

**Graphical abstract.** One model, three machines. Trained and tested on the same rig, the
classifier is at ceiling (balanced accuracy 1.000). Moved to another rig it falls to 0.043 — below
guessing, because healthy windows are labelled as inner-race faults. How far it falls is not
arbitrary: across the six ordered pairs of the three rigs, transfer accuracy decreases with the
distance between the two machines in a standardised descriptor space (Spearman ρ = −0.96,
p = 0.003), so the risk of a deployment can be estimated before the model is installed.

**Fig. 1** Study design. Top row: what is recorded and at what resolution. A rolling-element
bearing with a seeded inner-race defect (orange) is mounted in a test rig; an accelerometer on the
housing records vibration, here shown for 171 ms with the periodic impulse train of the fault. The
record is cut into overlapping windows of 2,048 samples (50% overlap), and each window becomes one
training example. Bottom row: the three evaluation protocols; each row is one experimental run,
marked by a bearing icon at the left. (a) Random window split: overlapping windows are shuffled
before division, so the windows of a run fall on both sides (dashed box). (b) Split by
run: every window of a run is assigned to one side, so the test set contains only unseen
recordings. (c) Cross-dataset transfer: the model is trained on all runs of a source rig; the
target rig is divided by run into a calibration pool, used only for few-shot adaptation, and a
held-out test pool that is never labelled. The two rigs are drawn as separate blocks and labelled
at the right of each row. Colours: blue, training data; light blue, calibration data; orange,
held-out test data.

**Fig. 2** What the three classes look like in one dataset (CWRU, drive-end accelerometer,
resampled to 12 kHz). Rows: healthy, inner-race fault, outer-race fault. (a) The first 40 ms of a
171 ms window. The black trace is the raw signal and the blue lines are the magnitude of the
analytic envelope, reflected about zero. The double-headed arrow spans two consecutive impulses,
with the period taken from the autocorrelation of the envelope — the quantity a fault-frequency
analysis turns into a defect frequency^5. In the healthy row no such periodicity is present, and the
envelope simply tracks the signal. (b) Envelope spectrum of the full 171 ms window, normalised to
its own maximum. Grey dashed lines mark the theoretical ball-pass frequencies of the drive-end
bearing (BPFO, 3.585 × shaft rate; BPFI, 5.415 × shaft rate), labelled once in the top row; the
marker gives the dominant peak and how it stands to those orders. The outer-race record also shows
a clear second harmonic, as expected for a localised spall. (c) Spectrogram (128-sample window, 75%
overlap) on a common 38 dB scale across the three rows, so that the panels can be compared in
absolute terms rather than each against its own maximum. The fault classes differ from the healthy
class in impulse rate and in the harmonics of the envelope, not in any single frequency that a
network could recognise independently of shaft speed.

**Fig. 3** The same model, the same data, two split conventions. Each row is one dataset–model
pair; the open circle is mean accuracy under a random window split, the filled circle under a
split by run, and the connecting line shows the paired change. Horizontal bars give the
difference in percentage points. Error bars are ±1 s.d. over the runs available for that
configuration. On CWRU and on PU the two conventions are indistinguishable (difference 0 pp,
markers coincide at 1.000); on MFPT, with global normalisation, the change of split unit costs 13–20 pp. The dashed vertical
line marks the chance level for a balanced three-class problem (0.333). Accuracy, not balanced
accuracy, is shown because the random-split condition is the one reported in the literature.

**Fig. 4** Cross-dataset transfer, and what 20 labelled target samples recover. (a) Balanced
accuracy of a frozen 1D-ResNet trained on the row domain and tested on the column domain, both
normalised globally. The diagonal is the within-dataset result under a split by run, on the same
balanced-accuracy scale; Table S1 reports plain accuracy for the within-dataset protocols,
which is slightly lower for MFPT because its classes are unevenly sized. (b) Gain in
balanced accuracy obtained by fine-tuning on k = 20 windows per class from the target calibration
pool; the arrow in each cell gives the level reached after adaptation. Diagonal cells are masked
because the within-dataset benchmark is already saturated at k = 0 and has nothing to gain. The gain is largest exactly where the frozen model fails worst (CWRU → MFPT, +0.74),
which is the behaviour expected if adaptation restores a scale rather than teaching a new feature.

**Fig. 5** Few-shot recovery in six transfer directions. Balanced accuracy against the number of
labelled target windows per class, k, for three model families; shaded bands are ±1 s.d. over
three random seeds. The dashed horizontal line is chance (0.333); the dotted lines mark the
target-domain ceiling, i.e. the accuracy the same model reaches on the target dataset with a
split by run, which bounds what adaptation can achieve. Note that the ranking at k = 20 differs
from the ranking within a dataset: the 21-feature model (grey) overtakes
both deep models once target labels are scarce.

**Fig. 6** Why the transfer fails, and why it fails to different degrees. (a) Class-conditional
distributions of window RMS for the three datasets (density on a log10 scale; ticks below each
ridge show individual windows). The amplitude separation between healthy and faulty windows is
5.1–8.6× in CWRU, 1.4–3.4× in MFPT and 1.4–1.9× in PU: a threshold on energy that is correct in
one rig is misplaced in every other. (b) Empirical cumulative distribution of excess kurtosis for
inner-race windows. The median is 3.0 for CWRU, 36.9 for MFPT and 1.1 for PU, so the same nominal
fault class is impulsive in one rig and nearly sinusoidal in another. (c) Transfer balanced
accuracy against a scalar measure of how far apart two rigs are in descriptor space
(root-mean-square Euclidean distance between class-conditional mean vectors of the 21 standardised
features). Each point is one direction, error bars are ±1 s.d. across the three model families, colour
indicates the target domain, and the label gives source → target (C, CWRU; M, MFPT; P, PU). The dashed line is a least-squares fit; the
rank correlation is Spearman ρ = −0.956 (p = 0.003, n = 6 directions). The severity of the collapse
is a property of the pair of machines, not of the model.

**Fig. 7** What the model represents, before and after adaptation. (a) t-SNE projection of the 21
features for 700 windows per dataset, globally standardised. The three datasets occupy different
regions, but CWRU and PU interleave: the two rigs that transfer best are also the two whose
descriptors overlap most. (b) Confusion matrix of the frozen ResNet on CWRU → MFPT, row-normalised:
healthy windows are all assigned to the inner-race class, and the two fault classes are exchanged
(94% each way). (c) The same model after fine-tuning on 20 windows per class, for the best of three
seeds: the diagonal is restored and the residual error moves to the outer-race row. Over the three
seeds this configuration gives 0.928, 0.740 and 0.667 balanced accuracy, which is the instability
discussed in the text. Axes: H, healthy; IR, inner race; OR, outer race.

**Fig. S1** Robustness of the conclusions to additive white Gaussian noise injected into the test
set. Balanced accuracy of the 1D-CNN under global normalisation, for within-dataset evaluation
(split by run) and for transfer without adaptation, at the SNR indicated. Within-dataset
performance degrades at rig-specific rates — CWRU loses half its accuracy between 0 dB and −4 dB,
PU already at 0 dB, and MFPT almost not at all — while cross-dataset performance stays at or near
chance in every CWRU–MFPT direction at every SNR tested. Neither the failure nor its ranking is an
artefact of clean recordings.
