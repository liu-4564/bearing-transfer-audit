# Pre-submission enquiry (to the editor of Current Science)

> Send two to three weeks before submitting. Keep the subject line short and specific,
> keep the body to one screen, and do not attach the manuscript.
> Use the editorial address printed on `currentscience.ac.in` itself, never an address
> offered by a third-party site.
> If there is no reply after two weeks, submit through the normal channel; do not wait.

---

**Subject:** Pre-submission enquiry — does deep learning for bearing fault diagnosis transfer across machines?

Dear Editor,

I am writing to enquire whether the following study would be suitable for consideration as a Research Article in *Current Science*.

**Title:** How much of "99% accuracy" survives a change of machine? A three-rig audit of bearing-fault benchmarks

**What we did.** Deep learning papers on rolling-element bearing diagnosis almost universally train and test on data from a single test rig. We assembled three publicly available vibration datasets recorded on three different machines, with different sampling rates and different fault-seeding procedures, and re-trained three representative models — a 21-feature gradient-boosted classifier, a shallow one-dimensional CNN, and a 1D ResNet — under three evaluation protocols: the conventional random split, a split that keeps each experimental run wholly within one partition, and full cross-dataset transfer in all six directions.

**What we found.** On a single dataset the benchmark is saturated — all three models reach 0.996–1.000 accuracy under every split protocol. Where the recorded runs genuinely differ, moving from a random window split to a run-level split lowers accuracy by 13–20 points. Under cross-dataset transfer, balanced accuracy ranges from 0.04 to 0.61 against a chance level of 0.333 — collapse is total in one direction and partial in others, and the severity is predictable: it tracks the distance between the two rigs in a standardised descriptor space (Spearman ρ = −0.96, p = 0.003, six directions). The mechanism is a change of scale in the physical cue (the median amplitude ratio between healthy and faulty windows is 5–8× in one rig and 1.4–1.9× in another; median excess kurtosis of inner-race windows is 3.0, 36.9 and 1.1 in the three rigs). With 20 labelled samples per class from the target machine, the feature-based classifier reaches a mean balanced accuracy of 0.88 across directions, against 0.78 and 0.71 for the two deep models.

**Why it may interest readers outside the field.** The result is not about bearings as such. It is a concrete, quantified demonstration of how a widely used evaluation convention inflates reported performance in applied machine learning, of what governs whether that performance survives a change of setting, and of what a minimal reporting standard would look like for this class of studies. We end with a five-item checklist — the unit of the train/test split, per-class recall alongside aggregate accuracy, the chance level of the label set, the normalisation scheme, and one result on data recorded elsewhere — that we believe general readers and referees in any data-driven applied field can apply.

All three datasets are public, all code and result files are already released at
<https://github.com/liu-4564/bearing-transfer-audit>, and the study involves no human or animal
subjects.

I would be grateful for your guidance on whether this falls within the journal's scope, and on the most appropriate article category. I can send the full manuscript immediately.

Yours sincerely,

Jiarui Liu
Qingdao University of Technology, Linyi, China
ages257@outlook.com | ORCID: 0009-0002-3035-9295
