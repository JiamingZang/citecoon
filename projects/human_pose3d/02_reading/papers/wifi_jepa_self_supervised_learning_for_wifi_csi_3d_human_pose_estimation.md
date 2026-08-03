# WiFi-JEPA: Self-supervised Learning for WiFi-CSI 3D Human Pose Estimation

> 2026 · id: W7168432887 · arXiv: 2607.11064 · pdf: https://arxiv.org/pdf/2607.11064 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Human pose estimation (HPE) is a fundamental component of systems for health
monitoring, human–computer interaction, and safety-critical perception. Despite
rapid progress in camera-based HPE under line-of-sight conditions [20, 21, 27],
vision is often limited in camera-denied scenarios: (i) occlusion by walls or ob-
stacles, (ii) low or no illumination, and (iii) privacy and regulatory constraints
that prohibit visual capture in sensitive spaces.
⋆These authors contributed equally to this work.
arXiv:2607.11064v1  [cs.CV]  13 Jul 2026

2
D. Kim, J. Lee et al.
WiFi-JEPA
Simulation
Real
CSI Generation Scene
GT
Prediction
Input CSI
Fig. 1: Overall framework of WiFi-JEPA. Left: Pre-training data — sim-object and
real CSI from PiW3D. Center: Generated CSI input and WiFi-JEPA. Right: GT and
predicted 3D poses.
WiFi channel state information (CSI) provides a compelling alternative.
WiFi signals are ubiquitous indoors and can propagate through many common
obstructions. Human motion modulates multipath propagation, which is cap-
tured as CSI—a factored, complex-valued measurement over subcarriers (fre-
quency), time, and multiple Tx–Rx antenna links. Prior work has shown that
WiFi sensing can recover human-centric outputs without visual imagery, includ-
ing fine-grained person perception and pose-related estimates [19, 23, 28]. Most
notably, PiW3D [28] demonstrates the feasibility of multi-person 3D pose estima-
tion with commodity WiFi even under visual occlusion, reporting a ∼90K-frame
dataset collected in multiple indoor areas and 3D joint localization errors on the
order of ∼100 mm.
However, robust WiFi-based HPE still faces three bottlenecks. First, cross-
domain generalization is fragile: performance can degrade when deployment
conditions differ from training, such as changes in transceiver placement, sur-
rounding objects and furniture. Second, label scalability is limited: CSI 3D pose
datasets typically rely on camera-based annotation pipelines and are collected
in a small number of rooms with fixed hardware [28]. Third, CSI is noisy and
hardware-dependent. A common approach is to reshape CSI into image-like 2D
grids to leverage ViT/CNN backbones; this axis-mixing can conflate the physical
semantics of subcarriers, time, and links and encourage reliance on device-specific
artifacts over pose-relevant dynamics.
These challenges motivate CSI-native representation learning that (i) is ro-
bust to environment and hardware shifts, (ii) scales without dense camera-
derived labels, and (iii) preserves CSI structure while exploiting multi-link spatial
diversity.

WiFi-JEPA
3
We propose WiFi-JEPA, a self-supervised learning (SSL) framework that
learns transferable CSI representations without reconstructing raw CSI. WiFi-
JEPA uses a CSI-specific tokenization and predicts masked latent representa-
tions. We further introduce link masking, which masks entire Tx-Rx link obser-
vations and forces prediction from the remaining links, explicitly leveraging the
multi-link spatial views intrinsic to CSI and central to pose recoverability. To re-
duce dependence on scarce labels, we also propose a CSI simulation pipeline
via ray-tracing (NVIDIA Sionna [15]).
Our main contributions are:
– WiFi-JEPA: a CSI-native SSL framework with CSI-specific tokenization
and link masking that learns CSI representations by predicting masked latent
embeddings, improving cross-domain robustness.
– CSI simulation pipeline (sim-object): a ray-tracing-based pre-training
paradigm providing evidence that dynamics diversity may matter more than
geometric realism for transfer; geometric primitives outperform human meshes
(sim-human, 100.1 vs. 110.3 mm MPJPE) and ∼90K simulated frames match
∼90K real frames in independent pre-training value.
– State-of-the-art WiFi-based results: combining real and simulated pre-
training achieves 76.8 mm single-person and 93.5 mm multi-person MPJPE
on PiW3D, improving over all prior WiFi HPE baselines under the same
evaluation protocol [9,28,29].
2

## method
Metric
Off.
Class.
Corr.
WiFi-JEPA
(ours)
MPJPE
248.4
428.2
296.0
Mean
324.2
PiW3D
(baseline)
Mean
626.4
Fig. 7:
Amplitude distribution shift
across 3 environments.

12
D. Kim, J. Lee et al.
room, and 296.0 mm on Corridor, yielding a mean MPJPE of 324.2 mm—a 48.2%
reduction over the PiW3D baseline (626.4 mm). While absolute errors remain
high, WiFi-JEPA halves the baseline error. This is consistent with the masking
ablation in Section 5.4, where link masking consistently outperforms alternative
strategies, indicating that cross-link correlations are key to environment-robust
representations. Furthermore, we evaluate WiFi-JEPA against a standard su-
pervised Domain Adaptation (DA) baseline, DANN [11], in a few-shot cross-
environment setup utilizing only 10% of target environment labels. WiFi-JEPA
consistently outperforms Supervised+DANN across all held-out environments,
reducing the macro mean MPJPE from 208.6 mm to 171.3 mm (a 17.9% er-
ror reduction). This demonstrates that our SSL pre-training yields substantial
generalization benefits that go beyond standard adversarial DA techniques.
Table 3: Per-person-count MPJPE
(mm). Performance degrades gracefully as
the number of simultaneous persons in-
creases.
Metric
Method 1P
2P
3P
MPJPE↓PiW3D 91.7 108.1 125.3
Ours
76.8 96.0 110.7
Table 4: Extremity error breakdown.
Mean L/R elbow and hand errors with
directional decomposition (mm). PiW3D:
Person-in-WiFi-3D baseline [28].
Metric
Elbows
Hands
PiW3D Ours PiW3D Ours
MPJPE↓
128.5
60.6
192.5
68.7
Horiz↓
68.2
31.1
94.4
34.4
Vert↓
56.8
35.0
104.8
40.0
Depth↓
73.9
23.2
92.9
26.4
Improv.
↓52.8%
↓64.3%
Multi-Person Scenes and Per-joint Analysis A known limitation of WiFi-CSI
pose estimation is performance degradation in multi-person scenarios, where
overlapping signals make it difficult to disentangle individual poses. WiFi-JEPA
improves across all person counts: single-person MPJPE drops from 91.7 mm to
76.8 mm (−16.2%), two-person from 108.1 mm to 96.0 mm (−11.2%), and three-
person from 125.3 mm to 110.7 mm (−11.7%) (Table 3).
The PiW3D baseline struggles most with elbows and hands (160.5 mm), be-
cause arms exhibit faster, more varied motion and occupy a smaller spatial
cross-section [28]. WiFi-JEPA reduces this to 64.7 mm, a 59.7% improvement
(Table 4), with depth error dropping from 92.9 mm to 26.4 mm for hands.
5.3
Effect of Simulated CSI
We now examine the hypothesis from Sec. 3. Table 5 presents a dataset-level
comparison: all rows use the same WiFi-JEPA encoder, PETR decoder, and
fine-tuning protocol; only the pre-training data varies. We additionally compare

WiFi-JEPA
13
against sim-human, a human-mesh variant that uses Blender-animated char-
acters with motion-captured animation clips instead of geometric primitives.
Both simulated datasets use the same number of clips and are sampled to 90K
frames, so the comparison isolates the effect of scene content: geometric primi-
tives with randomized physics trajectories (sim-object) versus animated human
meshes with motion-capture sequences (sim-human).
Table 5: Effect of pre-training data on downstream MPJPE (mm). All rows share the
same encoder, decoder, and fine-tuning protocol. ∆: improvement over no pre-training.
Pre-train data
Frames MPJPE↓
∆
Real Sim
None (scratch)
—
—
102.4
—
sim-object
—
90K
100.1
−2.3
sim-human
—
90K
110.3
+7.9
Real only
90K
—
97.1
−5.3
Real + sim-object 45K 45K
97.2
−5.2
Real + sim-object 90K 90K
93.5
−8.9
sim-object suffices for pre-training. Pre-training on sim-object (100.1 mm)
outperforms sim-human (110.3 mm), with the latter causing negative transfer
(+7.9 mm worse than no pre-training), likely because the fixed motion-capture
sequences lack the trajectory variation needed to learn generalizable channel
features. Because sim-human uses anatomically realistic scatterers yet performs
worse, the result directly supports the hypothesis that trajectory diversity—
randomized physics with varied velocities and elastic reflections—is more impor-
tant than scatterer realism for CSI pre-training. Furthermore, sim-object alone
(100.1 mm) is close to real-data-only pre-training (97.1 mm), with a gap of only
3.0 mm (2.3 vs. 5.3 mm reduction from scratch). That is, ∼90K simulated frames
provide comparable pre-training value to ∼90K real frames.
Complementarity of Simulated and Real Data. Combining 90K simulated
and 90K real frames reduces MPJPE by 8.9 mm over training from scratch,
compared with a 5.3 mm reduction from real-only pre-training. This suggests
that the simulation pipeline provides channel variation patterns absent from the
real dataset, such as diverse room geometries and wall materials. The simulated
data is generated at a cost of ∼10 GPU-hours on one RTX 4090 (Sec. 3), while
the real frames simply reuse the existing training set as unlabeled data.
5.4
Effect of WiFi-JEPA Pre-training

14
D. Kim, J. Lee et al.
Table 6: All methods use the same ViT backbone and PETR-style decoder, pre-
trained on ∼90K real frames for 100 epochs, then fine-tuned identically. Best in bold,
second-best underlined.

## experiments
We evaluate WiFi-JEPA on the PiW3D dataset [28], the only public WiFi-CSI
benchmark that includes multi-person 3D pose annotations. Our experiments
address three questions: (i) How much does WiFi-JEPA advance the state of
the art in WiFi-based 3D pose estimation? (ii) Why JEPA over alternative SSL
objectives, and do the proposed structure-aware tokenization and link masking
contribute to downstream performance? (iii) Can sim-object complement or even
replace real data for pre-training?
5.1
Experimental Setup
Dataset. PiW3D [28] provides synchronized WiFi-CSI and 3D pose labels (14
joints) captured in three indoor environments—an office, a classroom, and a
corridor—each measuring approximately 4 m × 3.5 m, using one transmitter and
three Intel 5300 receivers (three antennas each) at 5.64 GHz with 30 subcarriers.
Seven volunteers performed eight daily actions (e.g., walking, stretching, bend-
ing over, sitting down) across these environments, yielding diverse multipath
conditions. Each CSI sample has raw dimensions 1×3×3×30×20 (#Tx, #Rx,
#Ant, #Subcarrier, #Time). The training set contains 89,946 frames; the test
set contains 7,824 frames spanning single-person (2,586), two-person (3,184), and
three-person (2,054) scenarios.
Metrics. We report Mean Per-Joint Position Error (MPJPE, mm) as the
primary metric—the mean Euclidean distance between predicted and ground-
truth 3D joint positions. We additionally report Procrustes-aligned MPJPE
(PA-MPJPE), which removes global translation, rotation, and scale to isolate
articulated pose accuracy, and Percentage of Correct Keypoints (PCK@τ) at
τ=20, 50 mm. For finer-grained analysis, we also report per-dimension absolute
errors along the horizontal (x), vertical (y), and depth (z) axes, following [28].

10
D. Kim, J. Lee et al.
Table 1: Comparison with prior WiFi-CSI pose estimation methods on PiW3D
dataset. Best in bold, second-best underlined. “–” (gray) indicates metrics not re-
ported in the original paper. † Values taken directly from the original publications.

## related_work
2.1
Self-supervised representation learning
Self-supervised learning has evolved through three paradigms: contrastive meth-
ods (SimCLR [7], MoCo v3 [8]), momentum-teacher approaches (BYOL [13],
DINO [6]), and masked prediction. MAE [14] reconstructs masked inputs in
pixel space, retaining any low-level noise, whereas JEPA [3] predicts targets in a
learned latent space that need not preserve such artifacts. This latent objective
is particularly relevant to WiFi CSI, where raw measurements carry hardware-
specific distortions (clock offsets, quantization noise) irrelevant to downstream
sensing. Structured masking has proven effective beyond images: V-JEPA [2,
5] showed that structured spatio-temporal block masking outperforms random
masking for learning temporal structure in video.
Several recent works apply SSL to WiFi Channel State Information (CSI).
SSLCSI [26] systematically benchmarked four categories of SSL algorithms for
CSI-based activity recognition. CIG-MAE [18] learns to allocate masking based
on per-patch information density but does not differentiate masking strategies
across distinct physical axes of CSI. AM-FM [30] scales WiFi pre-training to
9.2 M samples across nine downstream tasks including activity recognition, ges-
ture recognition, and WiFi imaging; however, none addresses multi-person 3D
skeletal pose estimation. In the broader wireless domain, WirelessJEPA [10]

4
D. Kim, J. Lee et al.
applies JEPA to raw multi-antenna in-phase/quadrature streams for communi-
cation and RF classification; its inputs and objectives differ fundamentally from
estimated CSI for human sensing. Many CSI-SSL methods adapt image-domain
augmentation or masking schemes without explicitly modeling the distinct phys-
ical semantics of CSI’s frequency, spatial, and temporal axes.
2.2
WiFi-based Human Pose Estimation
Human motion modulates WiFi multipath propagation, enabling through-wall
sensing without cameras. WiPose [16] first demonstrated single-person 3D WiFi
pose estimation with a CNN–LSTM architecture. PiW3D [28] introduced the
first multi-person 3D benchmark with a Transformer-based pose decoder.
MetaFi++ [29] processed each receiver through a shared CNN and fused features
via Transformer self-attention. HPE-Li [12] used dual selective-kernel CNNs with
teacher–student distillation. These supervised methods all require synchronized
visual supervision (motion capture or camera-based pose annotations), limiting
data scalability.
To our knowledge, DT-Pose [9] is the only prior SSL method for WiFi pose
estimation. It combines MAE pretraining with temporal contrastive learning
and uniformity regularization, and uses a GCN–Transformer decoder that en-
forces skeleton topology constraints. Its tokenization flattens CSI into a 2D grid
with fixed sinusoidal positional encoding, jointly encoding all three axes into a
single sequence without preserving their independence. Because DT-Pose fur-
ther employs a decoder with explicit skeletal-topology constraints (an inductive
bias absent from our PETR-based decoder), we control decoder architecture and
isolate the effect of the pre-training objective in Sec. 5.4.
2.3
Simulation data for representation learning
Domain randomization [22] showed that training on diverse synthetic data with
randomized scene parameters enables sim-to-real transfer without photorealistic
rendering. FractalDB [17] and Dead Leaves [4] extended this idea to representa-
tion learning, showing that vision backbones pretrained on procedurally gener-
ated images can learn transferable features, confirming that structural diversity
matters more than geometric realism. This principle is particularly relevant for
CSI, where multipath fading causes even simple moving scatterers to produce
rich channel variations that encode spatial dynamics—making the diversity of
motion patterns more informative than the geometric fidelity of the scatterer
itself. In the wireless domain, ray-tracing tools such as Sionna [15] and synthetic
channel frameworks like DeepMIMO [1] can generate physically grounded chan-
nel data and have been used for supervised tasks (channel estimation, beam
prediction), but not for self-supervised pretraining of sensing models. Across
these three threads, no prior work combines axis-aware masking with latent-
space prediction tailored to the physical structure of CSI, nor has ray-tracing
simulation been used for self-supervised pretraining in WiFi sensing.

WiFi-JEPA
5
3
Simulated CSI from Geometric Primitives
Collecting large-scale WiFi CSI datasets is expensive, requiring dedicated in-
door environments, synchronized receivers, and co-located motion capture for
labeling. We bypass this bottleneck with a simulation pipeline that generates
benchmark-compatible CSI from scenes of simple geometric shapes—no human
models or external motion data required. We refer to this geometric-primitive
dataset as sim-object throughout; a human-mesh variant (sim-human) is evalu-
ated as a comparison in Sec. 5.3.
CPU-only, <1s / clip
Stage 1 : Scene Generation
20 independent RT passes per frame
Stage 2 : Ray-Tracing Simulation
x20
FractalDB [17]
Dead Leaves [4]
Sim-object (Ours)
Fig. 2: sim-object pipeline. Top: Analogy to FractalDB and Dead Leaves—geometric
primitives replace human models. Bottom: Stage 1 generates randomized scenes
(CPU-only); Stage 2 runs Sionna RT with ×20 independent passes per frame.
Inspired by the success of non-semantic pre-training in vision (Sec. 2.3), we
randomize room geometry, object count, trajectory, and wall materials to cover
a wide range of channel conditions.
Hypothesis. We hypothesize that SSL pre-training may not require CSI from
real human poses; rather, it needs exposure to how WiFi signals vary across
time, frequency, and space. At 5.64 GHz (wavelength ≈5 cm), a sphere and
a human body differ substantially in scattering behavior—the body is articu-
lated, non-convex, and has complex dielectric properties. Nevertheless, SSL pre-
training may still succeed with geometric primitives, because the learning objec-
tive benefits from diverse spatio-temporal channel variation, rather than faithful
reproduction of body-specific multipath. Specifically, we test whether dynam-
ics diversity—the range of positions, velocities, and directions across training
scenes—matters more than geometric fidelity in Sec. 5.3.
Stage 1: Scene generation. A pure-Python generator (CPU-only, <1 s/clip) cre-
ates randomized indoor scenes. Room dimensions are sampled uniformly (3–8 m
per side, 2.5–4 m height), with walls assigned ITU-standard materials (concrete,

6
D. Kim, J. Lee et al.
plasterboard, wood, glass). Each scene contains 1–4 geometric primitives (sphere,
cube, cylinder, ellipsoid; radius 0.1–0.5 m) that follow physics-based trajecto-
ries: initial velocities are sampled up to 3 m/s—exceeding typical indoor speeds
per [22]—with elastic wall reflections producing varied spatial coverage. A single
transmitter and three receivers are placed at randomized wall positions match-
ing the SIMO configuration of the target benchmark (1 Tx, 3×3 Rx antennas =
9 links). The real PiW3D training set contains ∼90K frames; our 90K simulated
frames thus constitute a comparable pre-training corpus.
Stage 2: Ray-tracing simulation. NVIDIA Sionna RT [15] simulates radio propa-
gation in each scene, configured to match the target benchmark. A critical design
choice is performing 20 independent ray-tracing passes per frame: each pass
computes the channel for a static scene snapshot, so we sample object positions
at 20 sub-frame locations within each measurement window (∼50 ms). Without
this, a single pass yields a near-constant time axis, rendering temporal masking
ineffective; 20 passes produce realistic temporal variation matching real data.
The output H ∈C20×30×3×3 (time × 30 subcarrier groups × 3 receivers × 3 an-
tennas per Rx) is decomposed into amplitude and phase, yielding a real-valued
tensor of shape (60, 20, 9): 60=2×30 subcarrier channels, 20 time samples, and
9=3 Rx×3 ant links—analogous to a multi-channel spectrogram—for the tok-
enizer (Sec. 4.1). Generating 90K frames takes ∼10 GPU-hours on one RTX
4090.
4
WiFi-JEPA
4.1
CSI-specific Tokenization
(a) Flat Spectrogram (1,60,180)
(b) Structured CSI (60,20,9)
Time (20)
Time x Ant x Rx (180)
Subcarrier (60)
amplitude (30)
phase (30)
Fig. 3: (a) Flattening mixes temporal and link dimensions, causing patches to cross
physical boundaries. (b) Our CSI-specific tokenization keeps (T, L) separate so each
token corresponds to a specific spatio-temporal coordinate.
The raw CSI data in the PiW3D dataset [28] is a tensor of shape Nrx ×Nant ×
T ×Nc = 3×3×20×30, where Nrx is the number of receivers, Nant is the number
of antennas per receiver, T is the number of time steps, and Nc is the number of
subcarriers. Following [28], we use amplitude directly and denoise phase values
using PhaseFi [24], resulting in a real-valued tensor of shape 3 × 3 × 20 × 60,
where 60 = 2 × Nc (30 ampl

## conclusion
We presented WiFi-JEPA, a self-supervised framework for WiFi-CSI-based 3D
human pose estimation that predicts masked latent embeddings on the factored
(C, T, L) CSI tensor. Structure-aware tokenization and link masking capture
cross-link spatial correlations central to body localization, while ray-tracing sim-
ulation provides scalable pre-training data without annotation. Across single-
and multi-person 3D pose estimation on PiW3D, WiFi-JEPA sets a new state of
the art and nearly halves cross-environment error, and it consistently improves
over training from scratch, whereas four vision-native SSL objectives degrade it.
Notably, simulated CSI from simple moving primitives offers pre-training value
comparable to real data, indicating that the diversity of channel dynamics mat-
ters more than the geometric realism of the scatterer.
Limitations. WiFi-JEPA improves global joint localization more than relative
joint configuration (PA-MPJPE stays close to the from-scratch baseline). Cross-
environment error, though substantially reduced, remains far higher than same-
environment performance. Our evaluation is confined to a single dataset with
fixed hardware, so generalization across antenna configurations and frequency
bands remains open.
More broadly, CSI-native self-supervised learning, which respects the physical
structure of the wireless channel rather than treating CSI as a generic image, is
a promising direction for WiFi sensing beyond pose estimation.
Acknowledgements This work was supported by the G-LAMP Program of the
National Research Foundation of Korea (NRF) grant funded by the Ministry of
Education (No. RS-2025-25441317); the Ministry of Science and ICT (MSIT)
and the National IT Industry Promotion Agency (NIPA) through the Advanced
GPU Utilization Support Program (02-26-01-0499); the NRF grant funded by the
MSIT (RS-2025-16071992); the Korea Institute for Advancement of Technology
(KIAT) grant funded by the Korea government(MOTIE) (RS-2026-25530975,
HRD Program for Industrial Innovation); the MSIT under the Convergence se-
curity core talent training business support program(IITP-2024 2024-RS-2024-
00426853) supervised by the IITP(Institute of Information & Communications
Technology Planning & Evaluation); and the IITP grant funded by the MSIT
(No. 2022-0-00986, Development of artificial intelligence-based base station elec-
tromagnetic wave human exposure prediction algorithm).

16
D. Kim, J. Lee et al.