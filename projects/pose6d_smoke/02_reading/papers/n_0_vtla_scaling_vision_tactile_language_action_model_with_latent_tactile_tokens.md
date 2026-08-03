# $N_0$-VTLA: Scaling Vision-Tactile-Language-Action Model with Latent Tactile Tokens

> 2026 · id: arxiv:2607.23782 · arXiv: 2607.23782 · pdf: https://arxiv.org/pdf/2607.23782 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

N0-VTLA: Scaling Vision–Tactile–Language–
Action Model with Latent Tactile Tokens
NeoteAI Team & Fudan TEAI Team
We present N0-VTLA, a vision–tactile–language–action (VTLA) foundation model capable of (1) fine-
grained contact-rich manipulation with tactile perception and tactile-feedback control, and (2) offline
policy improvement from stored deployment data. Stepping towards current visual-based backbones,
we propose an overall training recipe for tactile integration, consisting of visuo-tactile pre-training,
staged tactile-pathway integration, and advantage-conditioned offline policy improvement. During
pre-training, the policy learns broad contact priors from NeoData, our large-scale visuo-tactile robot
dataset. To our knowledge, N0-VTLA is the first VTLA model pretrained on tactile data at scale.
During post-training, we augment the policy with a predictive tactile pathway, distilling the contact
patterns learned at scale into the fine motion adjustments in downstream tactile-centric manipulation.
For offline policy improvement, we introduce ALTER, an advantage-conditioned offline Reinforcement
Learning (RL) method that converts relative progress and trajectory-event comparisons into binary
advantage labels for policy training on a fixed deployment corpus. This procedure further improves
task-specific learning on contact-rich skills such as deformable object manipulation. Across contact-
rich benchmarks, N0-VTLA outperforms strong baselines by wide margins: it wins all nine real-robot
NeoReal tasks and reaches 63.8% mean success on the twenty-task simulation suite against 44.0% for
the strongest baseline. N0-VTLA policies trained with ALTER reach 75–95% success on three long-
horizon real-robot tasks. Results lay a foundation for versatile tactile-driven manipulation policies.
Date: July 25, 2026
Code: https://github.com/neoteai/N0-VTLA
Website: https://research.neoteai.com/n0-vtla/
1
Introduction
Vision–language–action (VLA) models have made manipulation policies general. Fine-tuned from pretrained
vision–language backbones, they follow instructions and transfer across tasks, scenes, and embodiments
[9, 37, 58]. Touch, however, has remained largely absent from this progress, leaving current policies with a
persistent weakness in contact-rich manipulation: the tactile extensions attempted so far train on task-scale
collections, at most tens of hours gathered for a handful of skills. This report presents, to our knowledge,
the first VLA policy pretrained on tactile data at scale. We scope the report to vision-based tactile sensing
[39, 86], instrumenting each gripper finger with our self-developed sensor so that contact is read out as an
image. This signal is nothing like a camera view: tactile frames are noisy, nearly empty away from contact,
and informative almost only in the brief windows when contact forms or is about to change.
Existing systems integrate touch along one of two paths. The first concatenates tactile tokens into the
vision–language context and treats the tactile stream as one more camera [22, 36]; yet a signal that is sparse
and mostly silent buys little in a prefix built for information-dense views. The second injects the current
tactile reading into the action pathway to guide denoising [77, 91]; this placement mischaracterizes the role
of touch, since a tactile frame records contact that actions already taken have produced and, by itself, says
little about the contact the next actions must anticipate. Conditioned on it alone, the policy stays one step
behind its own contact events.
N0-VTLA, which we read as NeoVTLA, takes a third path: it keeps tactile out of the vision–language
prefix and conditions the action expert on a prediction of touch rather than the current reading. A small
predictor reads the vision–language context together with the current tactile tokens and emits latent tactile
1
arXiv:2607.23782v1  [cs.RO]  26 Jul 2026

𝒩!-VTLA
Scaling Vision–Tactile–Language Action 
Model with Latent Tactile Tokens
Model Architecture
Action
Expert
Latent 
Tactile 
Predictor
Vision 
Language
Model
Execution
z: Predicted Future Tactile 
ALTER Policy Training
Experiments
“Insert the Gears”
Vision
Tactile
Pairwise 
Progress 
Model
Aθ(xa,xb)
Dense 
Progress 
Pairs
Event 
Preference 
Pairs
1
2
3
Positive
Negative
··
·
Stage-Relative 
Advantage Labeling
𝒩!-VTLA
+
ALTER
6
;6
766

)/ -)g7
+$6`;
6g
<=`7
:7`:
>9`7
)$
6
;6
766
$*($g*6
+$6`;
6g
89`:
:;`>
;6`>
 *$(
6
;6
766

+$6`;
6g
76`8
:8`9
;<`>
 * '
Figure 1 N0-VTLA at a glance. N0-VTLA encodes vision, the instruction, and tactile difference images, predicts latent
tactile tokens z, and conditions a flow-matching action expert on them. Beyond demonstrations, ALTER converts
deployment experience into stage-relative advantage labels for offline policy learning. The experiment panels plot the
headline means against the base VLA policy and the strongest specialist baseline on UniVTAC [18], NeoSim, and
NeoReal. N0-VTLA leads every panel.
tokens z that estimate the net tactile change over the coming action chunk, so the policy acts on the contact
its own actions are about to cause. The tactile frames are contact-difference images encoded by a frozen
pretrained visual encoder through a lightweight trainable projection, and a three-stage recipe brings this
newly initialized pathway online, as Figure 1 shows.
Beyond supervised task adaptation, we formulate
learning from stored deployment experience as advantage-conditioned offline RL. ALTER trains a pairwise
progress model from clean demonstrations, tactile-detected object-drop events, and logged human corrections,
then assigns stage-relative binary advantage conditions for policy learning.
The full system rests on a vision–language–action backbone built on PaliGemma [6], a canonical cross-
embodiment action space, and NeoData, a multi-platform visuo-tactile corpus spanning single and dual-arm
configurations, documented in a companion data report [51]. Before evaluating the full system, we verify the
latent pathway itself: after Stage 1, the latent tokens retrieve their matching future-tactile targets at 92.3%
top-1 accuracy, where chance sits at 3.2%. Whether this grounded representation translates into better task
performance is the more demanding test. We evaluate N0-VTLA against external baselines on the NeoReal
real-robot benchmark and the simulated contact-rich suite under identical protocols. On the twenty-task
simulation suite N0-VTLA leads the strongest baseline by a wide margin, and it wins all nine real-robot
NeoReal tasks.
In summary, this report makes three contributions:
• Large-scale tactile pretraining (§2.1). N0-VTLA is pretrained on NeoData, the large-scale visuo-tactile
robot data across multiple robot platforms, made trainable as one model by a canonical cross-embodiment
action space and a quality-verified data pipeline [51].
• Latent tactile tokens (§2.2). Touch is treated as a prediction target rather than as observation context,
a predictor estimating the tactile change over the coming action chunk and conditioning the action
expert directly, brought online stably by a three-stage recipe.
• OfflinepolicyimprovementwithALTER(§4.4). A pairwise progress model, supervised by tactile-grounded
stage annotations, tactile-detected object-drop events, and logged human corrections, produces stage-
2

relative advantage labels for offline policy learning. The method applies to both the base VLA policy
and N0-VTLA, with N0-VTLA+ALTER achieving the highest success on all three tasks.
2
Model
N0-VTLA is a policy for contact-rich manipulation. It reads camera views, a language instruction, robot
state, and touch, and it generates a chunk of future actions. The model consists of a pretrained vision–
language–action backbone built on PaliGemma [6] and one added component, a latent tactile pathway
between perception and action. The backbone carries the views, instruction, and state in its vision–language
prefix and generates the action chunk with a flow-matching action expert. In the added pathway, a frozen-
backbone tactile encoder turns each finger’s contact-difference image into tokens, and a small predictor
distills those tokens, in the context of the scene and instruction, into latent tactile tokens z that estimate the
net contact change expected over the coming action chunk. The action expert is conditioned on z directly,
and tactile never enters the vision–language prefix. Touch therefore enters the policy as a prediction target
rather than as one more observation. Figures 2–4 lay out this design as a three-step recipe, and the section
follows them. Section 2.1 fixes the base policy, and Section 2.2 walks through the tactile pathway step by
step.
2.1
Base Architecture
The base policy pairs a PaliGemma vision–language backbone [6] with a flow-matching action expert [43].
Camera views, the instruction, and the robot state form the model prefix, state entering that prefix in
discretized form rather than as a separate continuous input. Conditioned on the prefix, the expert denoises
an action chunk over a horizon of H = 50 steps in the canonical 32-dimensional container of Section 3, whose
width and slot layout we inherit unchanged from the pretrained action head so that its weights load directly.
The flow-matching objective is unmasked over all 32 dimensions. Each platform populates the dimensions its
embodiment uses, the rest carry zero targets the model learns to reproduce, and single- and dual-arm data
therefore coexist in one model under a single fixed-width objective. All other aspects, including architecture,
tokenization, and training procedure, are inherited unchanged from the pretrained backbone.
2.2
Latent Tactile Tokens
Figure 2 shows Step 1, in which the tactile predictor is trained against a future-tactile target. Figures 3
and 4 show Steps 2 and 3, in which the action expert first learns to consume the resulting latents while the
vision–language pathway is masked, and the full policy then trains end to end. The model that leaves Step 3
is the deployed controller. The paragraphs below introduce each component in the same order.
The tactile encoder.
Every panel of both figures begins the same way. The policy never sees a raw tactile
frame. For view k we subtract the episode-start baseline frame tack
0, the zero-contact reference established in
Section 3, from the current frame tack
τ in pixel space, and encode the difference with a frozen self-supervised
visual encoder, followed by a trainable linear projection to the shared token width d of the vision–language
backbone:
gk = fenc
(
tack
τ −tack
0
)
∈R10×d,
(1)
where fenc denotes the frozen encoder composed with the trainable projection. Each tactile image yields
10 tokens, one class token and nine spatial tokens from a 3 × 3 adaptive average pool over the encoder’s
16 × 16 patch grid, and the tokens of the n active views are concatenated into g = [ g1; . . . ; gn ] ∈R10n×d.
Differencing against a per-episode baseline, rather than encoding the absolute gel image, removes the static
gel appearance and much of the mount-specific imprint, making the representation robust to, though not
strictly invariant under, differences in sensor placement. Freezing the encoder is deliberate. It preserves
the self-supervised representation intact, it lets a previously unseen sensor be onboarded by training only
the lightweight projection, and it removes the encoder’s activations and optimizer state from the training
memory budget.
3

Vision
Language
Vision 
Language 
Model
Latent 
Tactile 
Predictor
Tactile
Encoder
Tactile
Observed
····
z: Predicted 
Tactile Latent
Loss Computation
(z* - z)
Back Propagate
···
Tactile
Encoder
Future
Tactile
Latent Tactile Predictor Training
Tactile Token
Action Token
Vision Language Context
Predicted Tactile Latent
Figure 2 Step 1: latent tactile predictor training. The current tactile difference is encoded into tokens g. The predictor
reads g together with the contextualized vision–language prefix and emits the latent tactile tokens z. The future-
tactile target z∗comes from the same tactile encoder applied to the coming tactile change, and the loss on z∗and z
backpropagates into the predictor alone.
Step 1: the predictor and its future-tactile target.
The predictor is a lightweight module that reads
the current tactile difference tokens g in the context of the scene and instruction, carried by the contextualized
vision–language prefix, and distills them into a compact set of learned latent queries that become the latent
tactile tokens z. When an episode carries no tactile at all, a learned null token stands in for g, so z is always
produced and the policy falls back to vision–language control rather than failing. What makes z predictive
rather than merely descriptive is the target Figure 2 attaches to it,
z∗= 1
n
n
∑
k=1
fenc
(
tack
τ+H −tack
τ
)
∈R10×d,
H = 50,
(2)
obtained by applying the same tactile encoder to each view’s tactile change over the next H steps and
averaging the resulting tokens across the n active views. The predictor output z ∈R10×d is trained to match
z∗. The supervision combines a symmetric InfoNCE[66] contrastive loss that pulls the predicted latent toward
its matching future-tactile target and an auxiliary L1 reconstruction of a coarse future-tactile-difference field.
This is the supervision with which the three-stage recipe grounds the predictor. A free-latent simplified
variant omits the Step 1 supervision entirely, shaping z through the action gradient alone. It requires no
future frames during training.
Step 2: conditioning the action expert.
The latent tokens z are projected to the action-expert width
and prepended to the action suﬀix, ahead of the noisy action tokens. The current-contact tokens g never
enter the action expert. They reach action generation only through the predictor, which distills them into
z. The latent tokens form their own conditioning block. The action tokens attend to z and, as in the base
policy, to the vision–language prefix, the prefix never attends back to the latent tokens, and the z positions
are sliced off before the action output head so that they never emit actions. Because the pretrained expert
has never consumed such a token, Step 2 (Figure 3) trains this interface in isolation. The vision–language
pathway is masked so that action prediction must draw on z, aligning the latents with the expert before
anything else moves.
Step 3: training the full policy end to end.
Step 3 (Figure 4) removes that mask and opens the
whole policy to joint training. The direct prefix-to-expert path is restored, so the expert again sees scene
and instruction alongside z. Every component except the frozen tactile encoder backbone then adapts under
the action objective. What is frozen at each step, and why the order matters, is the subject of the training
chapter.
4

Action Expert Alignment
Action
Expert
Latent 
Tactile 
Predictor
Vision 
Language 
Model
····
Tactile
Encoder
Tactile
Observed
Vision
Language
z
Tactile Token
Action Token
Vision Language Context
Predicted Tactile Latent
Masked
Predicted 
Action
Figure 3 Step 2: aligning the action expert with the latent tokens. With the predictor and the vision–language backbone
frozen, the vision–language context is masked before the action expert, so action prediction must draw on the latent
tactile tokens z while the expert learns the interface.
Why predict, not react? The design choice is what conditions the action expert. Handing the policy its
current tactile reading, whether concatenated into the prefix or injected alongside the actions, hands
it a record of contact already made, and spends capacity built for information-dense views on a signal
that is sparse and mostly silent. The latent predictor instead asks the model to form an explicit internal
estimate of the contact state, and, under supervision, of the net contact change expected over the chunk
horizon. It then conditions the action head on that estimate. The behaviors where tactile matters most
are anticipatory, such as the millimeter-scale pre-load before a grasp closes and the catch of incipient
slip before the object moves. These live in the pre-contact blind spot, where a purely reactive signal
arrives too late to shape the action that caused it. We therefore hypothesize that conditioning actions
on a predictive latent, rather than on raw current tokens, is the better inductive bias for contact-rich
control. This choice to predict in a learned latent space rather than reconstruct raw sensory signals
follows the joint-embedding predictive principle [1] explored for self-supervised representation learning
in LeJEPA [4] and for latent world modeling from pixels in LeWorldModel [49].
Two observations ground this design choice. What decides a 50-step chunk is the contact the chunk
itself is about to create, and no encoding of the present frame contains it. Ranking future-tactile targets by
the current tactile encoding alone retrieves 57% top-1 where the predictor reaches 92.3%, with the margin
widening as the candidate pool grows, as Section 5.5 details. And the prediction objective pins the latent to
touch before the policy ever optimizes through it, where features shaped by the action gradient alone would
be free to drift into an appearance cue rather than contact state.
3
Data
N0-VTLA is pretrained on NeoData [51], our large-scale curated multi-platform visuo-tactile corpus, span-
ning single- and dual-arm robot manipulators as well as a UMI-style handheld collection gripper [24]. Every
gripper finger that participates in a manipulation task carries our self-developed visuo-tactile sensor, so
contact is read out as a stream of tactile images rather than as a low-dimensional force signal. Collection
protocols, corpus composition, and sensor specifications are documented in the companion data report [51].
This section states only the conventions the rest of the report depends on.
Canonical action and state schema.
All embodiments are unified into one fixed 32-dimensional state
and action container, inherited from π0.5 [58]. The container is laid out for two arms, with the first 20
dimensions split into one 10-dimensional slot per arm and the remaining 12 left unused and always zero.
Within a slot, the 10 dimensions comprise a 3-dimensional end-effector position, a 6-dimensional rot6d
5

End to End Training
Action
Expert
Latent 
Tactile 
Predictor
Vision 
Language 
Model
····
Tactile
Encoder
Tactile
Observed
Vision
Language
z
Tactile Token
Action Token
Vision Language Context
Predicted Tactile Latent
Action
Expert
Predicted 
Action
Figure 4 Step 3: end-to-end training. Everything except the tactile encoder backbone unfreezes and the full policy
trains end to end, emitting the predicted action chunk. The free-latent simplified variant omits the Step 1 supervision,
shaping z through the action gradient alone.
rotation [95], and a 1-dimensional gripper channel.
Dual-arm episodes populate both slots.
Single-arm
episodes populate the first slot only and leave the second zero-filled as well. Single- and dual-arm data
therefore coexist in one fixed-width container, and what the policy does with the zero-filled dimensions is
fixed by the objective of Section 2.1. Actions are stored as absolute end-effector poses. At training time each
chunk is rewritten relative to its own first frame, so the model predicts motion relative to the pose at which
the chunk begins. At deployment the predicted chunk is mapped back to absolute poses through the inverse
of that transform, and inverse kinematics resolves those poses into the joint commands sent to the robot.
Normalization statistics are computed on the chunk-relative representation, separately for each pairing of
robot and action schema.
Tactile collection convention.
Each participating gripper finger contributes one tactile stream, captured
on the same clock as the RGB and proprioceptive channels. The per-platform stream counts and the common
frame rate are listed in the data card of Appendix A. Each episode begins with a short zero-contact baseline,
the gripper open and static for at least 0.5 s. That baseline frame is the episode’s zero-contact reference, and
every tactile frame recorded afterwards is interpreted relative to it.
Data quality verification.
Every converted repository is verified for data quality before it enters training.
Verification checks that a repository is complete and internally valid, that its stored conventions match the
schema above, and that its statistics and media are consistent with what the training pipeline assumes.
Repositories that fail are repaired or excluded rather than trained on. The individual invariants, and the
symptom each produces when it is violated, are catalogued in Appendix C.
Simulated data.
Simulated data enters through the same door. Episodes from the UniVTAC visuo-tactile
simulator [18], which supplies the NeoSim suite evaluated in Section 5.3, are converted into the canonical
schema above and verified alongside real data, so that one policy interface applies to both.
4
Training
N0-VTLA reaches deployment through three core phases. A three-stage recipe then brings the latent tactile
pathway online. Supervised post-training specializes the resulting generalist to individual tasks. After this
core recipe, an optional procedure post-trains the task policy on its own deployment data. Throughout, the
trainable surface grows only after each new interface has been grounded, so the tactile pathway comes online
without destabilizing the pretrained policy.
6

0%
25%
50%
75%
100%
Training progress
0.03
0.05
0.10
0.20
Training loss
warmup
0.027
0%
25%
50%
75%
100%
Training progress
0.1
1.0
Gradient norm
warmup
0.07
Figure 5
Multi-platform visuo-tactile pretraining.
Training loss and gradient norm over multi-platform visuo-tactile
pretraining. The loss descends smoothly and gradient norms stay flat throughout, consistent with the pretrained
initialization transferring cleanly to the visuo-tactile action space.
4.1
Base Pre-training
Base pre-training is conducted at cluster scale on the NeoData corpus. Stability at this scale is achieved
by design, through choices validated in controlled comparisons, and is visible in the smooth loss and flat
gradient norms of Figure 5.
4.2
Three-Stage Latent-Tactile Training
The tactile pathway, newly initialized, attaches to the pretrained multi-platform checkpoint, and the three
steps of Figures 2–4 bring it online, each stage proceeding from the checkpoint the previous one produces.
The free-latent configuration corresponds to collapsing this recipe into a single joint stage with no auxiliary
supervision.
Stage 1: grounding the predictor.
With the entire base policy frozen, we train only the predictor,
the tactile projection, and a lightweight reconstruction head. The predictor output z is pulled toward the
future-tactile target z∗of Eq. 2 by a symmetric InfoNCE[66] objective. Write h(·) for the mean pooling over
the ten latent tokens followed by ℓ2 normalization, and, for a batch of B samples,
sij =
⟨
h(zi), h(z∗
j )
⟩
(3)
for the cosine similarity between the i-th prediction and the j-th target. The contrastive term
LNCE = −1
2B
B
∑
i=1
[
log
esii
∑B
j=1 esij + log
esii
∑B
j=1 esji
]
(4)
matches each predicted latent to its own future target, taking the other targets in the batch as negatives,
and is symmetrized over both retrieval directions. In parallel a reconstruction head rψ decodes z back to a
coarse future-tactile-difference field and is trained with an ℓ1 term Lrec = ∥rψ(z) −¯Dτ→τ+H∥1 against the
same horizon, where ¯Dτ→τ+H is the downsampled contact-change field over the coming chunk. The stage
minimizes
L1 = LNCE + λrec Lrec,
(5)
where λrec > 0 balances the two terms.
The contrastive term supplies the discriminative pressure that
makes z retrieve the right future contact; the reconstruction term anchors it to the spatial layout of that
contact, discouraging a shortcut latent that separates batches without encoding where contact forms. Because
gradients touch only the shallow predictor stack, the stage is cheap and converges quickly. At convergence
the latent tokens are strongly grounded in touch: the latent z retrieves its matching future-tactile target
with 92.3% top-1 accuracy against a 3.2% random baseline, analyzed in Section 5.
7

Stage 2: aligning latents with the action expert.
The pretrained action expert has never consumed
a latent tactile token, so we next teach it the interface. We hold the tactile perception stack frozen at its
Stage 1 checkpoint and train only the latent-to-expert projection and the action expert, under the base
action objective. Concretely, in the expert’s attention the keys and values from the vision–language prefix
are masked out for the action queries, leaving the latent tokens z and the noised action tokens as the only
conditioning the expert can attend to. Masking the prefix removes the shortcut of predicting actions from
scene and instruction alone, so the only route to lowering the action loss runs through z. The expert learns
to read touch through z before any joint training loosens the rest of the policy, in the spirit of the staged
alignment strategies explored for language–action models [67].
Stage 3: end-to-end joint training.
With the predictor grounded and its interface aligned, we unfreeze
everything except the always-frozen tactile encoder backbone and train the full policy jointly under the
standard pre-training recipe, on the action objective alone. The vision–language mask of Stage 2 is removed,
so the direct prefix-to-expert path is restored and the expert again sees scene and instruction alongside z.
Gradients from the action objective now flow together through the predictor, the two projections, the action
expert, and the vision–language backbone, letting the perception stack adapt to what the expert actually
needs. The contrastive and reconstruction targets of Stage 1 are no longer applied. The predictor keeps its
grounding through the action gradient alone, which the perturbation probe of Section 5.5 confirms it retains
rather than reroutes.
Why three stages? The staging turns one hard joint optimization into a curriculum, each stage handing
the next a better starting point. Stage 1 fixes what z means: trained only against the future-tactile
target, the bottleneck is forced to encode contact rather than a second copy of the prefix it already sees.
Stage 2 fixes how the expert uses z: with the prefix masked, the only way to lower the action loss is to
read touch through z, so the expert commits to the new interface while the rest of the policy stays put.
Stage 3 then trains everything jointly, but from an initialization where z is already grounded and already
consumed, so joint optimization refines a working pathway instead of having to discover grounding and
interface at once. The free-latent variant collapses all three into a single joint stage: with no Stage 1
target, the action gradient alone shapes z, and nothing stops the bottleneck from degenerating into
a copy of the prefix or being bypassed altogether. After joint training the latent stays tactile-leaning
(Section 5.5), consistent with Stage 3 refining the grounded pathway rather than rerouting around it. A
randomly initialized tactile pathway can therefore be grafted onto a pretrained VLA and brought online
without destabilizing it.
4.3
Supervised Task Adaptation
We adapt a pretrained N0-VTLA checkpoint to each downstream task by supervised fine-tuning(SFT) on
a few hundred demonstrations, warm-starting from that checkpoint and reusing the pre-training recipe at
reduced scale. Normalization statistics are always recomputed on the task’s own data and never reused from
pre-training, per Section 3. The same recipe covers both real-robot tasks and the simulated task suite, with
one policy per task.
4.4
Offline RL from Deployment Data with ALTER
We call our method ALTER, short for Advantage Labeling from Trajectory Events and Relative Progress.
Given a fixed deployment corpus, ALTER performs advantage-conditioned offline RL [57] without additional
environment interaction, as summarized in Figure 6. Clean demonstrations provide dense progress super-
vision from signal-grounded stage intervals, whose boundaries are localized using tactile contact changes,
end-effector kinematics, gripper state, and visual event cues.
Imperfect deployment trajectories instead
provide sparse before-and-after preferences from tactile-detected object-drop events and logged human-in-
the-loop corrections. These dense and sparse signals jointly train a task-specific pairwise progress model.
We then freeze the model and apply it to every trajectory retained for offline policy learning. Comparing
each observation with the episode start estimates global task phase, while comparing it with the observation
8

Event Preference Pairs
xbad
xgood
Preference: xgood > xbad
From tactile-detected 
object-drop events and 
HIL corrections
Pairwise 
Progress 
Model
Aθ(xa,xb)
Stage-Relative 
Advantage Labeling
1
2
3
Positive
Negative
···
Text 
Condition
𝒩!-VTLA
+
ALTER
Deployment on Positive 
Conditions
Global Task Phase
Initial Frame
Current Frame
Assign Global 
Task Stage
Stage1
Stage2
······
Local Execution Change
Current Frame xt
Future Frame xt+H
Assign In-Stage 
Advantage
Advantage Score: 0.06
Dense Progress Pairs
φ(xb)>φ(xa)
xb
xa
Δφ
In-stage
Advantage
Stage 
Assignment
Stage Annotation
Action Boundary Detection
Boundary 
Fusion
Arm
Kinematics
Gripper 
Events
Motion 
Offsets
Axis
Switch
Tactile
Features
Visual
Clues
Time
Figure 6 ALTER for offline policy improvement. Tactile contact changes, complemented by kinematic and visual cues,
ground stage annotations of clean demonstrations and yield dense progress pairs. Tactile-detected object-drop events
and logged HIL corrections yield sparse preference pairs. Both supervise a pairwise progress model, which is then
frozen to estimate global task phase and local execution change for each stored trajectory. Within each predicted
stage, local-change estimates produce binary advantage labels that are appended to the task prompt during policy
training. At deployment, the task prompt uses the positive label.
one action-chunk later estimates local execution change. Within each predicted stage, we rank samples by
estimated local change, assign positive labels to higher-ranked samples and negative labels to lower-ranked
ones, and append the resulting label to the task prompt used for policy training.
Deployment corpus and offline annotations.
After deploying task-adapted policies, we retain clean
teleoperation demonstrations, autonomous rollouts, and human-in-the-loop (HIL) rollouts. The HIL corpus
also includes staged-recovery episodes that start from selected error states and record human-teleoperated
recovery trajectories. Clean demonstrations receive dense stage-progress annotations from a signal-grounded
pipeline. From representative demonstrations, Gemini-3.5-Flash[25] generates a shared task template com-
prising the L3 objective, ordered L2 stages, and their L1 steps. A human reviews this task-level template
once, after which its vocabulary and ordering remain fixed across demonstrations of that task. For each
demonstration, tactile contact changes identify candidate transition times, supplemented by end-effector
motion, gripper state, and visual GEBD cues [59]. After these candidates are merged and deduplicated, the
VLM maps each resulting interval to an entry in the task template. It therefore assigns stage semantics to
pre-segmented intervals rather than predicting timestamps from the full video. We do not assign monotone
labels to complete autonomous or HIL trajectories because they may regress or retry. Instead, tactile contact
loss localizes object-drop events, while HIL logs mark correction intervals. These timestamps define local
event comparisons. Appendix F reports the task-wise composition of this corpus and provides annotation
examples.
Duration-calibrated stage progress.
For each stage k, we first compute its mean duration ¯dk across
clean demonstrations. Its share of the task progress is
wk =
¯dk
∑K
j=1 ¯dj
,
ϕt =
∑
j<k
wj + wkut.
(6)
Here K is the number of stages, and ut ∈[0, 1] is the fraction of the current stage completed at frame t.
Thus ϕt combines the cumulative weights of completed stages with the duration-scaled fraction of the current
stage. Weighting stages by their mean durations avoids forcing a brief transition and a long manipulation
stage to occupy equal portions of the [0, 1] range.
9

Event-aware pairwise progress learning.
We implement Aθ(xa, xb) ∈[−1, 1] with a π0.5-based paired-
observation progress architecture [83]. Conditioned on the task prompt, the model jointly encodes obser-
vations from two time points, each represented by synchronized RGB images from multiple cameras. A
three-layer MLP head then predicts the relative task progress between them. Tactile, kinematic, and event
signals construct the offline targets but are not inputs to the progress model. For demonstration pairs, we
regress Aθ(xa, xb) toward the target ϕ(xa) −ϕ(xb). We label an observation immediately before an object
drop as higher-progress than one immediately after it. For a HIL correction, we label an observation near
the end of the intervention as higher-progress than one near its start. Let Devent contain triples (xa, xb, y),
where y = +1 indicates that xa is assigned higher progress than xb, and y = −1 indicates the reverse. We
optimize
Lprog = EDstage [Aθ(xa, xb) −(ϕ(xa) −ϕ(xb))]2
+ λeventEDevent [max(0, m −yAθ(xa, xb))]2 .
(7)
Here Dstage contains demonstration pairs with dense progress-difference targets, m > 0 specifies the
minimum signed score yAθ(xa, xb) required of an event pair, and λevent controls the overall contribution of
event comparisons relative to dense demonstration supervision. We randomly reverse event-pair input order
during sampling to prevent a fixed input slot from becoming a shortcut.
Advantage-conditioned offline policy learning.
For every stored episode, the frozen pairwise progress
model produces global progress ˆϕt and local change ˆrt. Samples are assigned to the duration-calibrated stage
containing ˆϕt, then ranked only against other samples from that stage:
ˆϕt = δe + Aθ(xt, xe
0),
ˆrt = Aθ(xmin(t+H,Te−1), xt),
qk = Quantile1−ρ{ˆri : si = k},
ct = 1[ˆrt ≥qst].
(8)
Here xe
0 is the first observation of episode e, Te is its number of frames. The offset δe is zero for episodes
that begin at the nominal task start; staged-recovery episodes use the initialization described in Appendix F.
The estimated progress ˆϕt determines stage index st, and qk is the within-stage threshold.
We use the
action-chunk horizon H = 50 and retain the top ρ = 0.3 within each stage as positive, as indicated by
ct = 1.
We represent this binary indicator as an additional text input.
Samples with ct = 1 receive
Advantage: positive, while the remaining samples receive Advantage: negative. The tag is appended
to the existing task prompt. For each base policy, ALTER training starts from its pretrained checkpoint.
The policy architecture and training objective remain unchanged, including the same flow-matching loss used
for supervised task adaptation. At deployment the prompt always uses Advantage: positive. Filtering,
recovery offsets, short-horizon normalization, and sampling hyperparameters are reported in Appendix F.
5
Experiments
We evaluate N0-VTLA on the NeoReal real-robot benchmark and the twenty-task simulation suite, test fur-
ther improvement from deployment data, and close with analyses of the learned representation. Throughout,
the base policy of Section 2.1 is initialized from the released π0.5 weights [58], and the frozen tactile encoder
of Section 2.2 is DINOv2 [55].
5.1
Evaluation Protocol
Every comparison in this report is decided by rollout success rate on the target hardware or simulator.
Alongside binary success we report a 100-point progress score that awards partial credit for intermediate
milestones. Each task is decomposed into a fixed sequence of subtask checkpoints, a rollout earns credit for
the deepest checkpoint it reaches, and near-miss behavior on contact-rich tasks stays visible. Representative
rubrics appear in Appendix E, with the full set in the companion data report [51].
On NeoReal, each task carries its own trial budget, and outcomes are reported as success rate in percent
together with the 100-point stage-rubric progress score. On NeoSim, each policy is trained on 100 demon-
strations and evaluated as a success percentage under the simulator’s task-completion criterion with a fixed
per-task language prompt. Comparisons against the no-tactile base policy are isolated structurally. With
10

Cup
Stacking
Fruit
Collection
Board
Wiping
Bottle
Standing
Towel
Folding
Socket
Plugging
Board
Insertion
Cardboard
Box Folding
Bag
Packing
Task
mean
0
20
40
60
80
100
Success rate (%)
0
50
75
0
50
60
0
40
45
0
0
30
0
40
50
0
60
85
0
0
25
0
5
20
0
20
35
0
29.4
47.2
ACT
π0.5
N0-VTLA
Cup
Stacking
Fruit
Collection
Board
Wiping
Bottle
Standing
Towel
Folding
Socket
Plugging
Board
Insertion
Cardboard
Box Folding
Bag
Packing
Task
mean
0
20
40
60
80
100
Progress score
18
66
75.75
21
67.5
69
12
55.2
52
15
0
45
9
56.8
69.75
7
73.5
77
5
0
34
0
19
37.2
5
43
51.25
10.2
42.3
56.8
Figure 7 NeoReal benchmark: real-world results. Simulation success rate in the upper panel and the 100-point progress
score in the lower panel, on the nine NeoReal contact-rich tasks for ACT, π0.5, and N0-VTLA, with exact values
printed above each bar and the nine-task means in the rightmost group. Measured zeros appear as thin baseline ticks.
N0-VTLA beats the strongest baseline on every task in success rate and leads the progress-score mean.
its tactile flag disabled, the model reduces to the base policy, so any measured difference in success rate is
attributable to the tactile pathway alone.
5.2
Real-World Results: NeoReal
We evaluate N0-VTLA on nine tasks from NeoReal, a real-world benchmark of fine-grained contact-rich
manipulation tasks defined in the N0-Foundation data report [51]. Tasks run on the corpus’s robot-arm
platforms under the shared protocol. Figure 11 shows representative rollouts. We deploy the post-trained
checkpoints of N0-VTLA and compare against ACT [93] and π0.5 [58], both reproduced internally under one
aligned evaluation protocol.
Figure 7 plots per-task success rate and progress score. N0-VTLA beats the strongest baseline on every
task in success rate, averaging 47.2% against 29.4% for π0.5. On the progress score it leads on eight of the nine
tasks and in the mean, 56.8 points against 42.3. ACT completes no task and averages 10.2 progress points,
stalling in the earliest checkpoints. The margin is clearest on Socket Plugging, the precision outlet insertion,
where N0-VTLA reaches 85% against 60% for π0.5, and its successful rollouts hold up under daylight lighting
shifts and recover from failed insertion attempts. On the long-horizon tasks the progress score separates the
systems more sharply than the success rate alone. On Cardboard Box Folding, N0-VTLA earns 37.2 points
of stage credit at a 20% success rate, against 19 points for π0.5.
5.3
Simulation Results: UniVTAC and NeoSim
The simulation evaluation covers 20 fine-grained contact-rich tasks on the UniVTAC framework, the eight
original UniVTAC tasks [18] and the twelve NeoSim tasks of the companion report [51], four single-arm and
eight dual-arm. Both suites stress the pre-contact and in-contact regimes where an anticipatory tactile signal
11

Table 1 UniVTAC benchmark: per-task success rate (%). Closed-loop success on the eight UniVTAC tasks for N0-VTLA
and external baselines, all under one aligned protocol. ACT (Vision Only) drops the tactile stream; ACT + UniVTAC
and VITaL [29] add it. Best per task in bold.
Method
Lift
P