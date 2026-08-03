# Interactive Post-Training for Vision-Language-Action Models

> 2025 · id: arxiv:2505.17016 · arXiv: 2505.17016 · pdf: https://arxiv.org/pdf/2505.17016 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

arXiv:2505.17016v1  [cs.LG]  22 May 2025
Interactive Post-Training for
Vision-Language-Action Models
Shuhan Tan1, Kairan Dou2, Yue Zhao1, Philipp Krähenbühl1
UT Austin1, Nankai University2
Code & Model:
https://ariostgx.github.io/ript_vla/
Abstract
We introduce RIPT-VLA, a simple and scalable reinforcement-learning-based inter-
active post-training paradigm that fine-tunes pretrained Vision-Language-Action
(VLA) models using only sparse binary success rewards. Existing VLA training
pipelines rely heavily on offline expert demonstration data and supervised imita-
tion, limiting their ability to adapt to new tasks and environments under low-data
regimes. RIPT-VLA addresses this by enabling interactive post-training with a
stable policy optimization algorithm based on dynamic rollout sampling and leave-
on-out advantage estimation. RIPT-VLA has the following characteristics. First,
RIPT-VLA applies to various VLA models, resulting in an improvement on the
lightweight QueST model by 21.2%, and the 7B OpenVLA-OFT model to an
unprecedented 97.5% success rate. Second, RIPT-VLA is computationally efficient
and data-efficient: With only one demonstration, RIPT-VLA enables an unworkable
SFT model (4%) to succeed with a 97% success rate within 15 iterations. Fur-
thermore, we demonstrate that the policy learned by RIPT-VLA generalizes across
different tasks and scenarios and is robust to the initial state context. These results
highlight RIPT-VLA as a practical and effective paradigm for post-training VLA
models through minimal supervision.
80
90
100
88.6
94.3
QueST
+ RIPT
LIBERO 90
40
60
80
50.2
71.4
QueST
+ RIPT
LIBERO LONG (5 shot)
60
70
80
63.6
76.0
QueST
+ RIPT
MetaWorld45 (5 shot)
94
96
98
96.7
97.5
OpenVLA-OFT
+ RIPT
LIBERO Suites
1
2
3
4
5
Number of Demos
0
20
40
60
80
100
Cross-Scenario
RIPT
SFT
Figure 1: Overview of RIPT-VLA. While VLA models are typically trained with two supervised
stages, we propose a third stage: Reinforcement Interactive Post-Training for VLA. RIPT-VLA sets
state-of-the-art results across diverse benchmarks. It also presents remarkable improvement under the
low-data regime, transforming an SFT model from near failure to 97% with one demonstration.

1
Introduction
Vision-Language-Action (VLA) models [40] aim to enable agents to perceive, reason, and act in the
physical world with a unified interface. Current VLA models are trained with two supervised stages:
large-scale pretraining on diverse human demonstrations, followed by supervised fine-tuning (SFT)
on smaller-scale task-specific datasets. This paradigm has some distinct advantages: Pre-training
enables the VLA model to build general visuomotor skills while SFT allows it to specialize in specific
environments [15]. Supervised training allows VLA models to learn from large-scale pre-recorded
vision-language-action datasets. However, this supervised approach also has two core limitations:
First, data are collected offline. The VLA model learns to imitate interactions with the environment,
but never sees the consequences of its own actions. As a result, the learned policy often fails to handle
the complexities of real-world scenarios, especially for long-horizon tasks. Second, task-specific SFT
via imitation learning relies heavily on large-scale, high-quality human demonstrations. These data
are expensive and time-consuming to collect, and performance degrades significantly when only a
small number of demonstrations are available.
In this paper, we propose RIPT-VLA: a third stage for VLA training paradigm with Reinforcement
Interactive Post-Training. After pretraining and supervised fine-tuning, we allow the VLA model to
interact with the multitask environment and receive binary success/failure rewards. We then optimize
the VLA model to directly improve its success rate across multiple tasks through reinforcement
learning. Inspired by prior RL frameworks for LLMs reasoning [9], we propose a stable and efficient
RL framework for VLA finetuning in Section 4. Specifically, we extend the LOOP framework [4]
which combines REINFORCE leave-one-out (RLOO) advantage estimation [16] and proximal policy
optimization (PPO) [28]. Unlike LOOP, we construct uniform batches of non-zero advantage samples,
filtering out any group of trajectories with zero-advantage, and sampling rollouts until sufficient
samples exist. This uniform batch construction leads to improved training stability, especially as
training progresses and the VLA becomes more successful. RIPT-VLA allows efficient and stable VLA
policy update without relying on shaped or learned rewards, or critic models. Using Reinforcement
Learning in a third training stage has a few distinct advantages: It is more data efficient, yielding
close to state-of-the-art performance with only a single SFT demonstration. The resulting VLA
model achieves a much higher performance on the end-task, as it gets to see interactions with the
environment during training. RIPT-VLA works with both tokenized [22] and continuous actions [14].
RIPT-VLA resonates with the recent trend of paradigm shift in LLM training [24, 9]. While pretraining
on large-scale text corpora equips LLMs with broad knowledge and powerful skills, they often struggle
with challenging tasks that require precise reasoning, multi-step planning, or tool using [34]. To
address these limitations, reinforcement learning has emerged as a critical post-training stage to
reactivate and steer pretrained knowledge with only a small amount of interactive feedback [24].
Similarly, we observe that pretrained VLA models also encode rich visuomotor skills, yet struggle to
apply them effectively for new tasks and scenarios. RIPT-VLA bridges this gap by using only sparse
binary rewards to unlock these latent skills with a small number of optimization steps.
Through comprehensive experiments in Section 5, we demonstrate that RIPT-VLA achieves state-of-
the-art results when combined with both large-scale and lightweight VLA models across a diverse set
of tasks. On the LIBERO benchmark [21], RIPT-VLA improves QueST [22], the best lightweight
VLA model, on all four task suites by 10.9% absolute success rate (SR) on average (Table 1). When
evaluated on OpenVLA-OFT [14], the best-performing large VLA model with an already high success
rate (96.7%), RIPT-VLA still helps by further reducing the failure rate from 3.3% to 2.5%. We also
achieve top performance on many-task benchmarks LIBERO-90 (94.3%) and MetaWorld45 [36]
(92.2%), showing the effectiveness of RIPT-VLA in improving multi-task (up to 90) performance
with a single model (Table 2). Most notably, in the extreme low-data regime with only a single
training demo, RIPT-VLA adapts pretrained knowledge to new task goals or scenarios with remarkable
efficiency, boosting success rate from below 4% to over 97% within only 15 RL iterations.
2
Related Works
Vision-Language-Action Models.
Vision-Language-Action (VLA) models empower embodied
agents to interpret multimodal inputs—such as visual observations and natural-language instruc-
tions—and translate them into meaningful actions within the physical world [40]. Seminal works
like RT-2 [40], RT-1 [2], PaLM-E [7], Octo [32], Dita [11], π0 [1], and π0.5 [13], together with
2

OpenVLA [15], showcase VLAs achieving emergent semantic reasoning and generalization to novel
tasks and environments. These models are typically developed through a two-stage supervised-
learning paradigm that begins with an initial pre-training phase on extensive, web-scale datasets [7, 3],
which is crucial for acquiring generalizable visuomotor skills, grounding language in perception, and
building robust internal representations. While this two-stage approach has advanced the field, its
offline nature imposes key limitations. The supervised fine-tuning (SFT) stage typically requires vast
expert demonstrations for new tasks or environments, thereby degrading few-shot performance. This
highlights a critical gap: the need for methods that adapt pretrained VLAs beyond static imitation by
leveraging interactive experience and reducing reliance on extensive expert data.
Reinforcement Learning for Large Language Models.
Large Language Models (LLMs) offer
a precedent for enhancing pretrained models. While LLMs gain broad capabilities via pre-training
and SFT, they often struggle with complex reasoning, planning, or constraint satisfaction [34]. To
address this, Reinforcement Learning (RL) has emerged as a transformative third stage in LLM
training—enabling learning from interactive feedback rather than static datasets [24]. Recent progress
shows RL can unlock latent capabilities for math [19, 29], self-verifiable proofs [20], long-horizon
planning through tree-of-thoughts [35], and preference-aligned generation with AI feedback [17].
This paradigm, in which pretrained knowledge is steered by targeted feedback, strongly motivates a
similar approach for Vision-Language-Action models: RL has the potential to adapt pretrained VLAs
more effectively to the interactive and consequential nature of embodied tasks.
Reinforcement Learning for VLA.
Recent works have explored applying reinforcement learning
to pretrained VLA models to overcome limitations of supervised fine-tuning and adapt to novel
tasks without collecting new demonstrations. iRe-VLA [10] addresses optimization instability by
alternating between PPO-based updates on a frozen VLM backbone and supervised distillation stages.
However, it still relies on a learned value critic during PPO, and requires shaped reward functions or
success weighting to guide policy learning. ConRFT [5] further combines offline Q-learning with
online consistency-policy updates, but similarly depends on a parameterized value function. Both
methods require careful coordination between offline and online stages to stabilize critic learning.
In contrast, RIPT-VLA introduces a fully critic-free optimization framework with simpler training
dynamics under sparse binary rewards.
3
Preliminary
3.1
Vision-Language-Action Models
Autoregressive VLA rollout.
A vision-language-action (VLA) model πθ maps a sequence of obser-
vations and previous actions (o1:t, a1:t−1), along with a natural language goal g, to a probability dis-
tribution over the next action at. These models operate autoregressively: at ∼πθ(· | o1:t, g, a1:t−1).
Given an initial observation-goal pair context c = (o1, g), the model generates a sequence of actions
conditioned on past information in an autoregressive way:
πθ(a1:T | o1:T , g) =
T
Y
t=1
πθ(at | o1:t, g, a1:t−1).
(1)
We denote this sampling process as a = a1:T ∼πθ(· | c), the observation sequence as o = o1:T .
Sequences terminate upon task success or reaching a time limit. For each rollout sequence and task
goal g, the environment E returns a binary reward R = 1 when the task goal is successfully reached,
and R = 0 otherwise. The environment E can be either a simulator [36, 21] or the real world.
There are two common ways of action prediction in VLA models. The tokenized action head
represents actions as discrete tokens from a fixed vocabulary and predicts actions via classification
over the token set. In contrast, the regression action head directly predicts real-values action vectors
via regression.
Current VLA training paradigm.
Current Vision-Language-Action (VLA) models are typically
trained in two stages: Stage 1: Pretraining and Stage 2: Supervised Fine-tuning.
In Stage 1, a base policy πθ is pretrained on a large-scale, diverse dataset of real-world demonstrations,
denoted by Dpretrain = {(o, a, g)}N
i=1. The policy is trained to imitate the ground-truth actions given
3

offline data in Dpretrain. For VLA models with a tokenized action head, the loss is:
Lpre(θ) = −E(o,a,g)∼Dpretrain
" T
X
t=1
log πθ(at | o1:t, g, a1:t−1)
#
,
(2)
while for regression action head Lpre(θ) is implemented as an MSE or L1 loss. This stage enables
VLA models to capture strong representations and learn general visuomotor and instruction-following
capabilities.
In Stage 2, the pretrained policy is supervised fine-tuned on a smaller, multitask dataset to improve
performance on a small set of target tasks, denoted by Dsft = {(o, a, g)}N′
i=1. Typically, Dsft contains
around 50 high-quality human demonstrations per task [22]. The VLA model is trained with the
same objective function as in Stage 1. This stage enables the model to adapt its learned skills from
Stage 1 to a specialized set of skills for the target tasks.
Although being the standard process of VLA training, this two-stage process has two significant issues.
Firstly, it relies only on offline supervision and lacks interactive feedback from the environment.
Therefore, the learned policy may often fail in real rollouts due to distribution shift and cascading
errors, especially for long-term rollouts. Furthermore, the performance of VLA heavily relies on the
high quality and quantity of the task-specific data in Dsft, which is often hard and costly to obtain.
VLA as Markov decision processes.
To better optimize VLA models, we define its task as a
Markov decision process (MDP). Each episode is initialized with a context c = (o1, g). The state is
represented as [o1:t, g, a1:t−1], which includes the language goal g, the sequence of past observations
o1:t, and past actions a1:t−1. At each timestep t, the VLA policy produces an action sampled
from the policy distribution: at ∼πθ(· | o1:t, g, a1:t−1). The environment transitions to the next
observation ot+1 based on hidden environment dynamics, producing a new state [o1:t+1, g, a1:t].
After a sequence of actions a1:T , the agent receives a binary reward R(c, a) ∈{0, 1} from the
environment E, indicating task success or failure. The objective of VLA optimization is essentially
learning a policy πθ that maximizes expected task success reward:
Rθ(c) = Ea∼πθ(·|c) [R(c, a)] .
(3)
3.2
Reinforcement Policy Optimization
We consider the reinforcement learning setting where an agent interacts with an environment E
to learn a policy πθ(a | c) that maximizes the expected return: Ec∼Dcontext, a∼πθ[R(c, a)], where c
is the context (e.g., goal and initial observation), a is a trajectory (e.g., sequence of actions), and
R(c, a) ∈{0, 1} is a sparse binary reward returned by the environment. To optimize this objective, a
standard approach is policy gradient, which updates πθ with:
∇θLθ(c) = Ea∼πθ[∇θ log πθ(a | c) · A(c, a)],
(4)
where A(c, a) is the advantage function indicating how much better the action a is compared to a
baseline. In practice, it is hard to compute A(c, a), especially under sparse rewards. To address this
issue, a recent work proposed a critic-free optimization framework called Leave-One-Out Proximal
Policy Optimization (LOOP) [4]. Specifically, it combines the two methods below.
Leave-One-Out Advantage Estimation (RLOO) [16].
For each sampled context c, we draw K
rollouts {ak ∼πψ(· | c)}K
k=1 under a fixed sampling policy πψ. Each rollout receives a binary
reward Rk = R(c, ak). The leave-one-out baseline for rollout k is computed by averaging the other
rewards:
bk =
1
K −1
X
j̸=k
Rj,
Ak = Rk −bk.
(5)
This group-normalized advantage indicates how much better or worse a rollout performance is relative
to others given the same context. This allows us to efficiently compute a stable advantage signal from
sparse binary rewards, without requiring learning value functions.
Proximal Policy Optimization (PPO) [28].
To update πθ using collected rollouts {(ck, ak, Ak)},
we compute the importance ratio rk = πθ(ak | ck)/πψ(ak | ck), where πθ is the current updating
policy and πψ is the fixed sampling policy (normally set to the latest checkpoint of πθ). We then
optimize πθ with the following clipped objective:
LPPO = −min (riAi, clip(ri, 1 −ϵ, 1 + ϵ)Ai) ,
(6)
4

Algorithm 1 RIPT-VLA: Reinforcement Interactive Post-Training for VLA Model
Input: Pretrained VLA πθ, reward function R(c, a), context dataset Dcontext
1: for step = 1 to M do
2:
Update sampling VLA πψ ←πθ
3:
Initialize empty dataset Drollout ←∅
4:
while |Drollout| < B do
▷Rollout Collection
5:
Sample a context c ←(g, o1) ∼Dcontext
6:
Generate K rollouts {ak ∼πψ(· | c)}K
k=1
▷Group Sampling
7:
Compute rewards {Rk ←R(c, ak)}K
k=1
8:
Compute baselines: bk ←
1
K−1
P
j̸=k Rj
▷Leave-One-Out Baseline
9:
Compute advantages: Ak ←Rk −bk for each k
10:
if all A = 0 then
▷Dynamic Rejection
11:
continue
12:
end if
13:
Add (c, ak, Ak) for all k to Drollout
14:
end while
15:
for iteration = 1 to N do
16:
Update πθ with PPO loss (Equation 6) over Drollout
▷Policy Optimization
17:
end for
18: end for
where ϵ is a small updating threshold (we use 0.2). This objective encourages rollouts with positive
advantages while preventing unstable updates when πθ deviates too far from its previous version πψ.
LOOP adopts PPO to optimize the advantage estimated by RLOO, which enables sample-efficient
policy optimization in sparse reward settings without critics. It serves as an out-of-box working
implementation for our interactive post-training framework in Section 4.
4
RIPT-VLA
As mentioned above, there is a gap between the current VLA training paradigm and our essential
goal of making it work in our downstream tasks. On one hand, pure supervised training on offline
data makes the policy fragile in real rollout due to compounding errors and the distribution gap
between the offline dataset and online rollout. Furthermore, one has to collect a sufficient number
of high-quality demonstrations for offline datasets, especially Dsft, the model can easily overfit to
the training distribution. In other words, optimizing VLA through Equation 2 does not necessarily
improve the VLA’s task execution success rate in Equation 3. To bridge this gap, we propose a
new VLA training paradigm that directly optimize pretrained VLA through interaction with the
environment E through Reinforcement Interactive Post-Training, or RIPT-VLA for short.
4.1
Reinforcement Interactive Post-Training for VLA Models
The first two stages of our VLA training paradigm are the same as the standard setting. In Stage 1,
we pretrain the VLA model on a large, diverse dataset Dpretrain to learn visual-language representation
and general visuomotor skills. Then, in Stage 2, we finetune VLA on a small dataset Dsft to adapt it
to follow instructions to solve a small set of target tasks. These stages produce a pretrained VLA
policy πθ that can achieve a non-zero success rate (can be very low) on the target tasks.
In RIPT-VLA, we then conduct Stage 3: Reinforcement Interactive Post-Training. In this stage,
we assume we can rollout πθ in an environment E and receive a binary reward R(c, a) ∈{0, 1}
given a ∼πθ(· | c), where c is the initial context. In addition, we use an initial context dataset
Dc = {(o1, g)} to set up task initializations for model rollouts. Typically, we obtain Dc by directly
extracting the initial states from sequences in Dsft. For each optimization step, we iterate between
two steps: rollout collection and policy optimization.
During rollout collection, we randomly sample contexts ci ∼Dc and let πθ interact with the
environment E to output a sequence ai. For each rollout we collect its reward R(ci, ai) and compute
its advantage Ai = A(ci, ai), which indicate how strong the model should be encouraged (A > 0) or
5

penalized (A < 0) for generating rollout a. We add all rollouts and rewards (ci, ai, Ai) to a rollout
dataset Drollout until we obtain B rollouts: Drollout = {(ci, ai, Ai)}B
i=1
During policy optimization, we optimize πθ with reinforcement learning algorithms on Drollout to
maximize its expected task success rate in Equation 3 for N iterations. After optimization, we use
the updated VLA policy π′
θ to collect new rollouts and a new step begins. This process repeats until
we reach M steps and outputs the final policy π∗
θ, concluding the full VLA training paradigm. We
then deploy π∗
θ in the environment for testing.
Although RIPT-VLA is simple in concept, it presents several challenges. First, we only have sparse
binary rewards from each rollout sequence, no shaped reward is available. Training a learned reward
model to predict shaped reward values can easily lead to reward hacking [30], especially with limited
rollout data. Second, as VLA models operate over long-horizon, multi-task environments, credit
assignment becomes highly ambiguous. This causes the value target (e.g., from TD error) to be
extremely noisy and uninformative. Third, training a stable value function for VLA requires a model
of comparable capacity to the VLA itself, which significantly increases GPU memory usage and
training cost for large VLA models [38]. Finally, in multitask environments, different task contexts
can vary significantly in difficulty: some lead to trivial success while others consistently fail across
all rollouts. This results in highly imbalanced success rates and unstable policy gradient updates.
4.2
Dynamic-Sampling Leave-One-Out Proximal Policy Optimization
To implement RIPT-VLA in a stable and sample-efficient way, we propose a simple yet effective
policy optimization framework in Algorithm 1. First, we adopt LOOP (Section 3.2) as the foundation
of our implementation. LOOP is particularly well-suited for our VLA setting, where rollouts are
long-horizon and efficient advantage estimation is required for its sparse reward signal. Furthermore,
for VLA in multitask environments, we design a dynamic rollout sampling mechanism to filter out
uninformative contexts for more stable and efficient policy optimization.
LOOP for RIPT-VLA.
We apply LOOP [4] for both the rollout collection and policy optimization
stage. During rollout collection, we conduct RLOO [16] advantage estimation. In this step, we
use the most recent policy πθ as the sampling policy πψ. Given a single context c ∼Dc, we
collect K trajectories by repeatingly sampling K times from the policy given the same context:
{ak ∼πψ(· | c)}K
k=1. We obtain their corresponding rewards {Rk}K
k=1 from the environment E. For
each rollout k, we compute the advantage Ak with Equation 5. For each epoch, we conduct group
sampling on B/K contexts sampled from Dc, obtaining Drollout with B rollouts.
During policy optimization, we use PPO [28] to stabilize policy gradient updates. For each rollout
sample (ci, ai, Ai) ∈Drollout, we can compute its training objective LPPO with Equation 6. We
perform this update over the collected rollout dataset Drollout using mini-batches for N optimization
steps each epoch. When N = 1, the method corresponds to on-policy RLOO; when N > 1, the same
samples are reused for additional updates, resulting in a partially off-policy optimization.
Dynamic rollout sampling.
VLA models often operate in multitask environments [15, 22, 31],
where task difficulty varies widely across different contexts. Some contexts have already been
well solved by the VLA model, leading to trivial success across K-group sampling, while others
consistently fail due to inherent task complexity or distribution gap. Both cases result in rollout
groups where all rollout samples receive identical rewards (all ones or all zeros), producing all-zero
advantages in Equation 5. Therefore there is no gradient signal from Equation 6. Adding these
samples to Drollout makes unstable gradient updates during batch optimization, as they contribute zero
gradients that can dilute meaningful learning signals.
To address this, we apply a simple yet effective dynamic rejection strategy: we discard any sampled
context for which all K rollouts receive the same reward and resample a new context from Dcontext
for group sampling. As training progresses and the policy improves, an increasing number of task
contexts yield uniformly successful rollouts. Dynamic rejection naturally filters out these solved
contexts, allowing optimization to concentrate on the remaining harder contexts. Importantly, this
method make the batch optimization of the PPO loss (Equation 6) to have the same effective batch
size over all the minibatches across Drollout, which we empirically found to be important for stable
policy optimization in RIPT-VLA.
The full implementation of our optimization procedure is summarized in Algorithm 1.
6

4.3
Generalizing RIPT-VLA to Different VLA models
RIPT-VLA is compatible with both discrete and continuous action representations commonly used in
VLA models. RIPT-VLA requires the VLA model being able to output a probability distribution over
actions at each step, which is used in two key steps. First, during rollout collection, to support diverse
rollouts in group sampling for the same initialization context, we need to randomly sample different
actions from its output distribution:
at ∼πθ(at | o1:t, g, a1:t−1),
(7)
On the other hand, to perform stable policy optimization, we compute the trust region ri = πθ(ai|ci)
πψ(ai|ci)
in Equation 6 to constrain policy updates within a small region of the original policy. A key component
in this formulation is computing the log-probability of the sampled action sequences under both
policies. We compute the log-probability of a sampled action sequence a = (a1, . . . , aT ) as the sum
of the per-step log-probabilities:
log πθ(a | c) =
T
X
t=1
log πθ(at | a<t, c).
(8)
In other words, we can apply RIPT-VLA to any VLA model πθ that we sample a random action at
from the per-step action distribution and compute log πθ(at | a<t, c).
Tokenized action head.
For VLA models with discrete action outputs, e.g. QueST [22], actions
are predicted as sequences of discrete tokens from a fixed vocabulary, where the action header is a
classification head trained with NLL loss. Therefore, log πθ(at | a<t, c) is directly obtained from
applying softmax function to the model’s classification head output logits. We can also simply sample
action tokens from the distribution after softmax.
Regression action head.
For continuous-action VLA models [14], actions are regressed using
MSE or L1 loss, which do not produce a log-probability. To enable policy gradient optimization, we
extend the model with a light-scale prediction head that estimates the scale σθ of the action value.
Assuming the original output head provides the mean µθ, we treat the policy as a factorized Gaussian
(MSE) or Laplace (L1) distribution and train the scale head using the NLL loss in Equation 2 for
a few iterations on Dsft. After that, we can sample action at and compute log πθ(at | a<t, c) with
predicted µθ and σθ in a closed form.
5
Experiments
We evaluate RIPT-VLA on two widely used benchmarks for VLA learning: LIBERO [21] and
MetaWorld [36]. We study several settings: (1) standard multitask (up to 90 tasks) setting in Sec. 5.2,
(2) few-shot (1 ∼5 demonstration) setting in Sec. 5.3, and (3) cross-task and cross-scenario setting
in Secs. 5.4 and 5.5 to showcase the ability of fast generalization leveraging prior knowledge during
pretraining. Additionally, we conducted studies to analyze the practical behavior of RIPT-VLA,
including training curves, ablation studies as well as its sensitivity to the variance and diversity of the
context dataset.
5.1
Setup
Benchmark.
LIBERO [21] is a lifelong learning benchmark with 5 task suites. Each suite consists
of a set of language-guided manipulation tasks across multiple object types, task definitions and
environment scenarios. Specifically, it includes 4 suites: Goal, Spatial, Object, and Long. Each
suite is designed to evaluate a specific aspect of object manipulation and containing 10 distinct tasks.
In addition, it also includes a LIBERO-90 suite that contains 90 different tasks to assess multitask
performance at scale. MetaWorld [36] is a manipulation task benchmark for few-shot learning models.
We use Meta-Learning 45 (ML45) suite that contains 45 training tasks and 5 held-out tasks.
For both benchmarks, each task comes with 50 expert demonstrations for training. At evaluation
time, a single VLA model is deployed across all tasks in a suite and performs rollouts on 50 held-out
test contexts per task. We measure performance with the average task success rate (SR).
7

Stage 1 + Stage 2 Models
Method
Goal
Spatial
Object
Long
Average
Octo [32]
84.6
78.9
85.7
51.1
75.1
OpenVLA [15]
79.2
84.7
88.4
53.7
76.5
Dita [11]
85.4
84.2
96.3
63.8
82.4
π0 + FAST [26]
88.6
96.4
96.8
60.2
85.5
π0 [1]
95.8
96.8
98.8
85.2
94.2
OpenVLA-OFT* [14]
97.9
97.6
98.4
92.9
96.7
OpenVLA-OFT + RIPT
99.0 (+1.1)
98.6 (+1.0)
98.6 (+0.2)
93.8 (+0.9)
97.5 (+0.8)
Stage-2 Models
Method
Goal
Spatial
Object
Long
Average
Diffusion Policy [6]
68.3
78.3
92.5
50.5
72.4
Seer [33]
–
–
–
78.7
–
MDT [27]
73.5
78.5
87.5
64.8
76.1
MDT+ [27]
–
95.2
97.8
83.0
–
QueST [22]
80.8
87.4
93.6
68.8
82.7
QueST + RIPT
92.7 (+11.9)
95.6 (+8.2)
98.4 (+4.8)
87.5 (+18.7)
93.6 (+10.9)
Table 1: Multitask SR(%) on the four LIBERO suites. Bold indicates best result and underline marks
the second-best. Improvements from RIPT-VLA are marked in red. *: OpenVLA-OFT results are
obtained from running official checkpoints for each suite.
Base models.
We conduct RIPT-VLA on two pretrained VLA models with different design choices.
OpenVLA-OFT [14] is an Optimized Fine-Tuned variant of the 7B OpenVLA model [15]. OpenVLA
is initialized from a multimodal backbone that combines a Llama-2 7B language model with dual
vision encoders [23, 37] and is pretrained on 970k robot-manipulation demonstrations. OFT replaces
the original tokenized action decoder with a continuous decoding head and trains with an L1 regression
loss. This architecture represents the large-scale regression action VLA.
QueST [22], on the other hand, is a small-scale tokenized action VLA model with 20 million
parameters. QueST first learns a VQ-VAE that compresses short motion segments into a discrete skill
codebook; a GPT-style transformer then autoregressively predicts these skill tokens conditioned on
images and language, and a small decoder turns tokens back into continuous joint commands.
Implementation details.
For OpenVLA-OFT, we fine-tune the model using the official checkpoints
provided for each task suite. Training is conducted on 4 NVIDIA RTX A5000 GPUs, each with 24
GB memory. We use LoRA [12] with rank 32, and set K = 8, B = 192(8×24), N = 1 and ϵ = 0.1,
PPO mini-batch size of 4 per GPU. We set a learning rate of 1e−4 for the LoRA modules and 1e−5
for the action head. Following Section 4.3, before applying RIPT-VLA, we first train a small Laplace
scale header from scratch (with the same architecture as the action header) with NLL loss on Dsft
for 500 steps. For rollout collection during training, we randomly sample actions according to the
Laplace distribution with scale predicted by this header. For evaluation, we directly use the mean
value predicted by the original action header.
For QueST, as official checkpoints are not provided, we first train the base model from scratch for
each task suite following the official code and hyperparameters. In the multitask setting, we conduct
RIPT-VLA on 3 GPUs with K = 16, B = 2880 (16 × 180). For the single-task setting, we use 1
GPU with K = 16, B = 160. For both settings, we set N = 20, PPO mini-batch size of 8 per GPU,
a learning rate of 1e−6, and the clipping parameter ϵ = 0.2.
5.2
Standard Multitask Training
In this section, we evaluate RIPT-VLA under standard multitask benchmarks. For each suite, we use
all 50 expert demonstrations per task as its SFT dataset Dsft. We conduct RIPT-VLA to finetune a
base model on the corresponding dataset for each task suite.
Table 1 compares multitask performance on four LIBERO suites for different VLA models. We
organize the results into two sets based on VLA training paradigm. In the Stage 1+ Stage 2 set,
we include 5 state-of-the-art large VLA models: Octo [32], OpenVLA [15], Dita [11], π0 [1] and
8

Full Data
5-shot Data
Method
LIBERO-90
ML45
LONG
ML45
ACT [8]
50.8
90.8
42.0
70.8
PRISE [39]
54.4
80.4
52.7
66.8
DP [6]
75.4
90.3
45.9
65.0
VQ-BeT [18]
81.3
87.6
41.8
65.6
ResNet-T [22]
84.4
88.4
51.9
54.0
QueST [22]
88.6
91.0
50.2
63.6
QueST + RIPT
94.3
92.2
71.4
76.0
(improvement)
(+5.7)
(+1.2)
(+21.2)
(+12.4)
Table 2: Mean Success Rate (SR%) across four evaluation settings:
LIBERO-90 and ML45 (Full data), LONG and ML45 (5-shot).
1
3
5
10
Number of Demos
40
60
80
LIBERO-LONG (5-shot)
RIPT
SFT
Figure 2: Few-shot curve
on LIBERO-LONG.
OpenVLA-OFT [14]. These models are typically larger than 500M parameters, pretrained (Stage-1)
on large-scale general-purpose datasets, e.g., Open-X Embodiment [25], and then finetuned using
50 demonstrations per task for each LIBERO suite (Stage-2). In contrast, the Stage 2 set includes
4 representative small models: Diffusion Policy [6], Seer [33], MDT [27] and QueST [22]. These
models are within 50M parameters and are directly trained on each LIBERO suite from scratch.
We show that RIPT-VLA significantly improves the best-performing VLA model in both types, setting
new state-of-the-art performance on the 4 LIBERO suites. Specifically, RIPT-VLA improves QueST
on all four task suites by 10.9 absolute SR on average, and yields even larger gains of 18.7 for the
challenging LONG suite. Notably, with RIPT-VLA, the small 20M QueST model achieves much
better performance with large models like Dita (334M) and comparable with π0 (2B). When applying
to OpenVLA-OFT, the best-performing large VLA model with already high SR, RIPT-VLA still
further reduces the average failure rate from 3.3% to 2.4%. By applying RIPT-VLA, we set new
state-of-the-art performance on 3 out of the 4 LIBERO suites (with only a 0.2 gap on the Object
suite), and achieve the highest average success rate across all tasks. These results show the RIPT-VLA
is broadly effective: it can both unlock latent capabilities in small-scale models and further push the
limits of the high-performing ones.
In addition, in the left two columns of Table 2, we show the results on LIBERO-90 and ML45,
which contain 90 and 45 diverse tasks respectively. These benchmarks assess the scalability and
generalization of a single VLA model across many skills. We apply RIPT-VLA to QueST and
compare with representative imitation learning methods: ACT [8], PRISE [39], Diffusion Policy [6],
VQ-BeT [18] and ResNet-T [22]. We show that RIPT-VLA improves performance of QueST by
5.7 and 1.2 absolute SR for LIBERO-90 and ML45, again setting new SOTA performance for both
benchmarks. This confirms the utility of RIPT-VLA not only for improving performance on a few
related tasks, but also for scaling up to broader, more realistic scenarios where a single model solves
many different tasks.
5.3
Few-shot Multitask Training
In this section we evaluate RIPT-VLA under few-shot multitask setting. For each suite, we uniformly
sample 1 to 10 expert demonstrations from each task to constitute the few-shot SFT dataset Dsft. This
setting reflects practical situation where large-scale data collection is not available.
The right two columns of Table 2 show results under the 5-shot setting, where each task in the
LIBERO-LONG and ML45 suites is trained with only 5 demonstrations. While baseline models
struggle in this low-data regime, RIPT-VLA significantly improves QueST by 21.2 on LIBERO-
LONG and 12.4 on ML45. These results demonstrate that RIPT-VLA effectively addresses a key
limitation of standard VLA training with SFT: it enables strong performance even with minimal
demonstrations, alleviating concerns about data scarcity in real-world multitask deployment.
To further investigate the effect of the number of few-shot demonstrations, we conduct experiments
under varying few-shot settings with QueST, ranging from 1 to 10 demonstrations per task on
LIBERO-LONG. As shown in Figure 2, RIPT-VLA consistently largely improves the performance
of the standard SFT model across all data scales. Note that even for the extremely low-data regime,
where we only have 1 demonstration per task, RIPT-VLA can still achieve a 20.8 absolute gain. As
9

1
2
3
4
5
Number of Demos
0
20
40
60
80
100
Scene = 10
5:
close the top drawer
of the cabinet
1
2
3
4
5
Number of Demos
0
20
40
60
80
100
Scene = 9
3:
turn on the stove
and put the frying pan on it
1
2
3
4
5
Number of Demos
0
20
40
60
80
Scene = 10
5:
put the black bowl in_
the top drawer of the cabinet
2
3
4
5
Number of Demos
0
20
40
60
80
Scene = 5
10:
turn on the stove
and put the frying pan on it
RIPT
SFT
Figure 3: Cross-scenario task generalization from Scenario A to Scenario B with the same goal.
3
5
7
8
10
Number of Demos
0
20
40
60
80
Goal = Back
Front: 
 put the butter at the {Goal} 
in the top drawer of
 the cabinet and close it
3
5
7
8
10
Number of Demos
0
20
40
60
80
Goal = Left
Right: 
 put the chocolate 
pudding to the 
{Goal} of the plate
3
5
7
8
10
Number of Demos
0
20
40
60
80
Goal = Back
Front: 
 put the black 
bowl at the 
{Goal} on the plate
3
5
7
8
10
Number of Demos
0
20
40
60
80
100
Goal = Bottom
Top: 
 open the {Goal} 
drawer of the cabinet
RIPT
SFT
Figure 4: Cross-goal task generalization from Goal A to Goal B in the same scenario.
the number of demonstrations increases, RIPT-VLA continues to yield performance improvements,
indicating its strong sample efficiency and scalability. These results confirm that RIPT-VLA is robust
across different levels of data scarcity and is applicable in both low- and high-data settings.
5.4
Cross-scenario Generalization
Recent paradigm shift in LLM training demonstrates that reinforcement learning can reactivate
and steer pretrained knowledge with only a small amount of interactive feedback [24]. We adopt a
similar approach for VLA and ask: can RIPT-VLA enable sample-efficient pretrained visuomotor
skill transfer across scenarios and goals?
In this section, we experiment on the few-shot cross-scenario generalization setup. For each ex-
periment, we consider a pair of tasks that have the same task goal (e.g., ’turn on the stove and put
the frying pan on it’), but operate in different scenarios: Scenario A and Scenario B - with distanct
background layouts and object configurations. In Stage 1, we pretrain QueST on |Dpretrain| = 50
demonstrations from Scenario A to acquire a general visuomotor skill for this task goal. In Stage
2, we conduct SFT on |Dsft| = {1, 2, 3, 4, 5} demonstrations from Scenario B. Then, in Stage 3, we
apply RIPT-VLA to optimize the policy through interactive rollouts on contexts Dcontext extracted from
Dsft. We then evaluate the model on the 50 testing contexts of Scenario B. We conduct experiments
with 3 random seeds and plot the mean and variance across different sizes of Dsft.
Figure 3 show results on 5 scenario pairs. We observe that standard SFT on VLA models clearly
struggles in the 1-shot regime, achieving an average success rate of only around 5%, and in some
cases dropping as low as 2%. Clearly, SFT fails to generalize the task knowledge from the pretraining
stage to the new scenario. In contrast, RIPT-VLA dramatically improves performance, with absolute
SR gain as high as 93.7% (from 3.5% SFT to 97.2%). As the size of Dsft increases, both SFT and
RIPT-VLA performance improve, but RIPT-VLA consistently maintains a strong improvement, often
reaching near-100% performance with just 3-5 demonstrations. These results supports our core
assumption: RIPT-VLA enables pretrained VLA models to rapidly activate and adapt learned skills
with sparse binary reward feedback.
5.5
Cross-goal Generalization
In this section, we investigate RIPT-VLA in a cross-goal generalization setting. Here we focus on task
pairs that operate in the same scenario but with different goals. Specifically, we select Task A and
Task B such that they require the same visuomotor skills but have different goals. For example, Task
10

0
10
20
30
Iteration
0
25
50
75
1-Shot Cross-Goal
RIPT
w/o DS
Figure 5: Training curve analysis
of dynamic sampling.
1
5
10
25
40
Number of Rollout Contexts
0
20
40
60
1-Shot Cross-Scenario
RIPT
SFT
Figure 6: Analysis on context
dataset size.
0.0 0.1 0.5 1.0 2.0 3.0 5.0 6.0 7.0
Context Variance Scale (× 2.5 cm)
0
20
40
60
LIBERO-LON