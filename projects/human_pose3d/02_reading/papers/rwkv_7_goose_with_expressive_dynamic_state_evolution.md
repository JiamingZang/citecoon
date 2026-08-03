# RWKV-7 "Goose" with Expressive Dynamic State Evolution

> 2025 · id: arxiv:2503.14456 · arXiv: 2503.14456 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We present RWKV-7 "Goose", a new sequence modeling architecture with constant memory
usage and constant inference time per token. Despite being trained on dramatically fewer tokens
than other top models, our 2.9 billion parameter language model achieves a new 3B SoTA on
multilingual tasks and matches the current 3B SoTA on English language downstream performance.
RWKV-7 introduces a newly generalized formulation of the delta rule with vector-valued gating
and in-context learning rates, as well as a relaxed value replacement rule. We show that RWKV-7
can perform state tracking and recognize all regular languages, while retaining parallelizability of
training. This exceeds the capabilities of Transformers under standard complexity conjectures,
which are limited to TC0. To demonstrate RWKV-7’s language modeling capability, we also present
an extended open source 3.1 trillion token multilingual corpus, and train four RWKV-7 models
ranging from 0.19 billion to 2.9 billion parameters on this dataset.
To foster openness, reproduction, and adoption, we release our models1 and dataset component
listing2 on Hugging Face, and our training and inference code3 on GitHub; all under the Apache
2.0 License.
∗Equal first authorship. Others listed alphabetically.
1Model weights at https://huggingface.co/RWKV
2Dataset components listed at https://huggingface.co/RWKV
3Source code at: https://github.com/RWKV/RWKV-LM
1
arXiv:2503.14456v2  [cs.CL]  30 Mar 2025

Contents
1

## introduction
Autoregressive Transformers (Vaswani et al., 2023) have recently dominated sequence modeling
tasks, enjoying excellent in-context processing and highly parallelizable training due to their use
of softmax attention. However, softmax attention incurs quadratic computational complexity and
memory usage with respect to sequence length due to its linearly expanding key-value cache. For
short sequences, much of this cost can be covered by modern GPU parallelism techniques, but
Transformer inference becomes increasingly costly as sequence lengths grow.
This limitation has inspired significant research into the design of recurrent neural network (RNN)
architectures with compressive states that afford linear computational complexity and constant
memory usage, while still allowing highly parallel training. Two of the most commonly proposed
alternatives that satisfy these requirements are linear attention variant models (Katharopoulos
et al., 2020b; Sun et al., 2023; Peng et al., 2024b; Yang et al., 2023a) and State Space Models (Gu
& Dao, 2023). These architectures have grown more sophisticated, with many recent proposals
incorporating some form of the delta rule, as embodied by parallelized DeltaNet (Schlag et al., 2021;
Yang et al., 2024c). Such models have achieved impressive downstream performance results: since
RWKV-4 (Peng et al., 2023), RNN models have shown increasing potential to rival Transformers
when given equivalent model size and training compute, while dramatically reducing inference
costs.
We present a new architecture, RWKV-7 "Goose", which generalizes the delta rule for use in
sequence modeling. First, we add a vector-valued state gating mechanism, enhancing expressivity
and providing implicit positional encoding. Second, we expand the in-context learning rate from a
scalar to become vector-valued, allowing the model to selectively replace state data on a channel-
wise basis. Third, we decouple the keys at which the delta rule removes from and adds to the state.
Finally, we place these innovations within a modified RWKV-6 architecture, inheriting important
features such as token-shift, bonus, and a ReLU2 feedforward network. We also introduce an
expanded 3.1 trillion token RWKV World v3 corpus designed for enhanced English, code, and
multilingual task performance. We use this architecture and corpus to train new state-of-the-art
open-source language models, upgraded from preexisting RWKV-5/RWKV-6 checkpoints.
3

Our main contributions are as follows:
• The RWKV-7 "Goose" architecture, which dramatically improves downstream bench-
mark performance over RWKV-6 and demonstrates state-of-the-art multilingual perfor-
mance at 3B scale and near SoTA English language performance, despite being trained
on many fewer tokens than the top models in its class.
• The RWKV World v3 public dataset, comprised of 3.1 trillion tokens of publicly available
multilingual data.
• Public release of four pre-trained RWKV-7 World v3 language models, ranging from 0.19
to 2.9 billion parameters trained on 1.6 to 5.6 trillion tokens.
• Public release of three pre-trained RWKV-7 Pile language models, using the GPT-NeoX
tokenizer (Black et al., 2022), ranging from 0.17 to 1.47 billion parameters, useful for
comparative study with other architectures.
• Proofs that the generalized delta rule employed in RWKV-7 can solve problems outside of
TC0 under the widely held complexity conjecture that TC0̸ = NC1. This includes solving
an S5 state tracking problem known to be in NC1 using only a single layer, and recognizing
all regular languages using only a constant number of layers.
• A method for upgrading the RWKV architecture without pre-training from scratch,
producing increasingly competitive trained models at reduced computational expense.
Larger datasets and RWKV-7 models are under active preparation and construction and will be
released under the Apache 2 license whenever practical.
2

## method
In this section, we use D to denote the model dimension. Bold capital letters represent trainable
matrices, and vectors without a subscript t are trainable parameters. The first subscript denotes
sequence position and second subscript denotes layer index, where necessary. We use the con-
vention that all vectors are row vectors unless explicitly transposed, so all matrices operate on
the right side, therefore aT b is an outer product and abT is an inner one. We use the square
subscript to denote a placeholder for variable names and use the Q sign for cumulative matrix
multiplication. See Appendix G for a pseudocode implementation of these formulas.
4.1
Time Mixing
Weight Preparation
Along the lines of (Peng et al., 2024b), we introduce the following notation
templates for common operators in the model, using the square subscript to denote a variable:
lerp(a,b,x) = a +(b −a)⊙x,
(1)
loramlp□(f ,x,bias) = f (xA□)B□+(λ□if bias else 0),
(2)
Unless explicitly stated, all vectors appearing in this section are dimension D.
6

We extend the use of low-rank MLP (a 2-layer MLP with small hidden dimension compared to input
and output), abbreviated as loramlp, to implement data dependency using minimal parameters.
The replacement key ˜k, value v, decay w, removal key κ, in-context learning rate a, receptance r,
and rwkv gate g parameters are computed as follows (outputs annotated with ▷):
x□
t = lerp(xt,xt−1,µ□)
□∈{r,k,v,d,a,g},
token shifted inputs
(3)
at = sigmoid(loramlpa(Identity,xa
t ,bias=True)),
▷in-context learning rate
(4)
kt = xk
t W k,
key precursor
(5)
κt = kt ⊙ξ,
▷removal key
(6)
˜kt = kt ⊙lerp(1,at,α),
▷replacement key
(7)
νt = sigmoid(loramlpν(Identity,xv
t ,bias=True)),
value residual gate
(8)
v′
t,l = xv
t W v,
value precursor
(9)
vt =
(
v′
t,0,
layer l = 0
lerp(v′
t,0,v′
t,l,νt),
layer l ≥1 ,
▷value
(10)
dt = loramlpd(tanh,xd
t ,bias=True),
decay precursor
(11)
wt = exp(−e−0.5sigmoid(dt)),
▷decay
(12)
rt = xr
t W r ,
▷receptance
(13)
gt = loramlpg (sigmoid,xg
t ,bias=False)
▷rwkv gate
(14)
ξ is a learned parameter representing the removal key multiplier, which transforms the original
key into a version to be removed from the state. In practice, ξ lies in a range of approximately
[−5.3,9.4].
α is a learned parameter representing the replacement rate booster, which adjusts the amount
added back to the state after the transition matrix is applied.
Unlike r,k and v which are the main carriers of information, g,d,ν and a act like gates which
control the amount of information allowed to pass.
For comprehensive statistics of ξ, α and biases of dt observed in the released RWKV-7 model,
including extremum values, mean measurements, and distribution trends, see Appendix L.
For the computation of x□
t , we removed data dependency of linear interpolation from RWKV-6 to
improve training speed.
We adapted the idea of Value Residual Learning Zhou et al. (2024) for the computation of vt, which
has shown to improve the final language modeling loss. νt represents the value residual mix,
which interpolates between the layer zero and current layer value precursors: vt,0 and vt,l.
We also updated the formula for computation of wt, restricting all entries in (exp(−e−0.5),1) in
favor of a smaller condition number for diag(wt), which maintains better training stability, and
was beneficial to accuracy of the backward pass.
The ˜kt in the formula can be regarded as a "normalized key", a design to ensure that the state
of wkv contains columns of O(1) size. Normally, we expect ˜kt = kt ⊙(1 −wt), as employed in
RWKV-6c (see Appendix F), so that wkv t rows are linear interpolations between wkv t−1 and vT
t kt
controlled by wt. However, to further enhance expressivity, we decide to decouple wt and at. We
further decouple at from the amount actually added to the state, allowing the replacement rate
booster α to interpolate the amount added between the normal in-context learning rate and 1.0.
Importantly, all of these modifications operate on a per-channel basis. The numerical range of
RWKV-7’s wkv entries are generally stable as in RWKV-6c, unlike RWKV-6, where entries of the
states can accumulate to thousands (see Appendix J for a state visualization).
The Weighted Key Value State Evolution
After weight preparation, we reshape (r,w, ˜k,v,κ,a)t,
splitting them to h heads, with each head sized D/h. We always assume that h is a factor of D and
heads are equally split. All operations in this section are shown per-head.
7

Before mixing in the time dimension, κt is normalized per head:
ˆκt = κt/∥κt∥2
(15)
The wkv (Weighted Key Value) is a multi-headed matrix-valued state of fast weights that un-
dergoes dynamic evolution. The evolution of wkv is crucial for encoding context information
by learning at test time to map keys to values. We start by defining the WKV time mixing as the
recurrence relation
wkv0 = 0,
(16)
wkv t = wkv t−1
¡
diag(wt)−ˆκT
t (at ⊙ˆκt)
¢
+ vT
t · ˜kt
(17)
Compared to RWKV-5 and RWKV-6, the wkv in this paper is transposed to ensure consistency
with RWKV-7’s code.
The wkv t attention calculation can alternatively be written in a parallel manner:
wkv t =
tX
i=1
Ã
vT
i ˜ki
tY
j=i+1
³
diag(w j )−ˆκT
j (a j ⊙ˆκj )
´!
∈R(D/h)×(D/h)
(18)
The recurrent transition design has parallels with Schlag et al. (2021), but crucially the transition
matrix
Gt = diag(wt)−ˆκT
t (at ⊙ˆκt) =
µ
I −ˆκT
t ( at
wt
⊙ˆκt)
¶
diag(wt) ≈
¡
I −2ˆκT
t ˆκt
¢
diag(wt)
(19)
is no longer a Householder matrix but a scaled approximation of it, as ˆκt̸ = at
wt ˆκt. This mimics a
Householder matrix but with expanded dynamics, while still having all eigenvalues in a stable
range of [−1,1] and allows the network to decay information in all subspaces if necessary. It
contrasts with the case of a Householder-like matrix with learning rate (I −avT v), a ∈[0,1], as
used in Schlag et al. (2021); Yang et al. (2024c) where all eigenvalues are one except for the last one
corresponding to 1−a. Given these properties, we refer to wt as "in-context weight decay" and to
at as "in-context learning rate" (ICLR). The RWKV-7 transition matrix, therefore, allows for both
dynamic state evolution and approximation to a forget gate at the same time. See Appendix C for
the details on the eigenvalue of the transition matrix, and when the transition matrix is guaranteed
to be stable.
The original delta rule in Schlag et al. (2021) allows partial or full removal of pre-existing values
from the state at each time-step, with the amount removed being equal to the scalar a. Our
formulation extends this ability by making a a vector, allowing for different removal amount per
state column.
WKV Bonus and Output
All operations in this section are shown per-head unless otherwise
specified.
Receptance, which acts like the query found in transformers, is applied to the WKV state, and the
result is normalized. An added bonus, the amount of which is weighted by ρ, allows the model to
place extra attention on the current shifted input token without requiring it to store that token in
the state.
ut =
¡
rt ·(ρ ⊙˜kt)T ¢
vt
bonus
(20)
pt = LayerNorm(rt wkvT
t )+ut
▷attention result
(21)
Finally, the heads are recombined via reshaping so that pt ∈RD, gated, and transformed into the
output as follows:
ot = (gt ⊙pt)W o ∈RD
(22)
4.2
MLP
The MLP module of RWKV-7 is no longer identical to the Channel Mixing module of previous
RWKV-4,5,6 architectures (Peng et al., 2024b). We remove the gating matrix W r , making it a
two-layer MLP. In compensation for the removed gating parameters to satisfy the equi-parameter
condition, we set the hidden dimension to be 4 times the size of model dimension.
k′
t = lerp(x′
t,x′
t−1,µ′
k)W k′ ∈R4D
(23)
o′
t = ReLU(k′
t)2W v′ ∈RD
(24)
8

5
RWKV World v3 Dataset
We train our models on the new RWKV World v3 Dataset, a new multilingual 3.119 trillion token
dataset drawn from a wide variety of publicly available data sources. This dataset aims to help
close the gap with the amount of data used to train modern LLMs, which may consume as many as
15 - 18 trillion tokens (Qwen et al., 2025; Grattafiori et al., 2024). We select the data to approximate
the distribution of our previous World datasets, including English, multilingual, and code, while
slightly enhancing Chinese novels.We describe the composition of our dataset in Appendix B.
6
Pre-Trained Models
We have pre-trained and publicly released seven Apache 2.0 licensed RWKV-7 models:
1. Trained on Pile: RWKV7-Pile of sizes 0.1B, 0.4B, and 1.4B
2. Trained on RWKV World V3: RWKV7-World-3 of sizes 0.1B, 0.4B, 1.5B, and 2.9B
See Appendix E for detailed configurations.
The RWKV-7 Pile models all use the GPT-NeoX-20B tokenizer (Black et al., 2022), and were all
trained from scratch on the Pile dataset, which has 332 billion tokens.
All RWKV World dataset models use the RWKV World Tokenizer. Due to compute budget con-
straints, the Goose World 3 0.1B and 0.4B models were trained from pre-existing RWKV-5 World v1
and v2 checkpoints, and the Goose World 3 1.5B a

## related_work
Linear attention’s major advantage over softmax attention is that it can be formulated as a RNN
with constant running time per token and constant memory usage (Katharopoulos et al., 2020a),
while softmax attention takes O(N) time per token and O(N) memory with regard to sequence
length. Despite this dramatic efficiency improvement, linear attention has its own significant
drawbacks (Schlag et al., 2021; Han et al., 2024; Fan et al., 2025).
One such issue is that linear attention numerically adds to the fixed-size state at every time-step:
older state contents are never removed, only reduced by becoming a smaller proportion of the
numerically increasing state. Due to limitations on the state size, eventually such a system must
mix values together and muddy the outputs retrieved for a given key (Schlag et al., 2021; Yang
et al., 2024b). Modern linear attention architectures like RWKV-6 (Peng et al., 2024b), RetNet (Sun
et al., 2023), Gated Linear Attention (Yang et al., 2023a), and Mamba 2 (Dao & Gu, 2024) use per
time-step decay to remove some portion of such older values from the state in a data-dependent
manner. However, decay is a blunt tool that cannot remove only the values stored at specific keys.
Delta Rule.
DeltaNet (Schlag et al., 2021) sidesteps the problem of numerically increasing state
by partially replacing the value stored at the current key with the same amount of a new value,
allowing the model to both take away old memories and add new ones on a per-key basis. It
reformulates the state update as an explicit online learning problem where the goal is to retrieve
the correct value as output for a given key as input. DeltaNet was the first to apply the foundational
Error Correcting Delta Rule (Widrow et al., 1960) to key-value compressive states, akin to those
stored in the RNN formulation of linear attention. This update rule is equivalent to a single step of
stochastic gradient descent, training the state St at test time to output the desired values vt for
the keys kt as inputs using loss L = 1
2∥(Stkt −vt)∥2 and gradient ∂L
∂S = Sk⊤k −v⊤k, leading to a
recurrent update formula of St = St−1(I −akT
t kt)+ avT
t kt, where a is a scalar learning rate. The
ideas behind this internal state update can be traced back to fast weights (Schmidhuber, 1992).
There has been significant recent interest in improvements to DeltaNet, in order to bring its
efficiency and downstream performance in line with Transformers while still capturing the speed
and memory benefits of Linear Attention. Parallelizing DeltaNet (Yang et al., 2024c) showed that
DeltaNet used diagonal plus low-rank (DPLR) state evolution like S4 (Gu et al., 2022), and could be
parallelized across the time dimension, creating a path to efficiently train such models. Our work
further extends that parallelization to cover the generalized delta rule formulation introduced
herein, as well as the specific formula of RWKV-7.
4

Concurrent Work.
Concurrent work with our own has focused on architectural improvements
beyond DeltaNet while still using the delta rule or variations thereof. Longhorn (Liu et al., 2024)
employs an update rule that approximates a closed-form solution to a globally optimal update
objective, applied on an otherwise unchanged Mamba architecture. Gated Delta Networks (Yang
et al., 2024a) applies gating to the DeltaNet state, essentially multiplying the transition matrix by a
data-dependent scalar per head. This combines the DeltaNet update rule with the scalar decay
found in some modern RNNs like RetNet and Mamba-2. The delta rule gradient descent formula
with dynamic weight decay wt and learning rate at becomes St = St−1
¡
diag(wt)−k⊤
t kt diag(at)
¢
+
v⊤
t kt diag(at).
TTT (Test-Time Training) (Sun et al., 2024) and Titans (Behrouz et al., 2024) also both apply scalar
decay, but eschew per-step gradient descent update rules in favor of a batched multi-timestep
approach. Titans also adds momentum to the otherwise classical SGD update applied to the state.
Another concurrent work with our own, Unlocking State-Tracking in Linear RNNs Through Nega-
tive Eigenvalues (Grazzi et al., 2024), has demonstrated the potential for increased expressiveness
that comes from allowing the state transition matrix to contain negative eigenvalues. We show a
result significantly beyond this, proving that RWKV-7 and our generalized delta rule can recognize
all regular languages using only a small constant number of layers.4
3

## conclusion
19
10.1 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
10.2 Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
A Author Contributions
30
B Training Dataset Details
31
C Transition Matrix Eigenvalues and Stability
32
D Expressivity of RWKV-7
33
D.1 Warmup: Expressivity Beyond TC0 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
D.2 Main Result: RWKV-7 Can Recognize Any Regular Language . . . . . . . . . . . . . . . 34
D.3 Lemmas for Theorem 3
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
E Additional Architectural and Training Details
38
F
Additional Architecture Discussion
41
G Pseudocode For RWKV-7
44
H PyTorch code For Naive WKV7 Kernel (Forward and Backward)
46
I
Board Game Modeling
47
J
State Inspections
50
K Ablation Experiments
51
L
Parameters Statistics
52
M Initial Token Sensitivity
53
2

Name
State Evolution
Scalars
LS
FD
DD
GE
RWKV-4
st = e−w ⊙st−1 +ekt ⊙vt;
✗
✓
✗
✗
s′
t = e−w ⊙s′
t−1 +ekt
RetNet
St = wSt−1 + vT
t kt
w
✓
✗
✗
✗
RWKV-5
St = St−1diag(w)+ vT
t kt
✓
✓
✗
✗
Mamba
St = St−1 ⊙exp(−(wT
t 1)⊙exp(A))+(wt ⊙vt)T kt
✓
✓
✓
✗
RWKV-6 & GLA
St = St−1diag(wt)+ vT
t kt
✓
✓
✓
✗
HGRN-2
St = St−1diag(wt)+ vT
t (1−wt)
✓
✓
✓
✗
Mamba-2
St = wtSt−1 + vT
t kt
wt
✓
✗
✓
✗
TTT a
St = St−1 −at∇l(St−1,kt,vt)
a
✓
✗
✗
✓
Longhorn
St = St−1 ⊙(I −aT
t k2
t )+(at xt)T kt
✓
✓
✓
✗
Gated DeltaNet
St = wtSt−1(I −atkT
t kt)+ at vT
t kt
wt,at
✓
✗
✓
✓
Titans a
M t = (1−αt)M t−1 +St
wt,at
✓
✗
✓
✓
St = wtSt−1 −at∇l(M t−1,kt,vt)
Generalized ∆Rule
St = St−1(diag(wt)+ zT
t bt)+ vT
t kt
✓
✓
✓
✓
RWKV-7 (ours)
St = St−1(diag(wt)−ˆκT
t (at ⊙ˆκt))+ vT
t kt
✓
✓
✓
✓
Table 1: Recent RNN architectures used for language modeling.
LS (Large State): matrix-valued states, or state size at least 4 times larger than the model dimension.
FD (Flexible Decay): the dimension of the decay term w or wt is not smaller than the model dimension.
DD (Dynamic Dependence): the decay term wt is a function over the input xt .
GE (Generalized Eigenvalue): evolution matrix admits eigenvalues outside of the interval [0,1].
a Shown with mini batch size 1 for simplicity.
1