# Gated Delta Networks: Improving Mamba2 with Delta Rule

> 2024 · id: arxiv:2412.06464 · arXiv: 2412.06464 · pdf: https://arxiv.org/pdf/2412.06464 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Linear Transformers have gained attention as efﬁcient alternatives to standard
Transformers, but their performance in retrieval and long-context tasks has been
limited. To address these limitations, recent work has explored two distinct mech-
anisms: gating for adaptive memory control and the delta update rule for pre-
cise memory modiﬁcations. We observe that these mechanisms are complemen-
tary—gating enables rapid memory erasure while the delta rule facilitates targeted
updates. Building on this insight, we introduce the gated delta rule and develop a
parallel training algorithm optimized for modern hardware. Our proposed archi-
tecture, Gated DeltaNet, consistently surpasses existing models like Mamba2 and
DeltaNet across multiple benchmarks, including language modeling, common-
sense reasoning, in-context retrieval, length extrapolation, and long-context un-
derstanding. We further enhance performance by developing hybrid architectures
that combine Gated DeltaNet layers with sliding window attention or Mamba2 lay-
ers, achieving both improved training efﬁciency and superior task performance.
Code: https://github.com/NVlabs/GatedDeltaNet
1

## introduction
The Transformer architecture has signiﬁcantly advanced the capabilities of Large Language Models
(LLMs), showcasing exceptional performance across a wide range of tasks due to its effective atten-
tion mechanism. This mechanism excels in precise sequence modeling and leverages the parallel
processing capabilities of modern GPUs during training. However, the self-attention component
scales quadratically with sequence length, leading to substantial computational demands that pose
challenges for both training and inference.
To mitigate these issues, researchers have explored alternatives such as linear Transformers
(Katharopoulos et al., 2020a), which replace traditional softmax-based attention with kernelized
dot-product-based linear attention, substantially reducing memory requirements during inference
by reframing as a linear RNN with matrix-valued states. While early versions of linear Trans-
formers underperformed in language modeling tasks compared to standard Transformers, recent
enhancements—such as incorporating data-dependent gating mechanisms akin to those in LSTMs,
exempliﬁed by models like GLA (Yang et al., 2024a) and Mamba2 (Dao & Gu, 2024a)—have
shown promising improvements. However, challenges persist in managing information over long
sequences, particularly for in-context retrieval tasks where traditional Transformers maintain their
advantage (Arora et al., 2023a; 2024a; Jelassi et al., 2024; Wen et al., 2024; Akyürek et al., 2024).
This phenomenon is not surprising: linear Transformers can be interpreted as implementing an
outer-product-based key-value association memory, reminiscent of tensor product representation
(Smolensky, 1990). However, the number of orthogonal key-value pairs they can store is bounded by
the model’s dimensionality. When the sequence length exceeds this dimension, “memory collisions“
become inevitable, hindering exact retrieval (Schlag et al., 2021a).
Mamba2 addresses this limitation by introducing a simple gated update rule, St = αtSt−1 + vtk⊺
t ,
which uniformly decays all key-value associations at each time step by a dynamic ratio, αt ∈(0, 1).
∗Equation contribution. Work done during SY’s internship at NVIDIA.
1

Published as a conference paper at ICLR 2025
However, this approach does not account for the varying importance of different key-value associ-
ations, potentially leading to inefﬁcient memory utilization. If the model needs to forget a speciﬁc
key-value association, all key-value associations are equally forgotten, making the process less tar-
geted and efﬁcient.
In contrast, the linear Transformer with the delta rule (Widrow et al., 1960), known as DeltaNet
(Schlag et al., 2021a; Yang et al., 2024b), selectively updates memory by (softly) replacing an old
key-value pair with the incoming one in a sequential manner. This method has demonstrated im-
pressive performance in synthetic benchmarks for in-context retrieval. However, since this process
only modiﬁes a single key-value pair at a time, the model lacks the ability to rapidly clear outdated
or irrelevant information, especially during context switches where previous data needs to be erased.
Consequently, DeltaNet has been found to perform moderately on real-world tasks (Yang et al.,
2024b), likely due to the absence of a robust memory-clearing mechanism.
Recognizing the complementary advantages of the gated update rule and the delta rule in memory
management, we propose the gated delta rule, a simple and intuitive mechanism that combines both
approaches. This uniﬁed rule enables ﬂexible memory control: it can promptly clear memory by
setting αt →0, while selectively updating speciﬁc content without affecting other information by
setting αt →1 (effectively switching to the pure delta rule).
The remaining challenge lies in implementing the gated delta rule in a hardware-efﬁcient manner.
Building upon Yang et al. (2024b)’s efﬁcient algorithm that parallelizes the delta rule computation
using the WY representation (Bischof & Loan, 1985), we carefully extend their approach to incor-
porate the gating terms. Our extension preserves the beneﬁts of chunkwise parallelism (Hua et al.,
2022b; Sun et al., 2023a; Yang et al., 2024a;b), enabling hardware-efﬁcient training.
Our resulting architecture, Gated DeltaNet, consistently outperforms both Mamba2 and DeltaNet
across a comprehensive suite of benchmarks, including language modeling, commonsense reason-
ing, in-context retrieval, length extrapolation, and long-context understanding. Building on these
results, we also develop hybrid architectures that strategically combine Gated DeltaNet layers with
sliding window attention or Mamba2 layers, further enhancing both training efﬁciency and model
performance.
2
PRELIMINARY
2.1
MAMBA2: LINEAR ATTENTION WITH DECAY
It is known that the linear transformer (Katharopoulos et al., 2020b) can be formulated as the follow-
ing linear recurrence when excluding normalization and query/key activations:
St = St−1 + vtk⊺
t ∈Rdv×dk,
ot = Stqt ∈Rdv
where dk and dv represent the (head) dimensions for query/key and value, respectively. By expand-
ing the recurrence, we can express it in both vector form (left) and matrix form (right) as follows:
ot =
t
X
i=1
(vik⊺
i )qt =
t
X
i=1
vi(k⊺
i qt) ∈Rdv,
O = (QK⊺⊙M)V ∈RL×dv
where L is the sequence length, and M ∈RL×L is the causal mask deﬁned by Mij = 0 when i < j,
and 1 otherwise.
However, this vanilla linear attention underperforms Transformers in language modeling by a large
margin. To address this, it is common to add a decay term to forget historical information. Here we
take Mamba2 (Dao & Gu, 2024a) as an example, which can be represented by the following linear
recurrence (up to speciﬁc parameterization):
St = αtSt−1 + vtk⊺
t ,
ot = Stqt
where αt ∈(0, 1) is a data-dependent scalar-valued decay term that varies with t. Deﬁne the cumu-
lative decay product γj = Qj
i=1 αi, and by expanding the recurrence, we can express the result in
both a vector form (left) and a matrix parallel form (right):
ot =
t
X
i=1
γt
γi
vik⊺
i

qt =
t
X
i=1
vi
γt
γi
k⊺
i qt

,
O = ((QK⊺) ⊙Γ) V
2

Published as a conference paper at ICLR 2025
Here, Γ ∈RL×L is a decay-aware causal mask where Γij = γi
γj if i ≥j and Γij = 0 otherwise.
The equivalence between these parallel and recurrent forms is also referred to as the state space
duality (SSD) described in Dao & Gu (2024a). This recurrence structure appears in several other
architectures including Gated RFA (Peng et al., 2021), xLSTM (Beck et al., 2024), and Gated RetNet
(Sun et al., 2024b). When γt is data-independent, the formulation reduces to RetNet (Sun et al.,
2023a) and Lightning-Attention (Qin et al., 2024a). Furthermore, if γt is extended to be matrix-
valued rather than scalar-valued, efﬁcient training algorithms remain possible when parameterized
with an outer-product structure, as demonstrated by Yang et al. (2024a) and used by Yang et al.
(2024a); Peng et al. (2024); Qin et al. (2024b); Zhang et al. (2024); Chou et al. (2024); He et al.
(2025); Lu et al. (2025).
Chunkwise training
However, both the recurrent and parallel forms are not ideal for efﬁcient
training (Hua et al., 2022b; Yang et al., 2024a), which motivates the use of the chunkwise parallel
form (Hua et al., 2022b; Sun et al., 2023a) for hardware-efﬁcient, linear-time training, as introduced
below. To summarize, the chunkwise parallel form splits inputs and outputs into several chunks of
size C, and computes outputs for each chunk based on the ﬁnal state of the previous chunk and the
query/key/value blocks of the current chunk. Following the notation of Sun et al. (2023b); Yang
et al. (2024a;b), we take the query block, q, as an example. We denote Q[t] := qtC+1:(t+1)C+1 as
the query block for chunk t, and qr
[t] := qtC+r as the r-th query within chunk t. The initial state of
chunk t is deﬁned as S[t] := S0
[t] = SC
[t−1]. By partially expanding the recurrence, we have
Sr
[t] = S[t] +
r
X
i=1
vi
[t]ki⊺
[t] ∈Rdv×dk,
or
[t] = Sr
[t]qr
[t] = S[t]qr
[t] +
r
X
i=1
vi
[t]

ki⊺
[t]qr
[t]

∈Rdv
Equivalently, in matrix form:
S[t+1] = S[t] + V[t]K⊺
[t] ∈Rdv×dk,
O[t] = Q[t]S⊺
[t] +

Q[t]K⊺
[t] ⊙M

V[t] ∈RC×dv
where M ∈RC×C is the causal mask. The above equations are rich in matrix multiplications
(matmuls), allowing for tensor-core-based hardware optimization. This chunkwise algorithm could
be easily extended to linear attention with decay:
S[t+1] = −→
S[t] + V⊺
[t]
−−→
K[t] ∈Rdv×dk,
O[t] = ←−−
Q[t]S⊺
[t] +

Q[t]K⊺
[t] ⊙Γ[t]

V[t] ∈RC×dv
(1)
where (Γ[t])ij =
γi
[t]
γj
[t] , γj
[t] = QtC+j
j=tC+1 αj. 1 Here we use the left arrow (←−· ) or the right arrow (−→· )
to denote a variable decaying to the ﬁrst position and the last position of each chunk, respectively,
←−
qr
[t] = γr
[t]qr
[t]
decaying each vector to the ﬁrst position of chunk t
−→
kr
[t] =
γC
[t]
γr
[t]
kr
[t]
decaying each vector to the last position of chunk t
−→
S[t] = γC
[t]S[t]
decaying the state matrix over the entire chunk t
(2)
and likewise for othe

## method
1K
2K
4K
8K
1K
2K
4K
8K
1K
2K
4K
DeltaNet
97.4
96.8
99.0
98.8
98.4
45.6
18.6
14.4
85.2
47.0
22.4
Mamba2
99.2
98.8
65.4
30.4
99.4
98.8
56.2
17.0
64.4
47.6
4.6
Gated DeltaNet
98.4
88.4
91.4
91.8
100.0
99.8
92.2
29.6
86.6
84.2
27.6
On the other hand, Linear Attention (LA) and Mamba2 use a simple negative inner-product loss
-⟨Stkt, vt⟩, while Longhorn (Liu et al., 2024) uses a more expressive online regression objective
∥Stkt −vt∥2 for better modeling of key-value associations. The resulting Longhorn’s update rule
closely resembles the delta update rule, 3 suggesting the superiority of the (gated) delta rule over
Mamba2 in in-context associative recall.
From the perspective of fast weight programming (Irie et al., 2022a) and test-time training (Sun et al.,
2024a) and regression (Wang et al., 2025), the hidden state S can be interpreted as a (fast) weight
matrix, with the delta rule optimizing the online regression objective L(St) = 1
2∥Stkt −vt∥2 via
test-time stochastic gradient descent (SGD):
St+1 = St −βt∇L(St) = St −βt(Stkt −vt)k⊺
t = St (I −βtktk⊺
t ) + βtvtk⊺
t
where βt represents the (adaptive) learning rate. From this perspective, the gated delta rule can be
viewed as incorporating an adaptive weight decay term αt into the SGD update, a technique widely
used in deep learning (Krogh & Hertz, 1991; Andriushchenko et al., 2023). Concurrently, Titans
(Behrouz et al., 2024) demonstrated the effectiveness of incorporating weight decay mechanisms in
RNN test-time SGD updates.
3.2
CASE STUDY: SINGLE NEEDLE IN A HAYSTACK (S-NIAH)
To better understand the complementary strength between the delta rule and the gated rule, we
present a case study on the Single Needle-In-A-Haystack (S-NIAH) benchmark suite from RULER
(Hsieh et al., 2024), where a key-value pair acts as a needle in the haystack (context) and the model
must recall the value when given the key. Table 2 presents the results and we draw three main
observations:
Decay hurts memory retention.
In the simplest S-NIAH-1 setting with repeated synthetic con-
text, models memorize minimal information, testing long-term retention. DeltaNet achieves near-
perfect performance across all sequence lengths. Mamba2 degrades signiﬁcantly beyond 2K se-
quences since it decays historical information too quickly, while Gated DeltaNet’s degradation is
less severe thanks to the use of delta rule.
Gating facilitates ﬁltering.
In S-NIAH-2/3 with real-world-essay context, models store all po-
tentially relevant information, testing efﬁcient memory management. With ﬁxed state size, lack
of clearance causes memory collision—information becomes superimposed and indistinguishable.
DeltaNet’s performance drops signiﬁcantly at longer sequences due to poor memory clearance.
Mamba2 and Gated DeltaNet maintain better performance through gating mechanisms that ﬁlter
irrelevant information.
3The theoretical distinction lies in the optimization approach: Longhorn uses implicit online learning (Kulis
& Bartlett, 2010) to derive closed-form globally optimal updates, while DeltaNet optimizes the same objective
through one-step explicit gradient descent, as noted by Liu et al. (2024).
5

Published as a conference paper at ICLR 2025
Delta rule helps memorization.
In S-NIAH-3, values change from numbers to UUIDs, testing
complex pattern memorization. Mamba2’s performance drops quickly, while Gated DeltaNet per-
forms better, verifying that the delta rule indeed has better memorization ability.
3.3
ALGORITHM: HARDWARE-EFFICIENT CHUNKWISE TRAINING
In this subsection, we derive a hardware-efﬁcient chunkwise algorithm for training Gated DeltaNet.
By partially expanding the recurrence in Eq. 10, we have
Sr
[t] = S[t]
 r
Y
i=1
αi
[t]

I −βi
[t]ki
[t]ki⊺
[t]
!
|
{z
}
:=Fr
[t]
+
r
X
i=1

βi
[t]vi
[t]ki⊺
[t]
r
Y
j=i+1
αj
[t]

I −βj
[t]kj
[t]kj⊺
[t]



|
{z
}
:=Gr
[t]
It is easy to see that Fr
[t] = γr
[t]Pr
[t] = ←−
Pr
[t]. As for Gr
[t], we adapt Eq. 5 as follows,
Gr
[t] =
r
X
i=1
γr
[t]
γi
[t]
˜ui
[t]ki⊺
[t] ∈Rdv×dk
˜ur
[t] = βr
[t]
 
vr
[t] −
r−1
X
i=1
 
˜ui
[t](
γr
[t]
γi
[t]
ki⊺
[t]kr
[t])
!!
∈Rdv
(see §A for a proof). By UT transform, we have the matrix form:
g
U[t] =
h
I + strictLower

diag
 β[t]

(Γ[t] ⊙K[t]K⊺
[t])
i−1
diag
 β[t]

V[t]
∈RC×dv
Similar to how Mamba2 extends linear attention (Eq. 1), we can adapt DeltaNet’s chunkwise algo-
rithm (Eq. 8-9) for Gated DeltaNet to enable hardware-efﬁcient training as follows:
S[t+1] = −→
S[t] +

g
U[t] −←−−
W[t]S⊺
[t]
⊺−−→
K[t]
∈Rdv×dk
O[t] = ←−−
Q[t]S⊺
[t] + (Q[t]K⊺
[t] ⊙M)

g
U[t] −←−−
W[t]S⊺
[t]

∈RC×dv
where ←−
qr
[t] = γr
[t]qr
[t], ←−−
wr
[t] = γr
[t]wr
[t], −→
kr
[t] =
γC
[t]
γr
[t] kr
[t], and −→
S[t] = γC
[t]S[t] like we deﬁned in Eq. 2.
3.4
GATED DELTA NETWORKS AND HYBRID MODELS
Token mixer block.
The basic Gated DeltaNet follows Llama’s macro architecture, stacking to-
ken mixer layers with SwiGLU MLP layers, but replaces self-attention with gated delta rule token
mixing. Fig. 1 (right) shows its block design. For the gated delta rule (Eq. 10), queries, keys and
values {q, k, v} are generated through linear projection, short convolution and SiLU, with L2 nor-
malization applied to q, k for training stability. α, β use linear projection only.4 Following Sun et al.
(2023a), the output is processed through normalization and gating before applying output projection.
Hybrid models.
Linear transformers have limitations in modeling local shifts and comparisons,
and their ﬁxed state size makes it hard for retrieval tasks (Arora et al., 2024a). Following recent
hybrid architectures like Grifﬁn (De et al., 2024) and Samba (Ren et al., 2024), we combine linear
recurrent layers with sliding window attention (SWA), resulting in GatedDeltaNet-H1. We also stack
Mamba2, GatedDeltaNet and SWA, resulting in GatedDeltaNet-H2.
4

## experiments
Setup
Our experiments encompass a comprehensive comparison of recent state-of-the-art archi-
tectures, including pure Transformer models, RNN-based approaches, and hybrid architectures. We
evaluate against the following baselines: RetNet (Sun et al., 2023a), HGRN2 (Qin et al., 2024b),
Mamba (Gu & Dao, 2023), Mamba2 (Dao & Gu, 2024b), Samba (Ren et al., 2024), and DeltaNet
(Yang et al., 2024b). For fair comparison, all models are trained under identical conditions with
1.3B parameters on 100B tokens sampled from the FineWeb-Edu dataset (Penedo et al., 2024). We
use the AdamW optimizer with a peak learning rate of 4e-4, weight decay of 0.1, and gradient clip-
ping of 1.0. The learning rate follows a cosine annealing schedule with a 1B token warm-up period
4We use Mamba2’s parameterization for α but omit it for brevity.
6

Published as a conference paper at ICLR 2025
N×
Gated DeltaNet
MLP
SWA
MLP
Outputs
Gated DeltaNet-H2
Gated DeltaNet-H1
Mamba2
MLP
Gated DeltaNet
MLP
SWA
MLP
Outputs
Block Design
Gated Delta Rule
Inputs
Linear
Conv
v
Linear
Conv
L2
k
Linear
Conv
L2
q
Lin.
Lin.
α
β
Linear
Norm
Linear
Outputs
Figure 1: Visualization of the (hybrid) architecture and block design of Gated DeltaNet models.
Gated
DeltaNet-H1 and H2 use Gated DeltaNet + SWA and Mamba2 + Gated DeltaNet + SWA patterns, respec-
tively. In the block design, query/key paths consist of linear proj., shortconv., SiLU and L2 norm; value path
includes linear proj., shortconv. and SiLU; alpha/beta use linear proj.; and output gate applies linear proj. with
SiLU.

## related_work
Gated linear RNN.
Large linear recurrent language models have attracted signiﬁcant attention
due to their training and inference efﬁciency. The ﬁeld of linear RNNs has rapidly evolved from
using data-independent decay mechanisms, as exempliﬁed by models like S4 (Gu et al., 2022), S5
(Smith et al., 2023), LRU (Orvieto et al., 2023), RWKV4/5 (Peng et al., 2023), and RetNet (Sun
et al., 2023a), to incorporating data-dependent decay mechanisms in more recent architectures such
as HGRN1/2 (Qin et al., 2024b; 2023b), Mamba1/2 (Gu & Dao, 2023; Dao & Gu, 2024a), RWKV6
(Peng et al., 2024), GSA (Zhang et al., 2024). This transition stems from the proven advantages of
gating/forgetting mechanisms (termed selective mechanisms in Mamba)—a classical concept orig-
inating in the gated RNN literature (Gers et al., 2000) whose signiﬁcance has been consistently
reafﬁrmed (Greff et al., 2015; van der Westhuizen & Lasenby, 2018; Qin et al., 2024b; 2023b; Gu &
Dao, 2023).
Modern forget gates differ from traditional designs like those in LSTM by removing the depen-
dency on the previous hidden state, relying solely on input data. This modiﬁcation enables efﬁcient
parallelism across sequence lengths (Martin & Cundy, 2018; Qin et al., 2023b). The absence of a
forget gate has been a notable limitation in DeltaNet, and our gated extension addresses this gap in
9

Published as a conference paper at ICLR 2025
a natural, effective, and hardware-efﬁcient way. We also note a recent concurrent work RWKV-7 5
using a similar idea, but with a more relaxable formalism using diagonal-plus-low-rank transitions:
St = St−1(diag(dt) −atb⊤
t ) + vtk⊤
t where dt, at, bt ∈Rdk. The chunkwise algorithm could be
similarly adapted to this case, as implemented in Flash Linear Attention (Yang & Zhang, 2024). 6
Delta rule.
The delta learning rule demonstrates superior memory capacity compared to Heb-
bian learning (Gardner, 1988; Prados & Kak, 1989), an advantage DeltaNet leverages while linear
transformers rely on Hebbian-like rules. This memory capacity advantage is evident in synthetic
in-context learning tasks and extends to language modeling (Irie et al., 2021; Yang et al., 2024b),
reinforcement learning (Irie et al., 2022b), and image generation (Irie & Schmidhuber, 2023).
Yang et al. (2024b) parallelized delta rule computation and demonstrated how DeltaNet’s data-
dependent identity-plus-low-rank structure (I −βtktk⊺
t ) offers greater ﬂexibility than Mamba2’s
data-dependent diagonal matrices (αtI). This structural advantage could enable complex reason-
ing, including regular language recognition (Fan et al., 2024; Grazzi et al., 2024) and state-tracking
beyond TC0 complexity (Merrill et al., 2024)—crucial for coding and reasoning applications.
Despite these signiﬁcant advantages, the delta rule faces theoretical limitations (Irie et al., 2023) and
shows only moderate performance on real-world datasets (Yang et al., 2024b), suggesting room for
improvement. Previous attempts to enhance expressiveness through nonlinear recurrence (Irie et al.,
2021; 2022b) addressed some limitations but sacriﬁced training parallelism, creating a performance-
efﬁciency tradeoff. Recent work proposes some enhancements without compromising parallelism
for better state tracking performance, including using negative eigenvalues (Grazzi et al., 2024) and
multiple products of householder transition matrices (Siems et al., 2025) which enable high-rank
transformations. These methods could be applied to Gated DeltaNet seamlessly.
From a (online) learning objective perspective, alternative formulations could further extend expres-
siveness: nonlinear regression (L(St) =
1
2||fSt(kt) −vt||2) as in TTT (Sun et al., 2024a) and
Titans (Behrouz et al., 2024), where fS is a nonlinear function parameterized by S; or regression
considering the entire history (L(St) =
1
2
Pt
i=1 ||Stki −vi||2) as in Mesa layer (von Oswald
et al., 2024)—analogous to the difference between Least Mean Square and Recursive Least Square
algorithms. However, these more expressive variants introduce nonlinear recurrence and require
workarounds, such as performing nonlinear updates only after processing entire chunks (as in TTT
and Titans); or approximating nonlinear recurrence methods like Lim et al. (2024); Gonzalez et al.
(2024); Schöne et al. (2025).
Hybrid models.
In this work, we explore interleaving hybrid attention layers across layers, which
is commonly used such as in MiniMax-01 (MiniMax et al., 2025) and Hybrid Mamba2-Attention
(Waleffe et al., 2024). It is also interesting to investigate hybrid linear/softmax attention within a
single layer (Hua et al., 2022a; Zancato et al., 2024; Munkhdalai et al., 2024; Nunez et al., 2024;
Dong et al., 2025; Zhang et al., 2025).
6

## conclusion
In this work, we introduced Gated DeltaNet, which enables better key-value association learning
compared to Mamba2 and more adaptive memory clearance than DeltaNet, leading to consistently
better empirical results across various tasks. We extended the parallel algorithm from Yang et al.
(2024b) to enable hardware-efﬁcient training of Gated DeltaNet. Our hybrid Gated DeltaNet model
achieves even higher training throughputand overall performance, making it well-suited for practical
deployment.
ACKNOWLEDGMENT
We thank Yu Zhang for assistance with ﬁgure creation and model evaluation; Kazuki Irie for pro-
viding valuable feedback on the draft; Simeng Sun and Zhixuan Lin for insightful discussions on
5https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7
6https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/
generalized_delta_rule.
10

Published as a conference paper at ICLR 2025
long-sequence task evaluation settings; and Eric Alcaide and Volodymyr Kyrylov for their helpful
discussions on the online learning perspective of DeltaNet.