# Unlocking Feature Learning in Gated Delta Networks at Scale

> 2026 · id: arxiv:2606.04048 · arXiv: 2606.04048 · pdf: https://arxiv.org/pdf/2606.04048 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Training and scaling Large Language Models demand enormous computational resources,
motivating both efficient sub-quadratic architectures and principled hyperparameter tuning
methods. While the Maximal Update Parametrization (µP) has enabled zero-shot hyperparameter
transfer for standard Transformers, its extension to linear models, particularly those with
structured state transitions and complicated architectures, remains largely unexplored. By
rigorously propagating coordinate-size estimates through the forward pass, gating mechanisms,
and recurrent state dynamics, we derive the scaling rules for Gated Delta Network. Experiments
on language-model pre-training confirm that our configurations enable stable learning-rate
transfer across model widths under both AdamW and SGD, whereas standard parametrization
fails to transfer, validating the correctness and practical utility of our analysis.
1

## introduction
The rapid development of Large Language Models (LLMs) has demonstrated remarkable capabilities
across a wide range of downstream tasks (Brown et al., 2020; Touvron et al., 2023; Radford et al.,
2019; Vaswani, 2017). However, scaling these models to larger sizes introduces two challenges.
First, empirical scaling laws show that optimal performance requires increased model size, while
the computational budget required for training grows steeply with model scale (Kaplan et al.,
2020; Hoffmann et al., 2022). Second, the efficiency of standard Transformer architecture is limited
by the quadratic complexity of softmax self-attention with respect to sequence length, making it
increasingly costly for long-context inference and training (Katharopoulos et al., 2020).
Linear models have been proposed to address these issues. The original linear attention (Katharopou-
los et al., 2020) rewrites softmax attention as a linear kernel, enabling recurrent-form inference
at constant per-step cost. Structured state space models (SSMs) such as S4 (Gu et al., 2022),
Mamba (Gu and Dao, 2024) and Mamba-2 (Dao and Gu, 2024) utilize recurrent state spaces
to represent long-range dependencies within linear structures. A particularly promising family
of linear recurrent models is based on the delta rule (Widrow and Hoff, 1960), which updates
a fast-weight matrix by subtracting the prediction error of the current key-value pair. Further-
more, the DeltaNet (Yang et al., 2024c) introduced a hardware-efficient parallel training algorithm
for delta-rule Transformers, enabling scaling to large language models. Afterwards, Gated Delta
Network (Yang et al., 2025) augmented DeltaNet with the data-dependent gating mechanism of
1
arXiv:2606.04048v1  [cs.LG]  2 Jun 2026

Mamba-2, which achieves strong language modeling performance while maintaining linear-time
inference.
Simultaneously, training deep networks requires careful selection of hyperparameters such as
learning rates, which are expensive to tune through grid search (Snoek et al., 2012, 2015), and whose
optimal values often change greatly with model scale. Meta-learning approaches have been explored
to transfer hyperparameters across tasks and datasets (Yogatama and Mann, 2014; Perrone et al.,
2018; Horváth et al., 2021; Akiba et al., 2019). A more principled solution is offered by the Maximal
Update Parametrization (µP) (Yang and Hu, 2021), which identifies the valid parametrization of
a neural network that supports feature learning in the infinite-width limit, as formalized through
the Tensor Programs framework (Yang et al., 2022; Yang and Littwin, 2023; Yang et al., 2024a).
µP theories demonstrated that hyperparameters tuned on small proxy models transfer zero-shot to
large target models, with extensions to adaptive optimizers (Yang and Littwin, 2023; Ishikawa and
Karakida, 2024; Everett et al., 2024) and a spectral reformulation (Yang et al., 2023). Subsequent
work has successfully applied µP to other fields (Blake et al., 2025; Dey et al., 2024; Hajjar et al.,
2024), and even industrial models (Meta AI, 2025; Team et al., 2025).
Despite the development of efficient linear architectures, how to properly parametrize them for
feature learning at scale has received very limited attention. The core challenge is that their recurrent
state is updated through the sequence dimension, which does not fit the standard feedforward or
attention-based derivations of µP. The only prior work on this is Vankadara et al. (2024), which
shows that vanilla µP and spectral scaling conditions both fail to support feature learning in
diagonal SSMs like Mamba, and proposes a corrected scaling rule for them. However, the Gated
Delta Network differs from diagonal SSMs fundamentally, since its recurrent state is a full matrix
updated with additional data-dependent scalar gating through two separate weight matrices. These
differences make the SSM-specific analysis of Vankadara et al. (2024) inapplicable, leaving the µP
parametrization of Gated Delta Networks an open problem.
In this paper, we formally derive the complete µP formulation for Gated Delta Networks. Our
main contributions are:
• We theoretically derive coordinate-size estimates through the full forward pass. We also derive
principled initialization variances, forward multipliers, and learning-rate scalings for all weight
classes. We discover that the gating weight matrices require a non-standard Θ(1/
√
d) learning-
rate scaling, and the scalar gating parameters require a Θ(
√
d) scaling, both of which deviate
from standard µP setting.
• We pretrain Gated Delta Network language models across multiple widths and show that our µP
formulation enables zero-shot learning-rate transfer under both AdamW and SGD optimizers,
while standard parametrization fails to transfer, confirming both the theoretical derivation and
its practical efficiency.
2

## experiments
6.1
Experiment details
We implement LLM pre-training experiments to validate our µP derivation. All models use 8 layers
and 6 attention heads. We test five model widths d ∈{256, 512, 1024, 1536} for AdamW (Loshchilov,
2017) and d ∈{256, 512, 768, 1024} for SGD optimizer, which correspond to parameter counts
ranging from approximately 21M to 342M (non-embedding).
Architectural parameters.
We refer to (Yang et al., 2025) and its official repository for the
implementation of GDN and re-implement it on nanoGPT training framework (Karpathy, 2022). In
detail, we set the head dimension of queries and keys to d/8 and that of values to d/4. And we set
the kernel size of short convolutions in queries and keys to 4. Additionally, the intermediate size of
MLP is set to 4d, and we tied the input and output embeddings.
Initialization and optimizer.
For the base model with d0 = 256, the embedding layer and all input
projections are initialized with standard deviation 0.02, which also applies to larger models for
SP. In contrast, we initialize large models under original µP and our proposed µP according to
Tables 2 and 1. The scalar gating parameters alog and b follow the scheme of Yang et al. (2025):
alog ∼Uniform(0, 16) and b is set as described in Section 4. We apply a gradient clipping to 1.0
and set Dropout ratio (Srivastava et al., 2014) to 0.0. The minimum learning rate is fixed to 5e-5
throughout. All runs use a cosine learning-rate schedule with 2,000 warmup steps.
For AdamW experiments, we use a weight decay of 0.1 and set (β1, β2) = (0.9, 0.95). And for
SGD experiments, we use SGD with Nesterov momentum (Nesterov, 1983) with a momentum
of 0.98, since we notice there is great instability when using original SGD optimizer. For both
optimizers, we use the same learning rates for all the modules in models with d0 = 256 and all
models under SP, and applies different learning rates according to the scaling laws in Tables 2 and 1
in µP experiments.
We train models with each width at 5-7 different learning rates log-spaced with increased density
near optimal learning rates. The learning rate search grid ranges between 1e-3 and 2e-2 for AdamW
and between 1e-1 and 1 for SGD experiments. And we fix the training seed to 42.
13

Data and compute.
We train on the FineWeb-Edu 100B dataset (Lozhkov et al., 2024) for 20k
steps with a global batch size of 480 sequences and a sequence length of 1024 (approximately 9.83B
tokens in total). Moreover, we use 1 NVIDIA H100 80GB HBM3 GPU for all the experiments.
6.2
Experiment results
The final validation losses for models with different widths and peak learning rates under the µP
and SP configurations are shown in Figures 1 and 2 for AdamW and SGD, respectively. To remove
the trivial width-dependence of the absolute loss, we report shifted validation loss, defined as the
difference from the optimal loss value among all the experiments on the models with the same width
but different learning rates.
For AdamW, the optimal learning rate is consistently the same across all 4 model widths under
µP, demonstrating zero-shot learning-rate transfer. While under SP, the optimal learning rate
shifts substantially with width, confirming that SP fails to support feature learning at scale. SGD
experiments show the same qualitative pattern. Under SP, the optimal learning rate does not transfer
across widths and under original µP configuration, it varies a lot. And in our µP configuration,
the optimal learning rate transfers perfectly. These results validate that our theory works well in
practice.
1.0
0.3
0.4
0.5
0.6
0.7
0.8 0.9
2.0
Learning Rate (×10
2)
0.00
0.01
0.02
0.03
0.04
0.05
Shifted validation Loss
Width 256 (shifted by 3.4525)
Width 512 (shifted by 3.1107)
Width 1024 (shifted by 2.8613)
Width 1536 (shifted by 2.7554)
(a) Standard Parametrization (SP)
1.0
0.3
0.4
0.5
0.6
0.7
0.8 0.9
2.0
Learning Rate (×10
2)
0.00
0.01
0.02
0.03
0.04
0.05
Shifted validation Loss
Width 256 (shifted by 3.4525)
Width 512 (shifted by 3.1142)
Width 1024 (shifted by 2.8593)
Width 1536 (shifted by 2.7449)
(b) µP configuration
Figure 1 Shifted validation loss for Gated Delta Network trained with AdamW under varying peak learning
rates and model widths.
7

## related_work
3.1
Gated Delta Net
Proposed by Yang et al. (2025), Gated Delta Net is a variant of linear transformer (Katharopoulos
et al., 2020), based on the Mamba 2 architecture (Dao and Gu, 2024). For the query, key and value
vectors qt, kt and vt similar to the original Transformer, the update rule of the latent state is shown
as:
St = St−1(αt(I −βtktk⊤
t )) + βtvtk⊤
t ,
(3.1)
where αt ∈(0, 1) is the data-dependent gating scale and βt ∈(0, 1) is the “writing strength” of the
current input at time t, as proposed in Widrow and Hoff (1960); Schlag et al. (2021). And the
output is just direct readout of the latent state on the query:
ot = Stqt.
(3.2)
Different from the Transformer, Gated Delta Net added a short convolution after query, key and
value projections, followed by a SiLU activation layer. There are also L2 Normalization layer for
queries and keys. And there is also an RMSNorm layer before the output projection to stabilize the
training. As discussed in the original paper, these norm are crucial to the performance of Gated
Delta Net.
3.2
µP theory
In deep learning, models are frequently scaled by increasing their hidden dimension or width
d. Under the Standard Parameterization (SP), including He (He et al., 2015) or Xavier (Glorot
and Bengio, 2010) initialization, hidden weights are typically initialized with entries drawn from
N(0, σ2/d) and optimized using a uniform learning rate η across all layers. However, as d goes to
infinity, SP encounters fundamental limitations. If the learning rate remains constant, the network’s
activations and gradients diverge. To prevent this instability, η must be scaled down by O(1/d),
which forces the network into the Neural Tangent Kernel (NTK) or “lazy training” regime Yang and
Hu (2021), where the intermediate representations (features) seldom evolve from their initialized
state, meaning the network fails to perform real feature learning.
To resolve the trade-off between stability and feature learning in the infinite-width limit, Yang
and Hu (2021) proposed the Maximal Update Parameterization (µP) using the Tensor Programs
framework. µP provides rigorous configurations for scaling weight initializations and learning rates
as a function of the width d (sometimes a width-dependent multiplier on the weight is required;
refer to Tables 2 and 1 for AdamW and SGD configurations) to ensure feature learning. In this
setting, feature updates at every layer remain bounded and non-vanishing (i.e., ∆h = Θ(1)) as the
model expands to infinity width. To further illustrate this, the definition of coordinate size should
be first introduced:
Definition 3.1. A vector v ∈Rd has Θ(da)-sized coordinates if ∥v∥2/d = Θ(d2a), i.e., each entry of
v has variance Θ(d2a) as d →∞. When d is large, the coordinates of the vectors being studied are
regarded as roughly i.i.d. Gaussian.
4

Based on the definition above, µP theory proposes three desiderata. Firstly, every (pre)activation
vector should have Θ(1)-sized coordinates; and the output of a network should be O(1); moreover,
all parameters should be updated as much as possible without leading to divergence. And based on
these desiderata and the assumption of feature learning, there are some derivatives. For example,
the gradient to a hidden state is with Θ(1/d) coordinate size when optimized with SGD optimizer.
4
The µP Forward Analysis of Gated Delta Net
In this section, we will review the architecture of Gated Delta Net, and then derive the scaling law
of this architecture.
We derive the Maximal Update Parametrization (µP) conditions for Gated Delta Net by
propagating coordinate-size estimates through the forward pass and the gating mechanisms, then
conclude with the implications for the AdamW optimizer.
Notation and standing assumptions.
Following Yang et al. (2022), we say a vector z ∈Rd has Θ(1)
coordinate size if ∥z∥2 = Θ(
√
d), i.e. each coordinate is of order Θ(1) in magnitude. Equivalently,
the per-coordinate variance of z is Θ(1). For a matrix A ∈Rd×d, we say it has Θ(c) coordinate size
if each entry is of order Θ(c) in magnitude.
We assume throughout that the hidden state xt ∈Rd satisfies the µP feature-learning condition,
namely
∥xt∥2 = Θ(
√
d),
∥∆xt∥2 = Θ(
√
d),
so that xt has Θ(1) coordinate size and its update is of the same order. To isolate the effect of the
parametrization we temporarily ignore the SiLU activations (see Remark 4.1 below).
4.1
Coordinate sizes of the projected features
Let eqt = ShortConv(Wq xt), ekt = ShortConv(Wk xt), and vt = ShortConv(Wv xt), where
Wq, Wk, Wv ∈Rd×d are the query, key, and value projection matrices, respectively.
Under
the µP initialization of hidden weights (Yang et al., 2022), the products Wq xt, Wk xt, Wv xt each
have Θ(1) coordinate size; the short convolution preserves this order, so eqt, ekt and vt each have
Θ(1) coordinate size. The L2-normalized query and key are
qt =
eqt
∥eqt∥2
, kt =
ekt
∥ekt∥2
.
Since ∥eqt∥2 = Θ(
√
d), each coordinate of qt and kt is of order Θ(1)/Θ(
√
d) = Θ(1/
√
d), i.e., qt and
kt both have Θ(1/
√
d) coordinate size.
4.2
Coordinate size of the latent state
The rank-one write update in (3.1) is Ut = βt vtk⊤
t . Since βt ∈(0, 1) as a bounded scalar and
combining the Θ(1) coordinate size of vt with the Θ(1/
√
d) coordinate size of kt, each entry of Ut
satisfies
(Ut)ij = βt (vt)i (kt)j = Θ(1) · Θ
 1
√
d

= Θ
 1
√
d

.
5

For the cumulative latent state St, we apply the argument of Vankadara et al. (2024)1: unless
the write update Ut perfectly cancels the residual term in (3.1) at every step t, the steady-
state variance of St matches that of Ut. More precisely, the spectral contraction factor of the
map S 7→S αt(I −βtktk⊤
t ) is at most αt(1 −βt∥kt∥2
2) ≤αt, which is strictly less than 1 when
αt, βt ∈(0, 1) are both bounded away from 0 and 1. We assume this condition holds throughout the
analysis. Under this assumption the geometric sum of write updates converges and St has Θ(1/
√
d)
coordinate size.
4.3
Coordinate size of the readout
The output is ot = St qt, so the j-th coordinate is
(ot)j =
d
X
i=1
(St)ji (qt)i.
Treating the entries of St and qt as approximately independent and zero-mean, each with variance
Θ(1/d), the variance of the sum is
Var
(ot)j
 =
d
X
i=1
Var
(St)ji
 · Var
(qt)i
 = d · Θ
1
d

· Θ
1
d

= Θ
1
d

,
so ot has Θ(1/
√
d) coordinate size. Although the subsequent RMSNorm forces its output to Θ(1)
coordinate size, this implicit rescaling would disrupt the gradient scaling required by µP. We
therefore recommend inserting a
√
d-multiplier before RMSNorm so that the input to RMSNorm is
already Θ(1):
ot 7→
√
dot,
∥
√
dot∥2 = Θ(
√
d).
An equivalent alternative is to replace the L2-Normalization on qt with an RMSNorm layer, which
absorbs the same
√
d-factor. With either modification, the standard µP formulation applies to all
projection weights other than those governing αt and βt.
4.4
First-order analysis of the gating scalars
The gating parameters are defined as
βt = σ(Wβ xt) ∈(0, 1),
Wβ ∈R1×d,
and
αt = egt ∈(0, 1),
gt = −ealog ln(1 + e Wαxt+b),
with Wα ∈R1×d a trainable weight row and alog, b ∈R scalar parameters shared within each head.
Because both αt and βt are nonlinear transformations of a Gaussian-distributed pre-activation, they
are not themselves Gaussian, and traditional µP theory does not directly apply.
For αt, since it is bounded by (0, 1), it is naturally Θ(1). Define zα,t = Wαxt + b, when
zα,t + alog ≪0, |∂αt/∂zα,t| = αtezα,t+alog/(1 + ezα,t) is also Θ(1). Under the original initialization
of Yang et al. (2025), namely alog ∼Uniform(0, 16) and b = b0 + ln(1 −e−b0) with b0 = 102ϵb−3,
ϵb ∼Uniform(0, 1), the gradient |∂αt/∂zα,t| = αt · ezα,t+alog/(1 + ezα,t) = ezα,t+alog/(1 + ezα,t)ealog+1
is bounded by 1.
1The detailed derivations can be found in Appendix A.1.
6

Proof. Denote k = ealog > 0, p = ezα,t > 0, and f(k, p) = log(
kp
(1+p)k+1 ) = log(kp) −(k + 1) log(1 + p).
Then we have ∂f
∂p = 1
p −1+k
1+p =
1−pk
p(1+p). Therefore, ∀k > 0, f(k, p) ≤f(k, 1/k) = log(
1
(1+1/k)k+1 ) < 0,
and |∂αt/∂zα,t| = ef(k,p) < 1.
For βt, since zβ,t := Wβxt is Θ(1), βt ∈(0, 1) does not saturate and βt = σ(zβ,t), therefore,
∂βt
∂zβ,t = βt(1 −βt) = Θ(1).
Combining the two analyses, Θ(1) coordinate-size behavior of both αt and βt is maintained
under µP initialization. Consequently, Wα and Wβ may be treated as hidden weights with initial
variance 1/d, while the scalar parameters alog and b are assigned constant (i.e., width-independent)
initial variance.
Remark 4.1 (Effect of SiLU activations). The analysis above assumes that the SiLU activations
SiLU(x) = x/(1+e−x) following the short convolutions are suppressed. In practice, these activations
introduce non-Gaussian statistics in eqt, ekt, and vt. Therefore, the derivations above are only an
approximation to the ideal µP conditions. The approximation quality degrades if the pre-activations
are far from zero, but remains adequate for the initialization in µP.
5
The µP Analysis of Gated Delta Net under SGD
In this section, w

## conclusion
We have derived the µP-style parametrization for Gated Delta Networks. Our analysis reveals that
under SGD, scalings of the gating weight matrices and the scalar gating parameters are different
from the standard µP law. LLM pre-training experiments confirm that our µP formulation achieves
zero-shot learning-rate transfer under both AdamW and SGD, while standard parametrization fails
to transfer, empirically validating the correctness of our theoretical derivation. And we hope our
derivations would enlighten further research in the scaling laws of other linear or hybrid architectures.
14

1.0
10.0
0.9
2.0
3.0
4.0
5.0
6.0
7.0 8.0 9.0
Learning Rate (×10
1)
0.00
0.05
0.10
0.15
0.20
0.25
0.30
Shifted validation Loss
Width 256 (shifted by 3.3990)
Width 512 (shifted by 3.1699)
Width 768 (shifted by 3.1094)
Width 1024 (shifted by 3.0821)
(a) Standard Parametrization (SP)
1.0
2.0
3.0
4.0
5.0
6.0
7.0
8.0
Learning Rate (×10
1)
0.00
0.05
0.10
0.15
0.20
0.25
0.30
0.35
Shifted validation Loss
Width 256 (shifted by 3.3590)
Width 512 (shifted by 3.0667)
Width 768 (shifted by 3.0076)
Width 1024 (shifted by 3.0407)
(b) Original µP configuration
1.0
2.0
3.0
4.0
5.0
6.0
7.0
8.0
Learning Rate (×10
1)
0.00
0.05
0.10
0.15
0.20
0.25
0.30
Shifted validation Loss
Width 256 (shifted by 3.4490)
Width 512 (shifted by 3.1606)
Width 768 (shifted by 3.0830)
Width 1024 (shifted by 3.0345)
(c) Our µP configuration
Figure 2
Shifted validation loss (loss minus the best-achieved loss at width d = 1024) for Gated Delta
Network trained with SGD under varying peak learning rates and model widths.
Code Availability
The code for this paper can be accessed in https://github.com/lauyikfung/gated_delta_net_
mup.
Acknowledgement
Thank Fetch Compute program for their support of compute resources. Thank Songlin Yang
for discussion. Thank Amazon Trainium scholarship project for their funding support. And the
code specific for Trainium chips can be accessed in https://github.com/lauyikfung/Amazon_
Trainium_Optimizer/tree/main/gdn_mup_code.