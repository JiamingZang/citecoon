# Simple linear attention language models balance the recall-throughput tradeoff

> 2024 · id: arxiv:2402.18668 · arXiv: 2402.18668 · pdf: https://arxiv.org/pdf/2402.18668 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Recent work has shown that attention-based lan-
guage models excel at recall, the ability to ground
generations in tokens previously seen in context.
However, the efficiency of attention-based mod-
els is bottle-necked during inference by the KV-
cache’s aggressive memory consumption. In this
work, we explore whether we can improve lan-
guage model efficiency (e.g. by reducing mem-
ory consumption) without compromising on re-
call. By applying experiments and theory to a
broad set of architectures, we identify a key trade-
off between a model’s state size and recall abil-
ity. We show that efficient alternatives to atten-
tion (e.g. H3, Mamba, RWKV) maintain a fixed-
size recurrent state, but struggle at recall. We
propose BASED a simple architecture combining
linear and sliding window attention. By vary-
ing BASED window size and linear attention fea-
ture dimension, we can dial the state size and
traverse the Pareto frontier of the recall-memory
tradeoff curve, recovering the full quality of at-
tention on one end and the small state size of
attention-alternatives on the other. We train lan-
guage models up to 1.3b parameters and show that
BASED matches the strongest sub-quadratic mod-
els (e.g. Mamba) in perplexity and outperforms
them on real-world recall-intensive tasks by 10.36
accuracy points. We further develop IO-aware
algorithms that enable BASED to provide 24×
higher throughput on language generation than
FlashAttention-2, when generating 1024 tokens
using 1.3b parameter models. Overall, BASED ex-
pands the Pareto frontier of the throughput-recall
tradeoff space beyond prior architectures.
*Equal contribution 1Stanford University 2University of Buffalo.
Correspondence to: Simran Arora <simarora@stanford.edu>,
Sabri Eyuboglu <eyuboglu@stanford.edu>, Michael Zhang
<mzhang20@stanford.edu>.
Proceedings of the 2 nd Efficient Systems for Foundation Models
Workshop at the International Conference on Machine Learning
(ICML), Vienna, Austria. PMLR 235, 2024. Copyright 2024 by
the author(s).

## introduction
The choice of sequence mixer (e.g. attention, convolu-
tion) in a language model affects both its quality and ef-
ficiency (Arora et al., 2023a; Vaswani et al., 2017). Prior
work shows that attention excels at recall, the ability to
ground generations in previously seen tokens (Olsson et al.,
2022; Arora et al., 2023a). On the other hand, the throughput
of attention-based models is bottle-necked during training
by quadratic compute complexity and during inference by
aggressive memory consumption. The natural question is:
can we improve the real-world speed and memory-use of
language models without comprising on quality?
Recently, a number of architectures have been proposed
that enable substantially higher throughput while competing
with attention in perplexity (Wang et al., 2022; Gu and Dao,
2023; Yang et al., 2023; Poli et al., 2023; Peng et al., 2023).
However, coarse metrics like overall perplexity can obscure
important differences in model quality. For example, recent
work shows that a specific class of architectures, gated-
convolutions, despite complexity scaling sub-quadratically
in sequence length, are asymptotically less efficient than
attention at performing recall (Arora et al., 2023a). Building
on this analysis, we evaluate a broader class of architectures
across real-world recall-intensive tasks and show attention
improves over a currently-popular attention-free alternative,
Mamba, by 32.2 accuracy points (Table 1). 1
Motivated by these observations, we explore the Pareto fron-
tier of the tradeoff between high-recall and high-throughput
models. We evaluate a range of architectures on a popu-
lar synthetic associative recall task (Arora et al., 2023a;
Fu et al., 2023a; Olsson et al., 2022). Since generation
throughput is bottle-necked by memory consumption, we
vary hyperparameters (e.g. model dimension) that affect the
size of the recurrent state during generation and demonstrate
a fundamental recall-memory tradeoff that holds across ar-
chitecture classes (Figure 2). Attention performs associative
recall perfectly, but the recurrent state (i.e. the KV-cache)
grows linearly with the sequence length. Sliding window
1Examples of recall-intensive tasks include information extrac-
tion, reading comprehension, summarization and code generation.
These require using in context information (contrasting memorized
information) during generation.
1
arXiv:2402.18668v2  [cs.CL]  7 Mar 2025

Simple linear attention language models balance the recall-throughput tradeoff
Sliding window width 





Sliding window width 



Taylor approximation 
provides large memory for
recall
 
✔
Precise local token shifts
and comparison 
Limited memory for long
range recall
 
✔
✗
Precise local token shifts
and comparison 
Taylor approximation 
provides large memory for
recall
 
✔
Precise local token shifts
and comparison 
 
✔
✗
Figure 1. BASED overview. Combining linear attention with tiny sliding window softmax attention (e.g., 64 or 128 tokens in width)
enables improved recall accuracy with limited efficiency overhead vs. smaller tile sizes. (Left) Time to execute Cutlass GEMMs (y) vs.
sliding window attention size (x), with batch size 512 on tensor cores. (Center) Model recall accuracy (y) vs. sliding window attention
size (x). We compare linear attention alone (dark blue), sliding window attention alone (light blue), and their combination (BASED,
orange). (Right) Schematic diagram of BASED illustrating how the two components complement each other.
attention (SWA) can cap the size of the recurrent state at the
cost of worse long-range recall (Jiang et al., 2023). However,
Mamba, a recently proposed SSM architecture expands the
Pareto frontier beyond SWA. This begs the question: are
there other, perhaps simpler, models that can also expand
the Pareto frontier?
To reduce the memory consumption, we consider using two
simple techniques: SWA and softmax-approximating linear
attention. Our results on language modeling (Table 1) and
synthetic recall experiments (Figure 1, center) suggest nei-
ther primitive alone suffices to navigate the Pareto frontier.
1. We find that linear attention alone struggles to solve
associative recall (Figure 1, center). We hypothesize
that this is because linear attention lacks the precision
to perform local token shifts and comparisons (Fu et al.,
2023a; Arora et al., 2023a).
2. In sliding window attention, associative recall range is
limited by the width of the windows (Figure 1, center).
As we increase the window size, the recurrent state grows
linearly and has a non-linear affect on speed during par-
allel training and inference (Figure 1, left).
We combine these two techniques into a single architec-
ture, which we call BASED (Figure 1, right). We find that
SWA and linear attention complement each other, enabling
BASED to expand the pareto frontier of the recall-memory
tradeoff (Figure 2). We suspect that (1) the large recurrent
memory of linear attention could help model long-range
token interactions in the sequence and (2) SWA handles the
precise local shifts needed to perform associative recall.
To make BASED competitive with SoTA attention (Dao,
2023) and recurrent (Gu and Dao, 2023) models under
wall-clock and throughput metrics, we introduce several
IO-aware optimizations.
1. Despite the theoretical efficiency benefits, linear at-
tention implementations are often slower than well-
optimized attention implementations (Dao et al., 2022).
To make our attention competitive in real-world wall-
clock time and memory usage, we provide hardware-
efficient CUDA algorithms for liner attention generation
prefill (Algorithm 1) and decoding (Algorithm 2).
In BASED, we show that the 2nd-order Taylor approxi-
mation of softmax as the linear attention feature map is
hardware-efficient. With sequence length N and head di-
mension d, this na¨ıvely requires O(Nd3) time and space
complexity (Zhang et al., 2024; Keles et al., 2023). Rel-
ative to the baseline, our algorithm reduces data move-
ment from HBM (slower-to-access memory) to SRAM
(faster-to-access memory) by O(Nd2) bytes and from
SRAM to register by O(Nd3) bytes (Section 5).
2. Sliding window attention exploits tensor cores, special-
ized units on modern GPUs for performing matrix multi-
plications (GEMMs). While prior architectures use large
window sizes (e.g. 4096 for Mistral-7B (Jiang et al.,
2023)), we propose to use small 64 −128 windows,
guided by hardware properties. Size 64 −128 window
sizes keep the tensor cores occupied Figure 1 (left).
In experiments, we show that BASED competes in qual-
ity with strong Transformer++ (Touvron et al., 2023) and
SoTA sub-quadratic baselines in models up to the 1.3Bn
parameters across language modeling on the Pile language,
DNA modeling, and the LM Eval Harness (Gao et al., 2023).
Beyond this, BASED outperforms a strong sub-quadratic
architecture, Mamba, on the associative recall slice of the
Pile and in downstream recall-intensive tasks by 10.36 ac-
curacy points. In efficiency, BASED enables up to 24×
higher throughput than the strong FlashAttention-2 imple-
mentation on generation. Code for this work is provided at:
https://github.com/HazyResearch/based.
2

Simple linear attention language models balance the recall-throughput tradeoff
2. Preliminaries and Related Work
We discuss the key relevant work in this section and provide
an extended discussion in Appendix A.
Attention
The de facto language modeling primitive, soft-
max attention (Vaswani et al., 2017) takes inputs x ∈RN×d
of length N and head dimension d, and computes outputs
y ∈RN×d via the softmax over projections q, k, v =
xWq, xWk, xWv, i.e.,
yi =
i
∑︂
j=1
exp(q⊤
i kj/
√
d)vj
∑︁i
m=1 exp(q⊤
i km/
√
d)
(1)
in the causal case where Wq, Wk, Wv ∈Rd×d are learn-
able matrices . While effective at recall (Arora et al., 2023a)
and efficient to train (Eq 1 is parallelizable on GPUs and
O(N) in memory with recent advances (Dao et al., 2022)),
attention remains expensive for generation. For every new
output yn, we require nd operations over a growing KV-
cache of prior {ki, vi}n−1
i=1 . This results in larger memory
consumption and lower-throughput for longer sequences.
Efficient attentions
Various works thus try to improve
on attention’s efficiency without sacrificing quality. Sparse
attentions reduce attention’s time and memory requirements
by only attending over specific strided patterns or local slid-
ing windows (Parmar et al., 2018; Child et al., 2019; Beltagy
et al., 2020). While further popularized in large language
models (Mistral, Jiang et al. (2023)), prior works either un-
derperform full attention with sparse patterns that fail to
capture dense interactions, or use large window sizes that
still permit large KV-caches and subsequent inefficiency.
Meanwhile, linear attentions replace the softmax in standard
attention with alternative kernel functions (Kath

## method
Params/Tokens
Efficiency
Language Modeling (Pile)
Info. Extraction
QA
Common
Prefill
Generate
All
AR
Other
SWDE
FDA
SQUAD
LM-Evals
Tok./ms ↑
Tok./ms ↑
Ppl. ↓
Ppl. ↓
Ppl. ↓
Acc ↑
Acc ↑
F1 ↑
Avg. Acc. ↑
Transformer++
1.33b/10b
103.50
0.99
7.26
1.74
8.10
71.92
73.23
36.19
47.64
BASED
1.35b/10b
161.71
24.28
7.43
1.87
8.26
48.06
24.41
30.46
46.68
Mamba
1.32b/10b
112.22
25.69
7.48
1.96
8.29
34.74
12.89
28.20
46.84
Transformer++
1.33b/50b
103.50
0.99
6.28
1.65
6.82
76.50
80.47
43.47
53.33
BASED
1.35b/50b
161.71
24.28
6.30
1.71
6.82
64.45
30.40
41.62
53.81
Mamba
1.32b/50b
112.22
25.69
6.28
1.74
6.78
52.75
18.51
35.92
53.50
Transformer++
360m/10b
207.77
23.82
8.39
1.87
9.42
57.97
58.00
27.18
44.08
BASED
363m/10b
514.57
47.23
8.65
2.07
9.64
29.16
11.71
25.07
43.03
Mamba
358m/10b
267.09
39.95
8.64
2.21
9.59
23.67
6.53
24.06
43.51
GLA
362m/10b
—
—
9.12
2.36
10.68
—
—
—
—
RWKV v5
362m/10b
—
—
9.79
2.40
10.90
—
—
—
—
H3
362m/10b
—
—
10.60
4.88
11.23
6.75
0.64
7.87
39.35
Transformer++
360m/30b
103.50
0.99
7.68
1.80
8.40
70.75
63.79
25.07
44.75
BASED
363m/30b
161.71
24.28
7.77
1.93
8.46
45.01
16.45
32.67
45.36
Mamba
358m/30b
112.22
25.69
7.73
2.02
8.38
27.63
8.71
26.71
45.62
Table 1. Evaluation of pre-trained language models. Models were trained on the same sets of 10b to 50b tokens drawn from the Pile
corpus (Gao et al., 2020). We report inference throughput on 4, 096 tokens (16, 384 for 360m param.) of pre-fill and 2, 048 tokens of
recurrent generation for a subset of architectures. We report language model perplexity on the overall Pile test set as well as perplexity
on two slices of the test set: associative recall tokens and other tokens (see Section 6.1, (Arora et al., 2023a)). We report zero-shot
performance on three recall-intensive tasks: information retrieval on SWDE and FDA as well as question answering on SQUAD. Finally,
we report average performance on the set of LM Eval Harness (Gao et al., 2023) common sense reasoning tasks used in Gu and Dao
(2023), details in Appendix D. These tasks do not require significant recall capacity because the input text is typically very short. See
Section 6.1. Some proposed architectures that do not implement recurrent views for generation are marked with a —.









	


	






	

	
Figure 4. (Left) Throughput numbers for the varied prefill sequence lengths at a fixed batch size of 2. Right generation throughput at
varied batch sizes at a fixed generation length of 1024 tokens. The y-axis shows the in latency (ms). Lines are cutoff when the model runs
out of memory. We show results for both 360M and 1.3Bn, and all numbers are computed on a single NVIDIA H100 GPU.
put over FlashAttention-2 in generating 1024 tokens at
batch size 128 (see Figure 4).
Baselines
We compare to several key baselines at the
360m and 1.3b parameter scales, up to 50b tokens of train-
ing. We first consider Transformer++, Transformers with
modern improvements such as rotary encodings (Su et al.,
2023) and gated linear units (Touvron et al., 2023). We then
consider a class of popular efficient architectures built from
gating and long-convolution primitives including Hyena
(Poli et al., 2023), RWKV (Peng et al., 2023), and H3 (Fu
et al., 2023a). We finally compare to the recently popular
Mamba (Gu and Dao, 2023) and Gated Linear Attention
(Yang et al., 2023) linear recurrent architectures with input-
dependent recurrent-state updates. We give each architec-
ture the Transformer++ improvements as relevant and use
the implementations provided by prior work during training.
BASED combines familiar local and global sequence mixers
to achieve high quality. We train BASED as a hybrid of
≈20% linear attention, ≈20% sliding window attention,
and ≈60% gated convolution layers as discussed in Ap-
pendix E.1. In contrast to recent baselines, BASED requires
no input-dependent decays whatsoever.
6.1. Language Modeling Evaluations
Language Modeling Benchmarks
We pretrain language
models from scratch at two parameter scales (360m and 1.3b
parameters) on the Pile (Gao et al., 2020). Each model sees
the same tokens of pretraining data in the same order. The
Pile data is tokenized using the GPT-2 BPE tokenizer (Rad-
ford et al., 2019). We measure perplexity on the Pile and
8

Simple linear attention language models balance the recall-throughput tradeoff
report results in Table 1 and further experimental details are
provided in Appendix E.1.
We additionally evaluate the pretrained models on key natu-
ral language understanding downstream benchmarks using
the LM Eval Harness (SuperGLUE, ARC, PIQA, Wino-
Grande, HellaSwag, LAMBADA). A detailed breakdown
of tasks and metrics can be found in Appendix D.
In both pretraining and on the downstream tasks, BASED
consistently competes with the strongest Transformer++ and
Mamba baselines. While these overall metrics are helpful,
we next turn to a fine-grained analysis of recall and in-
context learning ability on real-world data.
Recall Evaluations
We evaluate our pretrained models
on a suite of in-context learning tasks selected to test the
downstream recall capacity in Table 1. These tasks fall into
three categories: (1) Real-world AR Beyond perplexity
scores, we slice the next token predictions on the Pile to un-
derstand each architecture’s AR quality ( Appendix E.1). (2)
Information extraction (IE) SWDE and FDA are popular
semi-structured and unstructured document IE benchmarks
respectively (Wu et al., 2021; Deng et al., 2022; Arora et al.,
2023b). SWDE has HTML for 8 Movie and 5 University
websites (e.g. IMDB, US News) and annotations for 8-
274 attributes per website (e.g., Movie runtime), and (3)
Question answering from in-context passages.
We find BASED outperforms the baseline sub-quadratic ar-
chitectures across these evaluations, closing the gap to Trans-
former++. These trends track the MQAR synthetic results
from Section 3.1. We further observe that as we train for
longer (more tokens), the improvements from BASED over
Mamba grow (from 3.9 to 9.0 points on average at 360m
scale and from 9.0 to 10.4 points at the 1.3b scale).
Quality Ablations
In Table 6, we ablate the feature maps,
feature dimensions, and sliding window and convolution
dimensions using the same Pile setting as prior experiments.
In feature maps, we consider replacing the Taylor approxi-
mation with CosFormer (Qin et al., 2022a) or Performers
(Choromanski et al., 2020), and varying the state size using
linear projections. We observe with larger sate size, Cos-
Former closes the gap to the Taylor map though note the
projections increase the parameter count. In feature dimen-
sion, we find 24 and 32 provide diminishing returns. Further
discussion is in Appendix E.1.
6.2. Efficiency Benchmarks
We benchmark the throughput of BASED, with and without
our proposed IO-Aware algorithms (Section 5, Figure 4).
We consider both the forward pass / generation prefill and
next token prediction stages. Experiments were run using
an H100 NVIDIA GPU and averaged over 20 repetitions.
End-to-end benchmarks
Using our efficient implemen-
tation (Section 5), BASED achieves 56% faster prefill than
FlashAttention-2 (Dao, 2023) and 44% faster than Mamba
at 4k sequence length and 1.3b parameters (28% faster than
FlashAttention-2 and 76% faster than Mamba at 360m pa-
rameters). We find that next token generation, with no pre-
fill, provides 24× higher throughput (tokens/second) over
the highly optimized FlashAttention-2 implementation and
achieves 95% and the throughput of the recurrent Mamba
architecture at batch size 128 and 1.3b parameters (98%
higher throughput vs. FlashAttention-2 and 118% higher
throughput vs. Mamba at 360m parameters). All bench-
marks is on a single NVIDIA H100 GPU, using CUDA
cache graphs during next token prediction (NVIDIA, 2019).
In Figure 4, we also include results for the baseline imple-
mentation of BASED that uses the popular Fast Transform-
ers CUDA kernel to compute the causal dot product (Vyas
et al., 2020) (discussed in Section 5). The custom kernel
introduced in our work unlocks the efficiency of BASED.
Micro benchmarks
As the end-to-end BASED architec-
ture is a hybrid architecture, we provide micro benchmarks
of the individual kernels against key baseline implementa-
tions in Appendix B. Kernels are accessible at: https://
github.com/HazyResearch/ThunderKittens.

## experiments
In Figure 2, we demonstrate a fundamental trade-
off between recurrent state size and accuracy on MQAR
that holds within and across architecture classes. Within
each architecture class (e.g. H3 models), increasing the
recurrent state size almost always leads to an improvement
in accuracy. Across architecture classes, we see a tradeoff
as well. Attention achieves perfect recall accuracy, but its
recurrent state size grows with the length of the sequence.
Other architecture classes like Mamba and H3 admit models
with much smaller recurrent states, but these models have
limited recall capacity.
Given a fixed recurrent state, not all architectures have the
same recall capacity. Among architectures proposed in
prior work, Mamba makes the best use of a limited memory
budget. Notably, architectures with a convolutional view
(e.g. Hyena and H3) fall well below the Pareto frontier. Our
proposed architecture, BASED (introduced in Section 4),
expands the Pareto-frontier beyond Mamba. By varying
hyper-parameters that determine its state size (e.g. feature
dimension and model dimension), we can smoothly navigate
the tradeoff between efficient models and memory-hungry
models with high recall capacity.
3.2. Theoretical Analysis
Our theoretical analysis provides further insight into the
empirical observations described above. First, using results
from communication complexity theory, we show that the
recall capacity of any causal model (e.g. Mamba, Attention)
is bounded by the size of its recurrent state (Theorem F.12
in Appendix F).
Theorem 3.1. Any recurrent model2 depending causally on
input u ∈{0, 1}N×d requires Ω(N)-bits3 in state size to
solve MQAR.
This result suggests that the tradeoff observed in Figure 2 is
fundamental, not an artifact of architectural quirks.
Next, we focus on gated-convolutions, a broad class of ar-
chitectures built from gating and convolutions (e.g. H3,
Hyena, RWKV v4). To make progress in theoretically an-
alyzing the broad set of gated convolution proposals, prior
work develops a canonical gated-convolution, referred to as
BaseConv which can provably simulate any architecture
built from gating and convolution primitives.
Building on this work, we show that BaseConv cannot
solve MQAR in constant-many layers (Theorem F.19 and
Theorem F.29 in Appendix F).
Theorem 3.2. Given an input sequence u ∈{0, 1}3N×d,
where N and d denote the sequence length and head dimen-
sion, respectively, a data-independent BaseConv model
needs log(2d)-layers to solve MQAR for d = log2(c),
where c denotes the vocabulary size4.
Remark 3.3. For a class of input encodings that general-
izes one-hot encodings termed as p-hot encodings (Def-
inition F.22), input-dependent BaseConv needs at least
⌊log(2p)⌋-layers to solve MQAR where d = p ·
p√c.
The above result is not as strong when c ≪N, for which
we prove a complementary lower bound (Theorem F.14 in
Appendix F):
2For Mamba (Gu and Dao, 2023), see Corollary F.13.
3Here, we need the entries of the state to be bounded.
4That is, each token from the vocabulary has the natural binary
encoding in {0, 1}log2(c)
4

Simple linear attention language models balance the recall-throughput tradeoff
Theorem 3.4. Given an input u ∈{0, 1}N×d to the
MQAR with any encoding such that log c
≤
d
≤
2(log N)1−ϵ for ϵ > 0, and c possible tokens from the vo-
cabulary with c ≤N, a data-independent BaseConv
model with model parameters taking O(log N) bits needs
Ω(ϵ log log N) layers to solve AR.
In contrast, Arora et al. (2023a) show that attention solves
MQAR in constant-many layers. This result helps to ex-
plain why the gated-convolution architectures (H3 and
Hyena) in Figure 2 lie below the Pareto frontier established
by newer architectures.
Note that Theorem 3.2 and Theorem 3.4 imply that we need
Ω(max(log log c, log log N)) many BaseConv layers to
solve MQAR. One might wonder if we can improve this
lower bound. In Theorem F.30, we show that this is the
best possible lower bound by showing that for certain set-
tings, O(max(log log c, log log N)) BaseConv layers are
enough to solve MQAR.
Finally, we show that we can simulate linear atten-
tion (Katharopoulos et al., 2020a), the foundation of BASED,
using BaseConv (Arora et al., 2023a) with a poly-log
blowup in the number of layers (Proposition F.8 in Ap-
pendix F), pointing to the relative efficiency of linear atten-
tion over gated-convolution architectures.
4. The BASED Architecture
In this section, we introduce BASED. Our objective in de-
signing this architecture is to demonstrate how we can navi-
gate the Pareto-frontier of the memory-recall tradeoff using
well-known architectural building blocks.
Softmax attention excels at recall, but since its recurrent
state, the KV-cache, grows unconstrained with the length of
sequence, it is stuck in the upper right quadrant of Figure 2.
We study two simple approaches for constraining the size
of attention’s recurrent state: linear attention and sliding
window attention. The recurrent state size of linear attention
(i.e. attention without softmax) does not grow with the
sequence length and can be modulated by changing simple
hyperparameters (Katharopoulos et al., 2020a). With sliding
window attention, we cap the recurrent state size to be the
width of the window.
However, our experiments on real-world language modeling
(Table 6) and synthetic associative recall (Figure 1 middle)
suggest that neither primitive alone suffices to navigate the
pareto frontier. Linear attention lacks the precision to per-
form local token shifts and comparisons (Fu et al., 2023b;
Arora et al., 2023a). In sliding window attention, associative
recall range is limited by the width of the windows (Figure
2, center). As we increase the window size, the recurrent
state grows linearly and has a non-linear effect on speed
during parallel training and inference (Figure 2, left).
BASED combines (1) softmax-approximating linear atten-
tion applied globally and (2) exact softmax attention applied
locally in small sliding windows (Figure 1, right). This al-
lows us to use softmax attention in surprisingly small sliding
windows (e.g., 64 −128 tokens) that recover 90.8% of full
softmax attention’s recall accuracy at 1e-5× its latency.
4.1. Taylor Linear Attention
By approximating softmax attention using linear feature
maps, we can constrain the size of the recurrent state while
maintaining global token interactions (i.e. each token de-
pends on every token before it in the sequence).
Katharopoulos et al. (2020a); Tsai et al. (2019); Choroman-
ski et al. (2020) show that we can select a feature map
ϕ : Rd →Rd˜ such that ϕ(qi)⊤ϕ(kj) ≈exp(q⊤
i kj/
√
d).
We can then rewrite the formula for softmax attention in
Equation (1) as
i
∑︂
j=1
ϕ(qi)⊤ϕ(kj)vj
ϕ(qi) ∑︁i
j=1 ϕ(kj)
=
ϕ(qi) ∑︁i
j=1
(︁
ϕ(kj)⊤vj
)︁
ϕ(qi) ∑︁i
j=1 ϕ(kj)
(2)
where every query attends to every past key in O(Nd2) time
and space complexity. Furthermore, Katharopoulos et al.
(2020b) show that linear attention has a fixed size recurrent
state during generation. Letting si = ∑︁i
j=1 ϕ(kj)⊤vj and
zi = ∑︁i
j=1 ϕ(kj)⊤be a “KV-state” and “K-state” respec-
tively, we can compute Equation (2) as
si = si−1 + ϕ(ki)⊤vi,
zi = zi−1 + ϕ(ki)⊤,
yi = ϕ(qi)si
ϕ(qi)zi
(3)
where si ∈Rd×d˜ and zi ∈Rd˜.
Feature map.
To approximate exp(q⊤
i kj/
√
d), we use
the 2nd-order Taylor series feature map, picking ϕ : Rd →
Rd2 such that
ϕ(qi)⊤ϕ(kj) = 1 + q⊤
i kj + (q⊤
i kj)2
2
(4)
While Zhang et al. (2024) note that picking a feature map
with d˜ = d2 results in linear attention with O(Nd3) time
and space complexity and large recurrent state of size O(d3),
we can tradeoff efficiency for recall capacity by projecting
queries and keys to smaller dimensions i.e., Wq, Wk ∈
Rd×d′ with d′ = 16. By changing d′ we modulate the size
of the recurrent state.
How does the choice of feature map affect the memory-recall
tradeoff? Prior work demonstrates the strong performance
5

Simple linear attention language models balance the recall-throughput tradeoff





		

		




	
	

Figure 3. Linear attention feature maps on AR. x: state size
(bytes) during generation or param. count; y: MQAR accuracy.
This setting is harder than Figure 2 (256 key-value pairs).
of the Taylor feature map on associative recall (Zhang et al.,
2024). Building on this analysis, we evaluate a broad set
of feature maps (ϕReLU(x) = max(x, 0), ϕPosELU(x) =
ELU(x) + 1, ϕSquare(x) = x2, ϕIdentity(x) = x, ϕCosFormer
as defined in (Qin et al., 2022a), and ϕPerformer as defined in
(Choromanski et al., 2020)) using the experimental setup
described in Section 3.1. In Figure 3 (top), we plot the
memory-recall tradeoff curves for these feature maps. The
Taylor series feature map, along with the simple ϕPosELU and
ϕReLU feature maps, sits at the Pareto frontier. One advan-
tage of the Taylor feature map over these alternatives is that
it expan

## related_work
GPU operations, or kernels, are executed by many paral-
lel threads. GPU streaming multiprocessors launch thread
blocks at the software level. These blocks are divided into
warps (e.g. 32 threads) that are assigned to cores at the hard-
ware level. Threads need to read inputs into their registers
to perform computations and write the outputs. The time
6

Simple linear attention language models balance the recall-throughput tradeoff
taken to read and write is referred to as the IO cost.
Operations could either be memory or compute bound, de-
pending on the time to load data vs. perform computations
on loaded data. In designing our IO-aware algorithms, we
would like to exploit two key properties of modern GPUs.
First, tensor core units (fast matrix multiply units) achieve
312 TFLOP/s speeds relative to 19 TFLOP/s for the non-
matrix multiply cores. Second, GPUs face a memory hi-
erarchy with large amounts of slow-to-access memory and
smaller amounts of fast-to-access memory. For instance,
the hierarchy on a modern NVIDIA 80GB A100 GPU is:
80GB of HBM with 2 TB/s bandwidth, 80MB of L2 cache,
192KB of L1 cache / shared memory (implemented via
SRAM) with 19 TB/s bandwidth per SM, and 256 KB of
register file per SM (NVIDIA, 2022). Register memory is
private to an executing thread, so threads need to write to
shared memory to communicate data to other threads in
the block. To reduce the IO cost, a key principle is to fuse
multiple operations on the same data slice while it’s in fast
memory before writing it back to slow memory.
5.2. Taylor Exponential Linear Attention
Despite the theoretical efficiency, the popular linear atten-
tion implementations are less efficient than well-optimized
softmax attention implementations when measured in real-
world wall-clock time and memory usage (Dao et al., 2022).
We next present hardware-aware algorithms to make Taylor
linear attention efficient. We focus on two operations: (1)
prefill (this section), corresponding to processing the prompt
during generation or the forward pass during training, and
(2) next token prediction during generation (Appendix B),
which also requires updating the recurrent hidden state state.
In this section, we refer to the batch size as B, number of
heads as H, head dimension as d, sequence length as N and
feature dimension as d′, following Section 4. For ease of
notation, let D = 1 + d′ + d′2 in this section. Additional
details for these algorithms are in Appendix B
Baseline Implementation
The na¨ıve implementation de-
tailed in Appendix B only uses a CUDA kernel to compute
the causal dot product between q, k, and v projections
(Vyas et al., 2020), but computes the feature maps in python
(non IO-aware). This is inefficient given the computation
required for the feature map computation.
Analysis In overall IO cost, ignoring the input and output pro-
jections in the linear attention layer, this procedure requires
2BHND bytes for writing featurized q, k to HBM. During
the causal dot product, this requires 2BHND + BHNd
bytes to read q, k, v tiles and BHNd bytes to write the
result. Throughout the computation, O(BHNDd) bytes
(note this is the shape KV state during the forward pass)
are read in and out of thread registers to SRAM to update
the running output and KV state at 19TB/s bandwidth.
Algorithm
Our kernel computes both the feature map
and causal dot product, detailed in Algorithm 1. Here we
describe the key insights. First, to handle causality in the
dot-product computation, for each tile of output yi ∈R16×d,
we split the computation as shown, where qi, ki, vi are also
now tiles of 16 tokens, handled in parallel by the kernel.
yi = Causal(qT
i ki)vi + qi
i−1
∑︂
j=0
(kjvj)
where the first term uses the quadratic attention view and
requires applying causal masking. The second term uses the
linear view and its causality has already been handled.
Second the large KV-state, ∑︁i−1
j=0(kjvj), ∈RD×d, needs to
be stored as we iterate over the length-16 tiles. By partition-
ing across workers (warps), we can store the state in thread
registers (fastest memory). The partitioning is restricted by
(1) each warp has a limited quantity of threads and (2) warps
cannot access the thread memory of other warps.
Analysis In IO cost, again ignoring the input and output pro-
jections in the linear attention layer, our procedure requires
2BHNd′ bytes for reading q, k and 2BHNd bytes for read-
ing v and writing output y between HBM and SRAM. Over-
all, our algorithm avoids in HBM O(2BHND) bytes in
HBM to SRAM data movement. We further improve upon
the baseline by storing the KV-state in-register to avoid the
O(BHNDd) bytes in SRAM to register data movement.
End-to-end benchmarks for BASED implemented with these
IO-aware algorithms are provided in Section 6. Micro-
benchmarks for each kernel against the baseline implemen-
tations are provided in Appendix B.

## conclusion
This work identifies a fundamental tradeoff between recall, a
critical skill for in-context learning, and throughput through
theory and experiments. Attention performs recall perfectly,
but requires retaining a KV cache that grows with the se-
quence length. As an alternative, we propose the BASED
architecture, which combines two simple techniques — lo-
cal fine-grained attention and long-range linear attention via
a Taylor approximation of the softmax exponential function –
that are sub-quadratic during training and permit an efficient
recurrent inference view. To enable wall clock efficiency,
we introduce IO-aware algorithms for the Taylor linear at-
tention inference that lead BASED to perform generation up
to 24× faster than FlashAttention-2 at the 1.3b parameter
scale (generating 1024 tokens, batch size 128). Beyond
competing in overall perplexity, BASED outperforms prior
sub-quadratic architectures in recall quality by 10.36 ac-
curacy points on average. Overall, our results show that
BASED extends the Pareto frontier of the recall-throughput
tradeoff space beyond prior architectures.
Acknowledgments
We thank Benjamin Spector, Dylan Zinsley, Songlin Yang,
Daniel Fu, Jessica Grogan, Eric Nguyen, Michael Wornow,
Alyssa Unell, and Gautam Machiraju for their helpful
feedback and discussion during this work. We thank the
9

Simple linear attention language models balance the recall-throughput tradeoff
Hazy Research lab and Together AI for supporting this
work.
We gratefully acknowledge the support of NIH
under No.
U54EB020405 (Mobilize), NSF under Nos.
CCF2247015 (Hardware-Aware), CCF1763315 (Beyond
Sparsity), CCF1563078 (Volume to Velocity), and 1937301
(RTML); US DEVCOM ARL under Nos. W911NF-23-2-
0184 (Long-context) and W911NF-21-2-0251 (Interactive
Human-AI Teaming); ONR under Nos. N000142312633
(Deep Signal Processing), N000141712266 (Unifying Weak
Supervision), N000142012480 (Non-Euclidean Geometry),
and N000142012275 (NEPTUNE); Stanford HAI under No.
247183; NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft,
NEC, Toshiba, TSMC, ARM, Hitachi, BASF, Accenture,
Ericsson, Qualcomm, Analog Devices, Google Cloud, Sales-
force, Total, the HAI-GCP Cloud Credits for Research pro-
gram, the Stanford Data Science Initiative (SDSI), and mem-
bers of the Stanford DAWN project: Facebook, Google, and
VMWare. The U.S. Government is authorized to reproduce
and distribute reprints for Governmental purposes notwith-
standing any copyright notation thereon. Any opinions,
findings, and conclusions or recommendations expressed in
this material are those of the authors and do not necessarily
reflect the views, policies, or endorsements, either expressed
or implied, of NIH, ONR, or the U.S. Government. AR’s
research is supported by NSF grant CCF#2247014.
Impact Statement
This paper presents work whose goal is to advance the field
of Machine Learning. We intend for BASED to aid in re-
ducing the costs of machine learning and in unlocking new
capabilities. There are many potential societal consequences
of our work, none which we feel must be specifically high-
lighted here. Detailed discussions of the risks of using and
developing LLMs are in Bommasani et al. (2021); Wei-
dinger et al. (2021).