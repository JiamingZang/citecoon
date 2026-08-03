# Large Language Diffusion Models

> 2025 · id: arxiv:2502.09992 · arXiv: 2502.09992 · pdf: https://arxiv.org/pdf/2502.09992 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Large Language Diffusion Models
Shen Nie1,2,3∗†
Fengqi Zhu1,2,3∗†
Zebin You1,2,3†
Xiaolu Zhang4‡
Jingyang Ou1,2,3
Jun Hu4‡
Jun Zhou4
Yankai Lin1,2,3‡
Ji-Rong Wen1,2,3
Chongxuan Li1,2,3‡ §
1 Gaoling School of Artificial Intelligence, Renmin University of China
2 Beijing Key Laboratory of Research on Large Models and Intelligent Governance
3 Engineering Research Center of Next-Generation Intelligent Search and Recommendation, MOE
4 Ant Group
{nieshen,fengqizhu,chongxuanli}@ruc.edu.cn
Abstract
The capabilities of large language models (LLMs) are widely regarded as relying
on autoregressive models (ARMs). We challenge this notion by introducing LLaDA,
a diffusion model trained from scratch under the pre-training and supervised fine-
tuning (SFT) paradigm. LLaDA employs a forward data masking process and
a reverse generation process, parameterized by a Transformer to predict masked
tokens. It provides a principled generative approach for probabilistic inference by
optimizing a likelihood lower bound. Across extensive benchmarks on general
tasks, math, code, and so on, LLaDA demonstrates strong scalability and performs
comparably to our self-constructed ARM baselines. Remarkably, LLaDA 8B is
competitive with strong LLMs like LLaMA3 8B in in-context learning and, after
SFT, exhibits impressive instruction-following abilities in case studies such as multi-
turn dialogue. Moreover, LLaDA addresses the reversal curse, surpassing GPT-4o
in a reversal poem completion task. Our findings show the promise of diffusion
models for language modeling at scale and challenge the common assumption that
core LLM capabilities discussed above inherently depend on ARMs. Project page
and codes: https://ml-gsai.github.io/LLaDA-demo/.
1
Introduction
Large language models (LLMs) [1] fall entirely within the framework of generative modeling.
Specifically, LLMs aim to capture the true but unknown language distribution pdata(·) by optimizing
a model distribution pθ(·) through maximum likelihood estimation, or equivalently KL divergence
minimization between the two distributions:
max
θ
Epdata(x) log pθ(x) ⇔min
θ
KL(pdata(x)||pθ(x))
|
{z
}
Generative modeling principles
.
(1)
The predominant approach relies on the autoregressive modeling (ARM)—commonly referred to as
the “next-token prediction” paradigm—to define the model distribution:
pθ(x) = pθ(x1)
L
Y
i=2
pθ(xi | x1, . . . , xi−1)
|
{z
}
Autoregressive formulation
,
(2)
∗Equal contribution.
†Work done during an internship at Ant Group.
‡Project leaders.
§Correspondence to Chongxuan Li.
39th Conference on Neural Information Processing Systems (NeurIPS 2025).
arXiv:2502.09992v3  [cs.CL]  18 Oct 2025

47
33
38
50
22
25
35
50
50
56
39
45
60
27
30
42
59
60
65
46
53
70
31
35
48
69
70
MMLU
TruthfulQA
ARC-C
GSM8K
Math
HumanEval
MBPP
CMMLU
C-Eval
General Tasks
Mathematics
Code
Chinese
LLaDA 8B Base
LLaMA 3 8B Base
LLaMA 2 7B Base
46
28
60
53
22
41
39
58
36
75
66
27
51
49
70
43
90
80
32
61
59
MMLU
MMLU-pro
ARC-C
GSM8K
Math
HumanEval
MBPP
General Tasks
Mathematics
Code
LLaDA 8B Instruct
LLaMA 3 8B Instruct
LLaMA 2 7B Instruct
Figure 1: Zero/Few-Shot Benchmarks. We scale LLaDA to 8B parameters from scratch and observe
competitive zero/few-shot performance compared with strong autoregressive LLMs [6].
where x is a sequence of length L, and xi is the i-th token. This paradigm has proven remarkably
effective [2–5] and has become the foundation of current LLMs. Despite its widespread adoption, a
fundamental question remains unanswered: Is the autoregressive paradigm the only path to achieving
the core capabilities of LLMs, such as scalability, in-context learning, and instruction-following?
We argue that the answer is not a simple “yes”. The key insight overlooked previously is: It is the
generative modeling principles (i.e., Eq. (1)), rather than the autoregressive formulation (i.e., Eq.
(2)) itself, that fundamentally underpin the essential properties of LLMs.
In particular, we argue that scalability is primarily a consequence of the interplay between Trans-
formers [7], model size, data size, and Fisher consistency5 [8] induced by the generative principles
in Eq. (1), rather than a unique result of the ARMs in Eq. (2). The success of diffusion trans-
formers [9, 10] on visual data [11] supports this claim. Furthermore, the instruction-following and
in-context learning [4] capabilities appear to be intrinsic properties of all conditional generative
models on structurally consistent linguistic tasks, rather than exclusive advantages of ARMs. In
addition, while ARMs can be interpreted as a lossless data compressor [12, 13], any sufficiently
expressive probabilistic model can achieve similar capabilities [14].
However, certain inherent limitations of LLMs can be directly attributed to their autoregressive nature.
For instance, the left-to-right generation process restricts their ability to handle reversal reasoning
tasks [15], highlighting a representative failure in the generalization capabilities of current models.
Motivated by these insights, we introduce LLaDA (Large Language Diffusion with mAsking) to inves-
tigate whether the capabilities exhibited by LLMs can emerge from generative modeling principles
beyond ARMs, thereby addressing the fundamental question posed earlier. In contrast to traditional
ARMs, LLaDA leverages a masked diffusion model (MDM) [16–20], which incorporates a forward
data masking process and trains a mask predictor to approximate its reverse process. This design
enables LLaDA to construct a model distribution with bidirectional dependencies and optimize
a variational lower bound of its log-likelihood, offering a principled and previously unexplored
perspective on the core capabilities of LLMs discussed above.
We adopt the standard pipeline of data preparation, pre-training, supervised fine-tuning (SFT), and
evaluation, scaling LLaDA to an unprecedented language diffusion of size 8B. In particular, LLaDA
8B was pre-trained from scratch on 2.3 trillion tokens using 0.13 million H800 GPU hours, followed
by SFT on 4.5 million pairs. Across diverse tasks, including language understanding, math, code,
and Chinese, LLaDA demonstrates the following contributions:
• LLaDA scales effectively to a compute budget of 1023 FLOPs, achieving comparable results to
ARM baselines trained on the same data across six tasks, e.g., MMLU and GSM8K.
5It suggests the ability to recover the true data distribution with infinite data, a sufficiently large network and
optimal training.
2

Mask predictor
Mask all tokens independently
Mask predictor
Prompt
Response
Mask predictor
Prompt
Response
... 
... 
An intermediate step
𝑡= 0
𝑡= 1
Non-mask token
Random mask
Remask
Mask token
(c)
(b)
(a)
Remask
Mask ratio 𝑡∼𝑈(0,1)
Figure 2: Overview of LLaDA. (a) Pre-training. LLaDA is trained on text with random masks
applied independently to all tokens at the same ratio t ∼U[0, 1]. (b) SFT. Only response tokens are
possibly masked. (c) Sampling. LLaDA simulates a diffusion process from t = 1 (fully masked) to
t = 0 (unmasked), predicting all masks simultaneously at each step with flexible remask strategies.
• The pre-trained LLaDA 8B Base surpasses LLaMA2 7B Base [21] on nearly all 15 standard
zero/few-shot learning tasks while performing on par with LLaMA3 8B Base [6], showcasing
effective in-context learning capability.
• LLaDA significantly enhances the ability to follow instructions after SFT, as demonstrated in case
studies such as multi-turn dialogue.
• LLaDA effectively breaks the reversal curse [15] with consistent performance across forward and
reversal tasks. Notably, it outperforms GPT-4o in a reversal poem completion task.
2
Approach
In this section, we introduce the probabilistic formulation6, along with the pre-training, supervised
fine-tuning, and inference procedures for LLaDA, as illustrated in Fig. 2.
2.1
Probabilistic Formulation
Unlike ARMs in Eq. (2), LLaDA defines a model distribution pθ(x0) through a forward process and
a reverse process [16–20]. The forward process gradually masks tokens independently in x0 until
the sequence is fully masked at t = 1. For t ∈(0, 1), the sequence xt is partially masked, with each
being masked with probability t or remaining unmasked with probability 1 −t. The reverse process
recovers the data distribution by iteratively predicting masked tokens as t moves from 1 to 0.
The core of LLaDA is a mask predictor, a parametric model pθ(·|xt) that takes xt as input and
predicts all masked tokens (denoted as M) simultaneously. It is trained using a cross-entropy loss
computed only on the masked tokens [18–20]:
L(θ) ≜−Et,x0,xt
"
1
t
L
X
i=1
1[xi
t = M] log pθ(xi
0|xt)
#
,
(3)
where x0 is a training sample, t is a continuous random variable drawn uniformly from [0, 1], xt is
sampled from the forward process and L is the sequence length. The indicator function 1[·] ensures
that the loss is computed only for masked tokens.
Once trained, we can simulate a reverse process (see Sec. 2.4 for details) parameterized by the mask
predictor and define the model distribution pθ(x0) as the marginal distribution induced at t = 0. The
loss function in Eq. (3) has been proven to be an upper bound on the negative log-likelihood of the
model distribution, making it a principled objective for generative modeling:
−Epdata(x0) [log pθ(x0)] ≤L(θ).
(4)
Notably, LLaDA employs a masking ratio that varies randomly between 0 and 1 while BERT [22]
uses a fixed ratio. The subtle differences have significant implications, especially at scale: as shown in
6Here, we focus on the approach of LLaDA. A rigorous formulation of MDM is provided in Appendix A for
interested readers.
3

Eq. (4), LLaDA is a principled generative model with the potential to perform in-context learning and
instruction-following naturally, akin to LLMs. Moreover, its generative perspective implies strong
scalability with large data and models as discussed in Sec. 1. In addition, MaskGIT [23] adopts a
heuristic training objective, which misses the 1
t term compared to Eq. (3), and lacks a theoretical link
to maximum likelihood. We emphasize that it is precisely the theoretical foundation of maximum
likelihood estimation that motivated us to scale discrete diffusion models for language modeling.
2.2
Pre-training
LLaDA employs a Transformer [7] as the mask predictor, similar to existing LLMs. However, LLaDA
does not use a causal mask, as its formulation allows it to see the entire input for predictions.
We trained two variants of LLaDA with different sizes: 1B and 8B. We summarize the model
architecture of LLaDA 8B and LLaMA3 8B [6] here, and details are provided in Appendix B.2. We
have ensured consistency in most hyperparameters while making several necessary modifications. We
use vanilla multi-head attention instead of grouped query attention [24] for simplicity, as LLaDA is
incompatible with KV caching, resulting in a different number of key and value heads. Consequently,
the attention layer has more parameters, and we reduce the FFN dimension to maintain a comparable
model size. Additionally, the vocabulary size differs due to a tokenizer [4] adapted on our data.
The LLaDA model is pre-trained on a dataset comprising 2.3 trillion (T) tokens, adhering to a data
protocol that aligns closely with existing LLMs [25, 26], without the incorporation of any special
techniques. The data are derived from online corpora, with low-quality content filtered through
manually designed rules and LLM-based approaches. Beyond general text, the dataset encompasses
high-quality code, math, and multilingual data. Please refer to Appendix B.1 for more details about
datasets. The mixing of data sources and domains is guided by scaled-down ARMs. The pre-training
process utilizes a fixed sequence length of 4096 tokens, incurring a total computational cost of 0.13
million H800 GPU hours, similar to ARMs of the same scale and dataset size.
For a training sequence x0, we randomly sample t ∈[0, 1], mask each token independently with the
same probability t to obtain xt (see Fig. 2 (a)) and estimate Eq. (3) via the Monte Carlo method for
stochastic gradient descent training. In addition, following Nie et al. [27], to enhance the ability of
LLaDA to handle variable-length data, we set 1% of the pre-training data to a random length that is
uniformly sampled from the range [1, 4096].
We adopted the Warmup-Stable-Decay [28] learning rate scheduler to monitor the training progress
without interrupting continuous training. Specifically, we linearly increased the learning rate from
0 to 4 × 10−4 over the first 2000 iterations and maintained it at 4 × 10−4. After processing 1.2T
tokens, we decayed the learning rate to 1 × 10−4 and held it constant for the next 0.8T tokens to
ensure stable training. Finally, we linearly reduced the learning rate from 1 × 10−4 to 1 × 10−5 for
the last 0.3T tokens. Furthermore, we utilized the AdamW optimizer [29] with a weight decay of 0.1,
a batch size of 1280, and a local batch size of 4 per GPU. The 8B experiment was executed once,
without any hyperparameter tuning.
2.3
Supervised Fine-Tuning
We enhance the capability of LLaDA to follow instructions by supervised fine-tuning (SFT) with
paired data (p0, r0), where p0 is the prompt and r0 denotes the response. This is the simplest and most
basic post-training method for LLMs. Technically, this requires to model the conditional distribution
pθ(r0|p0) instead of pθ(x0) in pre-training.
The implementation is similar to pre-training. As shown in Fig. 2 (b), we leave the prompt unchanged
and mask the tokens in the response independently, as done for x0. Then, we feed both the prompt
and the masked response rt to the pre-trained mask predictor to compute the loss for SFT:
−Et,p0,r0,rt

1
t
L′
X
i=1
1[ri
t = M] log pθ(ri
0|p0, rt)

,
(5)
where L′ denotes a dynamic length specified later, and all other notations remain the same as before.
Note that this approach is fully compatible with pre-training. Essentially, the concatenation of p0
and r0 can be treated as clean pre-training data x0, while the concatenation of p0 and rt serves as the
4

masked version xt. The process is identical to pre-training, with the only difference being that all
masked tokens happen to appear in the r0 portion.
The LLaDA 8B model undergoes SFT on a dataset comprising 4.5 million pairs. Consistent with the
pre-training process, both data preparation and training follow the SFT protocols utilized in existing
LLMs [25, 26], without introducing any additional techniques to optimize LLaDA’s performance.
The dataset spans multiple domains, including code, mathematics, and instruction-following. We
append |EOS| tokens to the end of short pairs in each mini-batch to ensure equal lengths across all
data. We treat |EOS| as a normal token during training and remove it during sampling, enabling
LLaDA to control the response length automatically. Please refer to Appendix B.1 for more details.
We train for 3 epochs on the SFT data using a similar schedule to the pre-training phase. The learning
rate is linearly increased from 0 to 2.5 × 10−5 over the first 50 iterations and then kept constant.
During the final 10% of iterations, it is linearly reduced to 2.5 × 10−6. Additionally, we set the
weight decay to 0.1, the global batch size to 256, and the local batch size to 2 per GPU. The SFT
experiment was executed once, without any hyperparameter tuning.
2.4
Inference
As a generative model, LLaDA can sample new text and evaluate the likelihood of candidate text in a
diffusion manner instead of the left-to-right autoregressive fashion.
We begin with the reverse generation process. As illustrated in Fig. 2 (c), given a prompt p0, we
discretize the reverse process to sample from the model distribution pθ(r0|p0), starting from a fully
masked response. The total number of sampling steps is a hyperparameter, which naturally provides
LLaDA with a trade-off between efficiency and sample quality, as analyzed in Sec. 3.3. We employ
uniformly distributed timesteps by default. In addition, the generation length is also treated as a
hyperparameter, specifying the length of the fully masked sentence at the beginning of the sampling
process. After generation, tokens appearing after the |EOS| token are discarded. As detailed in
Appendix B.5, since both pre-training and SFT are conducted using datasets with variable lengths,
the final results are insensitive to this length hyperparameter.
At an intermediate step from time t ∈(0, 1] to s ∈[0, t), we feed both p0 and rt into the mask
predictor and predict all masked tokens simultaneously. Subsequently, we remask s
t of the predicted
tokens in expectation to obtain rs, ensuring that the transition of the reverse process aligns with
the forward process for accurate sampling [18–20]. In principle, the remasking strategy should be
purely random. However, inspired by the annealing tricks of sampling in LLMs [4, 30], we adopt
a low-confidence remasking strategy, where s
t of predicted tokens with the lowest confidence are
remarked based on the predictions, same as the approach of Chang et al. [23].
We mention that LLaDA enables flexible sampling. In particular, it supports autoregressive and block
diffusion [31] sampling directly after the pre-training or SFT processes described above, without
requiring any further modifications or training. We provide a detailed analysis in Appendix B.4.
Nevertheless, the diffusion sampling (i.e., the reverse generation process) yields the best performance
and is adopted as the default throughout this paper, especially for all experiments presented in Sec. 3.
For conditional likelihood evaluation, we can naturally utilize the upper bound in Eq. (5). However,
we find that the following equivalent form [20] exhibits lower variance and is more stable:
−El,r0,rl
"
L
l
L
X
i=1
1[ri
l = M] log pθ(ri
0|p0, rl)
#
,
(6)
where L is the sequence length of r0, l is uniformly sampled from {1, 2, . . . , L}, and rl is obtained
by uniformly sampling l tokens from r0 without replacement for masking.
We present the training and inference algorithms, along with theoretical details, in Appendix A.
3
Experiments
We evaluate the scalability, instruction-following, and in-context learning capabilities of LLaDA on
standard benchmarks, followed by analyses and case studies to provide a comprehensive assessment.
5

10
20
10
21
10
22
10
23
FLOPs
20
30
40
50
60
MMLU (5-shot)
Autoregressive Baseline
LLaDA
10
20
10
21
10
22
10
23
FLOPs
20
30
40
50
60
ARC-C (0-shot)
Autoregressive Baseline
LLaDA
10
20
10
21
10
22
10
23
FLOPs
20
30
40
50
60
CMMLU (5-shot)
Autoregressive Baseline
LLaDA
10
20
10
21
10
22
10
23
FLOPs
50
60
70
80
PIQA (0-shot)
Autoregressive Baseline
LLaDA
10
20
10
21
10
22
10
23
FLOPs
0
20
40
60
GSM8K (4-shot)
Autoregressive Baseline
LLaDA
10
20
10
21
10
22
10
23
FLOPs
0
8
16
24
HumanEval (0-shot)
Autoregressive Baseline
LLaDA
Figure 3: Scalability of LLaDA. We evaluate the performance of LLaDA and our ARM baselines
trained on the same data across increasing pre-training computational FLOPs. LLaDA exhibits strong
scalability, matching the overall performance of ARMs on six tasks.
3.1
Scalability of LLaDA on Language Tasks
We first investigate the scalability of LLaDA on downstream tasks in comparison with the ARM
baselines we constructed. Specifically, at the 1B scale, we ensured that LLaDA and ARM shared
the same architecture, data, and all other configurations. At larger scales, we also report results
for LLaDA and ARM models of slightly different sizes trained on the same data due to resource
limitations. Please refer to Appendix B.2 for more details. We use the pre-training computational
cost as a unified scaling metric. For evaluation, we focused on six standard and diverse tasks.
Fig. 3 shows that LLaDA demonstrates impressive scalability, with its overall trend highly competitive
with ARMs. Notably, on tasks such as MMLU and GSM8K, LLaDA exhibits even stronger scalability.
Even on relatively weaker tasks like PIQA, the performance gap with ARMs narrows as scale increases.
To account for the influence of outliers, we opted not to fit quantitative curves, avoiding potential
misinterpretation. Nevertheless, the results clearly demonstrate the scalability of LLaDA.
Considering LLaDA’s advantages on certain benchmarks, we hypothesize that this performance gain
stems from a key architectural difference: while autoregressive models optimize only left-to-right
conditional probabilities, LLaDA is trained to consider multiple conditioning directions, as detailed in
Appendix A.2, which may offer greater flexibility and lead to better generalization. This hypothesis is
motivated by LLaDA’s strong performance on reversal reasoning in Sec. 3.3 and the ablation studies
on sampling strategies in Appendix B.4.
Nie et al. [27] suggests that MDM requires 16 times more computation than ARM to achieve the
same likelihood. However, key differences make our findings more broadly applicable. In particular,
likelihood is a relatively indirect metric for downstream task performance, and diffusion optimizes a
bound of the likelihood, making it not directly comparable to ARM. Additionally, we extended the
scaling range from 1018 ∼1020 FLOPs in Nie et al. [27] to 1020 ∼1023 FLOPs in this work.
3.2
Benchmark Results
To comprehensively evaluate the in-context learning and instruction-following capabilities of LLaDA
8B, we conducted detailed comparisons with existing LLMs [6, 21, 25, 26, 32, 33] of similar scale.
Task selection and evaluation protocols followed existing studies, covering popular benchmarks in
general tasks, mathematics, code, and Chinese. Further details are provided in Appendix B.6. For a
more direct comparison, we re-evaluated representative LLMs [6, 21] in our implementation.
As shown in Tab. 1, after pretraining on 2.3T tokens, LLaDA 8B Base demonstrates remarkable
performance, surpassing LLaMA2 7B Base on nearly all tasks, and is overall competitive with
LLaMA3 8B Base. LLaDA shows advantages in math and Chinese tasks. We conjecture that the
6

Table 1: Benchmark Results of Pre-trained LLMs. ∗indicates that models are evaluated under the
same protocol, detailed in Appendix B.6. Results indicated by † and ¶ are sourced from Yang et al.
[25, 26] and Bi et al. [32] respectively. The numbers in parentheses represent the number of shots
used for in-context learning. “-” indicates unknown data.
LLaDA 8B∗
LLaMA3 8B∗
LLaMA2 7B∗
Qwen2 7B†
Qwen2.5 7B†
Mistral 7B†
Deepseek 7B¶
Model
Diffusion
AR
AR
AR
AR
AR
AR
Training tokens
2.3T
15T
2T
7T
18T
-
2T
General Tasks
MMLU
65.9 (5)
65.4 (5)
45.9 (5)
70.3 (5)
74.2 (5)
64.2 (5)
48.2 (5)
BBH
49.7 (3)
62.1 (3)
39.4 (3)
62.3 (3)
70.4 (3)
56.1 (3)
39.5 (3)
ARC-C
45.9 (0)
53.1 (0)
46.3 (0)
60.6 (25)
63.7 (25)
60.0 (25)
48.1 (0)
Hellaswag
70.5 (0)
79.1 (0)
76.0 (0)
80.7 (10)
80.2 (10)
83.3 (10)
75.4 (0)
TruthfulQA
46.1 (0)
44.0 (0)
39.0 (0)
54.2 (0)
56.4 (0)
42.2 (0)
-
WinoGrande
74.8 (5)
77.3 (5)
72.5 (5)
77.0 (5)
75.9 (5)
78.4 (5)
70.5 (0)
PIQA
73.6 (0)
80.6 (0)
79.1 (0)
-
-
-
79.2 (0)
Mathematics & Science
GSM8K
70.3 (4)
48.7 (4)
13.1 (4)
80.2 (4)
85.4 (4)
36.2 (4)
17.4 (8)
Math
31.4 (4)
16.0 (4)
4.3 (4)
43.5 (4)
49.8 (4)
10.2 (4)
6.0 (4)
GPQA
25.2 (5)
25.9 (5)
25.7 (5)
30.8 (5)
36.4 (5)
24.7 (5)
-
Code
HumanEval
35.4 (0)
34.8 (0)
12.8 (0)
51.2 (0)
57.9 (0)
29.3 (0)
26.2 (0)
HumanEval-FIM
73.8 (2)
73.3 (2)
26.9 (2)
-
-
-
-
MBPP
40.0 (4)
48.8 (4)
23.2 (4)
64.2 (0)
74.9 (0)
51.1 (0)
39.0 (3)
Chinese
CMMLU
69.9 (5)
50.7 (5)
32.5 (5)
83.9 (5)
-
-
47.2 (5)
C-Eval
70.5 (5)
51.7 (5)
34.0 (5)
83.2 (5)
-
-
45.0 (5)
Table 2: Benchmark Results of Post-trained LLMs. LLaDA only employs an SFT procedure, while
other models have extra reinforcement learning (RL) alignment. ∗indicates models are evaluated
under the same protocol, detailed in Appendix B.6. Results indicated by † and ¶ are sourced
from Yang et al. [26] and Bi et al. [32] respectively. The numbers in parentheses represent the number
of shots used for in-context learning. “-” indicates unknown data.
LLaDA 8B∗
LLaMA3 8B∗
LLaMA2 7B∗
Qwen2 7B†
Qwen2.5 7B†
Gemma2 9B†
Deepseek 7B¶
Model
Diffusion
AR
AR
AR
AR
AR
AR
Training tokens
2.3T
15T
2T
7T
18T
8T
2T
Post-training
SFT
SFT+RL
SFT+RL
SFT+RL
SFT+RL
SFT+RL
SFT+RL
Alignment pairs
4.5M
-
-
0.5M + -
1M + 0.15M
-
1.5M + -
General Tasks
MMLU
65.5 (5)
68.4 (5)
44.1 (5)
-
-
-
49.4 (0)
MMLU-pro
37.0 (0)
41.9 (0)
4.6 (0)
44.1 (5)
56.3 (5)
52.1 (5)
-
Hellaswag
74.6 (0)
75.5 (0)
51.5 (0)
-
-
-
68.5 (-)
ARC-C
88.5 (0)
82.4 (0)
57.3 (0)
-
-
-
49.4 (-)
Mathematics & Science
GSM8K
69.4 (4)
78.3 (4)
29.0 (4)
85.7 (0)
91.6 (0)
76.7 (0)
63.0 (0)
Math
31.9 (0)
29.6 (0)
3.8 (0)
52.9 (0)
75.5 (0)
44.3 (0)
15.8 (0)
GPQA
33.3 (5)
31.9 (5)
28.4 (5)
34.3 (0)
36.4 (0)
32.8 (0)
-
Code
HumanEval
49.4 (0)
59.8 (0)
16.5 (0)
79.9 (0)
84.8 (0)
68.9 (0)
48.2 (-)
MBPP
41.0 (4)
57.6 (4)
20.6 (4)
67.2 (0)
79.2 (0)
74.9 (0)
35.2 (-)
strengths stem from the same factors as its relatively weaker performance in some tasks—differences
in data quality and distribution, largely due to the closed-source situation of LLM datasets.
Notably, we have carefully ruled out the possibility of data leakage by taking GSM8K as an example.
First, as shown in Fig. 3, LLaDA outperformed ARM baselines regarding GSM8K. Moreover, the
conclusion remains on a fully unseen GSM8K-like task [34] in Appendix B.8.
Further, Tab. 2 compares the performance of LLaDA 8B Instruct with existing LLMs. SFT improved
LLaDA’s performance on most downstream tasks. A few metrics, such as MMLU, showed declines,
possibly due to the suboptimal quality of the SFT data. Overall, since we did not perform alignment
7

Table 3: Visualization of the Sampling Process and a Generated Multi-round Dialogue. In the
response of LLaDA, darker colors indicate tokens predicted in the later stages of sampling, while
lighter colors correspond to earlier predictions.
Sampling Process
User
Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour.
How many kilometers can she run in 8 hours?
LLaDA
Multi-round Dialogue Case
User
Could you please share with me what the initial two lines of the renowned poem ‘The
Road Not Taken’?
LLaDA
Certainly! The initial two lines of the renowned poem “The Road Not Taken" by Robert
Frost are: “Two roads diverged in a yellow wood, And sorry I could not travel both." These
lines set the stage for the poem, introducing the idea of a speaker facing a choice between
two paths.
User
Please help me translate into Chinese.
LLaDA
Sure! The first two lines of “The Road Not Taken" by Robert Frost can be translated into
Chinese as: “两条路分岔在黄色的树林中，遗憾我不能同时走”
User
Please translate into German.
LLaDA
Sure! The first two lines of “The Road Not Taken" by Robert Frost can be translated into
German as: “Zwei Wege trennten sich im gelben Wald, und leider konnte ich nicht beide
tun."
User
Please also write a poem about life choices. I require 5 sentences, and the first word of
each sentence starts with C as the first letter.
LLaDA
Certainly! Here’s a poem about life choices, with each sentence starting with the letter “C":
Choices cascade through life’s journey, Creating connections with each decision, Careful
consideration leads to better paths, Courageous choices shape our destiny, Choosing wisely
guides us through the years.
with reinforcement learning (RL), our results are slightly behind LLaMA3 8B Instruct, though the
gaps in many metrics remain small. Notably, even with only SFT, LLaDA demonstrates impressive
instruction-following abilities, as detailed in Sec. 3.4. We leave RL-based alignment for future work.
All results in Sec. 3 are based on pure diffusion methods, as they achieve better overall performance
than approaches incorporating autoregressive components. Specifically, we use Eq. (6) for conditional
likelihood estimation and apply low-confidence remasking for sampling. For LLaDA 8B Instruct,
block diffusion style sampling performs better on GSM8K and Math, with scores of 78.6 and 42.2,
compared to 69.4 and 31.9 in Tab. 2. This gain is due to extensive |EOS| token padding in the SFT
data, causing early termination in low-confidence remasking. Please refer to Appendix B.4 for details.
Overall, despite the lack of data transparency, we have made every effort to adopt standardized
procedures and introduce diverse tasks, we believe they sufficiently demonstrate the extraordinary
capabilities of LLaDA, which is the only competitive non-autoregressive model to our knowledge.
3.3
Reversal Reasoning and Analyses
To quantify the reversal reasoning [15] ability of models, we follow the protocol established in Allen-
Zhu and Li [35]. Specifically, we construct a dataset of 496 famous Chinese poem sentence pairs.
Given a sentence from a poem, models are tasked with generating the subsequent line (forward) or the
preceding line (reversal) without additional fine-tuning. Examples can be found in Section B.9. This
setting provides a straightforward and more realistic evaluation compared to previous studies [27, 36].
As shown in Tab. 4, LLaDA effectively addresses the reversal curse [15], demonstrating consistent
zero-shot performance across both forward and reversal tasks. In contrast, both Qwen 2.5 and GPT-4o
exhibit a significant gap between the two. The results on forward generation confirm that both ARMs
are strong, benefiting from significantly larger datasets and greater computational resources than
LLaDA. However, LLaDA outperforms both by a large margin in the reversal task.
8

Table 4: Comparison on the Poem Comple-
tion task.
Forward
Reversal
GPT-4o (2024-08-06)
82.7
34.3
Qwen2.5-7B Instruct
75.9
38.0
LLaDA-8B Instruct
51.8
45.6
We did not design anything special for reversal
tasks. Intuitively, LLaDA treats tokens uniformly
without inductive bias, leading to balanced perfor-
mance. See Appendix A.2 for details.
We also analyze the effect of different sampling
strategies for LLaDA, including autoregressive
sampling, block diffusion [31] sampling, and pure
diffusion sampling, showing that pure diffusion
sampling achieves the best overall performance, as
detailed in Appendix B.4.
In addition, we examine LLaDA’s sampling speed and memory consumption, showing that it enables
a flexible trade-off between generation quality and speed. See Appendix B.7 for more details.
Classifier-free guidance (CFG) [37, 27] is a widely used technique in diffusion models to improve
generation quality. To ensure a fair comparison with ARMs, we do not apply CFG to LLaDA in the
main text. However, we show that LLaDA is compatible with CFG and consistently benefits from its
application. See Appendix B.3 for more details.
3.4
Case Studies
We present samples generated by LLaDA 8B Instruct in Tab. 3, showcasing its instruction-following
capabilities. First, the table illustrates LLaDA’s ability to generate coherent, fluent, and extended
text in a non-autoregressive manner. Second, it highlights the model’s multi-turn dialogue capability,
effectively retaining conversation history and producing contextually appropriate responses across
multiple languages. Such chat capabilities of LLaDA are impressive, as it departs from conventional
ARMs for the first time, to the best of our knowledge. See more case studies in Appendix B.10.
4
Related Work
Diffusion models [38–40] have achieved remarkable success in visual domains but remain unverified
for large-scale (e.g., models trained with over 1023 FLOPs) language modeling, despite growing
interest and extensive research efforts.
A simple approach is to continuousize text data and apply continuous diffusion models directly [41–
51]. Alternatively, some methods model continuous parameters of discrete distributions instead [52–
56]. However, scalability remains a significant challenge for these approaches. For instance, a 1B
model may require 64 times the compute of an ARM to achieve comparable performance [57].
Another approach replaces continuous diffusion with discrete processes featuring new forward and
reverse dynamics, leading to numerous variants [58–71]. The original diffusion model paper [38]
introduced both continuous-state and discrete-state transition kernels under a unified diffusion
framework. Austin et al. [16] was among the pioneering works that introduced discrete diffusion
models into language modeling, demonstrating the feasibility of this approach. Lou et al. [17] showed
that masked diffusion, as a special case of discrete diffusion, achieves perplexity comparable to
or surpassing ARMs at GPT-2 scale. Shi et al. [18], Sahoo et al. [19], Ou et al. [20] established
fundamental theoretical results, which motivated our model design, training, and inference (see
Appendix A for details). Nie et al. [27] introduced the scaling laws for MDMs in language modeling
and explored how MDMs can be leveraged for language tasks such as question answering at the
GPT-2 scale. Gong et al. [72] demonstrated the potential of fine-tuning an ARM within the MDM
framework. However, the improvements observed by Gong et al. [72] are limited to specific metrics,
and their approach does not address the performance achievable through pure diffusion-based training.
Concurrent work [73] demonstrates the potential of diffusion language models in code generation
and highlights their advantages in inference efficiency. Nonetheless, as it is a closed-source product,
specific details such as training procedures and sampling methods remain unknown.
In comparison, this study scales MDM to an unprecedented size of 8B parameters from scratch,
achieving performance comparable to leading LLMs such as LLaMA 3.
Additionally, a parallel line of work on image generation [23, 74, 75] aligns well with the application
of MDMs to text data. Moreover, MDMs have also shown promise in other domains such as protein
9

generation [76, 77], where they have achieved promising results. Notably, a series of studies [31, 78–
87] have explored techniques such as architectural optimization, distillation, and sampling algorithm
design to accelerate MDMs sampling.
5
Conclusion and Discussion
We introduce LLaDA, a diffusion language model trained from scratch with an unprecedented scale
of 8B parameters. LLaDA demonstrates strong capabilities in scalability, in-context learning, and
instruction-following, achieving performance comparable to strong LLMs such as LLaMA3. In
addition, LLaDA offers unique advantages, such as bidirectional modeling and enhanced robustness,
effectively addressing the relevant limitations of existing LLMs. Our findings show the promise
of diffusion models for language modeling at scale and challenge the common assumption that
these essential capabilities are inherently tied to ARMs. These results represent a new paradigm for
language modeling and uncover novel insights, demonstrating a high degree of scientific innovation.
Limitations. While promising, the full potential of diffusion models remains to be fully explored.
Several limitations of this work present significant opportunities for future research. The generation
length is a user-specified hyperparameter. Although LLaDA is insensitive to this hyperparameter
as detailed in Appendix B.5, we believe that adopting an adaptive generation length would offer
a more efficient solution. Due to computational constraints, direct comparisons between LLaDA
and ARMs—such as training on identical datasets—were restricted to a computational budget of
less than 1023 FLOPs. To allocate resources for training the largest possible LLaDA model and
showcasing its potential, we were unable to scale the ARM baseline to the same extent. Moreover,
no specialized attention mechanisms or position embeddings were designed for LLaDA, nor were
any system-level architectural optimizations such as KV cache applied. On the inference side, more
efficient and controllable [37, 88, 89] sampling algorithms remain preliminary. Furthermore, LLaDA
has yet to undergo alignment with reinforcement learning [90, 91], which is crucial for improving its
performance and alignment with human intent.
Looking ahead, both the model scale and the amount of training data for LLaDA remain smaller than
those of leading ARM counterparts [6, 26, 92–95], highlighting the need for further scaling to fully
evaluate its capabilities. In addition, LLaDA’s ability to process multi-modal data remains unexplored.
Its impact on prompt tuning techniques [96] and integration into agent-based systems [97, 98] is still
not fully understood. Finally, a systematic investigation into post-training for LLaDA (e.g., O1-like
systems [99, 100]) is needed to further unlock the potential of diffusion language models.
Acknowledgements
This work was supported by the National Natural Science Foundation of China (No. 92470118);
Beijing Natural Science Foundation (No. L247030); Beijing Nova Program (No. 20220484044); and
Ant Group Research Fund.
References
[1] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian
Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models.
arXiv preprint arXiv:2303.18223, 2023.
[2] Alec Radford. Improving language understanding by generative pre-training, 2018.
[3] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al.
Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
[4] Tom B Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165,
2020.
[5] OpenAI. ChatGPT: Optimizing Language Models for Dialogue. OpenAI blog, November
2022. URL https://openai.com/blog/chatgpt/.
10

[6] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783, 2024.
[7] Ashish Vaswani. Attention is all you need. arXiv preprint arXiv:1706.03762, 2017.
[8] Ronald A Fisher. On the mathematical foundations of theoretical statistics. Philosophical
transactions of the Royal Society of London. Series A, containing papers of a mathematical or
physical character, 222(594-604):309–368, 1922.
[9] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All
are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 22669–22679, 2023.
[10] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings
of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
[11] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe
Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video
generation models as world simulators. 2024. URL https://openai.com/research/
video-generation-models-as-world-simulators.
[12] Gregoire Deletang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein,
Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent
Orseau, et al. Language modeling is compression. In The Twelfth International Conference on
Learning Representations.
[13] Yuzhen Huang, Jinghan Zhang, Zifei Shan, and Junxian He. Compression represents intelli-
gence linearly. arXiv preprint arXiv:2404.09937, 2024.
[14] Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical
journal, 27(3):379–423, 1948.
[15] Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz
Korbak, and Owain Evans. The reversal curse: Llms trained on" a is b" fail to learn" b is a".
arXiv preprint arXiv:2309.12288, 2023.
[16] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg.
Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information
Processing Systems, 34:17981–17993, 2021.
[17] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language