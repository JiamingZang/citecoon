# LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures

> 2025 · id: arxiv:2509.14252 · arXiv: 2509.14252 · pdf: https://arxiv.org/pdf/2509.14252 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Preprint
LLM-JEPA: LARGE LANGUAGE MODELS MEET JOINT
EMBEDDING PREDICTIVE ARCHITECTURES
Hai Huang
Atlassian
hhuang3@atlassian.com
Yann LeCun
NYU
yann.lecun@nyu.edu
Randall Balestriero
Brown University
rbalestr@brown.edu
ABSTRACT
Large Language Model (LLM) pretraining, finetuning, and evaluation rely on
input-space reconstruction and generative capabilities. Yet, it has been observed
in vision that embedding-space training objectives, e.g., with Joint Embedding
Predictive Architectures (JEPAs), are far superior to their input-space counterpart.
That mismatch in how training is achieved between language and vision opens
up a natural question: can language training methods learn a few tricks from
the vision ones? The lack of JEPA-style LLM is a testimony of the challenge in
designing such objectives for language. In this work, we propose a first step in
that direction where we develop LLM-JEPA, a JEPA based solution for LLMs
applicable both to finetuning and pretraining. Thus far, LLM-JEPA is able to
outperform the standard LLM training objectives by a significant margin across
models, all while being robust to overfiting. Those findings are observed across
numerous datasets (NL-RX, GSM8K, Spider, RottenTomatoes) and various models
from the Llama3, OpenELM, Gemma2 and Olmo families. Code: https://
github.com/rbalestr-lab/llm-jepa.
Llama3
SYNTH
gemma2
SYNTH
OpenELM
SYNTH
Llama3
Spider
Llama3
GSM8K
Llama3
TURK
0
10
20
30
40
50
60
70
Accuracy (%)
Baseline
LLM-JEPA (Ours)
1
2
3
4
5
6
Epoch
0
10
20
30
40
50
60
70
80
Accuracy (%)
37.0
51.6
11.5
20.2
51.3
66.6
19.4
27.2
56.0
70.4
22.5
30.9
55.2
70.9
22.5
31.6
51.5
71.8
21.6
31.8
SYNTH
TURK
Figure 1: LLM-JEPA produces strong fine-tuned models across datasets and models.
1
INTRODUCTION
The research landscape around representation learning has been increasingly divided into two camps:
(i) generative or reconstruction-based methods Brown et al. (2020b); Chowdhery et al. (2023); He et al.
(2022); LeCun (2022), and (ii) reconstruction-free Joint Embedding Predictive Architectures (JEPAs)
Assran et al. (2023); Baevski et al. (2022); Bardes et al. (2024). While the former is self-explanatory,
the latter learns a representation by ensuring that different views, e.g., pictures of a same building
at different time of day, can be predicted from each other, all while preventing a collapse of the
embeddings. By moving away from input-space objectives, JEPAs training benefits from less biases
Littwin et al. (2024), at the cost of potential dimensional collapse of their representation Jing et al.
(2021); Kenneweg et al. (2025). That divide has been well studied in vision, where it was found
that JEPAs offer multiple provable benefits when it comes to knowledge discovery for perception
tasks. In the realm of Natural Language Processing however, reconstruction-based methods remain
1
arXiv:2509.14252v2  [cs.CL]  7 Oct 2025

Preprint
Natural Language to Regular Expression
Natural Language to SQL
Figure 2: Left: JEPA applied to NLP tasks that has Text and Code, where Text and Code are naturally
two views of the same thing. Right: (top): An illustration of the NL-RX-SYNTH dataset, where each sample
consists of a description of the regular expression in natural language (Text) and the regular expression itself
(Code). (bottom): The Spider dataset, where Text is the database ID and description of the SQL query and
Code is the SQL query itself.
predominant. In fact, today’s Large Language Models are mostly judged from their ability to generate
samples and answers in input space in text form–making it challenging to leverage JEPA objectives.
Yet, LLMs’ task also involve perception and reasoning where JEPA is known to be preferable. It
thus seems crucial to adapt JEPA solutions to LLMs in the hope to showcase the same benefits as
witnessed in vision. This first step is exactly what we present in this study. We propose to improve
the representation quality of LLMs by leveraging a novel objective combining both the original
reconstruction based loss–with an additional JEPA objective. To do so, we focus first on tasks and
datasets that are inherently suited for JEPA objectives: the ones providing multiple views of the same
underlying knowledge. One typical example is a git issue and the corresponding code diff (fig. 2)
Jimenez et al. (2024). The two samples are two views–one being plain English and one being in
code–of the same underlying functionality. Let’s use that particular example to highlight our core
contribution:
Viewing the (text,code) pairs as views of the same underlying knowledge enables JEPA objec-
tives to be utilized with LLMs, complementing the standard text →code generative task.
We strongly emphasize that being able to obtain non-trivial views, such as described above, is crucial
to the success of JEPA objectives. While we restrict ourselves to datasets offering those non-trivial
views, developing a mechanism akin to data-augmentation in vision would enable JEPA objectives to
be used on any dataset. Nonetheless, we believe that our proposed solution–coined LLM-JEPA–and
empirical study will serve as a first step towards more JEPA-centric LLM pretraining and finetuning.
We summarize our contributions below:
• Novel JEPA-based training objective: We present the first JEPA-based training objective for
LLMs operating in embedding space and with different views–perfectly following vision-based
JEPAs without sacrificing the generative capabilities of LLMs
• Improved SOTA: We empirically validate our formulation in various finetuning settings, where
we obtain improvements over standard LLM finetuning solutions. We also explore pretraining
scenarios showing encouraging results of LLM-JEPA
• Extensive empirical validation: on various model family (llama, gemma, apple/openelm, al-
lenai/olmo), dataset (NL-RX, GSM8K, Spider, RottenTomatoes), and size.
2
BACKGROUND
Large Language Models. Contemporary LLMs are mostly built from the same core principles:
stacking numerous layers of nonlinear operations and skip-connections–known as Transformers.
While subtleties may differ, e.g., about positional embeddings, initialization, normalization, the main
driver of performance remains the availability of high quality dataset during the pretraining stage. The
training objective in itself has also been standardize throughout methods: autoregressive token-space
reconstruction. Let’s first denote by LLLM the typical LLM objective used for the specific task and
2

Preprint
dataset at hand. In most cases, this will be a cross-entropy loss between the predicted tokens and the
ground-truth token to reconstruction. We note that our LLM-JEPA construction is agnostic of LLLM
hence making our method general to numerous scenarios.
LLLM(Text1:L−1, TextL) = XEnt (Classifier (Enc(Text1:L−1)) , TextL) ,
(1)
where Classifier predicts the logits of the next token TextL given the past tokens Text1:L−1. Com-
putation of eq. (1) is done at once over L through causal autoregression. Different stages and tasks
may vary the input and output of the loss.
Alternative training objectives. Next token prediction is the prevalent pretraining solution for
today’s latest LLMs. There exists a few alternatives, e.g., SimCSE leverages a contrastive loss in
the latent space by treating different dropout-induced views of the same sentence as positive pairs,
resulting in state-of-the-art sentence embeddings quality Gao et al. (2021). In a similar spirit, Wang
et al. (2022) uses weak supervision from text pairs to learn a joint embedding architecture. An
alternative relying on pretrained models instead of the supervised text pairs was explore in Ni et al.
(2021). While those approaches are powerful, they are all concern with producing text embeddings
without generative capabilities which greatly limits the applicability of those models since numerous
evaluations and use-cases require generation–which is core to our proposed method. Another solution
employs BERT pretraining coupled with a latent space semantic loss to ensure that representations of
semantically similar sentences are nearby in embedding space. This additional term led to improved
performance on semantic tasks compared to masked language modeling alone Reimers and Gurevych
(2019)–yet again by building atop BERT this solution prevents generative evaluation and use.
3
JEPA-LLM: IMPROVING LLMS’ REASONING AND GENERATIVE
CAPABILITIES
We propose the LLM-JEPA loss in section 3.1 along with extensive empirical validations in section 4
demonstrating clear finetuning and pretraining benefits.
3.1
THE LLM-JEPA OBJECTIVE
Throughout this section, we will use Text and Code as concrete examples of having different views
of the same underlying knowledge. It should be clear to the reader that our proposed LLM-JEPA
objective handles different types of views similarly.
The construction of our LLM-JEPA objective relies on two principles. First, we must preserve the
generative capabilities of LLMs and we therefore start with the LLLM from eq. (1). Second, we aim
to improve the abstraction capabilities of LLMs using the joint embedding prediction task. On top of
LLLM, we then propose to add the well-established JEPA objective leading to the complete loss L
defined as
LLLM−JEPA =
L
X
ℓ=2
LLLM(Text1:ℓ−1, Textℓ)
|
{z
}
generative capabilities (LLM)
+λ × d(Pred(Enc(Text)), Enc(Code))
|
{z
}
abstraction capabilities (JEPA)
,
(2)
where λ ≥0 is an hyperparameter balancing the contribution of the two terms, Pred and Enc are the
predictor and encoder networks respectively, and d is a metric of choice, e.g., the ℓ2 distance. Let’s
now precisely describe each of those components.
The encoder. We use the hidden_state of the last token from the last layer as the embedding
of an input sequence–as commonly done for LLM probing. We pack both Text and Code into
a single context window, applying an attention mask to ensure they do not reference each other.
Implementation-wise, most HuggingFace transformers support an additive attention mask,
where setting entry (i, j) = −∞prevents token j from attending to token i (for i < j). Using this
mechanism, we implement LLM-JEPA with only one additional forward pass. This introduces extra
cost during training, but not during inference—see section 6 for further discussion.
The metric. When it comes comparing embeddings, it is now widely accepted in vision to leverage
the cosine similarity. We thus propose to do the same for LLM-JEPA.
3

Preprint
The predictor. We leverage the auto-regressive nature of LLM and their internal self-attention
to define a tied-weights predictor. By introducing a special token [PRED] at the end of a given
input, we allow for further nonlinear processing of the input hereby producing Pred(·) at the final
embedding of the last layer. By reusing the internal weights of the LLM for the prediction task,
we greatly reduce the training overhead and architectural design choices. Practically, we append
k ∈{0, . . . , K} predictor tokens to an input prompt and use the embedding of the last predictor
token to be Pred(Enc(·)). When k = 0, the predictor is trivial, i.e., Pred(x) = x.
3.2
IMPLEMENTATION WITH CUSTOM ATTENTION MASK
The most important challenge in implementing our proposed LLM-JEPA objective lies in obtained
the embeddings of the different views, e.g., Text and Code in eq. (2). A priori, it is not possible to
get them in one forward pass because the current self-attention–albeit being causal. Even if we were
to concatenate the two views which is already done for the next token prediction part of the loss, it
would make the representation of the second view rely on the first view. As a result, we propose the
following custom self-attention mask that is causal per block, with number of blocks set to 2:
1
def additive_mask(k: int):
2
"""Returns a k by k triangle mask."""
3
mask = torch.zeros((k, k), dtype=torch.float32)
4
mask[torch.triu(torch.ones(k, k), diagonal=1) == 1] = -torch.inf
5
return mask
6
7
# Initialize all elemets to -inf.
8
mask = torch.full((batch_size * 2, 1, seq_length, seq_length), -torch.inf
).to(device)
9
# Text starts from t_start and is of size t_size, and Code starts
10
# from c_start and is of size c_size. Set for the i-th batch.
11
mask[i, :, t_start: t_start + t_size, t_start: t_start + t_size] =
additive_mask(t_size)
12
mask[i, :, c_start: c_start + c_size, c_start: c_start + c_size] =
additive_mask(c_size)
By leveraging the above implementation, we are able to obtain the LLM-JEPA loss value in two
forward passes instead of three. While this is still a substantial slowdown, we will explore later in the
manuscript a dropout version where the JEPA term is only applied some % of the mini batch–the end
result will be that are comparable FLOPS the proposed LLM-JEPA is still able to outperform the
baseline.
Relation to Previous Work. Because loss functions such as LLLM (input space reconstruction since
tokens are lossless compression of the original prompts) have been shown to be sub-optimal in vision,
a few LLM variations have started to employ embedding space regularizers and training objectives
Barrault et al. (2024); Wang and Sun (2025). Current solution however rely on intricate structural
constraints of the embedding space, e.g., hierarchical organization and cluster, and thus fall out of the
JEPA scope. We also note that our interpretation of views when it comes to LLM datasets, e.g., (text
issue, code diff), is something that has been leveraged as part of the LLM finetuning solutions–by
learning to generate one from the other–without a JEPA-style loss. This includes natural language
to regular expression translation (Locascio et al., 2016; Ye et al., 2020; Zhong et al., 2018), natural
language to SQL parsing (Guo et al., 2019; Iyer et al., 2017; Li et al., 2023; Wang et al., 2019; Yu
et al., 2018) and the more recent issue descriptions to code diffs (Cabrera Lozoya et al., 2021; Hoang
et al., 2020; Tian et al., 2020; Zhou et al., 2023). More intricate examples involve text-based problem
solving and their counterpart program induction (Amini et al., 2019; Cobbe et al., 2021; Hendrycks
et al., 2021; Ling et al., 2017).
3.3
A GOOD NEXT TOKEN PREDICTOR IS NOT A GOOD JEPA
Before validating the proposed LLM-JEPA to real task and models, we ask ourselves a simple
question. Is it really necessary to have an additional JEPA term? Is that term already implicitly
minimized by the original next token prediction objective?
4

Preprint
0
20
40
60
80
100
Index
101
102
103
Singular Value
Base Model
Regular Fine-tuning
LLM-JEPA k=1 (ours)
LLM-JEPA k=0 (ours)
Figure 3: left: The top 100 singular values of Enc(Text) −Enc(Code). The curves of LLM-JEPA (ours) are
a few magnitudes lower than that of base model and regular fine-tuning, meaning the mapping from Text to
Code are confined within a narrow subspace, fostering the nice structure we see in Figure 4. Right: Losses
in fine-tuning with LLLM loss (LLLM) and LLLM−JEPA loss (LLLM−JEPA, our method). We measure both
the cross-entropy loss for next token prediction (LossLLM, LLLM in chart) and JEPA prediction loss (D(·, ·),
pred in chart), although the latter does not contribute in the baseline case. The accuracy is 51.95% for LLLM
and 71.10% for LLLM−JEPA. Since LLLM and LLLM−JEPA share similar LLLM loss, the LLLM loss cannot
explain the gap between the accuracy. pred stays a constant in LLLM, while is minimized in LLLM−JEPA, hence
pred should be the main reason behind the accuracy gap.
To answer that, we compare two controlled settings. We will be using Llama-3.2-1B-Instruct and
NL-RX-SYNTH and have two training setup. In the first, we do the usual next-token prediction loss
but monitor the JEPA objective, i.e., no gradient comes from the JEPA loss. In the second, we will
use the proposed LLM-JEPA for gradient computation. We also monitor the prediction loss in both
cases. We obtain the following finding in fig. 3: minimizing LLLM does not implicitly minimize
LJEPA–indicating that it is required to add that term during training. This can be seen by comparing
the red and green lines. A natural follow-up question is are we trading off next token prediction
for JEPA?. In the same fig. 3 we obtain that the next token prediction capability is not hindered by
the presence of the JEPA term (compare the blue and yellow lines that are overlapping). This is a
very important observation echoing what was empirically observed in Balestriero and LeCun (2024)
where it was observed that autoencoders in image space could become much stronger classifiers
without sacrificing the reconstruction and generative objective. All in all we obtain that employed
LLM-JEPA only brings additional structure of the LLM latent space without altering its generative
capabilities. As we will see in the follows sections, we indeed obtain much better performances that
the baseline–in the generative evaluation setup.
4
EMPIRICAL VALIDATION: LLM-JEPAS OUTPERFORM LLMS
In this section, we first validate the proposed LLM-JEPA across four model families and four datasets
with natural two-view structures (section 4.1). The consistent improvements observed across the
board motivate us to further examine the internal representations of LLM-JEPA to better understand
the source of its strength (section 4.2). We conclude this section with a rigorous ablation study on
key design choices (section 4.3).
We adopt a universal protocol across all experiments by using five fixed random seeds,
{82, 23, 37, 84, 4}, for training. Each experiment is repeated five times, once with each seed. This
setup enables us to assess the stability of LLM-JEPA and to conduct paired one-tailed t-tests for
statistical significance.
4.1
FINE-TUNING AND PRETRAINING STRONGER GENERATIVE MODELS VIA JEPA
LLM-JEPA Improves Finetuning. We run experiments across multiple pretrained LLMs (Llama-
3.2-1B-Instruct (Grattafiori et al., 2024), gemma-2-2b-it (Team et al., 2024), OpenELM-1_1B-Instruct
(Mehta et al., 2024), and OLMo-2-0425-1B-Instruct (OLMo et al., 2024)) with various datasets
5

Preprint
(NL-RX-SYNTH, NL-RX-TURK (Locascio et al., 2016), GSM8K (Cobbe et al., 2021), Spider (Yu
et al., 2018)).
To demonstrate that LLM-JEPA improves from the strongest possible baseline, we first search for
the best learning rate lr ∈{1e −5, 2e −5, 4e −5, 8e −5} by selecting the value that yields
the highest accuracy of LLLM after 4 epochs for a given (model,dataset) pair. Then we tune the
hyperparameter specific to LLLM−JEPA, k and λ in a two dimensional grid defined by (k, λ) ∈
{0, 1, 2, 3, 4} × {0.5, 1, 2, 4} (fig. 7 and table 10 in the Appendix). For both NL-RX-SYNTH and
NL-RX-TURK, accuracy is exact match of the generated regular expression; for GSM8K, accuracy
is exact match of the final result; and for Spider, accuracy is exact match of the execution result of the
generated query.
We provide results demonstrating that LLM-JEPA improves performances across
• four model families, see fig. 1 left and table 12 in the Appendix.
• four datasets, see table 13 in the Appendix.
• 6 training epochs (fig. 1 right). We also observe that LLM-JEPA resists overfitting, whereas
standard fine-tuning does not.
• four different sizes: 1B, 3B, 7B, and 8B, see table 15 in the Appendix
Examples of inputs, targets, model predictions, and error analyses are provided in table 1 (more
examples can be found in table 11 in the Appendix). For LoRA fine-tuning, the performance gains
of LLM-JEPA hold consistently across different LoRA ranks (table 8 in the Appendix). We also
observe the same resistance to overfitting in the LoRA setting (fig. 8 in the Appendix).
Table 1: Regular expressions generated by Llama-3.2-1B-Instruct after fine-tuning with LLLM loss and
LLLM−JEPA loss (ours). Color code: wrong , extra , missing
Ground Truth
LLLM
LLLM−JEPA (ours)
lines not having the string ’dog’ followed by a number, 3 or more times
((dog.*[0-9].*)3,)
((dog.*[0-9].*)3,)
((dog.*[0-9].*){3,})
lines containing ending with a vowel, zero or more times
.*(.*)(([AEIOUaeiou])*).*
(.*)(([AEIOUaeiou])*) *
( .* ) (.*) { ([AEIOUaeiou])* }
lines with a number or a character before a vowel
(([0-9])|(.)).*([AEIOUaeiou]).*
(([0-9])|(.)).*([AEIOUaeiou]).* .*
(([0-9])|(.)).*([AEIOUaeiou]).*
lines with words with the string ’dog’, a letter, and a number
((([0-9])&(dog))|([A-Za-z]))*
((([0-9])&(dog))|([A-Za-z]))*
(( [0-9])&(dog))|( ( [A-Za-z]) * )
LLM-JEPA Improves Pretraining. A natural extension of our fine-tuning results is to examine
pretraining. We pretrain Llama-3.2-1B-Instruct from randomly initialized weights on the NL-RX-
SYNTH dataset. Owing to the limited size of the dataset, the pretrained model fails to reliably learn
how to terminate generation. To address this, we adjust the evaluation criterion, deeming a generated
solution valid as long as it begins with the ground-truth sequence. We obtain that LLM-JEPA also
improves the quality of the learned representation, as shown in table 2.
Table 2:
Pretraining accuracy on dataset NL-RX-SYNTH by Next Token Prediction (LLLM) loss vs.
LLLM−JEPA loss (our method). We inherit the best configuration from fine-tuning. Each case runs five
times. Average accuracy and standard deviation are reported. We also report p-value of paired, single-tailed
t-Test.
Model
Method
Accuracy (%) ↑
p-value ↓
Config
Llama-3.2-1B-Instruct
LLLM
54.38 ± 1.70
2.94e −4
lr = 8e −5
LLLM−JEPA (ours)
60.59 ± 1.01
λ = 2, k = 3, same lr
We then conduct a more advanced pretraining experiment on dataset cestwc/paraphrase containing
groups of 5 paraphrases. We leverage the five paraphrases to construct the JEPA loss by having the
6

Preprint
Text
Code
(a) Baseline: Fine-tuned by NTP loss
Text
Code
(b) LLM-JEPA (Ours) k = 0
Figure 4: t-SNE plot of Text and Code representations in (a) Baseline that is fine-tuned with NTP loss, (b)
LLM-JEPA (ours) with k = 0. Clearly LLM-JEPA (ours) induced nice structure on the representations while
fine-tuning with NTP loss disrupted the structure in the base model. A full version of this figure is in section A.2.
i-th version of a paraphrase predict the (i + 1)-th version. The goal is to encourage the JEPA loss to
tie the embeddings of all versions into a compact subspace, providing a geometric foundation to align
their representations. We pretrain the model for four epochs and then evaluate it on Rotten Tomatoes
and Yelp after one epoch of fine-tuning. Although there is no direct link between the pretraining and
evaluation datasets, we show that LLM-JEPA pretraining yields statistically significant improvements
in downstream performance (see table 9 in the Appendix). Note that fine-tuning does not employ the
JEPA loss—highlighting that the benefits arise specifically from the pretraining stage.
For Rotten Tomatoes and Yelp, we fine-tune the model to generate discrete sentiment labels: Good
and Bad for Rotten Tomatoes, and Very Good, Good, Mediocre, Bad, and Very Bad for
Yelp. A prediction is deemed correct if the generated output begins with the ground-truth label.
This evaluation approach—mapping free-form generation to categorical labels and applying prefix
matching—follows common practice in prior work on text classification with generative models (Mc-
Cann et al., 2018; Raffel et al., 2020; Wei et al., 2022).
Lastly, we provide in table 7 (in the Appendix) generated samples demonstrating that JEPA pretraining
does maintain the generative capabilities of the model when prompted with the first few tokens in the
cestwc/paraphrase dataset.
4.2
STRUCTURED REPRESENTATIONS INDUCED BY LLM-JEPA
We also examine the representation space to better understand how LLM-JEPA regularizes learned fea-
tures. Specifically, we plot t-SNE embeddings for both Text and Code across three settings: the base
model, a model fine-tuned with LLLM, and a model fine-tuned with LLLM-JEPA. As shown in fig. 4,
clear structure emerges after fine-tuning with LLLM-JEPA. We hypothesize that LLLM-JEPA enforces
structure in the representation space by constraining the mapping from Enc(Text) to Enc(Code)
within a narrow subspace. If this is the case, the SVD decomposition of Enc(Text) −Enc(Code)
should yield significantly smaller singular values, which is confirmed in fig. 3. Furthermore, we
hypothesize that the mapping is approximately linear. To test this, we compute the least-squares
regression error, and table 14 in the Appendix supports this hypothesis. Together, these results sug-
gest that LLM-JEPA promotes a near-linear transformation between Text and Code representations,
which may underlie its accuracy improvements.
4.3
ABLATION STUDY ON DESIGN CHOICES
In this section, we examine several alternative design choices. Specifically, we compare the use
of cosine similarity against ℓ2-norm and mean squared error (MSE); appending a [PRED] token to
the end of Text versus prepending it; and using Text to predict Code versus the reverse direction
(Code →Text). As shown in table 3, none of these alternatives perform as well as LLM-JEPA,
although some outperform standard fine-tuning.
7

Preprint
Table 3: Fine-tuning accuracy on NL-RX-SYNTH under different design choices. Reported are average
accuracy and standard deviation across runs. Learning rate lr = 2e−5, λ = 1.0, and k = 1.
Accuracy (%) ↑
Baseline
LLM-JEPA
ℓ2-norm
MSE
Prepend
Code →Text
InfoNCE loss
57.29 ± 5.32
71.46 ± 1.34
2.22 ± 0.07
70.64 ± 2.05
68.07 ± 2.57
65.70 ± 2.63
34.40 ± 6.10
Additionally, we replace our cosine similarity loss with the InfoNCE loss van den Oord et al. (2018),
which results in lower accuracy compared to the baseline. Moreover, its standard deviation is
substantially higher than that of other alternatives. We use the commonly adopted temperature of
τ = 0.07 for the InfoNCE loss (Chen et al., 2020; Radford et al., 2021).
5
TOWARDS FASTER AND MORE GENERAL LLM-JEPAS
The structured representations induced by LLM-JEPA have the potential to provide universal benefits
across diverse LLM applications. In this section, we explore its limits by evaluating datasets without
natural two-view structures and models with richer capabilities (section 5.1). The promising results
further motivate us to investigate methods for accelerating LLM-JEPA (section 5.2), as computational
overhead remains a key obstacle to broad adoption.
5.1
BEYOND CODE: APPLYING LLM-JEPA TO Q&A DATASETS AND REASONING MODELS
We evaluate Llama-3.2-1B-Instruct on two QA benchmarks: NQ-Open (Lee et al., 2019a) and
HellaSwag (Zellers et al., 2019). Our results show that LLM-JEPA achieves statistically significant
improvements on both datasets, demonstrating its capability beyond the canonical setup where Text
and Code are treated as two complementary views of the same object.
For NQ-Open, we regard Text as the question and Code as the answer span, typically consisting of
only a few tokens, which contrasts with the more balanced sequence lengths found in other datasets.
HellaSwag, by contrast, is a context-completion multiple-choice task. Rather than defining Code
as the answer label (a single token from A, B, C, D), we instead let Text denote the context and
Code represent the correct continuation. This formulation differs from prior setups in two important
ways: (i) Both Text and Code are now integral components of the question, and (ii) The relationship
between context and completion is more diverse than the near-equivalence seen in NL→Regex or
NL→SQL mappings.
Despite these differences, LLM-JEPA consistently improves accuracy on both benchmarks table 4.
For NQ-Open, generated answers may include additional syntactic or supporting tokens. Following
prior work, we deem a prediction correct if any ground-truth answer appears as a substring of the
generated output (Lee et al., 2019b; Guu et al., 2020; Izacard and Grave, 2021). For HellaSwag, we
compute the relative probabilities of the four candidate options A, B, C, D and select the answer with
the highest probability, consistent with standard practice in multiple-choice language understanding
benchmarks (Zellers et al., 2019; Brown et al., 2020a; OpenAI, 2023).
An additional observation is that performance continues to improve as we scale λ up to 1024, without
encountering a plateau. While table 10 in the Appendix suggests that extreme values of λ can degrade
accuracy, in certain cases it remains beneficial to push λ further, yielding extra gains.
We then evaluate two strong reasoning models—Qwen3-1.7B (Yang et al., 2025) and DeepSeek-
R1-Distill-Qwen-1.5B (DeepSeek-AI et al., 2025)—on GSM8K. As shown in table 5, both models
achieve statistically significant improvements when augmented with LLM-JEPA. These results offer
promising evidence that LLM-JEPA extends its benefits to large reasoning models (LRMs).
5.2
FASTER LLM-JEPAS VIA LOSS DROPOUT
To further reduce compute, we introduce random JEPA-loss dropout (LD). During training or
fine-tuning, we randomly drop the JEPA loss at a specified LD rate. Loss dropout is applied at the
8

Preprint
Table 4:
Fine-tuning accuracy of Llama-3.2-1B-Instruct with Next Token Prediction (LLLM) loss vs.
LLLM−JEPA loss (our method). Each case runs five times. Average accuracy and standard deviation are
reported. We also report p-value of paired, single-tailed t-Test.
Dataset
Method
Accuracy (%) ↑
p-value ↓
Config
NQ-Open
LLLM
20.12 ± 0.41
2.44e −3
lr = 2e −5
LLLM−JEPA (ours)
21.59 ± 0.40
λ = 1024, k = 0, same lr
HellaSwag
LLLM
69.40 ± 0.99
0.0136
lr = 4e −5
LLLM−JEPA (ours)
70.51 ± 1.20
λ = 1, k = 3, same lr
Table 5: Fine-tuning accuracy of GSM8K with Next Token Prediction (LLLM) loss vs. LLLM−JEPA loss (our
method). Each case runs five times. Average accuracy and standard deviation are reported. We also report
p-value of paired, single-tailed t-Test.
Model
Method
Accuracy (%) ↑
p-value ↓
Config
Qwen3-1.7B
LLLM
44.32 ± 0.39
0.0115
lr = 4e −5
LLLM−JEPA (ours)
45.00 ± 0.40
λ = 1, k = 0, same lr
R1-Distill-Qwen-1.5B
LLLM
13.87 ± 1.01
0.0396
lr = 8e −5
LLLM−JEPA (ours)
15.04 ± 0.15
λ = 0.5, k = 1, same lr
4.83
9.66
14.49
19.32
24.15
28.98
PFLOPs
20
30
40
50
60
70
80
Accuracy (%)
LD=0.5, =2.0
LD=0.5, =1.0
LD=0
Baseline
Figure 5: LLM-JEPA converges faster than regular
fine-tuning at the same PFLOPs. Furthermore, random
JEPA-loss dropout (LD) helps save PFLOPs and boost
accuracy at the same amount of compute. LD = 0 is
the regular LLM-JEPA. Learning rate lr = 2e−5 and
k = 1. λ varies.
batch level. When active, it eliminates the need for an extra forward pass to compute Enc(Text) and
Enc(Code), thereby saving compute. If LD = α, the per-epoch cost becomes 2 −α times that of
standard fine-tuning, since each batch saves α forward passes. As shown in Figure 5 and Table 6,
LLM-JEPA tolerates aggressive loss dropout rates (e.g., 0.5 or 0.75), which leads to higher accuracy
under the same compute budget. Moreover, increasing λ in proportion to the dropout rate can further
improve performance. Empirically, we observe that keeping λ × (1 −α) approximately constant
provides a useful guideline for co-tuning λ and α to balance compute efficiency and accuracy. The
use of the loss dropout coupled with our custom attention mask offers some positive perspectives to
further scale LLM-JEPA to full scale pretraining with minimal computational overhead.
6
CONCLUSION AND FUTURE WORK
We introduced an alternative training objective for LLMs leveraging JEPAs. Our formulation is
an exact replicate of the JEPA objective extensively used in vision–but that hadn’t been adapted to
language yet. Crucially, our proposed LLM-JEPA maintains the generative capabilities of LLMs
while improving their abstract prompt representation as empirically validated across datasets and
models. While our experiments mostly focus on finetuning, preliminary pretraining experiment are
promising which we plan to scale and more thoroughly test in future work. Regarding the limitations
of LLM-JEPA, the primary bottleneck at present is the 2-fold increase in compute cost during training,
which is mitigated by random loss dropout.
9

Preprint
Table 6: Random JEPA-loss dropout (LD) help save PFLOPs and at the same time, boost accuracy. LD = 0 is
the regular LLM-JEPA. Reported are average accuracy and standard deviation across runs. Each row is 4.83
PFLOPs apart. Learning rate lr = 2e−5 and k = 1. λ varies.
Accuracy (%) ↑
LD=0, λ=1
LD=0.5, λ=1
LD=0.5, λ=2
LD=0.75, λ=1
LD=0.75, λ=2
LD=0.75, λ=4
19.85 ± 2.44
25.00 ± 3.73
24.50 ± 4.40
32.46 ± 3.32
32.10 ± 3.11
31.45 ± 3.34
40.70 ± 2.67
48.96 ± 6.03
50.71 ± 6.52
53.77 ± 5.53
57.03 ± 3.51
56.75 ± 4.63
55.60 ± 5.16
64.08 ± 2.75
63.79 ± 5.96
64.51 ± 7.28
67.03 ± 3.08
65.32 ± 3.54
60.43 ± 3.21
69.87 ± 2.48
70.11 ± 3.15
67.80 ± 4.94
66.80 ± 4.06
68.93 ± 4.59
63.57 ± 1.01
69.74 ± 2.99
71.20 ± 2.17
70.00 ± 4.74
70.77 ± 4.08
72.42 ± 1.28
63.96 ± 1.07
70.60 ± 3.05
72.11 ± 2.18
70.31 ± 4.64
70.92 ± 4.62
73.08 ± 1.28
Limitations Despite its strong accuracy gains, LLM-JEPA introduces two additional hyperparameters.
As shown in fig. 7, the optimal configuration may occur at any point in a grid (λ, k), which imposes
a significant cost for hyperparameter tuning. While we have not identified an efficient method to
explore this space, we empirically observe that adjacent grid points often yield similar accuracy,
suggesting the potential for a more efficient tuning algorithm.
REFERENCES
Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi.
Mathqa: Towards interpretable math word problem solving with operation-based formalisms.
arXiv preprint arXiv:1905.13319, 2019.
Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat,
Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding
predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 15619–15629, 2023.
Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A
general framework for self-supervised learning in speech, vision and language. In International
conference on machine learning, pages 1298–1312. PMLR, 2022.
Randall Balestriero and Yann LeCun. Learning by reconstruction produces uninformative features
for perception. arXiv preprint arXiv:2402.11337, 2024.
Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud
Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from
video. arXiv preprint arXiv:2404.08471, 2024.
Loïc Barrault, Paul-Ambroise Duquenne, Maha Elbayad, Artyom Kozhevnikov, Belen Alastruey,
Pierre Andrews, Mariano Coria, Guillaume Couairon, Marta R Costa-jussà, David Dale, et al.
Large concept models: Language modeling in a sentence representation space. arXiv preprint
arXiv:2412.08821, 2024.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot
learners. In Proceedings of NeurIPS, 2020a.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020b.
Rocío Cabrera Lozoya, Arnaud Baumann, Antonino Sabetta, and Michele Bezzi. Commit2vec:
Learning distributed representations of code changes. SN Computer Science, 2(3):150, 2021.
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for
contrastive learning of visual representations. In Proceedings of the 37th International Conference
on Machine Learning, pages 1597–1607. PMLR, 2020. URL https://arxiv.org/abs/
2002.05709.
10

Preprint
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam
Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm:
Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113,
2023.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, John
Schulman, Jacob Hilton, Melanie Knight, Adrian Weller, Dario Amodei, et al. Training verifiers to
solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu,
Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu,
Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao
Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao,
Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding,
Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang
Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong,
Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao,
Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang,
Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang,
Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L.
Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang,
Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng
Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng
Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan
Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang,
Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen,
Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li,
Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang,
Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan,
Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia
He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong
Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha,
Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang,
Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li,
Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen
Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
URL https://arxiv.org/abs/2501.12948.
Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence
embeddings. arXiv preprint arXiv:2104.08821, 2021.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of
models. arXiv preprint arXiv:2407.21783, 2024.
Jiaqi Guo, Zecheng Zhan, Yan Gao, Yan Xiao, Jian-Guang Lou, Ting Liu, and Dongmei Zhang.
Towards complex text-to-sql in cross-domain database with intermediate representation. arXiv
preprint arXiv:1905.08205, 2019.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Retrieval augmented
language model pre-training. In Proceedings of ICML, 2020.
Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked
autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 16000–16009, 2022.
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song,
and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv
preprin