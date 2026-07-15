# Process-mining objects and assessment metrics

This section describes the process-mining objects used in the dataset and the metrics used in the test-time assessment. The implementation uses paired synthetic samples composed of an event log, a canonical process tree, and a Petri-net graph. The process tree is the decoder target and canonical behavior description; the event log is sampled from the behavior; and the Petri graph is either the deterministic PM4Py conversion of the tree or an alternate exact-equivalent representation from the same behavior family. Thus, each sample is a cross-modal process triple representing one underlying process behavior:

\[
x_i = (T_i, L_i, G_i, \mathrm{id}_i),
\]

where \(T_i\) is a process tree, \(L_i\) is a finite event log, \(G_i\) is a typed Petri-net graph, and \(\mathrm{id}_i\) is a behavior-family equivalence identifier. This design follows the standard process-mining view in which observed executions are represented as event logs and process behavior can be represented by formal process models such as Petri nets and process trees [vanDerAalst2016]. PM4Py is used for process-tree conversion, Petri-net handling, event-log simulation, and some deterministic baseline features [Berti2019, Berti2023].

## 1. Process-mining objects considered in the data

### 1.1 Event logs

A classical event log contains cases, each case contains a trace, and each trace is an ordered sequence of events. The XES standard formalizes a richer event-log exchange format in which logs, traces, and events may carry attributes such as case identifiers, activity names, timestamps, lifecycle transitions, resources, and extensions [IEEE1849-2023]. The present dataset uses a deliberately simpler control-flow representation: an event log is stored only as a finite collection of activity-label sequences. Formally, for a finite activity alphabet \(\Sigma\), a trace is a finite word

\[
\sigma = \langle a_1, a_2, \ldots, a_m \rangle, \qquad a_j \in \Sigma,
\]

and an event log is a finite multiset of traces,

\[
L = [\sigma_1, \sigma_2, \ldots, \sigma_n] \in \mathcal{B}(\Sigma^*).
\]

The multiset interpretation is important: if the same trace occurs multiple times, the repeated occurrences represent frequency information and are preserved. In the implementation, this object is represented as

```text
traces: tuple[tuple[str, ...], ...]
```

and serialized in JSON as a list of lists of strings. For example:

```json
{
  "traces": [
    ["A0", "A1", "A2"],
    ["A0", "A2"],
    ["A0", "A1", "A2"]
  ]
}
```

No case-level or event-level attributes are stored in the main sample representation. In particular, the dataset does not store timestamps, resources, lifecycle attributes, costs, or object identifiers. During one baseline computation, namely `pm4py_log_case_features_mean_std`, the implementation constructs a temporary Pandas event table with synthetic case identifiers and synthetic timestamps derived from the event position. This temporary table exists only to call `pm4py.extract_features_dataframe`; it is not part of the canonical data object.

Activities are canonicalized as `A0`, `A1`, ..., so that synthetic samples are invariant to arbitrary original activity names. The activity tokenizer therefore has the vocabulary

```text
<pad>, A0, A1, ..., A(max_activities-1)
```

and a trace is encoded as a sequence of integer activity identifiers. Padding is used only in batches; it is not a semantic activity.

### 1.2 Process trees

The process tree is the central structural representation in the project. Process trees are block-structured process models: each internal node is an operator, each leaf is either an activity or a silent step, and the subtree rooted at each operator defines a structured behavioral block. Process trees are closely connected to inductive process discovery and are commonly used because they provide a structured representation that can be converted to sound workflow models under appropriate assumptions [Leemans2013]. Recent work also treats process trees as a distinct formalism for conformance checking and alignment computation [Schwanen2025].

The implementation defines a process tree node as an immutable object with

```text
kind: activity | tau | seq | xor | and | loop
label: optional string
children: tuple[ProcessTreeNode, ...]
```

The grammar used by the project can be written as

\[
\begin{aligned}
T ::= {} & a \\
      \mid{} & \tau \\
      \mid{} & \operatorname{SEQ}(T_1,\ldots,T_k), \quad k \ge 2 \\
      \mid{} & \operatorname{XOR}(T_1,\ldots,T_k), \quad k \ge 2 \\
      \mid{} & \operatorname{AND}(T_1,\ldots,T_k), \quad k \ge 2 \\
      \mid{} & \operatorname{LOOP}(T_{body},T_{redo}) \\
      \mid{} & \operatorname{LOOP}(T_{body},T_{redo},T_{exit}).
\end{aligned}
\]

Here, \(a \in \Sigma\) is a visible activity leaf and \(\tau\) is an invisible/silent leaf. The operator meanings are the standard block-structured meanings:

- `SEQ` executes children in order.
- `XOR` selects one branch.
- `AND` executes branches in parallel/interleaving semantics.
- `LOOP` represents repeated execution of a body with a redo branch and, in the current synthetic generation, an explicit exit branch.

The data model accepts two- or three-child loop nodes. However, the current grammar-masked decoder uses a tokenizer whose maximum arity is forced to be at least 3 and whose next-token mask always emits `ARITY_3` for `LOOP`. The synthetic generator also constructs loops with an explicit third exit child. Thus, the effective generated-and-decoded loop form in the current experimental setting is the three-child form:

\[
\operatorname{LOOP}(T_{body}, T_{redo}, T_{exit}).
\]

Process trees are serialized recursively as dictionaries. An activity has kind `activity` and a label, a silent node has kind `tau`, and an operator has a kind and a list of children. For example:

```json
{
  "kind": "seq",
  "children": [
    {"kind": "activity", "label": "A0"},
    {
      "kind": "xor",
      "children": [
        {"kind": "activity", "label": "A1"},
        {"kind": "tau"}
      ]
    }
  ]
}
```

For neural encoding and decoding, the tree is represented by a prefix token sequence with explicit operator arities:

```text
<bos> SEQ ARITY_2 A0 XOR ARITY_2 A1 TAU <eos>
```

The process-tree tokenizer contains the special tokens `<pad>`, `<bos>`, `<eos>`, `TAU`; the operator tokens `SEQ`, `XOR`, `AND`, `LOOP`; arity tokens `ARITY_2`, ..., `ARITY_max`; and activity tokens `A0`, ..., `A(max_activities-1)`. The grammar mask restricts decoding to syntactically valid next tokens. For example, after an operator token, an arity token must follow; once all open nodes are completed, the only valid semantic terminator is `<eos>`.

The implementation applies two canonicalization rules before tree tokenization. First, activity labels are renamed in order of first occurrence to `A0`, `A1`, ... while preserving repeated activity identity. Second, child order is sorted for the commutative operators `XOR` and `AND`. These rules avoid treating behaviorally irrelevant label names or branch permutations as distinct targets in the synthetic setting.

### 1.3 Petri nets and Petri-net graphs

Petri nets are a classical formalism for concurrent, asynchronous, nondeterministic, and distributed discrete-event systems [Murata1989], and they are widely used for workflow and process modeling [vanDerAalst1998]. The logical Petri-net object underlying the project can be described as an accepting labeled Petri net

\[
PN = (P, T, F, \ell, M_0, M_f),
\]

where \(P\) is a finite set of places, \(T\) is a finite set of transitions, \(P \cap T = \emptyset\), \(F \subseteq (P \times T) \cup (T \times P)\) is the bipartite flow relation, \(\ell : T \rightarrow \Sigma \cup \{\tau\}\) maps transitions to visible labels or invisibility, \(M_0\) is the initial marking, and \(M_f\) is the final marking. In the implementation, the Petri net is not learned or decoded directly. Instead, each process tree is converted deterministically to a PM4Py Petri net, and the resulting net is extracted as a typed graph.

The stored object is therefore `PetriGraph`, not a raw PM4Py object. It contains:

```text
node_types: tuple[int, ...]
node_names: tuple[str, ...]
transition_labels: tuple[str | None, ...]
edges: tuple[tuple[int, int, int], ...]
initial_marking: tuple[float, ...]
final_marking: tuple[float, ...]
```

The node type coding is:

```text
0 = place
1 = visible transition
2 = invisible transition
```

The edge type coding is:

```text
0 = place-to-transition
1 = transition-to-place
```

The graph is explicitly bipartite: each edge must connect a place to a transition or a transition to a place. Visible transitions have an activity label in `transition_labels`; places and invisible transitions have `None` where appropriate. Initial and final markings are stored as numeric vectors aligned with the node list. Places may have non-zero marking values, while transition positions are zero in ordinary accepting nets.

For deterministic serialization, places and transitions are sorted by their PM4Py names, and arcs are sorted by source and target names before conversion to the graph representation. This ensures that the same converted Petri net is represented consistently across runs.

## 2. Metrics considered during test-time assessment

The test report evaluates the model from four complementary perspectives: neural loss on the held-out split, grammar-constrained decoding quality, behavioral similarity, and embedding-space quality. It also compares the learned ProcRosetta embeddings with deterministic event-log features, deterministic Petri-net structural features, and a PM4Py Petri-net embedding baseline based on PetriNet2Vec/Node2Vec ideas [Colonna2024, Grover2016].

Let \(N\) be the number of test samples. For each sample \(i\), the three encoders produce Gaussian latent distributions

\[
q_m(z \mid x_i^m) = \mathcal{N}(\mu_i^m, \operatorname{diag}(\exp(\lambda_i^m))),
\qquad m \in \{tree, trace, petri\},
\]

where \(\mu_i^m\) is the latent mean and \(\lambda_i^m\) is the log-variance vector. At test time, the deterministic latent mean \(\mu_i^m\) is used for embeddings and greedy decoding. The fused representation used in the report is the arithmetic mean of the three modality-specific means:

\[
\mu_i^{fused} = \frac{1}{3}\left(\mu_i^{tree} + \mu_i^{trace} + \mu_i^{petri}\right).
\]

### 2.1 Neural test-loss metrics

The `loss_metrics` block reports the same objective components used during validation. These metrics assess whether the model can reconstruct or translate each modality into the canonical process-tree token sequence.

#### 2.1.1 Tree reconstruction loss

The `tree_reconstruction` metric is the grammar-masked sequence cross-entropy for reconstructing the target process tree from the tree encoder. Given the target tree token sequence \(y_i = (y_{i,1},\ldots,y_{i,T_i})\), the decoder predicts each next token autoregressively:

\[
p_\theta(y_{i,t} \mid y_{i,<t}, z_i^{tree}).
\]

Padding positions are ignored. With label smoothing coefficient \(\epsilon\), the loss uses a smoothed target distribution rather than a point mass on the correct token. Label smoothing is a known regularization technique introduced in the neural-network literature to reduce overconfident predictions [Szegedy2016]. In this project, smoothing mass is distributed only over tokens allowed by the grammar mask at that position.

#### 2.1.2 Trace-to-tree translation loss

The `trace_to_tree` metric is the same grammar-masked sequence cross-entropy, but the latent vector is produced by the trace encoder:

\[
p_\theta(y_{i,t} \mid y_{i,<t}, z_i^{trace}).
\]

This evaluates whether the event log contains enough information for the learned representation to decode the original canonical process tree.

#### 2.1.3 Petri-to-tree translation loss

The `petri_to_tree` metric is the corresponding cross-entropy when the latent vector is produced by the Petri-graph encoder:

\[
p_\theta(y_{i,t} \mid y_{i,<t}, z_i^{petri}).
\]

This evaluates whether the converted Petri-net graph representation can be mapped back to the process-tree representation from which it was generated.

#### 2.1.4 Latent alignment loss

The `latent_alignment` metric is a within-sample alignment loss between modality-specific latent means. For each sample, the three modality encodings should be close because they represent the same underlying process. The implementation averages the mean-squared error over all unordered modality pairs:

\[
\mathcal{L}_{align}
= \frac{1}{3}\left(
\operatorname{MSE}(\mu^{tree},\mu^{trace})
+ \operatorname{MSE}(\mu^{tree},\mu^{petri})
+ \operatorname{MSE}(\mu^{trace},\mu^{petri})
\right).
\]

This is a project-specific alignment component.

#### 2.1.5 Cross-modal contrastive loss

The `contrastive` metric is a symmetric cross-modal contrastive loss. For each pair of modalities \((m,n)\), latent means are L2-normalized, and similarities are computed as

\[
s_{ij}^{m,n} = \frac{\cos(\mu_i^m, \mu_j^n)}{\tau},
\]

where \(\tau = 0.2\) is the temperature. The implementation uses a multi-positive InfoNCE variant: every row with the same behavior-family identifier is a valid positive, and rows from different behavior families are negatives. The loss applies this objective in both directions, \(m \rightarrow n\) and \(n \rightarrow m\), and averages over all modality pairs. This objective is closely related to the InfoNCE family of contrastive objectives [Oord2018].

#### 2.1.6 KL divergence loss

The `kl` metric regularizes each approximate posterior \(q_m(z \mid x_i^m)\) against a standard normal prior, as in variational autoencoders [KingmaWelling2014]. For each modality and sample, the diagonal-Gaussian KL term is

\[
D_{KL}\left(q_m(z \mid x_i^m) \parallel \mathcal{N}(0,I)\right)
= \frac{1}{2}\sum_d \left(\exp(\lambda_{i,d}^m) + (\mu_{i,d}^m)^2 - 1 - \lambda_{i,d}^m\right).
\]

The reported value averages this expression over modalities and samples.

#### 2.1.7 Total test loss

The `loss` metric is the weighted sum

\[
\begin{aligned}
\mathcal{L} ={}& 1.0\,\mathcal{L}_{tree\_reconstruction}
+ 1.0\,\mathcal{L}_{trace\rightarrow tree}
+ 1.0\,\mathcal{L}_{petri\rightarrow tree} \\
&+ 0.1\,\mathcal{L}_{align}
+ 0.1\,\mathcal{L}_{contrastive}
+ 0.001\,\mathcal{L}_{KL}.
\end{aligned}
\]

The test output reports all components, including `latent_alignment`, although the human-readable console summary emphasizes the main reconstruction, translation, contrastive, and KL terms.

### 2.2 Decode-quality metrics

The `decode_quality` block evaluates whether latent vectors can be decoded into valid process trees and whether those trees preserve the intended behavior. Greedy decoding is performed from four latent sources:

```text
proc_rosetta_tree_mu
proc_rosetta_trace_mu
proc_rosetta_petri_mu
proc_rosetta_fused_mu
```

The decoder is grammar-masked, and the maximum generated token length is 128 by default. For each decoded token sequence, the following per-sample quantities are computed.

| Metric | Meaning |
|---|---|
| `terminated` | Whether the decoded sequence contains the `<eos>` token. |
| `valid_tree` | Whether the token sequence can be parsed by the process-tree tokenizer into a valid `ProcessTreeNode`. |
| `exact_tree_match` | Whether the decoded tree dictionary exactly equals the canonical target tree dictionary. |
| `petri_convertible` | Whether the decoded process tree can be converted to a Petri net through PM4Py. |
| `behavior_evaluable` | Whether traces can be simulated from the decoded tree and compared with the original log. |
| `token_edit_distance` | Levenshtein edit distance between the target token sequence and the decoded token sequence [Levenshtein1966]. |
| `normalized_token_edit_distance` | Edit distance divided by `max(len(target_tokens), len(decoded_tokens), 1)`. |
| `behavior_l1` | Behavioral distance between the original traces and traces simulated from the decoded tree, using the project behavioral L1 metric defined below. |
| `error` | First decoding, Petri-conversion, or behavior-evaluation error, if any. |

The report aggregates these rows into rates and means:

```text
count
terminated_rate
valid_tree_rate
exact_tree_match_rate
petri_conversion_rate
behavior_eval_success_rate
mean_token_edit_distance
mean_normalized_token_edit_distance
mean_behavior_l1
median_behavior_l1
invalid_decode_count
petri_conversion_error_count
behavior_error_count
first_error
```

The validity, exact-match, Petri-conversion, and behavior-evaluation metrics are introduced in this project as task-specific decode-quality measures. They are necessary because a low token loss does not by itself guarantee that the generated sequence is a valid process tree, that it can be converted into a Petri net, or that it yields similar simulated behavior.

### 2.3 Process-discovery quality metrics

The `discovery_quality` block evaluates the log-to-model use case directly. For each test sample, the event log \(L_i\) is used in two discovery pathways:

```text
proc_rosetta_trace_mu
inductive_miner
```

For `proc_rosetta_trace_mu`, the trace encoder maps \(L_i\) to its deterministic latent mean, the grammar-masked decoder greedily decodes a process tree, and PM4Py converts the decoded tree to a Petri net. For `inductive_miner`, PM4Py's Inductive Miner discovers a process tree from the same log, and that tree is also converted to a Petri net. Both Petri nets are then evaluated against the original log using PM4Py's alignment-based conformance functions.

The per-sample fields are:

```text
model_discovered
alignment_evaluable
fitness
precision
f1
error
```

`fitness` is the alignment fitness reported by `pm4py.fitness_alignments`. It measures how much of the observed log behavior can be replayed by the discovered model. `precision` is the alignment precision reported by `pm4py.precision_alignments`. It penalizes models that allow too much behavior beyond what is observed in the log. The F1 score combines them as the harmonic mean:

\[
F1 = \frac{2 \cdot fitness \cdot precision}{fitness + precision},
\]

with \(F1 = 0\) when the denominator is zero. The report aggregates these rows by method:

```text
count
model_discovered_rate
alignment_evaluable_rate
mean_fitness
mean_precision
mean_f1
median_f1
alignment_error_count
first_error
```

This block is the classical process-discovery comparison in the test report. It differs from `decode_quality`: decode quality checks whether neural outputs are valid and behaviorally close under sampled-log simulation, whereas discovery quality asks how well the log-derived model scores under alignment fitness and precision against the source log.

### 2.4 Behavioral distance metrics

The test suite uses a lightweight behavioral distance between two event logs. It is not a classical alignment-based conformance measure; rather, it is a project-introduced distributional proxy that combines three standard log abstractions: trace variants, directly-follows relations, and trace lengths. Directly-follows relations are a common abstraction in process mining, although their limitations as a complete behavioral representation are well known [vanDerAalst2019DFG].

Given a log \(L\), define the empirical trace-variant distribution as

\[
p_{var}^L(\sigma) = \frac{\#\{\sigma' \in L : \sigma'=\sigma\}}{|L|}.
\]

Define the empirical trace-length distribution as

\[
p_{len}^L(k) = \frac{\#\{\sigma \in L : |\sigma|=k\}}{|L|}.
\]

For directly-follows behavior, each trace is augmented with boundary symbols:

\[
\tilde{\sigma} = \langle \langle start \rangle, a_1,\ldots,a_m, \langle end \rangle \rangle.
\]

The empirical directly-follows distribution is the normalized count of adjacent pairs in these augmented traces:

\[
p_{df}^L(a,b)
= \frac{\#\text{ occurrences of adjacent pair }(a,b)\text{ in all }\tilde{\sigma}\in L}
{\#\text{ adjacent pairs in all }\tilde{\sigma}\in L}.
\]

For any two discrete distributions \(p\) and \(q\), the L1 distance used in the implementation is

\[
d_1(p,q) = \sum_{u \in \operatorname{supp}(p) \cup \operatorname{supp}(q)} |p(u)-q(u)|.
\]

The project behavioral distance is then

\[
D_{beh}(L,L')
= \frac{1}{3}\left(
 d_1(p_{var}^L,p_{var}^{L'})
 + d_1(p_{df}^L,p_{df}^{L'})
 + d_1(p_{len}^L,p_{len}^{L'})
\right).
\]

Since each component is an L1 distance between probability distributions, each component lies in \([0,2]\), and the mean also lies in \([0,2]\). Lower values indicate more similar sampled behavior. The report exposes the three components as

```text
variant_l1
directly_follows_l1
length_l1
```

and their average as

```text
mean_l1
```

This metric is introduced in the project. Its purpose is to provide a fast, model-independent behavioral comparison for synthetic logs, decoded-model simulations, nearest-neighbor evaluation, and embedding-distance correlation. A Jensen-Shannon divergence helper exists in the code, but it is not used in the rich test report.

The report also computes all pairwise behavioral distances among test logs. For each component and for `mean_l1`, it summarizes the upper triangle of the pairwise distance matrix using

```text
pair_count
mean
std
min
max
```

These summaries appear in `behavioral_distance_summary` and `behavioral_component_summaries`.

### 2.5 Embedding-method metrics

The `embedding_methods` block evaluates each learned or baseline vector representation against the behavioral distance matrix. For any method producing vectors \(e_1,\ldots,e_N\), the embedding distance is cosine distance:

\[
D_E(i,j) = 1 - \frac{e_i^\top e_j}{\|e_i\|_2\,\|e_j\|_2}.
\]

Cosine similarity and cosine distance are standard in vector-space information retrieval [Manning2008].

#### 2.5.1 Vector statistics

The `vector_statistics` field reports basic properties of the embedding matrix:

```text
count
dimension
l2_norm_mean
l2_norm_std
feature_variance_mean
```

These statistics are descriptive. They indicate whether a method produced a vector for each test sample, the dimensionality of the representation, the scale of vector norms, and the average variance across vector dimensions.

#### 2.5.2 Pairwise distance statistics

The `pairwise_statistics` field summarizes the upper triangle of the cosine-distance matrix:

```text
pair_count
mean
std
min
max
```

These values describe the spread of the embedding geometry. For example, if all vectors are nearly identical, the mean and variance of pairwise distances will be small.

#### 2.5.3 Behavior alignment through correlation

The `behavior_alignment` field compares embedding distances with behavioral distances over all unordered test-sample pairs. It reports:

```text
spearman_embedding_distance_vs_behavior_l1
pearson_embedding_distance_vs_behavior_l1
```

The Spearman coefficient compares the rank ordering of distances and is therefore insensitive to monotone transformations [Spearman1904]. The Pearson coefficient compares linear association [Pearson1896]. In this setting, a high positive correlation means that pairs of samples that are far apart in the embedding space also tend to be behaviorally far apart according to `mean_l1`.

#### 2.5.4 Nearest-neighbor behavior

For each sample \(i\), the report identifies its nearest neighbor under the embedding distance:

\[
\operatorname{NN}_E(i) = \arg\min_{j \ne i} D_E(i,j).
\]

It then measures how behaviorally close these embedding-nearest neighbors are:

\[
\operatorname{NNBeh}(E) = \frac{1}{N}\sum_{i=1}^N D_{beh}(L_i,L_{\operatorname{NN}_E(i)}).
\]

The report exposes this as

```text
mean_behavior_l1_at_nearest_neighbor
```

and compares it with the mean behavioral distance over all unordered test pairs:

```text
random_pair_behavior_l1_mean
```

The improvement is

\[
\operatorname{Improvement}(E)
= \operatorname{RandomMean} - \operatorname{NNBeh}(E).
\]

It is reported as

```text
improvement_over_random
```

Higher improvement means that the embedding retrieves behaviorally closer samples than a random pair would, on average. This nearest-neighbor behavioral evaluation is introduced in the project.

#### 2.5.5 Method ranking

The `method_ranking` block sorts available embedding methods by descending Spearman behavior alignment and then by ascending nearest-neighbor behavioral distance. The ranking fields are:

```text
method
behavior_spearman
nearest_neighbor_behavior_l1
improvement_over_random
```

This ranking is project-specific. It is intended as an empirical summary of how well each vector representation preserves the behavior proxy, not as a universal ranking of process-mining representations.

### 2.6 Cross-modal retrieval metrics

The `cross_modal_retrieval` block evaluates whether embeddings from one modality retrieve the matching embeddings from another modality. The six query-target directions are:

```text
tree_to_trace
trace_to_tree
tree_to_petri
petri_to_tree
trace_to_petri
petri_to_trace
```

For a query modality \(m\) and candidate modality \(n\), the similarity matrix is

\[
S_{ij}^{m,n} = \cos(\mu_i^m, \mu_j^n).
\]

For query \(i\), the correct candidate is \(j=i\), because both embeddings come from the same synthetic process triple. Candidates are ranked by decreasing cosine similarity. The report computes:

```text
count
top1_accuracy
mean_rank
mrr
```

`top1_accuracy` is the fraction of queries for which the correct cross-modal counterpart is ranked first. `mean_rank` is the average rank of the correct counterpart. `mrr`, the mean reciprocal rank, is

\[
\operatorname{MRR} = \frac{1}{N}\sum_{i=1}^N \frac{1}{\operatorname{rank}_i}.
\]

Mean reciprocal rank is a standard retrieval evaluation metric [VoorheesTice2000]. Its use here for paired process-mining artifacts is introduced in the project.

### 2.7 Agreement with the fused ProcRosetta geometry

The report contains a direct comparison between each available embedding method and the fused ProcRosetta latent representation. The reference method is

```text
proc_rosetta_fused_mu
```

For each other method, the report computes:

```text
pairwise_distance_spearman_agreement
pairwise_distance_pearson_agreement
top1_neighbor_overlap
top3_neighbor_overlap
behavior_spearman_delta_vs_reference
nearest_neighbor_behavior_l1_delta_vs_reference
```

The pairwise agreement metrics correlate the upper-triangle distance vector of the method with the upper-triangle distance vector of the fused representation. The nearest-neighbor overlap metrics compare the top-\(k\) nearest-neighbor sets induced by the two distance matrices:

\[
\operatorname{Overlap}_k(i)
= \frac{|\operatorname{TopK}_{ref}(i) \cap \operatorname{TopK}_{method}(i)|}{k},
\]

and then average over all test samples. The delta metrics compare behavior alignment and nearest-neighbor behavioral distance against the fused representation. For `behavior_spearman_delta_vs_reference`, higher is better. For `nearest_neighbor_behavior_l1_delta_vs_reference`, negative values indicate that the method has lower nearest-neighbor behavioral distance than the fused reference.

These geometry-agreement and delta metrics are introduced in the project. They are particularly useful for comparing the learned multimodal latent space with deterministic baselines and the PM4Py Petri-net embedding baseline.

### 2.8 Baseline representations evaluated in the test report

The report evaluates both learned and deterministic vector representations.

#### 2.8.1 Learned ProcRosetta representations

The learned representations are:

```text
proc_rosetta_tree_mu
proc_rosetta_trace_mu
proc_rosetta_petri_mu
proc_rosetta_fused_mu
```

The first three are the latent means from the tree, trace, and Petri encoders. The fused representation is the arithmetic mean of the three means. The fused representation is introduced in the project as a simple multimodal aggregate.

#### 2.8.2 Deterministic event-log baselines

The deterministic event-log baselines map each log to a sparse count or frequency vector and then use the same embedding-method evaluation metrics described above.

`trace_activity_counts` uses normalized activity occurrence counts:

\[
f_L(a) = \frac{\#\text{ occurrences of activity }a\text{ in }L}{\#\text{ events in }L}.
\]

`trace_variant_distribution` uses the normalized trace-variant distribution \(p_{var}^L\), as defined above.

`trace_directly_follows` uses the normalized directly-follows distribution \(p_{df}^L\), including `<start>` and `<end>` boundary symbols.

`trace_eventually_follows` counts ordered pairs \((a,b)\) such that \(a\) occurs before \(b\) later in the same trace. If a trace is \(\sigma=\langle a_1,\ldots,a_m\rangle\), then all pairs \((a_r,a_s)\) with \(r<s\) contribute. Counts are normalized by the total number of such ordered pairs in the log.

`pm4py_log_case_features_mean_std` converts the traces to a temporary event table and calls `pm4py.extract_features_dataframe`. Numeric case-level features are aggregated per log by mean and standard deviation. This baseline relies on PM4Py’s feature extraction functionality [Berti2019, Berti2023].

#### 2.8.3 Deterministic Petri-net structural baseline

`petri_structural_counts` maps each Petri graph to the following structural counts:

```text
number of places
number of visible transitions
number of invisible transitions
number of place-to-transition edges
number of transition-to-place edges
sum of initial marking tokens
sum of final marking tokens
number of unique visible transition labels
number of duplicate visible transitions
```

This baseline is introduced in the project. It is intentionally simple and tests whether coarse structural Petri-net size and label information can approximate behavioral similarity.

#### 2.8.4 PM4Py Petri-net embedding baseline

`pm4py_colonna_petri_node2vec` uses the PM4Py helper `pm4py.objects.petri_net.utils.embeddings_similarity.petri_net_embedding`, when available. The report describes this baseline as the Petri-net Node2Vec/Word2Vec-style embedding associated with Colonna et al.’s PetriNet2Vec work [Colonna2024]. PetriNet2Vec is an unsupervised method for learning vector representations of Petri nets, inspired by document embeddings such as Doc2Vec/Paragraph Vector [LeMikolov2014] and graph random-walk embeddings such as Node2Vec [Grover2016].

The default baseline configuration in the project is:

```text
dimensions = 64
num_walks = 5
walk_length = 20
window = 5
epochs = 5
seed = 42
```

If the PM4Py embedding helper or its dependencies are unavailable, the method remains in the report with `available = false` and a diagnostic reason.

## 3. Metrics already published versus metrics introduced in this work

The assessment combines established metrics and new task-specific metrics.

### 3.1 Established or literature-based components

The following components are standard or directly derived from published work:

| Component | Role in the report | Reference |
|---|---|---|
| Event-log, process-tree, and Petri-net concepts | Defines the process-mining artifacts used by the dataset. | [vanDerAalst2016, Murata1989, vanDerAalst1998, Leemans2013] |
| PM4Py conversion, simulation, and feature extraction | Used for process-tree to Petri-net conversion, trace simulation, and a log-feature baseline. | [Berti2019, Berti2023] |
| Label-smoothed cross-entropy | Used in grammar-masked tree-token prediction. | [Szegedy2016] |
| KL regularization against a standard normal prior | Used for variational latent distributions. | [KingmaWelling2014] |
| Symmetric contrastive alignment | Used to align modalities in the shared latent space. | [Oord2018] |
| Levenshtein edit distance | Used to measure decoded tree-token sequence error. | [Levenshtein1966] |
| Cosine similarity/distance | Used for embedding distances and retrieval. | [Manning2008] |
| Spearman and Pearson correlations | Used for behavior-alignment and method-agreement statistics. | [Spearman1904, Pearson1896] |
| Mean reciprocal rank | Used for cross-modal retrieval evaluation. | [VoorheesTice2000] |
| Alignment fitness and precision | Used to evaluate discovered Petri nets against the source test logs. | [Adriansyah2011] |
| Inductive Miner | Used as the process-discovery baseline for log-to-process-tree discovery quality. | [Leemans2013] |
| Node2Vec/Doc2Vec-style Petri-net embeddings | Used through the PM4Py Petri embedding baseline. | [Colonna2024, Grover2016, LeMikolov2014] |
| Directly-follows abstraction | Used in behavioral distance and deterministic log features. | [vanDerAalst2019DFG] |

### 3.2 Metrics introduced in this project

The following metrics or metric combinations are introduced by the project for this multimodal process-mining setting:

| Introduced metric | Definition and purpose |
|---|---|
| `mean_l1` behavioral distance | Average of L1 distances over trace-variant, directly-follows, and trace-length distributions. It provides a fast behavioral proxy for comparing two sampled logs. |
| Decode validity package | `terminated_rate`, `valid_tree_rate`, `exact_tree_match_rate`, `petri_conversion_rate`, and `behavior_eval_success_rate` jointly evaluate whether a latent decodes into a syntactically valid, semantically usable process-tree/Petri-net artifact. |
| Decode behavioral distance | Behavioral L1 between the original log and traces simulated from the decoded tree. It measures behavioral preservation after neural decoding and deterministic Petri conversion. |
| ProcRosetta-vs-Inductive-Miner discovery summary | Per-log comparison of the trace-decoded ProcRosetta process model and the Inductive Miner process model using alignment fitness, alignment precision, and F1. |
| Cross-modal retrieval over paired process artifacts | Top-1 accuracy, mean rank, and MRR for retrieving the matching tree, trace, or Petri representation of the same synthetic process. |
| Nearest-neighbor behavioral distance | Mean behavioral distance from each sample to its nearest embedding neighbor. It evaluates whether local neighborhoods in the embedding space correspond to behaviorally similar logs. |
| Improvement over random | Difference between random-pair behavioral distance and nearest-neighbor behavioral distance. It quantifies the behavioral advantage of embedding-based retrieval. |
| Fused latent representation | Arithmetic mean of tree, trace, and Petri latent means. It is a simple multimodal representation used as a reference. |
| Agreement against fused geometry | Pairwise distance Spearman/Pearson agreement and top-k neighbor overlap between each method and `proc_rosetta_fused_mu`. |
| Behavior deltas against fused reference | Differences in behavior Spearman and nearest-neighbor behavioral L1 relative to the fused ProcRosetta representation. |
| Petri structural-count baseline | Coarse Petri graph statistics used as a deterministic process-model baseline. |

These introduced summaries complement classical process-discovery and conformance metrics. The report now includes alignment fitness and precision for the log-to-model discovery setting, while the other metrics evaluate the specific claims of the project: cross-modal representation learning, grammar-valid process-tree decoding, deterministic conversion of decoded trees to Petri nets, and behavioral preservation in a synthetic paired-triple setting.

## 4. Relation to classical process-discovery quality metrics

Classical process-discovery evaluation often discusses four quality dimensions: fitness, precision, generalization, and simplicity [Buijs2012]. These dimensions are important when a discovered model is compared with an event log through replay or alignment-based conformance checking. The current test suite computes alignment-based fitness and precision, plus their F1 score, for the specific log-to-model discovery comparison between ProcRosetta and PM4Py's Inductive Miner. It does not yet compute generalization, simplicity, token-based replay scores, or broader real-life benchmark suites.

Consequently, `discovery_quality` should be interpreted as the classical conformance-oriented discovery comparison in the report, while `behavior_l1`, `mean_l1`, nearest-neighbor behavioral distance, and cross-modal retrieval remain representation-learning and sampled-behavior metrics. A broader evaluation should still add generalization, simplicity, runtime, and real-life log benchmarks before making strong process-discovery performance claims.

## 5. Complete list of test-report fields

For completeness, the rich test report contains the following top-level fields:

```text
split
sample_count
loss_metrics
behavioral_distance_summary
behavioral_component_summaries
decode_quality
discovery_quality
cross_modal_retrieval
embedding_methods
method_ranking
method_comparisons_against_proc_rosetta_fused_mu
references
```

The `loss_metrics` field contains:

```text
loss
tree_reconstruction
trace_to_tree
petri_to_tree
latent_alignment
contrastive
kl
```

The `decode_quality.methods` field contains one summary for each of:

```text
proc_rosetta_tree_mu
proc_rosetta_trace_mu
proc_rosetta_petri_mu
proc_rosetta_fused_mu
```

Each decode summary contains:

```text
count
terminated_rate
valid_tree_rate
exact_tree_match_rate
petri_conversion_rate
behavior_eval_success_rate
mean_token_edit_distance
mean_normalized_token_edit_distance
mean_behavior_l1
median_behavior_l1
invalid_decode_count
petri_conversion_error_count
behavior_error_count
first_error
```

The `discovery_quality.methods` field contains one summary for each of:

```text
proc_rosetta_trace_mu
inductive_miner
```

Each discovery-quality summary contains:

```text
count
model_discovered_rate
alignment_evaluable_rate
mean_fitness
mean_precision
mean_f1
median_f1
alignment_error_count
first_error
```

The `cross_modal_retrieval` field contains:

```text
tree_to_trace
trace_to_tree
tree_to_petri
petri_to_tree
trace_to_petri
petri_to_trace
```

Each retrieval direction contains:

```text
count
top1_accuracy
mean_rank
mrr
```

The `embedding_methods` field may contain the learned methods, deterministic baselines, and PM4Py Petri embedding baseline:

```text
proc_rosetta_tree_mu
proc_rosetta_trace_mu
proc_rosetta_petri_mu
proc_rosetta_fused_mu
trace_activity_counts
trace_variant_distribution
trace_directly_follows
trace_eventually_follows
pm4py_log_case_features_mean_std
petri_structural_counts
pm4py_colonna_petri_node2vec
```

For available methods, each method summary contains:

```text
available
kind
vector_statistics
pairwise_statistics
behavior_alignment
nearest_neighbor_behavior
```

The `method_comparisons_against_proc_rosetta_fused_mu` field contains the reference method name, an availability flag, and per-method comparisons with:

```text
pairwise_distance_spearman_agreement
pairwise_distance_pearson_agreement
top1_neighbor_overlap
top3_neighbor_overlap
behavior_spearman_delta_vs_reference
nearest_neighbor_behavior_l1_delta_vs_reference
```

## 6. Current base-experiment results

The current `testing_results.txt` report evaluates the final checkpoint on the held-out test split:

```text
split = test
rows = 1024
behavior families = 512
```

Neural test losses are:

| loss | tree | trace->tree | Petri->tree | contrastive | KL |
|---:|---:|---:|---:|---:|---:|
| 1.4158 | 0.4282 | 0.4495 | 0.4362 | 0.7995 | 19.9026 |

Greedy decoding succeeds structurally for every latent source:

| latent source | ended | valid tree | exact tree | Petri ok | behavior L1 | norm edit |
|---|---:|---:|---:|---:|---:|---:|
| ProcRosetta tree | 100.0% | 100.0% | 85.9% | 100.0% | 0.322 | 0.063 |
| ProcRosetta trace | 100.0% | 100.0% | 85.2% | 100.0% | 0.336 | 0.083 |
| ProcRosetta Petri | 100.0% | 100.0% | 85.7% | 100.0% | 0.310 | 0.065 |
| ProcRosetta fused | 100.0% | 100.0% | 85.9% | 100.0% | 0.320 | 0.066 |

Process-discovery quality against the source logs is:

| method | model ok | align ok | fitness | precision | F1 |
|---|---:|---:|---:|---:|---:|
| ProcRosetta trace | 100.0% | 100.0% | 0.950 | 0.937 | 0.938 |
| Inductive Miner | 100.0% | 100.0% | 1.000 | 0.969 | 0.980 |

The mean pairwise behavior L1 among test logs is 1.2093 over 523,776 pairs.

The embedding-quality ranking by Spearman behavior correlation is:

| method | behavior rho | NN behavior | improvement | dim |
|---|---:|---:|---:|---:|
| eventually-follows | 0.889 | 0.026 | 1.184 | 148 |
| trace activity counts | 0.848 | 0.023 | 1.186 | 14 |
| pm4py log features | 0.846 | 0.146 | 1.063 | 28 |
| ProcRosetta trace | 0.841 | 0.000 | 1.209 | 48 |
| directly-follows | 0.764 | 0.000 | 1.209 | 146 |
| ProcRosetta fused | 0.757 | 0.000 | 1.209 | 48 |
| pm4py Petri Node2Vec | 0.722 | 0.531 | 0.678 | 64 |
| trace variants | 0.692 | 0.000 | 1.209 | 665 |
| ProcRosetta Petri | 0.686 | 0.000 | 1.209 | 48 |
| ProcRosetta tree | 0.673 | 0.000 | 1.209 | 48 |
| Petri structural counts | 0.528 | 0.000 | 1.209 | 9 |

Exact row-level cross-modal retrieval is above chance but remains the largest source of headroom:

| query -> target | top1 | MRR | mean rank |
|---|---:|---:|---:|
| Petri -> trace | 0.029 | 0.076 | 102.382 |
| Petri -> tree | 0.062 | 0.121 | 100.907 |
| trace -> Petri | 0.028 | 0.075 | 102.301 |
| trace -> tree | 0.020 | 0.066 | 102.281 |
| tree -> Petri | 0.063 | 0.121 | 100.971 |
| tree -> trace | 0.025 | 0.070 | 102.465 |

The behavior-family equivalence check reports within-family cosine similarities of 0.995--1.000 across the four learned embeddings. Family top-1 retrieval is strongest for the trace encoder (0.881) and lower for fused, Petri, and tree embeddings (0.146, 0.136, and 0.154 respectively).

## BIBLIOGRAPHY

[Adriansyah2011] A. Adriansyah, B. F. van Dongen, and W. M. P. van der Aalst. "Conformance Checking Using Cost-Based Fitness Analysis." In *Proceedings of the 2011 IEEE International Enterprise Distributed Object Computing Conference Workshops*, pp. 55--64. IEEE, 2011. DOI: 10.1109/EDOCW.2011.12.

[Berti2019] A. Berti, S. J. van Zelst, and W. M. P. van der Aalst. "Process Mining for Python (PM4Py): Bridging the Gap Between Process- and Data Science." *CEUR Workshop Proceedings*, Vol. 2374, 2019. Also available as arXiv:1905.06169.

[Berti2023] A. Berti, S. J. van Zelst, and D. Schuster. "PM4Py: A Process Mining Library for Python." *Software Impacts*, 17:100556, 2023. DOI: 10.1016/j.simpa.2023.100556.

[Buijs2012] J. C. A. M. Buijs, B. F. van Dongen, and W. M. P. van der Aalst. "On the Role of Fitness, Precision, Generalization and Simplicity in Process Discovery." In *On the Move to Meaningful Internet Systems: OTM 2012*, LNCS 7565, pp. 305--322. Springer, 2012. DOI: 10.1007/978-3-642-33606-5_19.

[Colonna2024] J. G. Colonna, A. A. Fares, M. Duarte, and R. Sousa. "Process Mining Embeddings: Learning Vector Representations for Petri Nets." *Intelligent Systems with Applications*, 23:200423, 2024. Also available as arXiv:2404.17129.

[Grover2016] A. Grover and J. Leskovec. "node2vec: Scalable Feature Learning for Networks." In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 855--864, 2016. DOI: 10.1145/2939672.2939754.

[IEEE1849-2023] IEEE. *IEEE Standard for eXtensible Event Stream (XES) for Achieving Interoperability in Event Logs and Event Streams*. IEEE Std 1849-2023, 2023. DOI: 10.1109/IEEESTD.2023.10267858.

[KingmaWelling2014] D. P. Kingma and M. Welling. "Auto-Encoding Variational Bayes." In *Proceedings of the 2nd International Conference on Learning Representations (ICLR)*, 2014. Also available as arXiv:1312.6114.

[LeMikolov2014] Q. V. Le and T. Mikolov. "Distributed Representations of Sentences and Documents." In *Proceedings of the 31st International Conference on Machine Learning (ICML)*, JMLR Workshop and Conference Proceedings, Vol. 32, pp. 1188--1196, 2014.

[Leemans2013] S. J. J. Leemans, D. Fahland, and W. M. P. van der Aalst. "Discovering Block-Structured Process Models from Event Logs: A Constructive Approach." In *Application and Theory of Petri Nets and Concurrency*, LNCS 7927, pp. 311--329. Springer, 2013. DOI: 10.1007/978-3-642-38697-8_17.

[Levenshtein1966] V. I. Levenshtein. "Binary Codes Capable of Correcting Deletions, Insertions and Reversals." *Soviet Physics Doklady*, 10(8):707--710, 1966.

[Manning2008] C. D. Manning, P. Raghavan, and H. Schütze. *Introduction to Information Retrieval*. Cambridge University Press, 2008. DOI: 10.1017/CBO9780511809071.

[Murata1989] T. Murata. "Petri Nets: Properties, Analysis and Applications." *Proceedings of the IEEE*, 77(4):541--580, 1989. DOI: 10.1109/5.24143.

[Oord2018] A. van den Oord, Y. Li, and O. Vinyals. "Representation Learning with Contrastive Predictive Coding." arXiv:1807.03748, 2018.

[Pearson1896] K. Pearson. "Mathematical Contributions to the Theory of Evolution. III. Regression, Heredity, and Panmixia." *Philosophical Transactions of the Royal Society of London. Series A*, 187:253--318, 1896. DOI: 10.1098/rsta.1896.0007.

[Schwanen2025] C. T. Schwanen, W. Pakusa, and W. M. P. van der Aalst. "Process Tree Alignments." In *Enterprise Design, Operations, and Computing: EDOC 2024*, LNCS 15409, pp. 300--317. Springer, 2025. DOI: 10.1007/978-3-031-78338-8_16.

[Spearman1904] C. Spearman. "The Proof and Measurement of Association Between Two Things." *The American Journal of Psychology*, 15(1):72--101, 1904. DOI: 10.2307/1412159.

[Szegedy2016] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. "Rethinking the Inception Architecture for Computer Vision." In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2818--2826, 2016. DOI: 10.1109/CVPR.2016.308.

[vanDerAalst1998] W. M. P. van der Aalst. "The Application of Petri Nets to Workflow Management." *Journal of Circuits, Systems and Computers*, 8(1):21--66, 1998. DOI: 10.1142/S0218126698000043.

[vanDerAalst2016] W. M. P. van der Aalst. *Process Mining: Data Science in Action*. 2nd ed. Springer, 2016. DOI: 10.1007/978-3-662-49851-4.

[vanDerAalst2019DFG] W. M. P. van der Aalst. "A Practitioner's Guide to Process Mining: Limitations of the Directly-Follows Graph." *Procedia Computer Science*, 164:321--328, 2019. DOI: 10.1016/j.procs.2019.12.189.

[VoorheesTice2000] E. M. Voorhees and D. M. Tice. "The TREC-8 Question Answering Track." In *Proceedings of the Second International Conference on Language Resources and Evaluation (LREC 2000)*. European Language Resources Association, 2000.
