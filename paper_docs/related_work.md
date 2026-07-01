# Related Work and State-of-the-Art Positioning

## Scope of the Contribution

The project considered here, **ProcRosetta**, addresses a multimodal representation-learning problem in process mining. Its central hypothesis is that heterogeneous process artifacts -- event logs, process trees, and Petri nets -- can be treated as different views of an underlying process behavior and embedded into a common latent space. The current implementation uses separate encoders for traces/logs, process trees, and Petri nets, projects them into a shared process-behavior latent space, and decodes the latent representation into a grammar-constrained process-tree representation. The decoded process tree is then converted deterministically into a Petri net using PM4Py. This design deliberately avoids direct arbitrary Petri-net generation in the first stage, because syntactic validity, soundness, and graph matching for generated Petri nets are harder than validity-preserving generation of block-structured process trees.

This contribution is best positioned as a **neural, multimodal process-artifact translation and embedding layer**, rather than as a direct replacement for mature process-discovery algorithms. Classical discovery techniques start from an event log and produce a model. ProcRosetta instead asks whether logs, trees, and nets can be encoded into a shared behavior-aware representation that supports retrieval, comparison, reconstruction, and cross-modal translation. The most distinctive aspects of the contribution are: (i) modality-specific encoders for the three major artifact types used in process mining; (ii) alignment of those modalities in one latent space; (iii) a process-tree decoder with grammar masking, so that generated outputs remain syntactically valid; and (iv) deterministic conversion from decoded process trees to Petri nets, thereby inheriting the validity advantages of a block-structured representation.

The related work is therefore broader than standard process discovery. It spans classical process-discovery algorithms, block-structured process models, expressive sound discovery formalisms, process-model similarity, event-log and model embeddings, neural discovery, multimodal representation learning, and grammar-constrained generation.

## Classical Process Discovery

The starting point for automated process discovery is the problem of deriving a process model from observed event data. The paper [Aalst2004] introduces one of the foundational discovery approaches, the alpha algorithm, which constructs Petri-net models from ordering relations extracted from event logs. This work established a direct connection between workflow mining and formal process-model synthesis, but it also made clear that purely relation-based discovery struggles with noise, incompleteness, short loops, duplicate labels, and complex control-flow constructs.

In [Weijters2006], the Heuristics Miner addresses some of these practical limitations by using frequency-based dependency measures rather than relying only on exact ordering relations. This made process discovery more robust on noisy real-world logs and helped shift the field from idealized complete logs toward practically useful discovery. However, heuristics-based discovery still focuses on a single input-output transformation: an event log is transformed into a process model, and the learned representation is not designed as a reusable latent space across several process-artifact modalities.

The paper [Gunther2007] introduces Fuzzy Mining, which targets highly unstructured or "spaghetti-like" processes by simplifying models according to significance and correlation metrics. Fuzzy Mining is important for real-world process analysis because it explicitly recognizes that a fully detailed discovered model may be unreadable. Its objective is interpretability and simplification rather than cross-modal representation learning.

A broader Petri-net-centered view is given in [AalstDongen2013], which surveys discovery of Petri nets from event logs and discusses alpha-style discovery, state-based region approaches, language-based region approaches, and practical issues such as noise and incompleteness. These methods are relevant to ProcRosetta because Petri nets are one of the target modalities. However, ProcRosetta does not directly synthesize arbitrary Petri nets. Instead, it decodes process trees and converts them to Petri nets, thereby trading Petri-net expressiveness for a simpler validity-preserving generation path.

The paper [Buijs2012] clarifies the quality dimensions that are still used to evaluate discovered models: fitness, precision, generalization, and simplicity. These criteria remain central for judging process-discovery systems. ProcRosetta's current evaluation is related but not identical: it measures whether latent distances agree with behavioral distances, whether cross-modal retrieval recovers corresponding artifacts, whether decoded process trees are valid, and whether simulated behavior from decoded models remains close to the source behavior. Thus, it complements standard discovery metrics with representation-learning metrics.

The survey and benchmark [Augusto2019Review] compares automated process-discovery algorithms on multiple real-life logs and quality metrics. It shows that no single discovery method dominates all quality dimensions and highlights persistent trade-offs among accuracy, simplicity, scalability, and robustness. ProcRosetta should therefore not be framed as surpassing the whole discovery state of the art at this stage. A more precise positioning is that it adds a latent multimodal layer that can be evaluated against discovery-oriented baselines and eventually combined with state-of-the-art miners.

## Block-Structured Process Models and Process Trees

Process trees are a key design choice in ProcRosetta. The paper [Leemans2013] introduces the Inductive Miner approach for discovering block-structured process models from event logs. Inductive Miner recursively decomposes a log into sublogs and constructs a process tree using operators such as sequence, exclusive choice, parallelism, and loop. A major advantage is that the resulting models are block-structured and sound by construction. This property is central for ProcRosetta: by decoding into a process-tree grammar rather than directly into arbitrary graph structures, the system can enforce syntactic well-formedness and then rely on deterministic conversion to Petri nets.

The overview [Leemans2026] summarizes process trees as hierarchical, compositional process models whose syntax induces block structure and whose semantics can be mapped to other process-modeling notations such as Petri nets and BPMN. This makes process trees a natural pivot representation for multimodal process translation: they are expressive enough for many structured control-flow patterns, easier to tokenize than arbitrary graphs, and have formal conversions into commonly used model types.

In [VanZelst2020], workflow nets are translated into process trees when such a translation is behaviorally possible. The algorithm identifies whether a workflow net corresponds to a process tree and, if it succeeds, returns a language-equivalent process tree. This work is closely related to ProcRosetta's cross-modal ambition because it connects Petri-net and process-tree representations. However, [VanZelst2020] is a symbolic algorithm with exact conditions, while ProcRosetta learns an approximate latent mapping that can translate noisy or sampled artifacts into a valid process tree even when exact symbolic inversion is not the immediate objective.

The PM4Py library described in [Berti2019] and [Berti2023PM4Py] provides practical infrastructure for event logs, process trees, Petri nets, simulation, conversion, and evaluation in Python. ProcRosetta relies on this ecosystem for deterministic tree-to-net conversion and for Petri-net handling. This is a pragmatic design choice: the neural model is used where learning is useful -- representation alignment and cross-modal decoding -- while mature symbolic process-mining tooling is used where formal conversions are already available.

Against this line of work, ProcRosetta's contribution is not that it invents a new block-structured process notation. Its novelty lies in using the process-tree formalism as the **validity-preserving output language** of a multimodal neural model. This places ProcRosetta close to Inductive Miner in terms of its structural bias, but different in objective: Inductive Miner is a deterministic discovery algorithm, while ProcRosetta learns a shared latent representation over logs, trees, and nets.

## Expressive and Sound Discovery Beyond Strict Block Structure

Block-structured process trees provide strong soundness and compositionality guarantees, but they are not the most expressive modeling formalism in current process-discovery research. The paper [Augusto2019Split] introduces Split Miner, which aims to discover accurate and simple process models by balancing fitness, precision, and complexity. Split Miner is important because it often produces compact and accurate models on practical logs, including behavior that may be less naturally represented by strictly nested process-tree blocks.

Recent work on partially ordered workflow models further extends the state of the art. The paper [Kourani2023POWL] introduces POWL, the Partially Ordered Workflow Language, as a sound process-modeling language that combines hierarchical operators with partial-order constructs. In [Kourani2023ScalablePOWL], this idea is connected to scalable discovery with formal guarantees. The journal version [Kourani2025POWL] further develops discovery of partially ordered workflow models, positioning POWL as a formalism that can express concurrency more flexibly than strictly tree-shaped block structure while still supporting soundness-preserving transformations.

The paper [Kourani2026] extends this direction with a discovery technique for expressive yet sound process models. It introduces POWL 2.0 with choice graphs, allowing more expressive decision structures and cyclic flows while preserving a hierarchical framework and soundness guarantees. This is particularly relevant for positioning ProcRosetta against the state of the art. On expressiveness, ProcRosetta's current process-tree target is narrower than POWL-style discovery and other advanced sound-discovery techniques. On representation learning, however, ProcRosetta addresses a different problem: it learns a shared vector space across logs, trees, and Petri nets and supports neural translation among artifact types.

This suggests a clear future direction. ProcRosetta's tree decoder could eventually be generalized to a richer sound target language such as POWL, provided that a grammar, canonicalization strategy, and deterministic conversion path are defined. In its current form, ProcRosetta chooses a conservative target formalism to make neural generation reliable. That is a reasonable first-stage engineering and research trade-off, but it should be acknowledged as a limitation relative to the most expressive sound discovery methods.

## Process-Model Similarity, Behavioral Distance, and Conformance

ProcRosetta evaluates whether learned embedding distances correlate with behavioral similarity. This connects it to a long line of work on comparing process models and measuring conformance.

The paper [VanDongen2008] proposes causal footprints as a way to measure similarity between business process models. It abstracts from concrete notation and compares models through behavioral relations. In [Dijkman2011], several similarity metrics for business process models are evaluated, including label, structural, and behavioral aspects. These works establish that process-model comparison cannot be reduced to graph isomorphism: two models may be structurally different but behaviorally close, or structurally similar but behaviorally different.

The paper [Kunze2011] defines behavioral similarity as a proper metric based on behavioral profiles. This is relevant for ProcRosetta because the project explicitly tries to make latent geometry reflect process behavior rather than merely structural resemblance. If two different process trees or Petri nets generate similar trace behavior, a behavior-aware latent space should place them close together even when their syntax differs.

Conformance checking provides another perspective. The alignment-based methods represented by [Adriansyah2012] compare observed log behavior with model behavior by finding optimal correspondences between traces and model executions. The paper [Leemans2019EMD] introduces Earth Movers' stochastic conformance checking, which compares stochastic behavior using distributional distance. These methods are more exact and interpretable than a learned latent embedding, but they are also explicit comparison procedures rather than a reusable representation over multiple artifact types.

ProcRosetta can be viewed as learning an approximate, continuous proxy for behavioral comparison. Instead of computing a symbolic or alignment-based distance for every pair of artifacts at query time, it maps each artifact into a vector space where distance should correlate with behavior. This is valuable for retrieval, clustering, anomaly search, and cross-modal matching. The trade-off is that learned embeddings do not provide the same formal guarantees as exact conformance or behavioral-profile metrics. Therefore, ProcRosetta should be evaluated not only by reconstruction accuracy, but also by rank correlation between latent distance and behavioral distance, nearest-neighbor behavioral quality, and cross-modal retrieval accuracy.

## Representation Learning for Event Logs and Process Models

Representation learning has already been applied to process-mining artifacts, but most existing work is single-modality or task-specific. The paper [DeKoninck2018] introduces act2vec, trace2vec, log2vec, and model2vec, adapting distributional representation-learning ideas to activities, traces, logs, and models. This work is one of the closest predecessors to ProcRosetta because it explicitly considers vector representations at multiple process granularity levels. However, the main focus is embedding artifacts for downstream tasks such as clustering, process comparison, predictive monitoring, and anomaly detection, rather than training a unified multimodal encoder-decoder that translates between event logs, process trees, and Petri nets.

The paper [Pfeiffer2022] also argues for business process representation learning as a generic layer for process-mining tasks. It emphasizes that learned event-log representations can support multiple downstream analyses. Surveys such as [Barbon2023] and [Rullo2025] show how many trace-encoding methods exist and how choices of representation can significantly affect predictive-monitoring and process-mining results. This literature is relevant because ProcRosetta's log encoder is one component in a broader representation-learning stack. However, trace encoding alone does not solve the multimodal translation problem: it does not usually align logs with process trees and Petri nets in the same latent space, nor does it generate a valid process model as output.

The predictive-monitoring literature demonstrates the power of deep sequence models on event logs. The paper [Tax2017] applies LSTM neural networks to predict next events and timestamps. The paper [Bukhsh2021] introduces a Transformer-based architecture for predictive business process monitoring. The review and benchmark [RamaManeiro2023] surveys deep-learning approaches for predictive business process monitoring and compares them empirically. These methods learn useful representations of traces, but their objective is prediction over continuing cases rather than discovery, model translation, or multimodal alignment.

ProcRosetta's state-of-the-art positioning relative to this literature is therefore as follows. It borrows the general idea that event logs and process artifacts can be embedded into vector spaces, but it extends the objective from **single-modality prediction or comparison** to **cross-modal representation and generation**. Its embeddings are trained so that equivalent logs, trees, and nets are close, and its decoder is trained to recover a process tree from any of those modalities. This makes ProcRosetta more ambitious than conventional trace encoding, while also more constrained than task-specific predictive models trained on large real-world event logs.

## Petri-Net and Graph Embeddings

Petri nets are graph-structured objects, so graph representation learning is an important background area. The paper [Perozzi2014] introduces DeepWalk, which learns node embeddings by treating random walks over a graph as sentences. The paper [Grover2016] introduces node2vec, which generalizes this idea with biased random walks that interpolate between breadth-first and depth-first exploration. The paper [Narayanan2017] introduces graph2vec, extending representation learning from nodes to whole graphs. The paper [Le2014] introduces paragraph vectors, a precursor to document-level embeddings that influenced several graph and process-embedding approaches.

Graph neural networks provide another family of graph encoders. The paper [Kipf2017] introduces graph convolutional networks for semi-supervised learning on graph-structured data. The survey [Battaglia2018] frames graph networks in terms of relational inductive biases, emphasizing why neural architectures should respect object-relation structure. ProcRosetta's Petri-net encoder follows this broad idea: places, transitions, and arcs form a structured object, and a graph-aware encoder is more appropriate than a flat feature vector when learning Petri-net representations.

The most directly related Petri-net embedding work is [Colonna2024], which introduces process-mining embeddings for Petri nets. It learns vector representations of Petri nets using an unsupervised, document-embedding-inspired approach and evaluates them for comparison, clustering, classification, and retrieval. This line of work is an important baseline for ProcRosetta because it already addresses learned Petri-net representations. The difference is that [Colonna2024] focuses on embedding Petri nets as a modality in their own right, whereas ProcRosetta aligns Petri-net embeddings with event-log and process-tree embeddings and uses the shared latent vector to decode a valid process tree.

Thus, relative to PetriNet2Vec-style baselines, ProcRosetta contributes **cross-modal semantics** and **generativity**. It does not merely ask whether two Petri nets are close in an embedding space. It asks whether a Petri net, its corresponding event log, and its corresponding process tree map to compatible latent vectors, and whether each of those vectors can generate a valid process tree. This is a stronger multimodal objective, but it is also harder to train and currently relies on synthetic paired triples.

## Neural Process Discovery and Learned Process-Model Generation

Neural process discovery attempts to learn mappings from data to process models. The paper [Sommers2021] uses graph neural networks for process discovery, training on synthetic log-model pairs to produce sound Petri nets. This is one of the closest neural predecessors to ProcRosetta because it treats discovery as a supervised learning problem and uses synthetic data to provide paired examples. However, its target is primarily log-to-model discovery, whereas ProcRosetta introduces a tri-modal setting: logs, process trees, and Petri nets are all encoded and aligned.

Large language models have recently been applied to process modeling from natural language. The paper [Kourani2024LLM] proposes generating process models from textual descriptions with quality guarantees, using formal intermediate representations. The benchmark [Kourani2025LLMEval] evaluates large language models on business process modeling tasks and studies variation in model quality and self-improvement. The paper [Berti2025RL] specializes large language models for process modeling using reinforcement learning with verifiable rewards based on structural and behavioral checks. These works show that learned generative models are becoming relevant for process modeling, but their input modality is natural language rather than logs, trees, and nets.

ProcRosetta differs from LLM-based process modeling in three ways. First, it is artifact-native: its inputs are process-mining artifacts rather than textual descriptions. Second, it is explicitly multimodal: the same target latent space is shared by logs, process trees, and Petri nets. Third, its decoder is grammar-constrained toward process-tree syntax, whereas LLM-based approaches often rely on text-to-formal-model generation pipelines and post-hoc validation or repair. These differences make ProcRosetta complementary rather than competitive with LLM-based process modeling. A plausible future integration would use an LLM to parse natural-language process descriptions into the same latent space or into the same process-tree/POWL target language.

Relative to neural discovery, ProcRosetta should be positioned carefully. It is not yet a mature end-to-end replacement for Inductive Miner, Split Miner, POWL discovery, or GNN-based discovery on real logs. Its first-stage contribution is a multimodal architecture and evaluation protocol showing that learned latent alignment across process artifacts is feasible. The next step toward state-of-the-art discovery performance would require larger, more diverse, and real-world training data; stronger behavioral objectives; and comparisons on standard discovery benchmarks.

## Multimodal Representation Learning and Cross-Modal Alignment

Outside process mining, multimodal representation learning provides the conceptual foundation for ProcRosetta. The paper [Ngiam2011] studies multimodal deep learning and cross-modality learning, showing that neural models can learn shared representations across different sensory modalities. The survey [Baltrusaitis2019] organizes multimodal machine learning into major challenges such as representation, translation, alignment, fusion, and co-learning. These categories map directly onto ProcRosetta: representation corresponds to the shared process-behavior latent space; translation corresponds to decoding logs or nets into process trees; alignment corresponds to making equivalent artifacts close; and fusion corresponds to combining modalities when several artifact views are available.

The paper [Radford2021] introduces CLIP, a contrastive image-text model that aligns images and natural-language captions in a shared embedding space. Although CLIP is not a process-mining model, it is relevant as a general example of paired multimodal alignment at scale. ProcRosetta adapts the same broad principle to process mining: paired artifacts generated from the same underlying process should have nearby embeddings, while artifacts generated from different processes should be separable.

The process-mining challenge is that modalities are not independent sensory streams but formal and semi-formal views of process behavior. An event log is a sample of executions, a process tree is a block-structured generative specification, and a Petri net is a graph-based execution model. These views can be behaviorally equivalent despite being structurally different. Therefore, ProcRosetta's latent space must learn behavioral equivalence rather than superficial syntactic similarity. This makes the problem more specialized than generic multimodal alignment and explains why behavioral-distance evaluation is essential.

## Grammar-Constrained Neural Decoding and Validity Preservation

Neural generation of structured objects often suffers from invalid outputs. Grammar-constrained generation addresses this by restricting the decoder to syntactically legal sequences. The paper [Kusner2017] introduces the Grammar Variational Autoencoder, which decodes according to a context-free grammar so that generated outputs are syntactically valid. The paper [Dai2018] extends this idea with syntax-directed variational autoencoders, incorporating semantic constraints through syntax-directed definitions. In [Geng2023], grammar-constrained decoding is applied to structured NLP tasks by using a grammar to restrict valid next tokens during decoding.

ProcRosetta applies this principle to process-tree generation. Instead of decoding arbitrary strings or arbitrary Petri-net graphs, it decodes a tokenized process-tree grammar with masks over valid next tokens. This design directly addresses a common failure mode of neural process-model generation: syntactically invalid or structurally malformed outputs. Once the decoded tree is valid, PM4Py can convert it into a Petri net.

This design is state-of-the-art in spirit, even though the target formalism is conservative. It follows the same validity-preserving philosophy as grammar-based molecular generation and grammar-constrained program generation, but applies it to process-mining models. The limitation is that syntactic validity is not the same as behavioral correctness. A decoded process tree can be well formed and convertible to a Petri net while still not matching the intended source process. For this reason, ProcRosetta's evaluation must include behavioral distance and not only grammar-validity or conversion-success metrics.

## Synthetic Data, Paired Triples, and Benchmarking

Synthetic data is common in neural process discovery because supervised training requires paired logs and ground-truth models. The GNN-based discovery approach [Sommers2021] uses synthetic log-model pairs for training, and process-discovery competitions have also used controlled generated models and logs to benchmark discovery algorithms. ProcRosetta similarly starts from synthetic paired triples: a process tree, a simulated event log, and a converted Petri net.

This is a reasonable first-stage methodology because it provides exact cross-modal supervision. Without paired triples, it would be difficult to train a model to know that a log sample, a tree, and a Petri net represent the same process. Synthetic triples also make it possible to canonicalize activity labels, balance model structures, and compute ground-truth reconstruction metrics.

However, synthetic training is also a limitation. Real event logs contain noise, missing behavior, infrequent paths, duplicate labels, resource and timestamp information, concept drift, lifecycle events, and data attributes. Real Petri nets and BPMN models may include non-block-structured fragments and modeling conventions not represented in a synthetic tree grammar. Therefore, ProcRosetta's current results should be interpreted as evidence for architectural feasibility rather than definitive state-of-the-art discovery performance. A stronger empirical claim would require evaluation on public real-life logs, discovery-contest datasets, and model collections beyond the synthetic block-structured distribution.

## Overall Positioning Against the State of the Art

ProcRosetta occupies a specific and promising niche between process discovery, process-model embeddings, and multimodal neural representation learning.

Against **classical process discovery** such as [Aalst2004], [Weijters2006], [Gunther2007], [Leemans2013], and [Augusto2019Split], ProcRosetta is not primarily another deterministic miner. Its contribution is a learned latent representation that supports cross-modal retrieval and translation. Classical miners generally produce one model from one log; ProcRosetta tries to make logs, trees, and nets mutually translatable through a shared behavior space.

Against **block-structured discovery** such as Inductive Miner [Leemans2013], ProcRosetta shares the process-tree bias and the desire for valid, sound, structured outputs. The difference is methodological: Inductive Miner derives a tree through recursive log decomposition, whereas ProcRosetta learns encoders and a grammar-masked decoder. ProcRosetta therefore gains the ability to embed and compare heterogeneous artifacts, but it gives up the interpretability and formal rediscovery guarantees of deterministic discovery.

Against **more expressive sound modeling formalisms** such as POWL [Kourani2023POWL], [Kourani2025POWL], and POWL 2.0 [Kourani2026], ProcRosetta is currently less expressive because it targets process trees. This is its main state-of-the-art gap. The positive interpretation is that the architecture is modular: a future version could replace the process-tree grammar with a richer sound grammar while retaining the multimodal latent-space objective.

Against **behavioral similarity and conformance methods** such as [VanDongen2008], [Dijkman2011], [Kunze2011], [Adriansyah2012], and [Leemans2019EMD], ProcRosetta does not provide exact symbolic guarantees. Instead, it attempts to learn a geometry in which distance approximates behavioral similarity. This is useful for scalable retrieval and downstream machine-learning tasks, but exact conformance metrics remain necessary for rigorous validation.

Against **process and Petri-net embeddings** such as [DeKoninck2018] and [Colonna2024], ProcRosetta extends the goal from embedding one artifact family to aligning several artifact families. In particular, [Colonna2024] is a strong Petri-net embedding baseline, but ProcRosetta adds event-log and process-tree encoders and a generative process-tree decoder. This makes ProcRosetta closer to a process-artifact "Rosetta stone" than to a single-modality embedding model.

Against **neural process discovery** such as [Sommers2021], ProcRosetta is broader in modalities but more conservative in output expressiveness. [Sommers2021] learns log-to-Petri-net discovery, while ProcRosetta learns log/tree/net-to-tree translation. The process-tree pivot improves validity control, but it restricts the output space to block-structured behavior.

Against **LLM-based process modeling** such as [Kourani2024LLM], [Kourani2025LLMEval], and [Berti2025RL], ProcRosetta is artifact-native rather than language-native. It does not attempt to transform natural-language descriptions into process models. Instead, it aligns formal and behavioral process-mining artifacts. These directions are complementary: LLMs can help with textual process descriptions, while ProcRosetta can help with artifact comparison, retrieval, and translation.

In summary, the strongest state-of-the-art claim for ProcRosetta is not that it outperforms mature discovery algorithms on all discovery metrics. The stronger and more defensible claim is that it introduces a **multimodal, validity-aware neural representation framework for process-mining artifacts**. Its state-of-the-art relevance comes from combining ideas that are usually separate: process-tree soundness, Petri-net graph encoding, event-log representation learning, behavioral embedding evaluation, contrastive multimodal alignment, and grammar-constrained structured decoding. Its main limitations are the current block-structured output bias, reliance on synthetic paired triples, lack of direct arbitrary Petri-net decoding, and the need for broader real-world benchmarking.

## Suggested Framing for a Paper Contribution

A concise contribution statement could be formulated as follows:

> ProcRosetta introduces a multimodal neural framework for process-mining artifacts that embeds event logs, process trees, and Petri nets into a shared behavior-oriented latent space and decodes latent representations into grammar-valid process trees, which can be deterministically converted to Petri nets. Unlike classical process discovery, which maps logs to models, ProcRosetta supports cross-modal retrieval, reconstruction, and translation among multiple process representations. Unlike existing Petri-net or trace-embedding methods, it learns an aligned latent space across modalities and couples representation learning with validity-preserving model generation.

The recommended positioning is therefore:

1. **Not a replacement for deterministic discovery yet**, because state-of-the-art miners have stronger guarantees and broader empirical validation.
2. **A new multimodal representation-learning layer**, because it aligns logs, process trees, and Petri nets in one behavioral latent space.
3. **A validity-aware neural generation approach**, because grammar-masked decoding and process-tree-to-Petri-net conversion reduce invalid model generation.
4. **A foundation for future expressive neural discovery**, because the process-tree decoder could later be replaced or extended by richer sound formalisms such as POWL.

# BIBLIOGRAPHY

[Aalst2004] W. M. P. van der Aalst, T. Weijters, and L. Maruster, "Workflow Mining: Discovering Process Models from Event Logs," *IEEE Transactions on Knowledge and Data Engineering*, vol. 16, no. 9, pp. 1128--1142, 2004. DOI: 10.1109/TKDE.2004.47.

[AalstDongen2013] W. M. P. van der Aalst and B. F. van Dongen, "Discovering Petri Nets from Event Logs," in *Transactions on Petri Nets and Other Models of Concurrency VII*, Lecture Notes in Computer Science, vol. 7480, Springer, 2013, pp. 372--422. DOI: 10.1007/978-3-642-38143-0_10.

[Adriansyah2012] A. Adriansyah, J. Munoz-Gama, J. Carmona, B. F. van Dongen, and W. M. P. van der Aalst, "Alignment Based Precision Checking," in *Business Process Management Workshops*, Lecture Notes in Business Information Processing, vol. 132, Springer, 2012, pp. 137--149.

[Augusto2019Review] A. Augusto, R. Conforti, M. Dumas, M. La Rosa, F. M. Maggi, A. Marrella, M. Mecella, and A. Soo, "Automated Discovery of Process Models from Event Logs: Review and Benchmark," *IEEE Transactions on Knowledge and Data Engineering*, vol. 31, no. 4, pp. 686--705, 2019. DOI: 10.1109/TKDE.2018.2841877.

[Augusto2019Split] A. Augusto, R. Conforti, M. Dumas, M. La Rosa, and A. Polyvyanyy, "Split Miner: Automated Discovery of Accurate and Simple Business Process Models from Event Logs," *Knowledge and Information Systems*, vol. 59, pp. 251--284, 2019. DOI: 10.1007/s10115-018-1214-x.

[Baltrusaitis2019] T. Baltrusaitis, C. Ahuja, and L.-P. Morency, "Multimodal Machine Learning: A Survey and Taxonomy," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 41, no. 2, pp. 423--443, 2019. DOI: 10.1109/TPAMI.2018.2798607.

[Barbon2023] S. Barbon Jr., P. Ceravolo, R. S. Oyamada, and G. M. Tavares, "Trace Encoding in Process Mining: A Survey and Benchmarking," arXiv:2301.02167, 2023.

[Battaglia2018] P. W. Battaglia, J. B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi, M. Malinowski, A. Tacchetti, D. Raposo, A. Santoro, R. Faulkner, and others, "Relational Inductive Biases, Deep Learning, and Graph Networks," arXiv:1806.01261, 2018.

[Berti2019] A. Berti, S. J. van Zelst, and W. M. P. van der Aalst, "Process Mining for Python (PM4Py): Bridging the Gap Between Process- and Data Science," in *Proceedings of the ICPM Demo Track 2019*, CEUR Workshop Proceedings, vol. 2374, pp. 13--16, 2019.

[Berti2023PM4Py] A. Berti, S. J. van Zelst, and D. Schuster, "PM4Py: A Process Mining Library for Python," *Software Impacts*, vol. 17, article 100556, 2023. DOI: 10.1016/j.simpa.2023.100556.

[Berti2025RL] A. Berti, X. Wang, H. Kourani, and W. M. P. van der Aalst, "Specializing Large Language Models for Process Modeling via Reinforcement Learning with Verifiable and Universal Rewards," *Process Science*, 2025. DOI: 10.1007/s44311-025-00034-4.

[Buijs2012] J. C. A. M. Buijs, B. F. van Dongen, and W. M. P. van der Aalst, "On the Role of Fitness, Precision, Generalization and Simplicity in Process Discovery," in *On the Move to Meaningful Internet Systems: OTM 2012*, Lecture Notes in Computer Science, vol. 7565, Springer, 2012, pp. 305--322. DOI: 10.1007/978-3-642-33606-5_19.

[Bukhsh2021] Z. A. Bukhsh, A. Saeed, and R. M. Dijkman, "ProcessTransformer: Predictive Business Process Monitoring with Transformer Network," arXiv:2104.00721, 2021.

[Colonna2024] J. G. Colonna, A. A. Fares, M. Duarte, and R. Sousa, "Process Mining Embeddings: Learning Vector Representations for Petri Nets," arXiv:2404.17129, 2024.

[Dai2018] H. Dai, Y. Tian, B. Dai, S. Skiena, and L. Song, "Syntax-Directed Variational Autoencoder for Structured Data," in *International Conference on Learning Representations*, 2018.

[DeKoninck2018] P. De Koninck, S. vanden Broucke, and J. De Weerdt, "act2vec, trace2vec, log2vec, and model2vec: Representation Learning for Business Processes," in *Business Process Management*, Lecture Notes in Computer Science, vol. 11080, Springer, 2018, pp. 305--321. DOI: 10.1007/978-3-319-98648-7_18.

[Dijkman2011] R. M. Dijkman, M. Dumas, B. F. van Dongen, R. Käärik, and J. Mendling, "Similarity of Business Process Models: Metrics and Evaluation," *Information Systems*, vol. 36, no. 2, pp. 498--516, 2011. DOI: 10.1016/j.is.2010.09.006.

[Geng2023] S. Geng, M. Josifoski, M. Peyrard, and R. West, "Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning," in *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 2023.

[Grover2016] A. Grover and J. Leskovec, "node2vec: Scalable Feature Learning for Networks," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 855--864. DOI: 10.1145/2939672.2939754.

[Gunther2007] C. W. Günther and W. M. P. van der Aalst, "Fuzzy Mining: Adaptive Process Simplification Based on Multi-perspective Metrics," in *Business Process Management*, Lecture Notes in Computer Science, vol. 4714, Springer, 2007, pp. 328--343. DOI: 10.1007/978-3-540-75183-0_24.

[Kipf2017] T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," in *International Conference on Learning Representations*, 2017.

[Kourani2023POWL] H. Kourani and S. J. van Zelst, "POWL: Partially Ordered Workflow Language," in *Business Process Management*, Lecture Notes in Computer Science, Springer, 2023, pp. 92--108. DOI: 10.1007/978-3-031-41620-0_6.

[Kourani2023ScalablePOWL] H. Kourani, D. Schuster, and W. M. P. van der Aalst, "Scalable Discovery of Partially Ordered Workflow Models with Formal Guarantees," in *International Conference on Process Mining*, IEEE, 2023, pp. 89--96. DOI: 10.1109/ICPM60904.2023.10271941.

[Kourani2024LLM] H. Kourani, A. Berti, D. Schuster, and W. M. P. van der Aalst, "Process Modeling With Large Language Models," in *Enterprise, Business-Process and Information Systems Modeling*, Springer, 2024, pp. 229--244. DOI: 10.1007/978-3-031-61007-3_18.

[Kourani2025LLMEval] H. Kourani, A. Berti, D. Schuster, and W. M. P. van der Aalst, "Evaluating Large Language Models on Business Process Modeling: Framework, Benchmark, and Self-Improvement Analysis," *Software and Systems Modeling*, 2025. DOI: 10.1007/s10270-025-01318-w.

[Kourani2025POWL] H. Kourani, S. J. van Zelst, D. Schuster, and W. M. P. van der Aalst, "Discovering Partially Ordered Workflow Models," *Information Systems*, vol. 128, article 102493, 2025. DOI: 10.1016/j.is.2024.102493.

[Kourani2026] H. Kourani, G. Park, and W. M. P. van der Aalst, "A Discovery Technique for Expressive Yet Sound Process Models," *Process Science*, vol. 3, article 14, 2026. DOI: 10.1007/s44311-026-00046-8.

[Kusner2017] M. J. Kusner, B. Paige, and J. M. Hernández-Lobato, "Grammar Variational Autoencoder," in *Proceedings of the 34th International Conference on Machine Learning*, PMLR, vol. 70, 2017, pp. 1945--1954.

[Kunze2011] M. Kunze, M. Weidlich, and M. Weske, "Behavioral Similarity -- A Proper Metric," in *Business Process Management*, Lecture Notes in Computer Science, vol. 6896, Springer, 2011, pp. 166--181.

[Le2014] Q. V. Le and T. Mikolov, "Distributed Representations of Sentences and Documents," in *Proceedings of the 31st International Conference on Machine Learning*, PMLR, vol. 32, 2014, pp. 1188--1196.

[Leemans2013] S. J. J. Leemans, D. Fahland, and W. M. P. van der Aalst, "Discovering Block-Structured Process Models from Event Logs: A Constructive Approach," in *Application and Theory of Petri Nets and Concurrency*, Lecture Notes in Computer Science, vol. 7927, Springer, 2013, pp. 311--329. DOI: 10.1007/978-3-642-38697-8_17.

[Leemans2019EMD] S. J. J. Leemans, A. F. Syring, and W. M. P. van der Aalst, "Earth Movers' Stochastic Conformance Checking," in *Business Process Management Forum*, Lecture Notes in Business Information Processing, vol. 360, Springer, 2019, pp. 127--143. DOI: 10.1007/978-3-030-26643-1_8.

[Leemans2026] S. J. J. Leemans, S. J. van Zelst, and X. Lu, "A Brief Overview of Process Trees," 2026.

[Narayanan2017] A. Narayanan, M. Chandramohan, R. Venkatesan, L. Chen, Y. Liu, and S. Jaiswal, "graph2vec: Learning Distributed Representations of Graphs," arXiv:1707.05005, 2017.

[Ngiam2011] J. Ngiam, A. Khosla, M. Kim, J. Nam, H. Lee, and A. Y. Ng, "Multimodal Deep Learning," in *Proceedings of the 28th International Conference on Machine Learning*, 2011, pp. 689--696.

[Perozzi2014] B. Perozzi, R. Al-Rfou, and S. Skiena, "DeepWalk: Online Learning of Social Representations," in *Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2014, pp. 701--710.

[Pfeiffer2022] P. Pfeiffer, "Business Process Representation Learning," in *Proceedings of the BPM 2022 Workshops*, CEUR Workshop Proceedings, 2022.

[Radford2021] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning Transferable Visual Models from Natural Language Supervision," in *Proceedings of the 38th International Conference on Machine Learning*, PMLR, vol. 139, 2021, pp. 8748--8763.

[RamaManeiro2023] E. Rama-Maneiro, J. C. Vidal, and M. Lama, "Deep Learning for Predictive Business Process Monitoring: Review and Benchmark," *IEEE Transactions on Services Computing*, vol. 16, no. 1, pp. 739--756, 2023. DOI: 10.1109/TSC.2021.3139807.

[Rullo2025] A. Rullo, F. Alam, and E. Serra, "Trace Encoding Techniques for Multi-Perspective Process Mining: A Comparative Study," *WIREs Data Mining and Knowledge Discovery*, vol. 15, no. 1, 2025. DOI: 10.1002/widm.1573.

[Sommers2021] D. Sommers, V. Menkovski, and D. Fahland, "Process Discovery Using Graph Neural Networks," in *International Conference on Process Mining*, IEEE, 2021.

[Tax2017] N. Tax, I. Verenich, M. La Rosa, and M. Dumas, "Predictive Business Process Monitoring with LSTM Neural Networks," in *Advanced Information Systems Engineering*, Lecture Notes in Computer Science, vol. 10253, Springer, 2017, pp. 477--492.

[VanDongen2008] B. F. van Dongen, R. M. Dijkman, and J. Mendling, "Measuring Similarity Between Business Process Models," in *Advanced Information Systems Engineering*, Lecture Notes in Computer Science, vol. 5074, Springer, 2008, pp. 450--464.

[VanZelst2020] S. J. van Zelst and S. J. J. Leemans, "Translating Workflow Nets to Process Trees: An Algorithmic Approach," *Algorithms*, vol. 13, no. 11, article 279, 2020. DOI: 10.3390/a13110279.

[Weijters2006] A. J. M. M. Weijters, W. M. P. van der Aalst, and A. K. Alves de Medeiros, "Process Mining with the HeuristicsMiner Algorithm," BETA Working Paper Series, WP 166, Eindhoven University of Technology, 2006.
