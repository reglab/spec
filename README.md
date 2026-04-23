# Learning When Not to Decide: A Framework for Overcoming Factual Presumptuousness in AI Adjudication

## Mohamed Afane, Emily Robitschek, Derek Ouyang, Daniel E. Ho

This is the code repository for ["Learning When Not to Decide: A Framework for Overcoming Factual Presumptuousness in AI Adjudication."](https://arxiv.org/abs/2604.19895)

A well-known limitation of AI systems is *presumptuousness*: the tendency to provide confident answers when information may be lacking. This challenge is particularly acute in legal applications, where a core task for attorneys, judges, and administrators is to determine whether evidence is sufficient to reach a conclusion. We study this problem in the important setting of unemployment insurance adjudication, which has seen rapid integration of AI systems and where the question of additional fact-finding poses the most significant bottleneck for a system that affects millions of applicants annually. We evaluate four leading AI platforms and show that standard RAG-based approaches achieve an average of only 15% accuracy when information is insufficient. We introduce SPEC (Structured Prompting for Evidence Checklists), a multi-stage framework requiring explicit identification of missing information before any determination. SPEC achieves 89% overall accuracy, while appropriately deferring when evidence is insufficient.

<p align="center">
  <img src="figures/spec_pipeline.png" width="100%" />
</p>

```
@inproceedings{afane2026spec,
  author    = {Mohamed Afane and Emily Robitschek and Derek Ouyang and Daniel E. Ho},
  title     = {Learning When Not to Decide: A Framework for Overcoming Factual Presumptuousness in {AI} Adjudication},
  booktitle = {Proceedings of the Twenty-First International Conference on Artificial Intelligence and Law (ICAIL '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {Singapore}
}
```

## Repository

This code repository is structured as follows.

* `backend_core/pipeline.py`: The SPEC multi-agent pipeline, implementing the three-agent architecture (Requirements Checklist, Fact Verification, Supervisory Review) with dynamic prompt generation.
* `backend_core/retriever.py`: RAG retrieval system for extracting relevant statutory provisions, administrative regulations, and adjudication guidance as complete passages.
* `backend_core/prompts.py`: Prompt templates for system prompts and the planner that generates dynamic, stage-specific instructions for each agent.
* `benchmarking/run_benchmark.py`: Evaluation script for running the benchmark dataset across models and prompting configurations.
* `benchmarking/UI Benchmark.xlsx`: The benchmark dataset of 250 questions based on Colorado UI law, including eligible, ineligible, and inconclusive cases with systematically varied information completeness.

## Dataset

We release our benchmark dataset of 250 carefully constructed questions based on Colorado unemployment insurance law. The dataset includes:

* **Complete cases** (44%): Cases containing all facts necessary for an eligibility determination (eligible or ineligible).
* **Inconclusive cases** (56%): Cases where one to four critical facts are intentionally withheld, testing whether systems appropriately defer judgment.

Questions address scenarios involving voluntary quit, discharge for misconduct, availability for work, and other eligibility criteria under Colorado Revised Statutes. The dataset is included in this repository as `benchmarking/UI Benchmark.xlsx`.
