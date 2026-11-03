# Review 2: 5-minute speaking script

Target: about 5 minutes, roughly 650 words. Times are cumulative. The **bold** phrases are the ones to say even if you're short on time.

---

## Slide 1: Title (0:00–0:10)
"Good morning. Our project is **VERITAS: Evidence-Driven Generative AI for Reliable Research**."

## Slide 2: Review 1 feedback (0:10–0:40)
"In Review 1, the panel asked us to **treat VERITAS as research, not only as a system to build**. We did four things.
- We reframed it around **five research questions**.
- We adopted a standard benchmark, **SciFact**, which has 300 expert-labelled scientific claims.
- We added baselines and ablations.
- We now report quantitative metrics.

The bottom row shows RQ1 to RQ5. Today I'll present results for RQ2 and a preview of RQ4."

## Slide 3: Problem statement (0:40–0:55)
"The problem is unchanged. **LLMs produce unsupported information with high confidence**, and current tools don't check individual claims or detect conflicts between sources. Our solution is grounded generation followed by claim-level verification."

## Slide 4: Refined pipeline (0:55–1:35)
"This is the refined pipeline, now with the actual models we implemented. There are three stages.
1. **Indexing:** uploaded documents are chunked and embedded with MiniLM into ChromaDB.
2. **Grounded generation:** we retrieve the top 20 passages, re-rank them with a cross-encoder down to 5, with at most 2 per document, and gpt-oss-120b writes an answer that cites every sentence.
3. **Verification:** the answer is split into atomic claims. Each claim is checked by **DeBERTa-v3 NLI** against the evidence and by **self-consistency** across 4 re-sampled answers. Then it is scored.

It's a **fixed, single-pass pipeline**, with no agents, planning or tool use."

## Slide 5: Concept justification (1:35–2:45), worth 10 marks, so don't rush
"Each concept layer builds on the one before it.
- **ML:** our confidence score is a weighted fusion of signals with decision thresholds, and we evaluate with accuracy, F1 and AUROC.
- **DL:** neural embeddings let us retrieve evidence by meaning, not keywords.
- **NLP:** chunking, claim decomposition, and **natural language inference** itself, which is a core NLP task.
- **Transformers:** we use them in four different roles. A **bi-encoder** for retrieval, a **cross-encoder** for re-ranking, an **encoder**, DeBERTa, for verification, and a **decoder**, the LLM, for generation.
- **GenAI:** retrieval-augmented generation with citations, **temperature sampling** for self-consistency, prompt engineering, and a one-pass self-correction.

All of the work happens inside models, which is why this is Generative AI and not Agentic AI."

## Slide 6: Generated content and analysis (2:45–3:55)
"On the left is **real output** from our system on 7 papers about LLM hallucination. Every claim has a verdict, a confidence score and a source page. Note the last claim. **The evidence fully supports it, with NLI at 1.00, but only 31% of the re-sampled answers repeat it**, so its confidence drops to 0.72. That is **self-consistency catching uncertainty that NLI alone would miss.**

On the right, each objective is linked to a measurement:
- **O1:** all 6 claims in this demo verified against their sources.
- **O2:** on SciFact, our NLI verifier reaches **0.70 accuracy** against expert labels. Random guessing would give 0.33.
- **O3:** contradiction recall is 0.59. This is our weakest point.
- **O4:** an **AUROC of 0.86**, so the confidence score separates supported from unsupported claims well.

Our analysis: most errors are cautious ones, and contradictions are the hardest, so our next step is to fine-tune the verifier on SciFact."

## Slide 7: Prompt engineering (3:55–4:55)
"We use eight prompt types across the pipeline: role-play for the system persona, instructional for grounding, completion with few-shot for claim extraction, classification for the LLM-judge baseline, reasoning for explanations, comparison for conflicts, information for the closed-book baseline, and creative for lay summaries.

The bottom row is the use case: **the same questions and the same evidence, with only the prompt changed.** A vague prompt hallucinated **74%** of the time. Our role-plus-instruction prompt hallucinated **0%** on these questions. **Prompt design alone removes almost all of the hallucination.** Chain-of-thought also reached 0%, but it gave much shorter answers, with only 6 claims, so there's a trade-off."

## Close (4:55–5:00)
"Next steps: the full RQ1, RQ3 and RQ5 experiments, and fine-tuning the verifier. Thank you."

---

## Likely questions and short answers

**Q: Isn't this agentic AI?**
No. The pipeline is fixed and runs once. The LLM never decides what to do next, which tool to use, or whether to loop. It only generates text, which we then verify.

**Q: Why NLI instead of just asking the LLM whether it's correct?**
An LLM judging its own output shares its own biases. NLI is a separate model trained specifically for entailment, and it's cheap enough to run locally. We also built an LLM-as-judge baseline to compare against it.

**Q: What is self-consistency?**
We ask the LLM the same question 4 more times at temperature 0.8. A fact the model actually knows reappears across samples, while a hallucinated detail usually changes. This method comes from SelfCheckGPT (Manakul et al., 2023).

**Q: How is the confidence score calculated?**
0.6 × NLI entailment + 0.4 × self-consistency − 0.5 × contradiction, clipped to the range 0–1. A claim is verified at 0.70 or above and unsupported below 0.40.

**Q: Why is accuracy only 0.70?**
The verifier reads whole abstracts, while the experts labelled specific sentences, and 3-way classification is hard. Most errors are cautious "not enough info" errors, which is the safer failure for a hallucination detector. Fine-tuning on SciFact is our next step.

**Q: Why is the 74% → 0% result only on 3 questions?**
It's a preview. The full RQ4 run uses 20 or more SciFact questions and is scheduled for the next review.

**Q: Which LLM do you use, and why?**
gpt-oss-120b, an open-weight model, served through Groq's free API. We'll compare it with gpt-oss-20b for RQ5, the model-size question.

**Q: What are the limitations?**
- Contradiction detection (recall 0.59).
- Text-only PDFs, so figures and tables aren't read.
- English only.
- Free-tier rate limits make a full run with self-consistency take about 1–2 minutes.

**Q: Did you find any bugs while building it?**
Yes. NLI sometimes labels a passage that just *doesn't mention* a fact as a "contradiction". At first our scoring took the strongest contradiction from any passage, which wrongly flagged true claims. Now, if a passage clearly supports the claim, we judge the claim by that passage, and real disagreement between different documents is handled by conflict detection. This fix changed the prompt results from 79%/6% to 74%/0%.
