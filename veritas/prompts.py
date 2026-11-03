"""Phase 3b: every prompt used by VERITAS, in one place.

Each prompt is tagged with the prompt-engineering type it demonstrates
(Role-play, Instructional, Information, Completion, Classification, Reasoning,
Comparison, Creative). The answer-generation prompts also come in several
strategies so RQ4 can measure how prompt design affects hallucination.
"""

# ---------------------------------------------------------------- Role-play
SYSTEM_ROLE = (
    "You are VERITAS, a meticulous research assistant. You never state a fact that is not "
    "supported by the evidence you are given, you cite the source of every factual sentence, "
    "and you clearly say when the evidence is insufficient or when sources disagree."
)

# ------------------------------------------------- Instructional (grounded answer)
GROUNDING_RULES = """Rules:
1. Use ONLY the evidence passages below. Do not use outside knowledge.
2. End every factual sentence with its source tag(s), e.g. [S1] or [S2][S4].
3. If the evidence does not answer part of the question, write: "The provided evidence is insufficient to determine ...".
4. If passages disagree, report both positions with their tags instead of choosing one.
5. Be concise: at most 8 sentences, no bullet points, no preamble."""

ANSWER_TEMPLATE = """Question: {question}

Evidence passages:
{evidence}

{rules}

Answer:"""

FEW_SHOT_EXAMPLE = """Example
Question: Does vitamin D supplementation reduce fracture risk?
Evidence passages:
[S1] (trial_a.pdf, p.2) In 25,871 adults, vitamin D3 at 2000 IU/day did not reduce total fractures compared with placebo.
[S2] (meta_b.pdf, p.5) Combined vitamin D and calcium reduced hip fracture risk by 16% in institutionalised elderly people.
Answer: Vitamin D alone did not reduce total fractures in a large trial of 25,871 adults [S1]. However, vitamin D combined with calcium reduced hip fractures by 16% in institutionalised elderly people [S2]. The provided evidence is insufficient to determine whether the effect applies to younger, community-dwelling adults.
---
"""

COT_SUFFIX = """First, under the heading REASONING:, list which passage supports each point you will make and note any disagreements.
Then, under the heading ANSWER:, write the final answer following the rules."""

# Strategies compared in RQ4 (prompt engineering experiment).
ANSWER_STRATEGIES = {
    "bad_vague": {
        "type": "Bad prompt (vague, no constraints)",
        "system": None,
        "user": "{question}\n\n{evidence}",
    },
    "zero_shot": {
        "type": "Instructional (zero-shot)",
        "system": None,
        "user": ANSWER_TEMPLATE,
    },
    "role_instruct": {
        "type": "Role-play + Instructional (VERITAS default)",
        "system": SYSTEM_ROLE,
        "user": ANSWER_TEMPLATE,
    },
    "few_shot": {
        "type": "Role-play + Instructional + Few-shot",
        "system": SYSTEM_ROLE,
        "user": FEW_SHOT_EXAMPLE + ANSWER_TEMPLATE,
    },
    "cot": {
        "type": "Role-play + Instructional + Chain-of-thought (Reasoning)",
        "system": SYSTEM_ROLE,
        "user": ANSWER_TEMPLATE.replace("Answer:", COT_SUFFIX),
    },
}

# ------------------------------------------------ Information (closed-book baseline)
CLOSED_BOOK = """Answer the following research question accurately and concisely in at most 8 sentences.

Question: {question}

Answer:"""

# ----------------------------------------------------- Completion (claim extraction)
CLAIM_EXTRACTION = """Break the answer below into atomic factual claims.

Instructions:
- Each claim must state exactly ONE checkable fact and be understandable on its own
  (replace pronouns like "it" or "this study" with what they refer to).
- Keep the source tags that support each claim, e.g. ["S1"]. Use [] if none.
- Skip statements that are not facts, such as "the evidence is insufficient to determine X".

Example
Answer: Aspirin lowered heart-attack risk by 22% in men [S1], but it raised bleeding risk [S1][S3].
Output: {{"claims": [
  {{"claim": "Aspirin lowered heart-attack risk by 22% in men.", "citations": ["S1"]}},
  {{"claim": "Aspirin raised bleeding risk.", "citations": ["S1", "S3"]}}
]}}

Answer: {answer}
Output (JSON only):"""

# ----------------------------------------------- Classification (LLM-as-judge baseline)
CLAIM_JUDGE = """Classify whether the evidence supports the claim.

Claim: {claim}

Evidence:
{evidence}

Reply in JSON: {{"label": "SUPPORTED" | "CONTRADICTED" | "NOT_ENOUGH_INFO", "reason": "<one sentence>"}}"""

# ----------------------------------------------------- Comparison (conflict explanation)
CONFLICT_EXPLAIN = """Two sources take different positions on the same claim.

Claim: {claim}

Source A ({source_a}): {text_a}

Source B ({source_b}): {text_b}

Compare the two sources in 2-3 sentences: what each one says, and a plausible reason they
differ (e.g. population, dosage, study design, time period). Do not decide who is right."""

# ----------------------------------------------------- Reasoning (confidence explanation)
CONFIDENCE_EXPLAIN = """Explain step by step, in at most 3 short sentences for a non-expert, why this claim
received its verdict.

Claim: {claim}
Verdict: {label} (confidence {confidence:.2f})
Evidence agreement (NLI): entailment {entail:.2f}, contradiction {contradict:.2f}
Self-consistency: {consistency}
Best evidence ({source}): {evidence}"""

# --------------------------------------------- Instructional (single-pass correction)
REGENERATE = """The answer below contains claims that the evidence does not support.

Question: {question}

Original answer:
{answer}

Problem claims:
{weak_claims}

Evidence passages:
{evidence}

Rewrite the answer once. Remove or correct every problem claim using only the evidence, keep all
supported sentences and their [S#] tags, and say "The provided evidence is insufficient to determine ..."
where needed. Output only the rewritten answer."""

# ----------------------------------------------------------- Creative (lay summary)
PLAIN_SUMMARY = """Rewrite these verified findings as a short, engaging explanation for a curious
15-year-old. Use one everyday analogy, at most 5 sentences, and do not add any new facts.

Findings:
{findings}"""


# Catalogue used by the Prompt Lab page and the Review 2 slides.
PROMPT_CATALOG = [
    {"name": "System persona", "type": "Role-play", "stage": "Generation", "template": SYSTEM_ROLE},
    {"name": "Grounded answer", "type": "Instructional", "stage": "Generation", "template": ANSWER_TEMPLATE},
    {"name": "Closed-book baseline", "type": "Information", "stage": "Baseline (RQ1)", "template": CLOSED_BOOK},
    {"name": "Claim extraction", "type": "Completion + Few-shot", "stage": "Verification", "template": CLAIM_EXTRACTION},
    {"name": "Claim judge", "type": "Classification", "stage": "Verification (LLM-judge baseline)", "template": CLAIM_JUDGE},
    {"name": "Confidence explanation", "type": "Reasoning", "stage": "Scoring", "template": CONFIDENCE_EXPLAIN},
    {"name": "Conflict explanation", "type": "Comparison", "stage": "Conflict detection", "template": CONFLICT_EXPLAIN},
    {"name": "Single-pass correction", "type": "Instructional", "stage": "Correction", "template": REGENERATE},
    {"name": "Lay summary", "type": "Creative", "stage": "Report", "template": PLAIN_SUMMARY},
]
