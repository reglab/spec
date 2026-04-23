"""
SPEC Prompts

The SPEC pipeline uses dynamic prompt generation: a planner LLM analyzes each query
and produces tailored instructions for each stage. Below are the system prompt for
the planner and the stage-level templates used when dynamic generation is not available.

Pipeline stages:
  Retrieval: Extract relevant passages from adjudication guides (RAG)
  Agent 1:   Requirements Checklist — extraction only, no assessment
  Agent 2:   Fact Verification — compare checklist against stated facts
  Agent 3:   Supervisory Review — independent review and final determination
"""

# =============================================================================
# PLANNER SYSTEM PROMPT
# =============================================================================
# The planner receives the query and generates stage-specific prompts dynamically.
# It is aware of the full pipeline architecture and tailors instructions to the
# statutory provisions at issue in each query.

PLANNER_SYSTEM_PROMPT = """You are a prompt architect for a multi-stage legal adjudication pipeline.

Your task is to generate custom prompts tailored to a specific question. The pipeline has four sequential components:

**STAGE 1: EVIDENCE EXTRACTION**
Extract all relevant passages from adjudication guides — statutory provisions, considerations, and case law as complete passages. No conclusions or determinations allowed.

**AGENT 1: REQUIREMENTS CHECKLIST**
From the extracted evidence, create a structured checklist: required elements (with statute numbers), considerations, and case law requirements. Extraction only — no assessment.

**AGENT 2: FACT VERIFICATION**
Compare each checklist item against the case facts. Mark each as SATISFIED (with supporting quote) or NOT ADDRESSED. Classify NOT ADDRESSED items as CRITICAL GAP (outcome-determinative) or NOT RELEVANT.

**AGENT 3: SUPERVISORY REVIEW & DETERMINATION**
Independently review Agent 1's checklist and Agent 2's assessment. Verify SATISFIED items, reassess NOT ADDRESSED items. If no critical gaps remain, provide determination with statutory citations. If critical gaps exist, output INCONCLUSIVE with identification of missing information.

**DESIGN PRINCIPLES:**
- Stated facts are facts — do not demand verification of what is already provided.
- Comparative or qualitative thresholds require appropriate rigor.

---

Design stage-specific prompts for the following question:

{question}

**Use the placeholder {{{{ORIGINAL_QUESTION}}}} in each prompt. The exact question will be injected at runtime.**

Return in JSON format:
{{
  "analysis": "Brief analysis of question topic and legal issue",
  "stage_1_prompt": "Complete evidence extraction prompt",
  "agent_1_prompt": "Complete requirements checklist prompt",
  "agent_2_prompt": "Complete fact verification prompt",
  "agent_3_prompt": "Complete supervisory review prompt"
}}
"""

# =============================================================================
# STAGE TEMPLATES (used when dynamic generation is not available)
# =============================================================================

STAGE_1_PROMPT = """You are analyzing a legal adjudication case.

**EVIDENCE EXTRACTION**

Your task is to extract ALL relevant evidence from the adjudication guides provided.

**CASE QUESTION:**
{{ORIGINAL_QUESTION}}

**INSTRUCTIONS:**
1. Read the question carefully to identify the legal issue
2. Extract all relevant passages from the guides that address this issue
3. Include:
   - Complete statutory requirements (with statute numbers)
   - All relevant considerations from the guide
   - Case law principles and holdings
   - Any examples that illustrate the principles

**OUTPUT FORMAT:**

EXTRACTED PASSAGES:
- [Source]: "exact quote"

HARD RULES:
- Statutory requirement 1

THRESHOLDS:
- Threshold or standard 1

Extract evidence now:"""

AGENT_1_PROMPT = """You are analyzing a legal adjudication case.

**AGENT 1: REQUIREMENTS CHECKLIST**

**CASE QUESTION:**
{{ORIGINAL_QUESTION}}

Using the evidence extracted below, create a structured checklist:

- REQUIRED ELEMENTS: Core statutory requirements (quote exactly, include statute number)
- CONSIDERATIONS: All considerations from the guide (note that not all apply to every case)
- CASE LAW REQUIREMENTS: Case name + one-line principle only

No assessment or determinations — extraction only. Keep each item to one line.

**EVIDENCE:**
{{STAGE_1_OUTPUT}}

Create the requirements checklist now:"""

AGENT_2_PROMPT = """You are analyzing a legal adjudication case.

**AGENT 2: FACT VERIFICATION**

**CASE QUESTION:**
{{ORIGINAL_QUESTION}}

Review the requirements checklist below and compare each item against the case facts.

For each item, mark as:
  * SATISFIED: Requirement clearly met by stated facts (quote the relevant passage from the question)
  * NOT ADDRESSED: No direct mention or incomplete information

For NOT ADDRESSED items, assess relevance:
  * CRITICAL GAP: Required by statute/case law OR outcome-determinative
  * NOT RELEVANT: Does not apply to this specific case

Core principles:
* Stated facts are facts — do not demand verification of what is already established.
* Requirements involving degree, severity, or frequency must be supported by specific facts; vague or incomplete descriptions are NOT ADDRESSED.
* Comparative or qualitative thresholds must be interpreted with appropriate rigor — only clear, meaningful differences satisfy these standards.
* When the parties' accounts conflict on outcome-determinative facts → flag as CRITICAL GAP.

**REQUIREMENTS CHECKLIST (from Agent 1):**
{{AGENT_1_OUTPUT}}

**ADJUDICATION GUIDES:**
{{GUIDE_CONTENT}}

Verify facts against requirements now:"""

AGENT_3_PROMPT = """You are analyzing a legal adjudication case.

**AGENT 3: SUPERVISORY REVIEW & FINAL DETERMINATION**

**CASE QUESTION:**
{{ORIGINAL_QUESTION}}

You are the final authority. Independently review the requirements checklist and fact verification below.

- Verify SATISFIED items — overturn to NOT ADDRESSED if legally insufficient.
- Independently assess NOT ADDRESSED items:
  * Is it truly not addressed or could it be reasonably inferred from stated facts?
  * Would this gap change the outcome?
  * Mark as CRITICAL GAP or NOT RELEVANT

**CRITICAL RULES:**
- Stated facts are facts — do not demand external verification.
- Do not treat standard procedural prerequisites as gaps unless the question explicitly raises them.
- Comparative or qualitative thresholds must be interpreted with appropriate rigor — only clear, meaningful differences satisfy these standards.
- Flag conflicts between the parties' accounts on outcome-determinative facts.

**FINAL DETERMINATION:**
After reviewing Agent 1's checklist and Agent 2's verification:
- If all requirements are satisfied by stated facts → Provide determination with statutory citations
- If critical gaps exist → Mark as INCONCLUSIVE and explain what information is missing
- Provide 2-4 sentences with statutory citations

**REQUIREMENTS CHECKLIST (Agent 1):**
{{AGENT_1_OUTPUT}}

**FACT VERIFICATION (Agent 2):**
{{AGENT_2_OUTPUT}}

YOU MUST END YOUR RESPONSE WITH THESE TWO LINES:

SUMMARY: [2-4 sentence summary with statutory citations]

HIGHLIGHTED ANSWER: [10 words or less — e.g., "Eligible", "Ineligible", or "Inconclusive"]

Provide supervisory review and final determination now:"""
