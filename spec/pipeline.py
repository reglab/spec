"""
SPEC Pipeline: Structured Prompting for Evidence Checklists

A multi-agent pipeline for legal adjudication with explicit gap detection:
  Retrieval: RAG extraction of relevant statutory provisions and guidance
  Agent 1:   Requirements Checklist — extraction only, no assessment
  Agent 2:   Fact Verification — compare checklist against stated facts
  Agent 3:   Supervisory Review — independent review and final determination

Each agent runs as a separate LLM call. Agent 2 receives Agent 1's output,
and Agent 3 receives both Agent 1 and Agent 2's outputs, creating a dependency
chain that ensures determination cannot proceed until all stages complete.

A planner LLM analyzes each query and produces tailored instructions for each
agent, encoding awareness of position in the pipeline, explicit reference to
outputs from prior components, and analysis strategies specific to the
statutory provisions at issue.
"""

from typing import Dict, Any, List
from spec.retriever import PDFRetriever
from spec.prompts import (
    PLANNER_SYSTEM_PROMPT,
    STAGE_1_PROMPT,
    AGENT_1_PROMPT,
    AGENT_2_PROMPT,
    AGENT_3_PROMPT,
)
import json
import re
import os


class SPECPipeline:
    """SPEC multi-agent pipeline for legal adjudication."""

    def __init__(self, llm_provider: str = "anthropic", api_key: str = None, model: str = None, docs_dir: str = "docs"):
        self.pdf_retriever = PDFRetriever(
            docs_dir,
            llm_provider="anthropic",
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            model="claude-sonnet-4-5-20250929",
            default_pdf_limit=3
        )
        self.provider = llm_provider

        if llm_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key or os.getenv('GEMINI_API_KEY'))
            self.model = genai.GenerativeModel(model or os.getenv('GEMINI_MODEL', 'gemini-3-pro-preview'))
            self._generate = self._generate_gemini
        elif llm_provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv('ANTHROPIC_API_KEY'))
            self.model_name = model or os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929')
            self._generate = self._generate_anthropic
        elif llm_provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
            self.model_name = model or 'gpt-4'
            self._generate = self._generate_openai
        else:
            raise ValueError(f"Unknown provider: {llm_provider}")

    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a question through the SPEC pipeline.

        The pipeline runs sequentially:
          1. Planner generates dynamic prompts tailored to the query
          2. Retrieval extracts relevant documents via RAG
          3. Stage 1 extracts evidence from retrieved documents
          4. Agent 1 builds a requirements checklist (receives Stage 1 output)
          5. Agent 2 verifies facts against checklist (receives Agent 1 output)
          6. Agent 3 provides supervisory review (receives Agent 1 + Agent 2 outputs)

        Returns:
            {
                "question": str,
                "retrieved_docs": List[str],
                "stage_1_result": str,
                "agent_1_output": str,
                "agent_2_output": str,
                "agent_3_output": str,
                "determination": str,
                "reasoning": str,
            }
        """
        result = {
            "question": question,
            "retrieved_docs": [],
            "stage_1_result": "",
            "agent_1_output": "",
            "agent_2_output": "",
            "agent_3_output": "",
            "determination": None,
            "reasoning": None,
            "error": None,
        }

        # Step 1: Generate dynamic prompts via planner
        prompts = self._generate_prompts(question)

        # Step 2: Retrieve relevant documents
        def stage_1_callback(q, docs):
            if not docs:
                return None
            return self._execute_stage_1(prompts["stage_1_prompt"], docs, q)

        retrieved_docs, stage_1_result = self.pdf_retriever.retrieve_with_adaptive_retry(
            question, execution_callback=stage_1_callback
        )

        if not retrieved_docs:
            result["error"] = "No adjudication guides available."
            result["determination"] = "ERROR: No guides available"
            return result

        result["retrieved_docs"] = [doc["filename"] for doc in retrieved_docs]
        result["stage_1_result"] = stage_1_result

        # Prepare guide content for agents
        guide_content = "\n\n---\n\n".join([
            f"**SOURCE: {doc['filename']}**\n\n{doc['content']}"
            for doc in retrieved_docs
        ])[:8000]

        # Step 3: Agent 1 — Requirements Checklist
        agent_1_prompt = prompts["agent_1_prompt"]
        agent_1_prompt = agent_1_prompt.replace('{{ORIGINAL_QUESTION}}', question)
        agent_1_prompt = agent_1_prompt.replace('{{STAGE_1_OUTPUT}}', stage_1_result or "")

        agent_1_output = self._generate(agent_1_prompt)
        result["agent_1_output"] = agent_1_output

        # Step 4: Agent 2 — Fact Verification (receives Agent 1 output)
        agent_2_prompt = prompts["agent_2_prompt"]
        agent_2_prompt = agent_2_prompt.replace('{{ORIGINAL_QUESTION}}', question)
        agent_2_prompt = agent_2_prompt.replace('{{AGENT_1_OUTPUT}}', agent_1_output)
        agent_2_prompt = agent_2_prompt.replace('{{GUIDE_CONTENT}}', guide_content)

        agent_2_output = self._generate(agent_2_prompt)
        result["agent_2_output"] = agent_2_output

        # Step 5: Agent 3 — Supervisory Review (receives Agent 1 + Agent 2 outputs)
        agent_3_prompt = prompts["agent_3_prompt"]
        agent_3_prompt = agent_3_prompt.replace('{{ORIGINAL_QUESTION}}', question)
        agent_3_prompt = agent_3_prompt.replace('{{AGENT_1_OUTPUT}}', agent_1_output)
        agent_3_prompt = agent_3_prompt.replace('{{AGENT_2_OUTPUT}}', agent_2_output)

        agent_3_output = self._generate(agent_3_prompt)
        result["agent_3_output"] = agent_3_output

        # Parse Agent 3's determination
        determination = self._parse_agent_3_determination(agent_3_output)
        result["determination"] = determination.get("determination", "Inconclusive")
        result["reasoning"] = determination.get("reasoning", "")

        return result

    def _generate_prompts(self, question: str) -> Dict[str, str]:
        """
        Generate dynamic prompts tailored to the query via the planner LLM.
        Falls back to static templates if dynamic generation fails.
        """
        try:
            planner_prompt = PLANNER_SYSTEM_PROMPT.format(question=question)
            response = self._generate(planner_prompt)

            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            prompts = json.loads(response)
            return {
                "stage_1_prompt": prompts.get("stage_1_prompt", STAGE_1_PROMPT),
                "agent_1_prompt": prompts.get("agent_1_prompt", AGENT_1_PROMPT),
                "agent_2_prompt": prompts.get("agent_2_prompt", AGENT_2_PROMPT),
                "agent_3_prompt": prompts.get("agent_3_prompt", AGENT_3_PROMPT),
            }
        except Exception:
            # Fall back to static templates
            return {
                "stage_1_prompt": STAGE_1_PROMPT,
                "agent_1_prompt": AGENT_1_PROMPT,
                "agent_2_prompt": AGENT_2_PROMPT,
                "agent_3_prompt": AGENT_3_PROMPT,
            }

    def _execute_stage_1(self, prompt: str, docs: List[Dict], question: str) -> str:
        """Execute Stage 1: Evidence Extraction. Returns raw text output."""
        pdf_contents = "\n\n---\n\n".join([
            f"**SOURCE: {doc['filename']}**\n\n{doc['content']}"
            for doc in docs
        ])[:15000]

        prompt = prompt.replace('{{ORIGINAL_QUESTION}}', question)

        full_prompt = f"""{prompt}

**ADJUDICATION GUIDES TO ANALYZE**:
{pdf_contents}

---

**CRITICAL**: You are ONLY extracting evidence. Do NOT provide any determination.

Extract evidence now (EXTRACTION ONLY):"""

        return self._generate(full_prompt)

    def _parse_agent_3_determination(self, agent_3_output: str) -> Dict:
        """Parse Agent 3's output for SUMMARY and HIGHLIGHTED ANSWER."""
        clean = re.sub(r'\x1b\[[0-9;]*m', '', agent_3_output)

        summary_match = (
            re.search(r'\*\*SUMMARY:\*\*\s*(.+?)\s*\*\*HIGHLIGHTED ANSWER:\*\*', clean, re.IGNORECASE | re.DOTALL)
            or re.search(r'SUMMARY:\s*(.+?)\s*HIGHLIGHTED ANSWER:', clean, re.IGNORECASE | re.DOTALL)
            or re.search(r'SUMMARY[:\s]*(.+?)(?:\n\n|\Z)', clean, re.IGNORECASE | re.DOTALL)
        )
        summary = summary_match.group(1).strip() if summary_match else ""

        highlighted_match = (
            re.search(r'\*\*HIGHLIGHTED\s+ANSWER:\*\*\s*(.+?)(?:\n|$)', clean, re.IGNORECASE)
            or re.search(r'HIGHLIGHTED\s+ANSWER[:\s]*(.+?)(?:\n|$)', clean, re.IGNORECASE)
        )
        highlighted = highlighted_match.group(1).strip().replace('"', '').replace("'", "").replace('*', '').strip() if highlighted_match else "Inconclusive"

        return {"determination": highlighted, "reasoning": summary}

    def _generate_gemini(self, prompt: str) -> str:
        return self.model.generate_content(prompt).text

    def _generate_anthropic(self, prompt: str) -> str:
        result_text = ""
        with self.client.messages.stream(
            model=self.model_name,
            max_tokens=32000,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                result_text += text
        return result_text

    def _generate_openai(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
