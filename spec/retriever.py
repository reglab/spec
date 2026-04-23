"""
Document Retriever: Retrieval of relevant adjudication guides using
topic classification with hard-coded mappings for common separation types.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2

from spec.pdf_parser import extract_full_text


TOPIC_MAPPINGS = {
    "JOB_PERFORMANCE": {
        "keywords": ["job performance", "performance", "quota", "sales quota", "performance standards",
                     "failed to meet", "did not meet", "poor performance", "underperforming",
                     "performance review", "performance issues", "not meeting expectations",
                     "productivity", "metrics", "KPIs", "targets"],
        "primary_docs": [
            "Failure to meet an established job performance or defined standards",
            "Claimant Not at Fault for Separation",
            "Physically or Mentally Unable to Perform the Work",
        ],
        "secondary_docs": [
            "Violation of a Statute or Company Rule",
            "Other defined standards",
        ],
    },
    "THEFT": {
        "keywords": ["theft", "stealing", "stole", "took money", "embezzlement", "misappropriation",
                     "missing money", "cash shortage", "took property", "pilfering"],
        "primary_docs": [
            "Theft",
            "Gross Misconduct",
            "Willful Neglect or Damage to an Employer_s Property or Interests",
        ],
        "secondary_docs": [
            "Failure to Safeguard, Maintain, or Account for the Employer_s Property",
        ],
    },
    "ATTENDANCE": {
        "keywords": ["attendance", "absent", "absences", "tardiness", "tardy", "late", "no call no show",
                     "missed work", "didn't show up", "excessive absences", "attendance policy"],
        "primary_docs": [
            "Attendance",
            "Claimant Not at Fault for Separation",
            "Violation of a Statute or Company Rule",
        ],
        "secondary_docs": [
            "Health Problem",
        ],
    },
    "DRUGS_ALCOHOL": {
        "keywords": ["drug", "drugs", "alcohol", "intoxicated", "drunk", "substance", "marijuana",
                     "failed drug test", "positive drug test", "under the influence", "impaired"],
        "primary_docs": [
            "On the Job Use of Drugs or Alcohol",
            "Off the Job use of Drugs or Alcohol Interfering with Job Performance",
            "Having Drugs or Alcohol in an Individual's System During Working Hours",
            "On the Job Distribution of Drugs or Alcohol",
            "Alcohol or Substance Use Disorder",
        ],
        "secondary_docs": [
            "Gross Misconduct",
            "Failure to Participate in or Complete an Approved Program to Deal with an Alcohol or Substance Use Disorder",
            "Drugs or Alcohol Procedures",
        ],
    },
    "INSUBORDINATION": {
        "keywords": ["insubordination", "refused", "disobeyed", "defied", "wouldn't follow",
                     "refused to do", "ignored instructions", "disregarded", "wouldn't comply"],
        "primary_docs": [
            "Disobedience of a Reasonable Instruction",
            "Claimant Not at Fault for Separation",
        ],
        "secondary_docs": [
            "Violation of a Statute or Company Rule",
            "Rudeness, Insolence, or Offensive Behavior",
        ],
    },
    "QUIT_VOLUNTARY": {
        "keywords": ["quit", "resigned", "voluntary quit", "gave notice", "walked out",
                     "left the job", "decided to leave", "chose to leave"],
        "primary_docs": [
            "Quitting for Personal Reasons",
            "Quitting to Seek or Accept Other Work",
            "Quitting to Move for Personal Reasons",
        ],
        "secondary_docs": [
            "Quit, No Reason Given",
            "Quitting to Get Married",
            "Voluntary Leave of Absence",
        ],
    },
    "QUIT_WORKING_CONDITIONS": {
        "keywords": ["hostile", "harassment", "unsafe", "dangerous conditions", "hazardous",
                     "working conditions", "intolerable", "unbearable", "toxic workplace"],
        "primary_docs": [
            "Hazardous Working Conditions",
            "Unsatisfactory Working Conditions",
            "Quitting Due to Personal Harassment by the Employer",
            "Substantial Change in Working Conditions",
        ],
        "secondary_docs": [
            "Claimant Not at Fault for Separation",
            "Working Conditions Change",
        ],
    },
    "HEALTH_RELATED": {
        "keywords": ["health", "medical", "illness", "sick", "disability", "injured", "injury",
                     "doctor", "unable to work", "medical condition", "health condition"],
        "primary_docs": [
            "Health Problem",
            "Physically or Mentally Unable to Perform the Work",
            "Separated to Care for Family Member",
            "Separating Due to the Health of a Family Member",
        ],
        "secondary_docs": [
            "Claimant Not at Fault for Separation",
            "Health-Related Separations Under (4)(b) Procedures",
        ],
    },
    "LAYOFF": {
        "keywords": ["layoff", "laid off", "reduction in force", "RIF", "downsizing",
                     "position eliminated", "lack of work", "no work available"],
        "primary_docs": [
            "Lack of Work",
            "Layoff Information",
            "Claimant Not at Fault for Separation",
        ],
        "secondary_docs": [
            "No Separation",
        ],
    },
    "MISCONDUCT_GENERAL": {
        "keywords": ["misconduct", "violation", "policy violation", "broke rules", "rule violation"],
        "primary_docs": [
            "Violation of a Statute or Company Rule",
            "Gross Misconduct",
        ],
        "secondary_docs": [
            "Claimant Not at Fault for Separation",
        ],
    },
    "ASSAULT_THREATS": {
        "keywords": ["assault", "attacked", "hit", "punched", "threatened", "threat", "violence",
                     "physical altercation", "fight", "fighting"],
        "primary_docs": [
            "Assault",
            "Threatening to Assault",
            "Gross Misconduct",
        ],
        "secondary_docs": [
            "Rudeness, Insolence, or Offensive Behavior",
        ],
    },
    "SLEEPING": {
        "keywords": ["sleeping", "fell asleep", "asleep on the job", "napping", "dozed off"],
        "primary_docs": [
            "Sleeping on the Job",
        ],
        "secondary_docs": [
            "Violation of a Statute or Company Rule",
            "Claimant Not at Fault for Separation",
        ],
    },
}


class PDFRetriever:
    """Retrieves relevant adjudication guide documents using topic classification."""

    def __init__(self, docs_dir: str = "docs", llm_provider="anthropic", api_key=None, model=None, default_pdf_limit: int = 3):
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.provider = llm_provider
        self.default_pdf_limit = default_pdf_limit

        if llm_provider == "anthropic":
            import anthropic
            import os
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv('ANTHROPIC_API_KEY'))
            self.model_name = model or os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929')
            self._generate = self._generate_anthropic
        elif llm_provider == "openai":
            import openai
            import os
            self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
            self.model_name = model or 'gpt-4'
            self._generate = self._generate_openai
        elif llm_provider == "gemini":
            import google.generativeai as genai
            import os
            genai.configure(api_key=api_key or os.getenv('GEMINI_API_KEY'))
            self.model = genai.GenerativeModel(model or os.getenv('GEMINI_MODEL', 'gemini-3-pro-preview'))
            self._generate = self._generate_gemini
        else:
            raise ValueError(f"Unknown provider: {llm_provider}")

    def get_available_docs(self) -> List[Dict[str, str]]:
        """Get list of available documents with metadata."""
        documents = []
        for ext in ["*.pdf", "*.docx", "*.doc"]:
            for path in self.docs_dir.glob(ext):
                documents.append({
                    "filename": path.name,
                    "path": str(path),
                    "topic": path.stem
                })
        return documents

    def retrieve_with_adaptive_retry(self, question: str, execution_callback=None, start_limit: int = None):
        """Retrieve documents with adaptive retry on context length errors."""
        current_limit = start_limit or self.default_pdf_limit
        min_limit = 1

        while current_limit >= min_limit:
            try:
                retrieved = self.retrieve_relevant_docs(question, top_k=current_limit)
                callback_result = None
                if execution_callback:
                    callback_result = execution_callback(question, retrieved)
                return retrieved, callback_result
            except Exception as e:
                error_str = str(e).lower()
                is_context_error = any(kw in error_str for kw in ['context', 'token', 'too long', 'length', 'maximum', 'limit', 'exceed'])
                if is_context_error and current_limit > min_limit:
                    current_limit = max(current_limit - 2, min_limit)
                    continue
                else:
                    raise

    def retrieve_relevant_docs(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the most relevant documents for a question."""
        available = self.get_available_docs()
        if not available:
            return []
        if len(available) <= 5:
            return [{ **doc, 'content': extract_full_text(doc['path']) } for doc in available]

        topic = self._classify_topic(question, available)
        ranked, used_hardcoded = self._rank_docs(question, available, topic)

        if used_hardcoded:
            filtered = [doc for doc in ranked if doc.get('relevance_score', 0) >= 8]
        else:
            filtered = ranked[:min(top_k, len(ranked))]

        return [{ **doc, 'content': extract_full_text(doc['path']) } for doc in filtered]

    def _detect_category(self, question: str) -> Optional[str]:
        """Detect separation category using keyword matching, then LLM fallback."""
        question_lower = question.lower()
        best_match, best_score = None, 0

        for category, config in TOPIC_MAPPINGS.items():
            score = sum(1 for kw in config["keywords"] if kw.lower() in question_lower)
            if score > best_score:
                best_score = score
                best_match = category

        if best_score >= 2:
            return best_match

        categories_desc = "\n".join(f"- {k}: {len(v['keywords'])} keywords" for k, v in TOPIC_MAPPINGS.items())
        prompt = f"""Classify this unemployment insurance case into ONE category.

**CASE**: {question}

**CATEGORIES**:
{categories_desc}
- OTHER: None of the above

Return ONLY the category key (e.g., "JOB_PERFORMANCE", "THEFT", "OTHER").
Category:"""
        response = self._generate(prompt).strip().upper().replace('"', '').replace("'", "")
        if response in TOPIC_MAPPINGS:
            return response
        for category in TOPIC_MAPPINGS:
            if category in response:
                return category
        return None

    def _get_docs_for_category(self, category: str, available: List[Dict]) -> List[Dict]:
        """Get relevant documents for a category from hard-coded mappings."""
        if category not in TOPIC_MAPPINGS:
            return []

        config = TOPIC_MAPPINGS[category]
        available_map = {doc['topic'].lower().replace("_", " ").replace("-", " "): doc for doc in available}

        scored = []
        for doc_name in config["primary_docs"]:
            normalized = doc_name.lower().replace("_", " ").replace("-", " ")
            for topic, info in available_map.items():
                if self._topics_match(normalized, topic):
                    copy = info.copy()
                    copy['relevance_score'] = 10
                    scored.append(copy)
                    break

        for doc_name in config.get("secondary_docs", []):
            normalized = doc_name.lower().replace("_", " ").replace("-", " ")
            for topic, info in available_map.items():
                if self._topics_match(normalized, topic) and not any(s['topic'] == info['topic'] for s in scored):
                    copy = info.copy()
                    copy['relevance_score'] = 8
                    scored.append(copy)
                    break

        return scored

    def _topics_match(self, doc_name: str, available_topic: str) -> bool:
        """Check if a document name matches an available topic."""
        if doc_name in available_topic or available_topic in doc_name:
            return True
        common = {'the', 'a', 'an', 'of', 'to', 'for', 'in', 'on', 'or', 'and'}
        doc_words = set(doc_name.split()) - common
        topic_words = set(available_topic.split()) - common
        if doc_words and len(doc_words & topic_words) / len(doc_words) >= 0.6:
            return True
        return False

    def _classify_topic(self, question: str, available: List[Dict]) -> str:
        """Classify the primary topic of a question."""
        category = self._detect_category(question)
        if category:
            docs = self._get_docs_for_category(category, available)
            if docs:
                return docs[0]['topic']

        topics_list = "\n".join(f"- {doc['topic']}" for doc in available)
        prompt = f"""Identify the single most relevant document topic for this UI adjudication question.

**AVAILABLE TOPICS**:
{topics_list}

**QUESTION**: {question}

Return ONLY the exact topic name from the list.
Topic:"""
        response = self._generate(prompt).strip().replace('"', '').replace("'", "")
        for doc in available:
            if doc['topic'].lower() == response.lower():
                return doc['topic']
        return "GENERAL"

    def _rank_docs(self, question: str, available: List[Dict], primary_topic: str) -> tuple:
        """Rank documents by relevance using hard-coded mappings first, then LLM."""
        category = self._detect_category(question)

        if category:
            relevant = self._get_docs_for_category(category, available)
            relevant_topics = {doc['topic'].lower() for doc in relevant}
            scored = list(relevant)
            for doc in available:
                if doc['topic'].lower() not in relevant_topics:
                    copy = doc.copy()
                    copy['relevance_score'] = 0
                    scored.append(copy)
            scored.sort(key=lambda x: x['relevance_score'], reverse=True)
            return scored, True

        docs_info = "\n".join(f"{i+1}. {doc['topic']}" for i, doc in enumerate(available))
        prompt = f"""Score each document for relevance to this question (0-10).

**QUESTION**: {question}
**PRIMARY TOPIC**: {primary_topic}

**DOCUMENTS**:
{docs_info}

**SCORING**: 9-10 = direct match, 7-8 = closely related, 3-6 = possibly relevant, 0-2 = irrelevant.

Return comma-separated number:score pairs.
Scores:"""
        response = self._generate(prompt)

        try:
            scored = []
            for pair in response.strip().split(','):
                parts = pair.strip().split(':')
                if len(parts) == 2:
                    idx = int(parts[0].strip()) - 1
                    score = int(parts[1].strip())
                    if 0 <= idx < len(available):
                        copy = available[idx].copy()
                        copy['relevance_score'] = score
                        scored.append(copy)
            scored.sort(key=lambda x: x['relevance_score'], reverse=True)
            return scored, False
        except Exception:
            result = []
            for doc in available:
                copy = doc.copy()
                copy['relevance_score'] = 10 if doc['topic'].lower() == primary_topic.lower() else 0
                result.append(copy)
            result.sort(key=lambda x: x['relevance_score'], reverse=True)
            return result, False

    def _generate_gemini(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    def _generate_anthropic(self, prompt: str) -> str:
        result_text = ""
        with self.client.messages.stream(
            model=self.model_name,
            max_tokens=1024,
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
