# backend/utils.py
from typing import Dict, Optional, Any
import json
import os
import logging

# ----------------- Logging Setup -----------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------- OpenAI Setup -----------------
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception as exc:  # catch broader errors (e.g., requests SSL issues)
    OPENAI_AVAILABLE = False
    OpenAI = None
    logger.warning("OpenAI SDK unavailable: %s", exc)

_openai_client: Optional[OpenAI] = None


def get_openai_client() -> Optional[OpenAI]:
    """Get or create OpenAI client instance"""
    global _openai_client
    if not OPENAI_AVAILABLE:
        return None

    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables.")
            return None
        _openai_client = OpenAI(api_key=api_key)

    return _openai_client


def is_openai_available() -> bool:
    """Check if OpenAI API is available and configured"""
    return OPENAI_AVAILABLE and get_openai_client() is not None


# ----------------- OpenAI Summarization -----------------
def openai_summarize(text: str, max_length: int = 150) -> Optional[str]:
    """Summarize text using OpenAI API"""
    client = get_openai_client()
    if not client:
        return None

    try:
        max_tokens = min(int(max_length * 1.5), 500)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an expert academic assistant. Summarize feedback or course content concisely."},
                {"role": "user",
                 "content": f"Summarize the following text in ~{max_length} words, synthesizing the main points:\n\n{text}"}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logger.error(f"OpenAI summarization error: {e}")
        return None


# ----------------- OpenAI Sentiment Analysis -----------------
NEGATIVE_INDICATORS = [
    # Workload, Pacing, and Effort
    "too fast", "rushed", "overwhelming", "difficult", "stressful",
    "too heavy", "overloaded", "heavy workload", "workload", "unsustainable",
    "crammed", "busywork", "pointless",
    
    # Clarity and Organization
    "confusing", "unclear", "vague", "poorly organized", "disorganized",
    "inconsistent", "needs improvement", "should improve", "could be better",
    "rambling", "disconnected",
    
    # Assessment and Fairness
    "unfair", "delayed" , "grading criteria", "minor details", "arbitrary",
    "subjective", "biased", "unjustified", "cryptic" ,
    
    # Quality of Content and Instruction
    "lack of", "not enough", "needed more explanation", "irrelevant",
    "outdated", "monotonous", "boring", "repetitive", "dull", "superficial",
    
    # Affective and Communication
    "problem", "issue", "concern", "complaint", "frustrating",
    "hard to", "unresponsive", "unapproachable", "patronizing", "generic",
    "inaccessible", "demoralizing"
]


def openai_predict_sentiment(text: str) -> Optional[Dict[str, Any]]:
    """Analyze sentiment using OpenAI API with heuristic safeguards for negative feedback"""
    client = get_openai_client()
    if not client:
        return None

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an expert sentiment analyst for academic feedback. Return only JSON with 'label' and 'score'."},
                {"role": "user",
                 "content": f"Analyze the sentiment of this text and return JSON:\n\n{text}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message["content"].strip())
        label = result.get("label", "POSITIVE").upper()
        score = float(result.get("score", 0.5))

        text_lower = text.lower()
        has_negative_indicators = any(phrase in text_lower for phrase in NEGATIVE_INDICATORS)

        # If OpenAI claims POSITIVE but we detect clear negative signals, override
        if label == "POSITIVE" and has_negative_indicators and score < 0.8:
            label = "NEGATIVE"
            score = min(score, 0.5)

        # Normalize score
        if label == "NEGATIVE":
            score = min(max(0.0, 0.5 if score > 0.5 else score), 0.5)
        else:
            score = min(max(0.5, score), 1.0)

        return {"label": label, "score": score}
    except Exception as e:
        logger.error(f"OpenAI sentiment analysis error: {e}")
        return None


# ----------------- OpenAI Keyword Extraction -----------------
def openai_extract_keywords(text: str, top_k: int = 10) -> Optional[Dict[str, Any]]:
    """Extract keywords using OpenAI API"""
    client = get_openai_client()
    if not client:
        return None

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Extract top keywords from academic feedback or text. Return only JSON with 'keywords' and 'unique_terms'."},
                {"role": "user",
                 "content": f"Extract top {top_k} keywords from the text:\n\n{text}"}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message["content"].strip())
        keywords = [{"term": kw.get("term", ""), "count": int(kw.get("count", 1))}
                    for kw in data.get("keywords", [])][:top_k]
        return {"keywords": keywords, "unique_terms": data.get("unique_terms", len(keywords))}
    except Exception as e:
        logger.error(f"OpenAI keyword extraction error: {e}")
        return None
