# backend/utils.py
from typing import Dict, Optional, Any
import re
import json
from collections import Counter
from pathlib import Path
import os
import logging

# ----------------- Logging Setup -----------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------- OpenAI Setup -----------------
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

MODEL_PATH = Path(__file__).parent / "model.joblib"
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


# ----------------- Fallback Summarizer -----------------
def simple_summarize(text: str, max_sentences: int = 2) -> str:
    """Simple fallback summarizer that extracts first sentences"""
    text = text.strip()
    if not text:
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if sentences:
        return " ".join(sentences[:max_sentences])

    if len(text) > 100:
        cutoff = text.rfind(' ', 0, 100)
        if cutoff == -1:
            cutoff = 100
        return text[:cutoff].strip() + "..."
    return text


# ----------------- Fallback Keyword Extraction -----------------
STOPWORDS = set([
    "the", "and", "is", "in", "to", "of", "a", "for", "on", "that", "this", "it",
    "with", "as", "are", "was", "an", "by", "be", "were", "but", "or", "if", "so",
    "at", "from", "not", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "must", "been", "being", "them",
    "they", "their", "there", "these", "those", "which", "what", "when", "where",
    "who", "why", "how", "all", "each", "every", "some", "any", "no", "more",
    "most", "other", "such", "than", "then", "too", "very", "just", "only",
    "also", "even", "much", "many", "well", "your", "you", "because", "often",
    "putting", "about", "onto", "upon", "within", "without", "during", "let",
    "lets", "make", "makes", "made", "get", "got", "go", "goes", "went", "come",
    "comes", "came"
])


def keyword_analysis(text: str, top_k: int = 10) -> Dict[str, Any]:
    """Extract keywords from text using simple frequency analysis"""
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    counts = Counter(tokens)
    most = counts.most_common(top_k)
    keywords = [{"term": t, "count": c} for t, c in most]
    return {"keywords": keywords, "unique_terms": len(counts)}


# ----------------- Fallback Sentiment Analysis -----------------
def predict_sentiment(text: str) -> Dict[str, Any]:
    """Simple fallback sentiment prediction"""
    text_lower = text.lower()
    positive_words = ["good", "great", "excellent", "helpful", "clear", "enjoyed", "useful", "interesting", "well"]
    negative_words = ["difficult", "rushed", "overwhelming", "too fast", "piling", "problem", "issue", "bad", "poor"]
    negative_phrases = ["too fast", "felt rushed", "overwhelming", "difficult to", "piling up"]

    pos_count = sum(word in text_lower for word in positive_words)
    neg_count = sum(word in text_lower for word in negative_words) + sum(phrase in text_lower for phrase in negative_phrases)

    if neg_count > pos_count:
        return {"prediction": 0, "positive_prob": 0.3, "confidence": 0.5}
    elif pos_count > neg_count:
        return {"prediction": 1, "positive_prob": 0.7, "confidence": 0.5}
    else:
        return {"prediction": 0, "positive_prob": 0.4, "confidence": 0.3}


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
def openai_predict_sentiment(text: str) -> Optional[Dict[str, Any]]:
    """Analyze sentiment using OpenAI API"""
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
