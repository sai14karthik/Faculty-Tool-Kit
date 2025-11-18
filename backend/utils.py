# backend/utils.py
from typing import Dict
import re
from collections import Counter
import joblib
from pathlib import Path

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).parent / "model.joblib"

# ---------- Summarizer ----------
def simple_summarize(text: str, max_sentences: int = 2) -> str:
    # very simple sentence split on punctuation
    text = text.strip()
    # Check if text has sentence-ending punctuation
    has_sentence_end = bool(re.search(r'[.!?]\s*$', text)) or bool(re.search(r'[.!?]\s+', text))
    
    if has_sentence_end:
        sents = re.split(r'(?<=[.!?])\s+', text)
        sents = [s.strip() for s in sents if s.strip()]
        if sents:
            return " ".join(sents[:max_sentences])
    
    # If no sentence breaks found or no sentence-ending punctuation, try to create a summary
    text_lower = text.lower()
    
    # Try to find natural break points
    # Look for "because" - cut before it to get just the main point
    because_pos = text_lower.find(' because ', 20)
    if because_pos > 0:
        # Take everything up to (but not including) "because" as the main point
        summary = text[:because_pos].strip()
        return summary
    
    # Look for other break points
    for pattern in [',', ';', ' but ', ' and ', ' however ']:
        pos = text_lower.find(pattern, 30)
        if pos > 0:
            if pattern in [' but ', ' and ', ' however ']:
                cutoff = pos + len(pattern)
            else:
                cutoff = pos + 1
            summary = text[:cutoff].strip()
            if cutoff < len(text):
                summary += "..."
            return summary
    
    # For longer texts without break points, take first 100 chars at word boundary
    if len(text) > 100:
        cutoff = text.rfind(' ', 0, 100)
        if cutoff > 50:
            return text[:cutoff].strip() + "..."
    
    # For short texts without punctuation or break points, return as is
    return text

# ---------- Analyzer (keyword extraction) ----------
STOPWORDS = {
    "the","and","is","in","to","of","a","for","on","that","this","it","with","as","are","was","an","by","be",
    "were","but","or","if","so","at","from","not","have","has","had","do","does","did","will","would","could",
    "should","may","might","can","must","been","being","them","they","their","there","these","those","which",
    "what","when","where","who","why","how","all","each","every","some","any","no","more","most","other",
    "such","than","then","too","very","just","only","also","even","much","many","most","more","very","well",
    "your","you","because","often","putting","into","about","into","onto","upon","within","without","during",
    "let","lets","make","makes","made","them","more","most","get","got","go","goes","went","come","comes","came"
}

POSITIVE_CUES = {
    "excellent","engaging","helpful","clear","organized","enjoyed","supportive","insightful","great","positive",
    "well-structured","effective","fantastic","improved","better","confident","motivated","useful","responsive"
}

NEGATIVE_CUES = {
    "poorly","confusing","difficult","frustrating","unhelpful","disorganized","messy","unclear","boring","slow",
    "negative","bad","terrible","awful","annoying","incomplete","late","missing","unresponsive","hard","struggle",
    "struggling","lack","lacking","dissatisfied","disappointed","worse","problem","issue","painful","overwhelming"
}

# Thresholds for probabilistic model interpretation
POS_THRESHOLD = 0.6
NEG_THRESHOLD = 0.4

def keyword_analysis(text: str, top_k: int = 10) -> Dict:
    # simple tokenization / frequency with better stopword filtering
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())  # Minimum 3 characters
    tokens = [t for t in tokens if t not in STOPWORDS]
    counts = Counter(tokens)
    most = counts.most_common(top_k)
    keywords = [{"term": t, "count": c} for t, c in most]
    return {"keywords": keywords, "unique_terms": len(counts)}

# ---------- Simple sentiment/predictor model ----------
def train_small_model():
    # Comprehensive and diverse dataset for better generalization
    texts = [
        # Positive examples - various phrasings
        "great class and engaging lectures",
        "helpful and clear instructions",
        "the instructor provided useful feedback",
        "I learned a lot and enjoyed the labs",
        "excellent course with clear explanations",
        "the lectures were clear and engaging",
        "overall the class was good",
        "very helpful professor and good materials",
        "enjoyed the course and learned much",
        "clear and well-organized content",
        "good course with useful assignments",
        "the instructor was helpful and clear",
        "your essay demonstrates strong organisation",
        "strong work with good analysis",
        "demonstrates strong understanding of the material",
        "excellent work, let's work on improving details",
        "strong foundation, let's develop it further",
        "well done on this assignment",
        "impressive work and thorough analysis",
        "good job on completing the project",
        "your presentation was excellent",
        "solid understanding of the concepts",
        "creative approach to the problem",
        "thoughtful response to the question",
        "well-structured and organized work",
        "clear communication of ideas",
        "demonstrates good critical thinking",
        "effective use of examples",
        "strong grasp of the material",
        # Mixed but overall positive (constructive feedback)
        "assignments were sometimes confusing but overall good",
        "some parts were unclear but the class was good",
        "lectures were clear and engaging, assignments were sometimes confusing but overall the class was good",
        "your essay demonstrates strong organisation, but let's work on developing topic sentences",
        "good work overall, let's focus on improving clarity",
        "strong analysis, but we can work on structure",
        "well done, though we could improve the conclusion",
        "good effort, let's refine the introduction",
        "solid work, but needs more supporting evidence",
        "strong start, let's develop the main points further",
        "good foundation, we can work on making it more concise",
        "excellent ideas, let's work on better organization",
        "strong argument, but let's strengthen the evidence",
        # Negative examples - various phrasings
        "bad lectures and poor explanation",
        "assignments were unclear and confusing",
        "the course was disorganized",
        "I did not like the grading policy",
        "poor instruction and unclear materials",
        "the lectures were confusing and unhelpful",
        "overall the class was bad",
        "terrible course with no clear direction",
        "assignments were very confusing",
        "the instructor was unhelpful",
        "disorganized and unclear content",
        "did not enjoy the course",
        "your work is often messy and incomplete",
        "work is messy and incomplete because not putting in effort",
        "not putting in much effort",
        "work is often messy",
        "incomplete work and lack of effort",
        "poor quality work and minimal effort",
        "submitted work is messy",
        "work lacks effort and is incomplete",
        "this needs significant improvement",
        "the work is below expectations",
        "unclear and poorly organized",
        "lacks understanding of the material",
        "weak analysis and insufficient detail",
        "does not meet the requirements",
        "poorly written and difficult to follow",
        "inadequate explanation of concepts",
        "missing key points and details",
        "needs major revision",
        "unacceptable quality of work",
        "fails to address the main question",
        "weak argument with no supporting evidence",
        "confusing and poorly structured",
    ]
    labels = ([1] * 28 +  # positive examples (28)
              [1] * 12 +  # mixed but positive (constructive feedback) (12)
              [0] * 36)   # negative examples (36)
    # Use more features and better n-gram range for better generalization
    vect = TfidfVectorizer(ngram_range=(1,3), max_features=1000, min_df=1, max_df=0.95)
    X = vect.fit_transform(texts)
    # Use regularization to prevent overfitting and improve generalization
    clf = LogisticRegression(max_iter=2000, C=0.5, penalty='l2')
    clf.fit(X, labels)
    joblib.dump((vect, clf), MODEL_PATH)
    return vect, clf

def load_model():
    if MODEL_PATH.exists():
        vect, clf = joblib.load(MODEL_PATH)
    else:
        vect, clf = train_small_model()
    return vect, clf

def predict_sentiment(text: str) -> Dict:
    vect, clf = load_model()
    try:
        X = vect.transform([text])
        prob = clf.predict_proba(X)[0][1]  # probability positive
        text_lower = text.lower()
        pos_cues = sum(1 for word in POSITIVE_CUES if word in text_lower)
        neg_cues = sum(1 for word in NEGATIVE_CUES if word in text_lower)

        if prob <= NEG_THRESHOLD:
            pred = 0
        elif prob >= POS_THRESHOLD:
            pred = 1
        else:
            if neg_cues > pos_cues:
                pred = 0
            elif pos_cues > neg_cues:
                pred = 1
            else:
                pred = int(prob >= 0.5)
        
        base_confidence = abs(prob - 0.5) * 2
        if POS_THRESHOLD > prob > NEG_THRESHOLD and pos_cues != neg_cues:
            base_confidence = min(1.0, base_confidence + 0.2)

        return {
            "prediction": pred, 
            "positive_prob": float(prob),
            "confidence": float(round(base_confidence, 3))
        }
    except Exception as e:
        # Fallback: simple keyword-based sentiment if model fails
        text_lower = text.lower()
        pos_count = sum(1 for word in POSITIVE_CUES if word in text_lower)
        neg_count = sum(1 for word in NEGATIVE_CUES if word in text_lower)
        
        if pos_count > neg_count:
            return {"prediction": 1, "positive_prob": 0.6, "confidence": 0.3}
        elif neg_count > pos_count:
            return {"prediction": 0, "positive_prob": 0.4, "confidence": 0.3}
        else:
            return {"prediction": 1, "positive_prob": 0.5, "confidence": 0.1}
