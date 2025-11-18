from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import json
import traceback
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

# Try to import transformers, but allow server to start without it
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"Warning: transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

# Import local modules
from db import log_request, reset_requests
from utils import keyword_analysis, predict_sentiment as utils_predict_sentiment, simple_summarize

# CSV file path
CSV_PATH = Path(__file__).parent.parent / "data" / "analysis_results.csv"

def init_csv():
    """Initialize CSV file with headers if it doesn't exist"""
    if not CSV_PATH.exists():
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'input_text', 'endpoint', 'summary', 'keywords', 
                'unique_terms', 'sentiment_label', 'sentiment_score', 
                'text_length', 'word_count'
            ])

def save_to_csv(data: dict, endpoint: str, input_text: str):
    """Save analysis results to CSV file"""
    try:
        init_csv()
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Extract data based on endpoint
            if endpoint == '/analyze-all':
                summary = data.get('summary', '')
                keywords = ', '.join([k['term'] for k in data.get('keywords', [])])
                unique_terms = data.get('unique_terms', 0)
                sentiment_label = data.get('sentiment', {}).get('label', '')
                sentiment_score = data.get('sentiment', {}).get('score', 0)
                text_length = data.get('text_length', 0)
                word_count = data.get('word_count', 0)
            elif endpoint == '/summarize':
                summary = data.get('summary', '')
                keywords = ''
                unique_terms = 0
                sentiment_label = ''
                sentiment_score = 0
                text_length = len(input_text)
                word_count = len(input_text.split())
            elif endpoint == '/analyze':
                summary = ''
                keywords = ', '.join([k['term'] for k in data.get('keywords', [])])
                unique_terms = data.get('unique_terms', 0)
                sentiment_label = ''
                sentiment_score = 0
                text_length = len(input_text)
                word_count = len(input_text.split())
            elif endpoint == '/predict':
                summary = ''
                keywords = ''
                unique_terms = 0
                sentiment_label = data.get('label', '')
                sentiment_score = data.get('score', 0)
                text_length = len(input_text)
                word_count = len(input_text.split())
            else:
                return
            
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                input_text[:500],  # Limit input text length
                endpoint,
                summary[:500],  # Limit summary length
                keywords[:500],  # Limit keywords length
                unique_terms,
                sentiment_label,
                sentiment_score,
                text_length,
                word_count
            ])
    except Exception as e:
        print(f"Warning: Failed to save to CSV: {e}")

# Initialize CSV on startup
init_csv()

# --------------------------
# MODELS
# --------------------------

class TextRequest(BaseModel):
    text: str

# --------------------------
# FASTAPI APP
# --------------------------

app = FastAPI()

# --------------------------
# CORS FIX (IMPORTANT)
# --------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow all origins
    allow_credentials=True,
    allow_methods=["*"],          # allow OPTIONS (preflight)
    allow_headers=["*"],
)

# --------------------------
# PIPELINES
# --------------------------

# Initialize pipelines with error handling
summarizer = None
sentiment = None
if TRANSFORMERS_AVAILABLE and pipeline is not None:
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        sentiment = pipeline("sentiment-analysis")
    except Exception as e:
        print(f"Warning: Could not initialize transformers pipelines: {e}")
        summarizer = None
        sentiment = None

# --------------------------
# ENDPOINTS
# --------------------------

@app.post("/summarize")
def summarize_text(req: TextRequest, save_csv: bool = True):
    try:
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Try transformers first, fallback to simple summarizer
        if summarizer is not None:
            try:
                result = summarizer(req.text, max_length=120, min_length=30, do_sample=False)
                summary = result[0]["summary_text"]
            except Exception as e:
                print(f"Warning: Transformers summarization failed, using fallback: {e}")
                summary = simple_summarize(req.text, max_sentences=2)
        else:
            # Use simple fallback summarizer
            summary = simple_summarize(req.text, max_sentences=2)
        
        if not summary:
            summary = req.text[:100] + "..." if len(req.text) > 100 else req.text
        
        # Log request to database
        try:
            log_request("/summarize", req.text[:500], summary[:500])
        except Exception as e:
            print(f"Warning: Failed to log request: {e}")
        
        # Save to CSV only if called directly (not from analyze-all)
        if save_csv:
            save_to_csv({"summary": summary}, "/summarize", req.text)
        
        return {"summary": summary}
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error during summarization: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/analyze")
def analyze_keywords(req: TextRequest, save_csv: bool = True):
    try:
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Use the better keyword_analysis from utils.py
        result = keyword_analysis(req.text, top_k=10)
        
        # Log request to database
        try:
            log_request("/analyze", req.text[:500], json.dumps(result)[:500])
        except Exception as e:
            print(f"Warning: Failed to log request: {e}")
        
        # Save to CSV only if called directly (not from analyze-all)
        if save_csv:
            save_to_csv(result, "/analyze", req.text)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error during keyword analysis: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/predict")
def predict_sentiment(req: TextRequest, save_csv: bool = True):
    try:
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Try transformers first, fallback to utils model
        if sentiment is not None:
            result = sentiment(req.text)[0]
            response = {"label": result["label"], "score": result["score"]}
        else:
            # Fallback to utils.py model
            result = utils_predict_sentiment(req.text)
            response = {
                "label": "POSITIVE" if result["prediction"] == 1 else "NEGATIVE",
                "score": result["positive_prob"]
            }
        
        # Log request to database
        try:
            log_request("/predict", req.text[:500], json.dumps(response)[:500])
        except Exception as e:
            print(f"Warning: Failed to log request: {e}")
        
        # Save to CSV only if called directly (not from analyze-all)
        if save_csv:
            save_to_csv(response, "/predict", req.text)
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error during sentiment prediction: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/stats")
def get_stats():
    """Get usage statistics from the database"""
    try:
        import sqlite3
        from pathlib import Path
        from datetime import datetime, timedelta
        
        DB_PATH = Path(__file__).parent / "faculty_toolkit.db"
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Total requests
        cur.execute("SELECT COUNT(*) FROM requests")
        total_requests = cur.fetchone()[0]
        
        # Requests by endpoint
        cur.execute("SELECT endpoint, COUNT(*) as count FROM requests GROUP BY endpoint")
        by_endpoint = {row[0]: row[1] for row in cur.fetchall()}
        
        # Recent activity (last 24 hours)
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("SELECT COUNT(*) FROM requests WHERE created_at > ?", (yesterday,))
        recent_requests = cur.fetchone()[0]
        
        # Sentiment distribution (from predict endpoint)
        cur.execute("""
            SELECT result FROM requests 
            WHERE endpoint = '/predict' 
            ORDER BY created_at DESC LIMIT 100
        """)
        sentiment_results = cur.fetchall()
        
        positive_count = 0
        negative_count = 0
        for (result_str,) in sentiment_results:
            try:
                result = json.loads(result_str)
                if result.get("label") == "POSITIVE":
                    positive_count += 1
                elif result.get("label") == "NEGATIVE":
                    negative_count += 1
            except:
                pass
        
        conn.close()
        
        return {
            "total_requests": total_requests,
            "by_endpoint": by_endpoint,
            "recent_requests_24h": recent_requests,
            "sentiment_distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "total_analyzed": positive_count + negative_count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")

@app.post("/analyze-all")
def analyze_all(req: TextRequest):
    """Run all analyses (summarize, analyze keywords, predict sentiment) at once"""
    try:
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Run all analyses without saving to CSV individually
        summary_result = summarize_text(req, save_csv=False)
        keywords_result = analyze_keywords(req, save_csv=False)
        sentiment_result = predict_sentiment(req, save_csv=False)
        
        # Combine results
        combined = {
            "summary": summary_result["summary"],
            "keywords": keywords_result["keywords"],
            "unique_terms": keywords_result["unique_terms"],
            "sentiment": {
                "label": sentiment_result["label"],
                "score": sentiment_result["score"]
            },
            "text_length": len(req.text),
            "word_count": len(req.text.split())
        }
        
        # Log as a special combined request
        try:
            log_request("/analyze-all", req.text[:500], json.dumps(combined)[:500])
        except Exception as e:
            print(f"Warning: Failed to log request: {e}")
        
        # Save to CSV
        save_to_csv(combined, "/analyze-all", req.text)
        
        return combined
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error during combined analysis: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "database": "connected"
    }

@app.get("/export-csv")
def export_csv():
    """Download the CSV file with all analysis results"""
    try:
        if not CSV_PATH.exists():
            raise HTTPException(status_code=404, detail="CSV file not found. No analyses have been performed yet.")
        return FileResponse(
            path=str(CSV_PATH),
            filename="analysis_results.csv",
            media_type="text/csv"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting CSV: {str(e)}")

@app.get("/csv-info")
def csv_info():
    """Get information about the CSV file"""
    try:
        if not CSV_PATH.exists():
            return {
                "exists": False,
                "message": "CSV file not created yet. It will be created on first analysis."
            }
        
        # Count rows (excluding header)
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            row_count = sum(1 for row in reader) - 1  # Subtract header
        
        file_size = CSV_PATH.stat().st_size
        
        return {
            "exists": True,
            "file_path": str(CSV_PATH),
            "total_records": row_count,
            "file_size_bytes": file_size,
            "file_size_kb": round(file_size / 1024, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting CSV info: {str(e)}")


@app.post("/reset-stats")
def reset_stats():
    """Clear all logged requests so stats reset to zero"""
    try:
        reset_requests()
        return {"status": "ok", "message": "All usage stats cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting stats: {str(e)}")


@app.get("/visualize-data")
def visualize_data():
    """Aggregate CSV data for front-end visualizations"""
    try:
        if not CSV_PATH.exists():
            raise HTTPException(status_code=404, detail="CSV file not found. Run an analysis first.")

        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return {
                "total_records": 0,
                "endpoint_counts": {},
                "sentiment_counts": {},
                "daily_word_counts": [],
                "top_keywords": [],
                "date_range": None,
            }

        endpoint_counts = Counter()
        sentiment_counts = Counter()
        keyword_counts = Counter()
        daily_word_counts = defaultdict(lambda: {"total": 0, "count": 0})
        timestamps = []

        sentiment_series = []

        for row in rows:
            endpoint = row.get("endpoint") or "unknown"
            endpoint_counts[endpoint] += 1

            sentiment_label = row.get("sentiment_label", "").strip().upper()
            if sentiment_label and sentiment_label not in ["UNLABELED", ""]:
                sentiment_counts[sentiment_label] += 1

            keywords = row.get("keywords") or ""
            for term in [k.strip() for k in keywords.split(",") if k.strip()]:
                keyword_counts[term] += 1

            try:
                word_count = int(row.get("word_count") or 0)
            except ValueError:
                word_count = 0

            timestamp_str = row.get("timestamp")
            ts = None
            if timestamp_str:
                try:
                    ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    timestamps.append(ts)
                    daily_key = ts.strftime("%Y-%m-%d")
                except ValueError:
                    ts = None
                    daily_key = "unknown"
            else:
                daily_key = "unknown"

            label_for_series = row.get("sentiment_label")
            if label_for_series:
                try:
                    sentiment_score = float(row.get("sentiment_score") or 0)
                except ValueError:
                    sentiment_score = 0
                sentiment_series.append({
                    "timestamp": timestamp_str,
                    "label": label_for_series.upper(),
                    "score": sentiment_score,
                    "ts_sort": ts.isoformat() if ts else ""
                })

            daily_word_counts[daily_key]["total"] += word_count
            daily_word_counts[daily_key]["count"] += 1

        timestamps.sort()
        date_range = None
        if timestamps:
            date_range = {
                "start": timestamps[0].strftime("%Y-%m-%d %H:%M:%S"),
                "end": timestamps[-1].strftime("%Y-%m-%d %H:%M:%S"),
            }

        daily_word_counts_list = []
        for day, stats in sorted(daily_word_counts.items()):
            if stats["count"] == 0:
                continue
            avg_word_count = stats["total"] / stats["count"]
            daily_word_counts_list.append({
                "date": day,
                "avg_word_count": round(avg_word_count, 2),
            })

        top_keywords = [
            {"term": term, "count": count}
            for term, count in keyword_counts.most_common(10)
        ]

        sentiment_series.sort(key=lambda item: item["ts_sort"])
        for idx, entry in enumerate(sentiment_series, start=1):
            entry["sequence"] = idx
            entry.pop("ts_sort", None)

        return {
            "total_records": len(rows),
            "endpoint_counts": dict(endpoint_counts),
            "sentiment_counts": dict(sentiment_counts),
            "daily_word_counts": daily_word_counts_list,
            "top_keywords": top_keywords,
            "sentiment_series": sentiment_series,
            "date_range": date_range,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building visualization dataset: {str(e)}")

# OPTIONAL: handles OPTIONS if browser still complains
@app.options("/{path:path}")
def options_handler(path: str):
    return {"status": "ok"}

# --------------------------
# MAIN
# --------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
