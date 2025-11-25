import os
import csv
import requests

API_URL = "http://127.0.0.1:8000"
CSV_FILE = os.path.join(os.path.dirname(__file__), 'course_evaluations.csv')


def read_feedback(csv_path):
    """Read feedback from CSV and return as a list of strings"""
    feedback_list = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            feedback_list.append(row['feedback'])
    return feedback_list


def analyze_feedback(feedback):
    """Send feedback to API and return analysis result"""
    response = requests.post(f"{API_URL}/analyze-all", json={"text": feedback})
    if response.status_code == 200:
        return response.json()
    return None


def main():
    feedback_list = read_feedback(CSV_FILE)
    results = []

    for i, feedback in enumerate(feedback_list, 1):
        print(f"Analyzing feedback {i}/{len(feedback_list)}...")
        result = analyze_feedback(feedback)
        if not result:
            continue

        summary = result.get('summary', 'No summary available')
        sentiment = result['sentiment']['label']
        confidence = result['sentiment']['score']
        top_keywords = [k['term'] for k in result['keywords'][:5]]

        results.append({
            'feedback_preview': feedback[:100] + "...",
            'summary': summary,
            'sentiment': sentiment,
            'confidence': confidence,
            'top_keywords': top_keywords
        })

    # --- Summary Report ---
    total = len(results)
    positive = sum(1 for r in results if r['sentiment'] == 'POSITIVE')
    negative = sum(1 for r in results if r['sentiment'] == 'NEGATIVE')

    print("\n=== COURSE EVALUATION SUMMARY ===")
    print(f"Total Responses: {total}")
    print(f"Positive: {positive} ({positive/total*100:.1f}%)")
    print(f"Negative: {negative} ({negative/total*100:.1f}%)")

    # Common keywords / concerns
    keyword_counts = {}
    for r in results:
        for kw in r['top_keywords']:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

    print("\nTop Concerns / Keywords:")
    for kw, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {kw}: mentioned {count} times")

    print("\Summaries Feedback:")
    for r in results:
        print(f"- {r['summary']} (Sentiment: {r['sentiment']})")


if __name__ == "__main__":
    main()
