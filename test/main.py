# analyze_course_evaluations.py
import os
import csv
import json
import requests

API_URL = "http://127.0.0.1:8000"

# Read student feedback from CSV
feedback_list = []
csv_path = os.path.join(os.path.dirname(__file__), 'course_evaluations.csv')
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        feedback_list.append(row['feedback'])

# Analyze each feedback
results = []
for i, feedback in enumerate(feedback_list, 1):
    print(f"Analyzing feedback {i}/{len(feedback_list)}...")
    
    response = requests.post(f"{API_URL}/analyze-all", 
        json={"text": feedback})
    
    if response.status_code == 200:
        result = response.json()
        results.append({
            'feedback': feedback[:100] + "...", 
            'sentiment': result['sentiment']['label'],
            'confidence': result['sentiment']['score'],
            'top_keywords': [k['term'] for k in result['keywords'][:5]]
        })

# Summary report
positive_count = sum(1 for r in results if r['sentiment'] == 'POSITIVE')
negative_count = sum(1 for r in results if r['sentiment'] == 'NEGATIVE')

print(f"\n=== COURSE EVALUATION SUMMARY ===")
print(f"Total Responses: {len(results)}")
print(f"Positive: {positive_count} ({positive_count/len(results)*100:.1f}%)")
print(f"Negative: {negative_count} ({negative_count/len(results)*100:.1f}%)")

# Find common concerns
all_keywords = {}
for r in results:
    for keyword in r['top_keywords']:
        all_keywords[keyword] = all_keywords.get(keyword, 0) + 1

print(f"\nTop Concerns:")
for keyword, count in sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  - {keyword}: mentioned {count} times")