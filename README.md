# 🎓 Faculty Toolkit

An AI-powered web application for analyzing academic feedback, course notes, and student evaluations. Built with FastAPI, the OpenAI API, and modern web technologies.

## ✨ Features

### Core Functionality
- **📝 Text Summarization** - Generate concise summaries of long texts
- **🔑 Keyword Extraction** - Identify important terms and concepts
- **😊 Sentiment Analysis** - Classify feedback as positive or negative
- **🚀 Combined Analysis** - Run all analyses at once for comprehensive insights

### Additional Features
- **📊 Usage Statistics** - Track API usage and sentiment distribution
- **💾 Request Logging** - All analyses are logged to SQLite database
- **🎨 Modern UI** - Beautiful, responsive web interface
- **⚡ Fast API** - RESTful API with automatic documentation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- OpenAI API key (optional, but recommended for better results)

### Installation

1. **Clone or navigate to the project directory**
```bash
cd faculty_toolkit
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure OpenAI API (Optional but Recommended)**
   
   The application works without OpenAI, but using OpenAI API provides significantly better summarization and sentiment analysis results.
   
   **Option A: Using environment variable**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   **Option B: Using .env file**
   Create a `.env` file in the project root:
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```
   
   Get your API key from: https://platform.openai.com/api-keys

4. **Start the backend server**
```bash
cd backend
python main.py
```

The API will be available at `http://127.0.0.1:8000`

4. **Open the frontend**
Open `frontend/index.html` in your web browser, or use:
```bash
open frontend/index.html  # macOS
# or
xdg-open frontend/index.html  # Linux
```

## 📖 API Documentation

### Endpoints

#### `POST /summarize`
Summarize text input.

**Request:**
```json
{
  "text": "Your text here..."
}
```

**Response:**
```json
{
  "summary": "Summarized text..."
}
```

#### `POST /analyze`
Extract keywords from text.

**Response:**
```json
{
  "keywords": [
    {"term": "keyword1", "count": 2},
    {"term": "keyword2", "count": 1}
  ],
  "unique_terms": 10
}
```

#### `POST /predict`
Analyze sentiment of text.

**Response:**
```json
{
  "label": "POSITIVE",
  "score": 0.85
}
```

#### `POST /analyze-all`
Run all analyses at once.

**Response:**
```json
{
  "summary": "...",
  "keywords": [...],
  "unique_terms": 10,
  "sentiment": {
    "label": "POSITIVE",
    "score": 0.85
  },
  "text_length": 150,
  "word_count": 25
}
```

#### `GET /stats`
Get usage statistics.

**Response:**
```json
{
  "total_requests": 100,
  "by_endpoint": {
    "/summarize": 30,
    "/analyze": 25,
    "/predict": 25,
    "/analyze-all": 20
  },
  "recent_requests_24h": 15,
  "sentiment_distribution": {
    "positive": 60,
    "negative": 40,
    "total_analyzed": 100
  }
}
```

#### `GET /health`
Health check endpoint.

#### `GET /docs`
Interactive API documentation (Swagger UI)

### Database

All requests are automatically logged to `backend/faculty_toolkit.db`.

**View database contents:**
```bash
cd backend
sqlite3 faculty_toolkit.db "SELECT * FROM requests LIMIT 10;"
```

## 🏗️ Project Structure

```
faculty_toolkit/
├── backend/
│   ├── main.py          # FastAPI application
│   ├── db.py            # Database operations
│   ├── utils.py         # ML utilities and models
│   └── faculty_toolkit.db  # SQLite database
├── frontend/
│   └── index.html       # Web interface
├── data/
│   └── feedback_sample.csv
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 Configuration

### OpenAI API Integration
All AI-powered features now rely exclusively on the OpenAI API. Make sure `OPENAI_API_KEY` is set (either via environment variable or `.env`) before starting the backend.  
Use `GET /health` to confirm OpenAI connectivity.

### Database
The SQLite database is automatically created on first run. Location: `backend/faculty_toolkit.db`

## 🧪 Testing

Test the API using curl:

```bash
# Summarize
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'

# Get stats
curl http://127.0.0.1:8000/stats
```

## 📊 Technologies Used

- **Backend:** FastAPI, SQLite, Uvicorn, OpenAI API
- **Frontend:** Vanilla JavaScript, modern CSS, Chart.js

## 🎯 Use Cases

- **Faculty:** Analyze student feedback and course evaluations
- **Instructors:** Summarize course notes and identify key concepts
- **Academic Administrators:** Track sentiment trends in feedback
- **Researchers:** Extract keywords and analyze text data



## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request.

## 📧 Support

For issues or questions, please open an issue on the project repository.

---

**Built with ❤️ for educators and researchers**

# Faculty-Tool-Kit
