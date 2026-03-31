# 📧 AI Email Classification Agent

## 🚀 Overview

This project is an AI-powered email classification agent built as part of the **Ciklum AI Academy Engineering Track**.

The agent automatically:

* Reads unread emails from Gmail
* Classifies them into categories:

  * High Priority
  * Meeting
  * Low Priority
  * CC
* Applies labels without marking emails as read
* Learns from past emails using a RAG (Retrieval-Augmented Generation) approach
* Self-corrects using a reflection mechanism

---

## 🧠 Key Features

* ✅ Rule-based + LLM hybrid classification
* ✅ RAG-based memory using FAISS
* ✅ Self-reflection for improved accuracy
* ✅ Configurable rules via `rules.json`
* ✅ Persistent vector database
* ✅ Evaluation framework with accuracy metrics
* ✅ Logging for traceability

---

## 🛠️ Tech Stack

* Python
* Gmail API
* FAISS (vector database)
* Ollama (local LLM - Llama3)
* JSON-based configuration

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Gmail API

* Create credentials in Google Cloud Console
* Download `credentials.json`
* Place it in the project root

### 3. Run Ollama locally

```bash
ollama run llama3
```

### 4. Run the agent

```bash
python read_emails.py
```

---

## 🧪 Run Evaluation

```bash
python read_emails.py eval
```

## 📊 Output

* Labels applied directly in Gmail
* Logs stored in `logs.json`
* Memory stored in `vector_db.index` and `email_store.json`

---

## 📁 Configuration

All rules and test cases are defined in:

```
rules.json
```

---

## 🎯 Categories

* High Priority → urgent/action required
* Meeting → scheduling/invites
* Low Priority → newsletters/promotions
* CC → informational only

---

## 📌 Notes

* Emails remain **unread**
* System improves over time using stored context
* Designed as a real-world agentic AI system

---
