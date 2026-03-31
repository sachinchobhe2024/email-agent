import os.path
import base64
import json
import re
import requests
import faiss
import os
import sys
import numpy as np
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
if os.path.exists("vector_db.index"):
    vector_db = faiss.read_index("vector_db.index")

    with open("email_store.json", "r", encoding="utf-8") as f:
        email_store = json.load(f)

    print("[VECTOR DB LOADED]")
else:
    vector_db = faiss.IndexFlatL2(4096)  # ⚠️ match your embedding dimension
    email_store = []
    print("[NEW VECTOR DB CREATED]")

with open("rules.json", "r", encoding="utf-8") as f:
    RULE_CONFIG = json.load(f)


# 🔐 Authentication
def authenticate():
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds


# 📩 Extract email body
def extract_body(payload):
    body = ""

    if 'parts' in payload:
        for part in payload['parts']:
            mimeType = part.get('mimeType')
            data = part['body'].get('data')

            if mimeType == 'text/plain' and data:
                return base64.urlsafe_b64decode(data).decode('utf-8')

            if mimeType == 'text/html' and data:
                return base64.urlsafe_b64decode(data).decode('utf-8')

            if 'parts' in part:
                body = extract_body(part)
                if body:
                    return body
    else:
        data = payload['body'].get('data')
        if data:
            return base64.urlsafe_b64decode(data).decode('utf-8')

    return body
def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "llama3",
            "prompt": text
        }
    )

    emb = response.json()["embedding"]

    # print("[EMBEDDING LENGTH] ->", len(emb))  # ✅ ADD THIS

    return np.array(emb, dtype="float32")

# def store_email(subject, body, category):
    text = subject + " " + body[:500]

    emb = get_embedding(text)

    vector_db.add(np.array([emb]))
    email_store.append({
    "text": text,
    "label": category
})
    print(f"[STORED EMAIL COUNT] -> {len(email_store)}") 
def store_email(subject, body, category):
    text = subject + " " + body[:500]

    emb = get_embedding(text)

    if emb is None:
        print("[STORE SKIPPED] -> embedding failed")
        return

    vector_db.add(np.array([emb]))

    email_store.append({
        "text": text,
        "label": category
    })

    print(f"[STORED EMAIL COUNT] -> {len(email_store)}")

    # ✅ SAVE FAISS INDEX
    faiss.write_index(vector_db, "vector_db.index")

    # ✅ SAVE METADATA
    with open("email_store.json", "w", encoding="utf-8") as f:
        json.dump(email_store, f, indent=4, ensure_ascii=False)

def retrieve_context(subject, body, k=2):
    if len(email_store) == 0:
        return ""

    query = get_embedding(subject + " " + body[:500])

    D, I = vector_db.search(np.array([query]), k)

    results = []
    for i in I[0]:
        if i < len(email_store):
            # ✅ Trim each retrieved email
            item = email_store[i]
            results.append(f"Previous Email:\n{item['text'][:300]}\nLabel: {item['label']}")

    return "\n---\n".join(results)
def reflect(subject, body, predicted_label):
    text = (subject + " " + body).lower()

    # 🔴 Fix false Meeting cases
    if predicted_label == "Meeting":
        if not any(word in text for word in ["invite", "calendar", "schedule"]):
            return "Low Priority", "Corrected: weak meeting signal"

    # 🔴 Fix false High Priority
    if predicted_label == "High Priority":
        if not any(word in text for word in ["urgent", "asap", "critical", "error"]):
            return "Low Priority", "Corrected: no urgency detected"

    return predicted_label, "No change"

def evaluate():
    test_cases = RULE_CONFIG.get("test_cases", [])

    correct = 0
    total = len(test_cases)

    print("\n🔍 Running Evaluation...\n")

    for test in test_cases:
        subject = test["subject"]
        body = test.get("body", "")
        expected = test["expected"]

        try:
            context = retrieve_context(subject, body)
            pred = llm_classify(subject, body, context)
        except:
            pred = classify_email(subject, body, [], "")

        final, _ = reflect(subject, body, pred)

        print(f"Subject: {subject}")
        print(f"Expected: {expected} | Predicted: {final}\n")

        if final == expected:
            correct += 1

    accuracy = correct / total if total > 0 else 0

    print(f"✅ Accuracy: {accuracy * 100:.2f}%")

# 🔍 Keyword matcher with word boundaries
def contains_keyword(text, keywords):
    for word in keywords:
        if re.search(rf"\b{re.escape(word)}\b", text):
            return True
    return False
def log_email(subject, predicted, final, reflection):
    log_entry = {
        "subject": subject,
        "predicted": predicted,
        "final": final,
        "reflection": reflection
    }

    with open("logs.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def classify_email(subject, body, headers, my_email):
    subject = subject or ""
    body = body or ""

    text = (subject + " " + body).lower().strip()

    if not text:
        return "Low Priority"

    rules = RULE_CONFIG["rules"]
    priority_order = RULE_CONFIG["priority_order"]

    cc_list = ""
    to_list = ""

    for h in headers:
        if h['name'].lower() == 'cc':
            cc_list = (h['value'] or "").lower()
        if h['name'].lower() == 'to':
            to_list = (h['value'] or "").lower()

    # ✅ CC logic (kept separate)
    if my_email.lower() in cc_list and my_email.lower() not in to_list:
        return "CC"

    # 🔁 Apply rules based on priority order
    for rule_key in priority_order:
        if rule_key == "CC":
            continue  # already handled

        keywords = rules.get(rule_key, [])

        if contains_keyword(text, keywords):
            # map config key → label
            if rule_key == "HIGH_PRIORITY":
                return "High Priority"
            elif rule_key == "MEETING":
                # extra strict condition (retain your logic)
                if any(x in text for x in ["invite", "calendar", "schedule"]):
                    return "Meeting"
            elif rule_key == "LOW_PRIORITY":
                return "Low Priority"

    return "Low Priority"


def call_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        },
        timeout=30
    )

    return response.json()["response"]


def normalize_label(label):
    label = label.lower()

    if "high" in label:
        return "High Priority"
    if "meeting" in label:
        return "Meeting"
    if "cc" in label:
        return "CC"
    if "low" in label:
        return "Low Priority"

    return "Low Priority"  # fallback


def llm_classify(subject, body,context):
    prompt = f"""
You are an email classification agent.
IMPORTANT RULES:
1. If the current email is similar to any previous email in the context,
   you MUST use the SAME label as shown in the context.
2. Context examples are more important than your general knowledge.
3. Be strict and consistent.

Classify the email into ONE of these categories:
- High Priority → urgent, requires action,asap, immediately", production issue,failure, error, down, critical, blocker
- Meeting → scheduling, invites, calendar, schedule, zoom, teams
- Low Priority → newsletters, hiring, promotions,refer, referral, reward, hiring, job, career,update, subscription, promo, offer
- CC → user is only copied
Use the context from similar past emails to improve your decision.

Context from similar past emails:
{context}
Rules:
- Be strict
- Return ONLY the category name (no explanation)

Email:
Subject: {subject}
Body: {body[:500]}
"""

   
    response = call_llm(prompt)

    # print(f"[RAW LLM RESPONSE] -> {repr(response)}")

    result = response.strip().split("\n")[0]
    result = result.replace("Category:", "").strip()
    result = result.replace(".", "").strip()

    # print(f"[CLEANED CATEGORY] -> {repr(result)}")

    # ✅ Normalize to valid labels
    result = normalize_label(result)

    # print(f"[FINAL CATEGORY] -> {repr(result)}")

    return result


def load_rules():
    with open("rules.json", "r", encoding="utf-8") as f:
        return json.load(f)

# 🧠 Reflection Layer
def reflect_classification(subject, body, predicted_category):
    text = (subject + " " + body).lower()

    if predicted_category == "Low Priority" and "urgent" in text:
        return "High Priority", "Corrected due to urgent keyword"

    return predicted_category, "No change"


# 🏷️ Sanitize label
def sanitize_label_name(name):
    if not name:
        return "Others"

    name = str(name).strip()
    name = "".join(c for c in name if c.isprintable())
    name = re.sub(r"[^a-zA-Z0-9 _-]", "", name)
    name = re.sub(r"\s+", " ", name)

    return name if name else "Others"


# 🏷️ Get or create label (with cache)
def get_or_create_label(service, label_name, labels_cache):
    # print(f"[RAW LABEL] -> {repr(label_name)}")

    label_name = sanitize_label_name(label_name)

    # print(f"[SANITIZED LABEL] -> {repr(label_name)}")

    for label in labels_cache:
        if label['name'].lower() == label_name.lower():
            return label['id']

    # Create new label
    label_object = {
        'name': label_name,
        'labelListVisibility': 'labelShow',
        'messageListVisibility': 'show'
    }

    label = service.users().labels().create(
        userId='me',
        body=label_object
    ).execute()

    # update cache
    labels_cache.append(label)

    return label['id']


# 🏷️ Apply label (DO NOT mark as read)
def apply_label(service, msg_id, new_label_id):
    msg = service.users().messages().get(userId='me', id=msg_id).execute()
    existing_labels = msg.get('labelIds', [])

    labels_to_remove = []

    for label in existing_labels:
        # ❗ Do NOT remove system labels
        if label in ['INBOX', 'UNREAD', 'STARRED', 'IMPORTANT']:
            continue

        # ❗ Do NOT remove the label we are about to add
        if label == new_label_id:
            continue

        labels_to_remove.append(label)

    service.users().messages().modify(
        userId='me',
        id=msg_id,
        body={
            'addLabelIds': [new_label_id],
            'removeLabelIds': labels_to_remove
        }
    ).execute()

# 📥 Main function
def get_emails():
    creds = authenticate()
    service = build('gmail', 'v1', credentials=creds)
    rules_config = load_rules()
    results = service.users().messages().list(
        userId='me',
        q='is:unread',
        maxResults=100
    ).execute()

    messages = results.get('messages')

    if not messages:
        print("No unread emails found.")
        return

    my_email = "sacc@ciklum.com"  

    # 🔥 Cache labels once
    labels_cache = service.users().labels().list(userId='me').execute().get('labels', [])

    email_list = []

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        payload = msg_data['payload']
        headers = payload['headers']

        subject = ""
        for h in headers:
            if h['name'] == 'Subject':
                subject = h['value']

        body = extract_body(payload)

        print(f"[DEBUG] Subject: {repr(subject)}")
        print(f"[DEBUG] Body: {repr(body[:100])}")

        # Step 1: classify
        try:
            context = retrieve_context(subject, body)
            # print("[RAG CONTEXT] ->", context)
            category = llm_classify(subject, body,context)            
        except Exception as e:
            # print("[LLM FAILED] -> Falling back", e)
            category = classify_email(subject, body, headers, my_email)
        store_email(subject, body,category)
        print(f"[LLM CATEGORY] -> {category}")

        # Step 2: reflect
        final_category, reflection_note = reflect(subject, body, category)

        # print(f"[DEBUG] Initial Category: {category}")
        # print(f"[DEBUG] Final Category: {final_category}")
        # print(f"[DEBUG] Reflection: {reflection_note}")
        log_email(subject, category, final_category, reflection_note)

        # Apply label
        label_id = get_or_create_label(service, final_category, labels_cache)
        apply_label(service, msg['id'], label_id)

        email_obj = {
            "subject": subject,
            "body": body[:200],
            "initial_category": category,
            "final_category": final_category,
            "reflection": reflection_note
        }

        email_list.append(email_obj)

        print("\n====================")
        print("Subject:", subject)
        print("Final Category:", final_category)

    # Save results
    with open("emails.json", "w", encoding="utf-8") as f:
        json.dump(email_list, f, indent=4, ensure_ascii=False)

    print("\nEmails saved to emails.json")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate()
    else:
        get_emails()
 
