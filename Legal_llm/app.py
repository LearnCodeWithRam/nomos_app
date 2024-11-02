from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import boto3
from botocore.config import Config
from pdfminer.high_level import extract_text
from langchain.llms import Bedrock
from datetime import datetime
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import os
import json
import logging
import sys 

import fitz  # PyMuPDF for PDF handling
import pytesseract
from PIL import Image



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# MySQL database configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'nomos_legal'
mysql = MySQL(app)

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Initialize AWS Bedrock Session
session_aws = boto3.session.Session(profile_name='default')
retry_config = Config(retries={'max_attempts': 10, 'mode': 'standard'})
boto3_bedrock_runtime = session_aws.client("bedrock-runtime", config=retry_config)

# Initialize ChromaDB client with configuration
chroma_client = chromadb.Client(Settings(persist_directory="chroma_db"))
collection = chroma_client.get_or_create_collection(name="pdf_embeddings")

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# AWS Bedrock model configurations
llm_configs = {
    "aws-bedrock": {
        "model_id": "mistral.mixtral-8x7b-instruct-v0:1",
        "client": boto3_bedrock_runtime,
        "model_kwargs": {"temperature": 0.7}
    },
    "meta-llama": {
        "model_id": "meta.llama3-8b-instruct-v1:0",
        "client": boto3_bedrock_runtime,
        "model_kwargs": {"temperature": 0.7}  # Lower temperature for stability in Llama output
    }
}

# Load both models
def load_llms():
    return {key: Bedrock(**config) for key, config in llm_configs.items()}

llms = load_llms()

def chunk_text(text, chunk_size=700):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to extract text from scanned PDFs using OCR
def extract_text_with_ocr(pdf_path):
    text = ""
    pdf = fitz.open(pdf_path)

    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)

    pdf.close()
    return text                       

# Enhanced function to extract text with chunking and OCR fallback
def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        if not text.strip():
            logging.info(f"OCR triggered for {pdf_path}")
            text = extract_text_with_ocr(pdf_path)
        return chunk_text(text)
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return []
    

def truncate_input(input_text, max_tokens=8000):
    """Truncate input to fit within token limits."""
    tokens = input_text.split()  # Naive tokenization by words
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return ' '.join(truncated_tokens)
    return input_text



def clean_response(response):
    """Remove repeated lines, questions from the response, and ensure sentence completion."""
    # Remove repeated lines
    lines = response.split('\n')
    cleaned_lines = []
    seen = set()

    for line in lines:
        stripped_line = line.strip()
        if stripped_line and stripped_line not in seen:
            cleaned_lines.append(stripped_line)
            seen.add(stripped_line)

    # Join cleaned lines and remove any question-like phrases
    cleaned_response = ' '.join(cleaned_lines)
    cleaned_response = re.sub(r'\b(Q|q)uestion:.*', '', cleaned_response).strip()

    # Ensure the last sentence is complete
    if not cleaned_response.endswith(('.', '!', '?')):
        last_period = cleaned_response.rfind('.')
        if last_period != -1:
            cleaned_response = cleaned_response[:last_period + 1]  # Cut off at the last full sentence

    return cleaned_response


def get_pdf_text_if_no_results(query_str, pdf_directory, max_chunks=5):
    query_embedding = embedding_model.encode(query_str).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    if not results['ids'] or not results['ids'][0]:
        all_pdf_chunks = []
        for file in os.listdir(pdf_directory):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_directory, file)
                chunks = extract_text_from_pdf(pdf_path)
                all_pdf_chunks.extend(chunks)

        # Return only a limited number of chunks
        return all_pdf_chunks[:max_chunks]

    relevant_texts = [metadata.get('content', 'No content available') for metadata in results['metadatas']]
    return relevant_texts


# Function to call AWS Bedrock model
def call_bedrock_model(model_client, model_id, context, query, model_kwargs):
    """Call the Bedrock model with refined input."""
    # Format the input prompt concisely
    input_text = f"Context: {context}\n\n{query}"
    
    # Truncate the input to fit token limits
    input_text = truncate_input(input_text, max_tokens=8000)

    payload = {
        "prompt": input_text,
        **model_kwargs
    }

    try:
        response = model_client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json"
        )
        response_body = response['body'].read().decode('utf-8')
        result = json.loads(response_body)

        # Extract the text output from the response
        if 'outputs' in result:
            raw_answer = result['outputs'][0].get('text', 'No output received')
        elif 'generation' in result:
            raw_answer = result.get('generation', 'No output received')
        else:
            raw_answer = 'No output received'

        # Clean the raw answer to remove redundant content
        return clean_response(raw_answer)

    except Exception as e:
        return f"Error calling model: {e}"


# Main route for querying PDFs
@app.route('/main', methods=['GET', 'POST'])
def main():
    if 'user_id' not in session:
        flash("Please log in first", "danger")
        return redirect(url_for('login'))

    pdf_directory = 'pdfs'
    results_mistral = None
    results_llama = None

    if request.method == 'POST':
        query_mistral = request.form.get('query_mistral')
        # query_llama = request.form.get('query_llama')
        query_llama = query_mistral

        # Get context from PDFs or ChromaDB
        chroma_results_mistral = get_pdf_text_if_no_results(query_mistral, pdf_directory)
        chroma_results_llama = get_pdf_text_if_no_results(query_llama, pdf_directory)

        context_mistral = "\n".join(chroma_results_mistral)
        context_llama = "\n".join(chroma_results_llama)

        # Call the Bedrock models with refined prompts
        results_mistral = call_bedrock_model(
            model_client=boto3_bedrock_runtime,
            model_id="mistral.mixtral-8x7b-instruct-v0:1",
            context=context_mistral,
            query=query_mistral,
            model_kwargs={"temperature": 0.7}
        )

        results_llama = call_bedrock_model(
            model_client=boto3_bedrock_runtime,
            model_id="meta.llama3-8b-instruct-v1:0",
            context=context_llama,
            query=query_llama,
            model_kwargs={"temperature": 0.7}
        )

        # Return JSON response with only the cleaned answers
        return jsonify({
            "results_mistral": results_mistral,
            "results_llama": results_llama
        })

    return render_template('main.html', results_mistral=results_mistral, results_llama=results_llama)



# Main route for deleting the chat
@app.route('/deleteChat', methods=['POST'])
def deleteChat():
    chat_id = request.form.get('chat_id')
    if not chat_id:
        return jsonify({"error": "Chat ID is missing"}), 400

    user_id = session.get('user_id')  # Retrieve user_id from session if necessary

    # Delete chat from database
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('DELETE FROM chat_history WHERE chat_id = %s and user_id= %s', (chat_id, user_id))
    mysql.connection.commit()
    cursor.close()

    # Return JSON response
    return jsonify({"result": "Chat deleted successfully"})

# Main route for saving chat PDFs
@app.route('/saveChat', methods=['POST'])
def saveChat():
    queries = None
    ans1_mist = None
    ans2_llama = None

    if request.method == 'POST':
        queries = request.form.get('query')
        ans1_mist = request.form.get('answer1')
        ans2_llama = request.form.get('answer2')
        user_id = session['user_id'] # Assuming user_id is static for now
        
        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d')

        # Insert into the database
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('INSERT INTO chat_history (user_id, user_query, answer1, answer2, time_stamp) VALUES (%s, %s, %s, %s, %s)',
                       (user_id, queries, ans1_mist, ans2_llama, timestamp))
        mysql.connection.commit()

        # Return JSON response
        return jsonify({"results_mistral": "done"})

    
# Main route for getting chat history
@app.route('/getChat', methods=['GET'])
def getChat():
    if request.method == 'GET':
        user_id = session['user_id']  # Use .get() to safely access session data

        # Get chat history from the database
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        #cursor.execute('SELECT * FROM chat_history WHERE user_id = %s ', (user_id,))
        cursor.execute('SELECT * FROM chat_history WHERE user_id = %s ORDER BY chat_id DESC', (user_id,))

        account = cursor.fetchall()  # Fetch all records

        if account:
            # Format each record as a dictionary
            chat_history = [
                {
                    "chat_id": record['chat_id'],
                    "user_id": record['user_id'],
                    "query": record['user_query'],
                    "answer1": record['answer1'],
                    "answer2": record['answer2'],
                    "time_stamp": record['time_stamp']
                } for record in account
            ]
            return jsonify(chat_history)
        else:
            return jsonify({"error": "No chat history found"}), 404

# Other routes for login, signup, etc.
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user_account WHERE user_name = % s AND user_password = % s', (username, password, ))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['user_id'] = account['user_id']
            session['username'] = account['user_name']
            msg = 'Logged in successfully !'
            flash("Logged in successfully!", "success")
            return render_template('main.html', msg = msg)
        else:
            msg = 'Incorrect username / password !'
    return render_template('login.html', msg = msg)

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('loggedin', None)
    session.pop('user_id', None)
    session.pop('username', None)
    # Respond with JSON to indicate success
    return jsonify(success=True)
    #return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form :
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user_account WHERE user_name = % s', (username, ))
        account = cursor.fetchone()
        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers !'
        elif not username or not password or not email:
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO user_account VALUES (NULL, % s, % s, % s, % s)', (username, email, password, timestamp ))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('signup.html', msg = msg)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/why')
def why():
    return render_template('why.html')

@app.route('/services')
def services():
    return render_template('service.html')

if __name__ == '__main__':
    app.run(debug=True)
