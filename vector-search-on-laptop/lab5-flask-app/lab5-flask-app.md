# Lab 5: Build a Flask Web Application for PDF Search

## Introduction

In this final lab you will build a complete Flask web application that combines everything from the previous labs: PDF upload, in-database chunking, vector embedding generation, and semantic similarity search. Users can upload multiple PDFs and search across all loaded documents using natural language queries.

Estimated Time: 20 minutes

### About This Application

The Flask app provides two core features: a PDF upload endpoint that stores the file, extracts text, chunks it with Oracle's `VECTOR_CHUNKS` function, and generates embeddings using the in-database ONNX model; and a search endpoint that accepts a natural language query, generates its embedding, and returns the top 10 most similar chunks across all uploaded documents using cosine distance.

### Objectives

In this lab, you will:
* Create the Flask application structure
* Implement PDF upload with in-database chunking and embedding
* Implement the semantic search endpoint
* Create an HTML interface for upload and search
* Test the complete end-to-end application

### Prerequisites

This lab assumes you have:
* Completed Labs 1 through 4
* Both `oracle-ai-db` and `python-runner` containers running (Flask and all packages are already installed in the Python container)
* The ONNX model loaded and at least one PDF in the database


## Task 1: Create the Flask Application

1. The `webapp/templates` and `webapp/uploads` directories already exist from Lab 1. Confirm they are present on your laptop:

    ```bash
    <copy>ls ~/hero/webapp/</copy>
    ```

    You should see `templates/` and `uploads/`. Inside the Python container these appear as `/workshop/webapp/templates/` and `/workshop/webapp/uploads/`.

2. Create the main application file `~/hero/webapp/app.py`:

    ```
    <copy>## app.py — Flask web application for PDF upload and AI vector search
    import oracledb
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import config

    from flask import (Flask, render_template, request,
                    jsonify, redirect, url_for, flash)
    from werkzeug.utils import secure_filename

    try:
        import PyPDF2
    except ImportError:
        print("Install PyPDF2: pip install PyPDF2")
        sys.exit(1)

    app = Flask(__name__)
    app.secret_key = "oracle-ai-workshop-secret"
    ## Use the container-side path; ~/hero/webapp/uploads is mounted at /workshop/webapp/uploads
    app.config["UPLOAD_FOLDER"] = "/workshop/webapp/uploads"
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  ## 50 MB limit
    ALLOWED_EXTENSIONS = {"pdf"}

    def allowed_file(filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def get_connection():
        """Return a new database connection."""
        return oracledb.connect(
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            dsn=config.DB_DSN
        )

    def extract_text_from_pdf(pdf_path):
        """Extract text from a PDF file."""
        all_text = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            page_count = len(reader.pages)
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    all_text.append(text.strip())
        return "\n\n".join(all_text), page_count

    def load_pdf_to_db(pdf_path, filename):
        """Extract, chunk, embed, and store a PDF in Oracle Database."""
        full_text, page_count = extract_text_from_pdf(pdf_path)

        if not full_text.strip():
            raise ValueError("No text could be extracted from this PDF.")

        file_size = os.path.getsize(pdf_path)
        connection = get_connection()
        cursor = connection.cursor()

        try:
            ## Insert document record
            doc_id_var = cursor.var(oracledb.NUMBER)
            cursor.execute("""
                INSERT INTO documents (filename, filepath, file_size, page_count)
                VALUES (:filename, :filepath, :file_size, :page_count)
                RETURNING doc_id INTO :doc_id
            """, {
                "filename": filename,
                "filepath": pdf_path,
                "file_size": file_size,
                "page_count": page_count,
                "doc_id": doc_id_var
            })
            ## DML RETURNING gives a 1-item list for the OUT bind
            doc_id = int(doc_id_var.getvalue()[0])

            ## Chunk the text using Oracle VECTOR_CHUNKS
            cursor.execute("""
                SELECT chunk_offset, chunk_length, chunk_text
                FROM VECTOR_CHUNKS(
                    :full_text
                    BY WORDS
                    MAX 150
                    OVERLAP 20
                    SPLIT BY RECURSIVELY
                    LANGUAGE american
                    NORMALIZE ALL
                )
                ORDER BY chunk_offset
            """, full_text=full_text)

            chunks = cursor.fetchall()
            inserted = 0

            for seq, (offset, length, text) in enumerate(chunks, 1):
                if not text or not text.strip():
                    continue

                cursor.execute(f"""
                    INSERT INTO doc_chunks
                        (doc_id, chunk_seq, chunk_offset, chunk_length, chunk_text, embedding)
                    VALUES (
                        :doc_id, :chunk_seq, :chunk_offset, :chunk_length, :chunk_text,
                        VECTOR_EMBEDDING({config.ONNX_MODEL_NAME} USING :embed_text AS DATA)
                    )
                """, {
                    "doc_id": doc_id,
                    "chunk_seq": seq,
                    "chunk_offset": offset,
                    "chunk_length": length,
                    "chunk_text": text,
                    "embed_text": text[:3000]
                })
                inserted += 1

            connection.commit()
            return doc_id, inserted, page_count

        finally:
            cursor.close()
            connection.close()

    def perform_search(query_text, top_k=10):
        """Search for relevant chunks using cosine similarity."""
        connection = get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(f"""
                SELECT
                    d.filename,
                    d.doc_id,
                    c.chunk_seq,
                    c.chunk_text,
                    ROUND(1 - VECTOR_DISTANCE(
                        c.embedding,
                        VECTOR_EMBEDDING({config.ONNX_MODEL_NAME} USING :query AS DATA),
                        COSINE
                    ), 4) AS similarity_score
                FROM doc_chunks c
                JOIN documents d ON c.doc_id = d.doc_id
                ORDER BY VECTOR_DISTANCE(
                    c.embedding,
                    VECTOR_EMBEDDING({config.ONNX_MODEL_NAME} USING :query AS DATA),
                    COSINE
                )
                FETCH FIRST :top_k ROWS ONLY
            """, query=query_text, top_k=top_k)

            rows = cursor.fetchall()
            results = []
            for filename, doc_id, chunk_seq, chunk_text, score in rows:
                results.append({
                    "filename": filename,
                    "doc_id": doc_id,
                    "chunk_seq": chunk_seq,
                    "text": str(chunk_text) if chunk_text else "",
                    "score": float(score) if score else 0.0
                })
            return results

        finally:
            cursor.close()
            connection.close()

    def get_document_list():
        """Return all loaded documents."""
        connection = get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute("""
                SELECT d.doc_id, d.filename, d.page_count,
                    COUNT(c.chunk_id) as chunk_count, d.loaded_at
                FROM documents d
                LEFT JOIN doc_chunks c ON d.doc_id = c.doc_id
                GROUP BY d.doc_id, d.filename, d.page_count, d.loaded_at
                ORDER BY d.loaded_at DESC
            """)
            rows = cursor.fetchall()
            docs = []
            for row in rows:
                docs.append({
                    "doc_id": row[0],
                    "filename": row[1],
                    "page_count": row[2],
                    "chunk_count": row[3],
                    "loaded_at": str(row[4])[:19] if row[4] else ""
                })
            return docs
        finally:
            cursor.close()
            connection.close()

    ## ===================== ROUTES =====================

    @app.route("/")
    def index():
        """Home page with upload form and document list."""
        documents = get_document_list()
        return render_template("index.html", documents=documents)

    @app.route("/upload", methods=["POST"])
    def upload():
        """Handle PDF file upload."""
        if "pdf_file" not in request.files:
            flash("No file selected.", "error")
            return redirect(url_for("index"))

        file = request.files["pdf_file"]

        if file.filename == "":
            flash("No file selected.", "error")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Only PDF files are allowed.", "error")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            doc_id, chunk_count, pages = load_pdf_to_db(filepath, filename)
            flash(
                f"Successfully loaded '{filename}': {pages} pages, "
                f"{chunk_count} chunks created.",
                "success"
            )
        except Exception as e:
            flash(f"Error loading PDF: {str(e)}", "error")

        return redirect(url_for("index"))

    @app.route("/search", methods=["GET", "POST"])
    def search():
        """Semantic similarity search."""
        query = ""
        results = []

        if request.method == "POST":
            query = request.form.get("query", "").strip()
        elif request.method == "GET":
            query = request.args.get("q", "").strip()

        if query:
            try:
                results = perform_search(query, top_k=10)
            except Exception as e:
                flash(f"Search error: {str(e)}", "error")

        documents = get_document_list()
        return render_template("search.html", query=query,
                            results=results, documents=documents)

    @app.route("/api/search")
    def api_search():
        """JSON API endpoint for search."""
        query = request.args.get("q", "").strip()
        top_k = int(request.args.get("top_k", 10))

        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400

        try:
            results = perform_search(query, top_k=top_k)
            return jsonify({"query": query, "count": len(results), "results": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if __name__ == "__main__":
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        print("Starting Oracle AI Vector Search Web App")
        print("Open your browser at: http://localhost:5500")
        app.run(debug=True, host="0.0.0.0", port=5500)
    </copy>
    ```

## Task 2: Create the HTML Templates

1. Create the base template `~/hero/webapp/templates/index.html`:

    ```html
    <copy>
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Oracle AI Vector Search</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
            h1 { color: #c74634; }
            h2 { color: #312d2a; border-bottom: 2px solid #c74634; padding-bottom: 6px; }
            .card { background: white; border-radius: 8px; padding: 24px; margin: 20px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
            .upload-form input[type=file] { margin: 10px 0; display: block; }
            .btn { background: #c74634; color: white; padding: 10px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
            .btn:hover { background: #a33024; }
            .btn-secondary { background: #312d2a; }
            .alert { padding: 12px 16px; border-radius: 4px; margin: 12px 0; }
            .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .alert-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            table { width: 100%; border-collapse: collapse; margin-top: 12px; }
            th { background: #312d2a; color: white; padding: 10px; text-align: left; }
            td { padding: 10px; border-bottom: 1px solid #ddd; }
            tr:hover td { background: #f9f9f9; }
            .search-link { display: inline-block; margin-top: 16px; }
            .nav { display: flex; gap: 16px; margin-bottom: 20px; }
            .nav a { color: #c74634; text-decoration: none; font-weight: bold; }
            .nav a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>Oracle AI Vector Search</h1>
        <div class="nav">
            <a href="/">Home</a>
            <a href="/search">Search</a>
        </div>

        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <div class="card">
            <h2>Upload PDF Document</h2>
            <p>Upload a PDF to extract text, chunk it with Oracle VECTOR_CHUNKS, and generate embeddings using the in-database ONNX model.</p>
            <form class="upload-form" action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="pdf_file" accept=".pdf" required>
                <button type="submit" class="btn">Upload and Process PDF</button>
            </form>
        </div>

        <div class="card">
            <h2>Loaded Documents ({{ documents|length }})</h2>
            {% if documents %}
                <table>
                    <tr><th>Filename</th><th>Pages</th><th>Chunks</th><th>Loaded At</th></tr>
                    {% for doc in documents %}
                    <tr>
                        <td>{{ doc.filename }}</td>
                        <td>{{ doc.page_count }}</td>
                        <td>{{ doc.chunk_count }}</td>
                        <td>{{ doc.loaded_at }}</td>
                    </tr>
                    {% endfor %}
                </table>
            {% else %}
                <p>No documents loaded yet. Upload a PDF above to get started.</p>
            {% endif %}
            <a href="/search" class="search-link"><button class="btn btn-secondary">Go to Search</button></a>
        </div>
    </body>
    </html>
    </copy>
    ```

2. Create the search template `~/hero/webapp/templates/search.html`:

    ```html
    <copy>
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Search - Oracle AI Vector Search</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
            h1 { color: #c74634; }
            h2 { color: #312d2a; border-bottom: 2px solid #c74634; padding-bottom: 6px; }
            .card { background: white; border-radius: 8px; padding: 24px; margin: 20px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
            .search-box { display: flex; gap: 12px; margin: 12px 0; }
            .search-box input { flex: 1; padding: 10px 14px; border: 2px solid #ddd; border-radius: 4px; font-size: 16px; }
            .search-box input:focus { border-color: #c74634; outline: none; }
            .btn { background: #c74634; color: white; padding: 10px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
            .btn:hover { background: #a33024; }
            .result-item { background: white; border: 1px solid #ddd; border-radius: 6px; padding: 16px; margin: 10px 0; }
            .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
            .score-badge { background: #c74634; color: white; padding: 4px 10px; border-radius: 12px; font-size: 13px; font-weight: bold; }
            .score-high { background: #28a745; }
            .score-medium { background: #fd7e14; }
            .result-file { font-size: 13px; color: #666; }
            .result-text { font-size: 14px; line-height: 1.6; color: #333; margin-top: 8px; max-height: 120px; overflow: hidden; text-overflow: ellipsis; }
            .nav { display: flex; gap: 16px; margin-bottom: 20px; }
            .nav a { color: #c74634; text-decoration: none; font-weight: bold; }
            .nav a:hover { text-decoration: underline; }
            .result-count { color: #666; font-size: 14px; margin: 8px 0; }
            .alert { padding: 12px 16px; border-radius: 4px; margin: 12px 0; }
            .alert-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .no-results { text-align: center; color: #666; padding: 40px 0; }
        </style>
    </head>
    <body>
        <h1>Oracle AI Vector Search</h1>
        <div class="nav">
            <a href="/">Home</a>
            <a href="/search">Search</a>
        </div>

        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <div class="card">
            <h2>Semantic Search</h2>
            <p>Search across {{ documents|length }} loaded document(s) using Oracle AI Vector Search with cosine similarity.</p>
            <form action="/search" method="post">
                <div class="search-box">
                    <input type="text" name="query" placeholder="Enter your search query..."
                        value="{{ query }}" autofocus>
                    <button type="submit" class="btn">Search</button>
                </div>
            </form>
        </div>

        {% if query %}
            <div class="card">
                <h2>Results for: "{{ query }}"</h2>
                {% if results %}
                    <p class="result-count">Showing top {{ results|length }} results (cosine similarity search)</p>
                    {% for result in results %}
                        <div class="result-item">
                            <div class="result-header">
                                <span class="result-file">
                                    {{ result.filename }} &nbsp;|&nbsp; Chunk #{{ result.chunk_seq }}
                                </span>
                                <span class="score-badge {% if result.score >= 0.75 %}score-high{% elif result.score >= 0.55 %}score-medium{% endif %}">
                                    Score: {{ result.score }}
                                </span>
                            </div>
                            <div class="result-text">{{ result.text }}</div>
                        </div>
                    {% endfor %}
                {% else %}
                    <div class="no-results">
                        <p>No results found for "{{ query }}".</p>
                        <p>Make sure you have uploaded PDFs on the <a href="/">Home page</a>.</p>
                    </div>
                {% endif %}
            </div>
        {% endif %}
    </body>
    </html>
    </copy>
    ```

## Task 3: Run the Flask Application

1. Start the Flask server inside the Python container. The `-it` flag attaches your terminal so you can see the log output and stop the server with **Ctrl+C**:

    ```bash
    <copy>podman exec -it python-runner python /workshop/webapp/app.py</copy>
    ```

    You should see:

    ```
    Starting Oracle AI Vector Search Web App
    Open your browser at: http://localhost:5500
     * Running on http://0.0.0.0:5500
    ```

    Because the Python container was started with `-p 5500:5500`, the Flask server is reachable directly on your laptop at port 5500.

2. Open your browser and navigate to:

    ```bash
    <copy>http://localhost:5500</copy>
    ```

3. Upload a PDF using the upload form on the home page. The uploaded file is saved to `/workshop/webapp/uploads/` inside the container, which maps to `~/hero/webapp/uploads/` on your laptop.

4. After the upload completes, click **Go to Search** or navigate to `http://localhost:5500/search`.

5. Enter a search query related to your PDF's content and click **Search**. You will see the top 10 most semantically similar chunks, color-coded by score:
    - Green badge (score >= 0.75): High similarity
    - Orange badge (score >= 0.55): Medium similarity
    - Red badge: Lower similarity

6. Test the JSON API endpoint from a second terminal window on your laptop:

    ```bash
    <copy>curl "http://localhost:5500/api/search?q=vector+embeddings&top_k=5"</copy>
    ```

7. To stop the Flask server, press **Ctrl+C** in the terminal running the server.

    > **Note:** The Flask development server is for local testing only. For production deployments, use a WSGI server such as Gunicorn. To restart the server later without restarting the containers, simply run the same `podman exec -it python-runner python /workshop/webapp/app.py` command again.

## Learn More

* [Flask Documentation](https://flask.palletsprojects.com/)
* [Oracle AI Vector Search - Similarity Search](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/perform-exact-similarity-search.html)
* [Oracle Vector Indexes - HNSW](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/manage-different-categories-vector-indexes.html)

## Acknowledgements
* **Author** - Oracle LiveLabs Team
* **Last Updated By/Date** - Oracle LiveLabs Team, February 2026
