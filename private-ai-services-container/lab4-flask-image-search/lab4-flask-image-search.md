# Lab 4: Image Search App with CLIP Models using Oracle Private AI Services Container

## Introduction

In this lab you build a small Flask application that performs semantic image search.

The workflow is:
- Download and unzip an image archive with `wget`
- Embed each image with `clip-vit-base-patch32-img`
- Embed user text queries with `clip-vit-base-patch32-txt`
- Return and display the top 10 most similar images from Oracle AI Database

Estimated Time: 30 minutes

### Objectives

In this lab, you will:
- Build and run a Flask web app in JupyterLab
- Load an image dataset into Oracle AI Database
- Generate image embeddings using Oracle Private AI Services Container CLIP image model
- Generate query embeddings using CLIP text model
- Perform top-10 cosine similarity search and render image results

### Prerequisites

This lab assumes:
- You completed Labs 1-3
- JupyterLab is available
- `privateai` and `aidbfree` are reachable from JupyterLab
- The `admin` database user can connect to `aidbfree:1521/FREEPDB1`

## Task 1: Prepare Project and Download Image Archive

1. Open **JupyterLab Terminal** and run:

    ```bash
    <copy>mkdir -p ~/image-search-lab/data
    cd ~/image-search-lab/data
    wget -O pdimagearchive.zip "https://c4u04.objectstorage.us-ashburn-1.oci.customer-oci.com/p/EcTjWk2IuZPZeNnD_fYMcgUhdNDIDA6rt9gaFj_WZMiL7VvxPBNMY60837hu5hga/n/c4u04/b/livelabsfiles/o/ai-ml-library/pdimagearchive.zip"
    unzip -o pdimagearchive.zip</copy>
    ```

    Verify images exist:

    ```bash
    <copy>find ~/image-search-lab/data/pdimagearchive -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -n 5</copy>
    ```

## Task 2: Create Flask App Files

1. Create project folders:

    ```bash
    <copy>mkdir -p ~/image-search-lab/templates</copy>
    ```

    Create `~/image-search-lab/app.py` with the following code:

    ```python
    <copy>
    import base64
    import json
    import mimetypes
    import os
    from pathlib import Path

    import oracledb
    import requests
    from flask import Flask, flash, redirect, render_template, request, url_for

    APP = Flask(__name__)
    APP.secret_key = os.getenv("FLASK_SECRET", "privateai-image-search-lab")

    LAB_ROOT = Path(os.getenv("LAB_ROOT", str(Path.home() / "image-search-lab")))
    DEFAULT_IMAGE_ROOT = LAB_ROOT / "data" / "pdimagearchive"

    PRIVATEAI_BASE_URL = os.getenv("PRIVATEAI_BASE_URL", "http://privateai:8080").rstrip("/")
    PRIVATEAI_EMBED_URL = f"{PRIVATEAI_BASE_URL}/v1/embeddings"
    IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "clip-vit-base-patch32-img")
    TEXT_MODEL_ID = os.getenv("TEXT_MODEL_ID", "clip-vit-base-patch32-txt")

    VECTOR_DIM = 512
    TOP_K = 10
    TABLE_NAME = "IMAGE_LIBRARY"
    DB_USER = "ADMIN"
    DB_DSN = "aidbfree:1521/FREEPDB1"


    def _load_env():
        db_password = (os.getenv("DBPASSWORD") or "").strip()
        if not db_password:
            raise RuntimeError("Database password not found in env variable DBPASSWORD.")

        return DB_USER, DB_DSN, db_password, "DBPASSWORD env"


    def _get_connection():
        user, dsn, password, _ = _load_env()
        return oracledb.connect(user=user, password=password, dsn=dsn)


    def _ensure_table():
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM user_tables
                    WHERE table_name = :table_name
                    """,
                    table_name=TABLE_NAME,
                )
                exists = cur.fetchone()[0] == 1

                if not exists:
                    cur.execute(
                        f"""
                        CREATE TABLE {TABLE_NAME} (
                            image_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                            filename VARCHAR2(500) UNIQUE NOT NULL,
                            category VARCHAR2(120),
                            mime_type VARCHAR2(120),
                            embedding VECTOR({VECTOR_DIM}, FLOAT32),
                            image_data BLOB,
                            loaded_at TIMESTAMP DEFAULT SYSTIMESTAMP
                        )
                        """
                    )
                    conn.commit()


    def _collect_images(root: Path):
        if not root.exists():
            return []
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        files = []
        for pattern in patterns:
            files.extend(root.rglob(pattern))
            files.extend(root.rglob(pattern.upper()))
        files = sorted({p.resolve() for p in files if p.is_file()})
        return files


    def _embed_with_privateai(model: str, input_payload: str):
        response = requests.post(
            PRIVATEAI_EMBED_URL,
            json={"model": model, "input": input_payload},
            timeout=120,
        )
        if not response.ok:
            raise RuntimeError(
                f"Private AI embedding call failed ({response.status_code}): {response.text[:800]}"
            )
        body = response.json()
        data = body.get("data", [])
        if not data:
            raise RuntimeError("Private AI embedding response had no vectors.")
        return data[0]["embedding"]


    def _load_images_into_db(image_root: Path):
        _ensure_table()
        images = _collect_images(image_root)
        if not images:
            raise RuntimeError(
                f"No images found under {image_root}. Download and unzip the image archive first."
            )

        inserted = 0
        skipped = 0

        with _get_connection() as conn:
            with conn.cursor() as cur:
                for image_path in images:
                    relative_name = str(image_path.relative_to(image_root))

                    cur.execute(
                        f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE filename = :filename",
                        filename=relative_name,
                    )
                    if cur.fetchone()[0] > 0:
                        skipped += 1
                        continue

                    image_bytes = image_path.read_bytes()
                    image_b64 = base64.b64encode(image_bytes).decode("ascii")
                    vector = _embed_with_privateai(IMAGE_MODEL_ID, image_b64)
                    vector_json = json.dumps(vector)

                    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
                    category = image_path.parent.name

                    sql_insert = (
                        "INSERT INTO"
                        + " "
                        + TABLE_NAME
                        + " (filename, category, mime_type, embedding, image_data) "
                        + "VALUES (:filename, :category, :mime_type, TO_VECTOR(:vector_json), :image_data)"
                    )
                    cur.execute(
                        sql_insert,
                        {
                            "filename": relative_name,
                            "category": category,
                            "mime_type": mime_type,
                            "vector_json": vector_json,
                            "image_data": image_bytes,
                        },
                    )
                    inserted += 1

                conn.commit()

        return inserted, skipped, len(images)


    def _blob_to_bytes(value):
        if value is None:
            return b""
        if hasattr(value, "read"):
            return value.read()
        return bytes(value)


    def _search_images(query_text: str, top_k: int = TOP_K):
        query_vector = _embed_with_privateai(TEXT_MODEL_ID, query_text)
        query_vector_json = json.dumps(query_vector)

        sql = f"""
            SELECT image_id,
                filename,
                category,
                mime_type,
                image_data,
                ROUND(1 - VECTOR_DISTANCE(embedding, TO_VECTOR(:query_vector), COSINE), 4) AS similarity
            FROM {TABLE_NAME}
            ORDER BY VECTOR_DISTANCE(embedding, TO_VECTOR(:query_vector), COSINE)
            FETCH FIRST {top_k} ROWS ONLY
        """

        results = []
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, query_vector=query_vector_json)
                for image_id, filename, category, mime_type, image_data, similarity in cur:
                    raw_bytes = _blob_to_bytes(image_data)
                    data_uri = (
                        f"data:{mime_type or 'image/jpeg'};base64,"
                        f"{base64.b64encode(raw_bytes).decode('ascii')}"
                    )
                    results.append(
                        {
                            "image_id": image_id,
                            "filename": filename,
                            "category": category,
                            "similarity": similarity,
                            "data_uri": data_uri,
                        }
                    )
        return results


    def _table_count():
        _ensure_table()
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                return int(cur.fetchone()[0])


    @APP.route("/", methods=["GET", "POST"])
    def index():
        _ensure_table()

        query = ""
        results = []
        image_root = request.form.get("image_root", str(DEFAULT_IMAGE_ROOT))

        if request.method == "POST":
            action = request.form.get("action")

            if action == "load":
                try:
                    inserted, skipped, total = _load_images_into_db(Path(image_root).expanduser())
                    flash(
                        f"Image load complete: inserted={inserted}, skipped={skipped}, discovered={total}.",
                        "success",
                    )
                except Exception as exc:
                    flash(f"Image load failed: {exc}", "error")
                return redirect(url_for("index"))

            if action == "search":
                query = (request.form.get("query") or "").strip()
                if not query:
                    flash("Enter a search query.", "error")
                else:
                    try:
                        results = _search_images(query, top_k=TOP_K)
                        if not results:
                            flash("No images found. Load the image archive first.", "error")
                    except Exception as exc:
                        flash(f"Search failed: {exc}", "error")

        return render_template(
            "index.html",
            query=query,
            results=results,
            total_images=_table_count(),
            default_image_root=str(DEFAULT_IMAGE_ROOT),
            image_model_id=IMAGE_MODEL_ID,
            text_model_id=TEXT_MODEL_ID,
        )


    if __name__ == "__main__":
        LAB_ROOT.mkdir(parents=True, exist_ok=True)
        db_user, db_dsn, _, password_source = _load_env()
        print(
            "DB config:",
            f"user={db_user}",
            f"dsn={db_dsn}",
            f"password_source={password_source}",
        )
        _ensure_table()
        print("Starting Flask image search app on port 5500")
        print("Open in browser: http://<server-ip>:5500/")
        APP.run(host="0.0.0.0", port=5500, debug=True)
    </copy>
    ```

2. Create `~/image-search-lab/templates/index.html` with the following code:

    ```html
    <copy>
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Private AI CLIP Image Search</title>
    <style>
        :root {
        --bg: #f3f5f7;
        --panel: #ffffff;
        --ink: #111827;
        --muted: #4b5563;
        --accent: #c74634;
        --line: #e5e7eb;
        --ok: #0f766e;
        --err: #b91c1c;
        }
        body {
        margin: 0;
        font-family: "Segoe UI", "Noto Sans", sans-serif;
        background: var(--bg);
        color: var(--ink);
        }
        .wrap {
        max-width: 1200px;
        margin: 0 auto;
        padding: 24px;
        }
        .hero {
        background: linear-gradient(120deg, #fff 0%, #f9e9e7 100%);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        }
        .hero h1 { margin: 0 0 8px 0; }
        .hero p { margin: 4px 0; color: var(--muted); }
        .row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 16px;
        }
        .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 16px;
        }
        label {
        display: block;
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        }
        input[type="text"] {
        width: 100%;
        box-sizing: border-box;
        padding: 10px 12px;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        margin-bottom: 10px;
        }
        .btn {
        border: 0;
        border-radius: 8px;
        padding: 10px 14px;
        cursor: pointer;
        background: var(--accent);
        color: #fff;
        font-weight: 600;
        }
        .btn.secondary {
        background: #111827;
        }
        .flash {
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 10px;
        font-size: 14px;
        }
        .flash.success { background: #ecfeff; color: var(--ok); border: 1px solid #99f6e4; }
        .flash.error { background: #fff1f2; color: var(--err); border: 1px solid #fecdd3; }
        .meta {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        color: var(--muted);
        font-size: 14px;
        }
        .grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 12px;
        margin-top: 14px;
        }
        .card {
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
        background: #fff;
        }
        .card img {
        width: 100%;
        height: 180px;
        object-fit: cover;
        display: block;
        background: #f8fafc;
        }
        .card .body {
        padding: 10px;
        font-size: 13px;
        }
        .small { color: var(--muted); font-size: 12px; }
        @media (max-width: 900px) {
        .row { grid-template-columns: 1fr; }
        }
    </style>
    </head>
    <body>
    <div class="wrap">
        <div class="hero">
        <h1>Private AI CLIP Image Search</h1>
        <p>Image embeddings model: <b>{{ image_model_id }}</b></p>
        <p>Text embeddings model: <b>{{ text_model_id }}</b></p>
        <div class="meta">
            <div>Total indexed images: <b>{{ total_images }}</b></div>
            <div>Top-K results: <b>10</b></div>
        </div>
        </div>

        {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            {% for category, msg in messages %}
            <div class="flash {{ category }}">{{ msg }}</div>
            {% endfor %}
        {% endif %}
        {% endwith %}

        <div class="row">
        <div class="panel">
            <h3>1) Load Image Library into Database</h3>
            <form method="post">
            <input type="hidden" name="action" value="load" />
            <label>Image Root Directory</label>
            <input type="text" name="image_root" value="{{ default_image_root }}" />
            <button class="btn" type="submit">Load + Embed Images</button>
            </form>
        </div>

        <div class="panel">
            <h3>2) Search by Text</h3>
            <form method="post">
            <input type="hidden" name="action" value="search" />
            <label>Query</label>
            <input type="text" name="query" value="{{ query }}" placeholder="e.g. old ship in ocean storm" />
            <button class="btn secondary" type="submit">Search Top 10</button>
            </form>
        </div>
        </div>

        {% if results %}
        <div class="panel">
            <h3>Search Results</h3>
            <p class="small">Query: <b>{{ query }}</b></p>
            <div class="grid">
            {% for r in results %}
                <div class="card">
                <img src="{{ r.data_uri }}" alt="{{ r.filename }}" />
                <div class="body">
                    <div><b>{{ r.filename }}</b></div>
                    <div class="small">Category: {{ r.category }}</div>
                    <div class="small">Similarity: {{ r.similarity }}</div>
                </div>
                </div>
            {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>
    </body>
    </html>
    </copy>
    ```

## Task 3: Start the Flask App

1. Run:

    ```bash
    <copy>cd ~/image-search-lab
    python app.py</copy>
    ```

2. Expected output includes:

    ```text
    Starting Flask image search app on port 5500
    Open in browser: http://<server-ip>:5500/
    ```

## Task 4: Open the Web UI

1. Open the Flask app directly:

    ```text
    http://<server-ip>:5500/
    ```

## Task 5: Load Images and Build Embeddings

1. In the app:
2. 
   1. Keep the default image root (`~/image-search-lab/data/pdimagearchive`) or adjust if needed.
   2. Click **Load + Embed Images**.
   3. Wait for completion message with inserted/skipped counts.

This step generates image embeddings with:
- `clip-vit-base-patch32-img`

## Task 6: Search with Text

1. Enter a query such as:
    - `ship in rough sea`
    - `old maps and navigation`
    - `harbor with boats`

2. Click **Search Top 10**.

The app embeds the query using:
- `clip-vit-base-patch32-txt`

Then it returns and displays the 10 most similar images using cosine similarity.

## Task 7: Clean Up Database Objects

1. Create `~/image-search-lab/cleanup_db.py` with:

    ```python
    <copy>
    import sys

    import oracledb

    from app import TABLE_NAME, _get_connection


    def _table_exists(cur):
        cur.execute(
            """
            SELECT COUNT(*)
            FROM user_tables
            WHERE table_name = :table_name
            """,
            table_name=TABLE_NAME,
        )
        return cur.fetchone()[0] == 1


    def main():
        with _get_connection() as conn:
            with conn.cursor() as cur:
                if not _table_exists(cur):
                    print(f"No cleanup needed. Table {TABLE_NAME} does not exist.")
                    return 0

                cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                row_count = cur.fetchone()[0]

                try:
                    cur.execute(f"DROP TABLE {TABLE_NAME} PURGE")
                except oracledb.DatabaseError as exc:
                    error_obj = exc.args[0]
                    if getattr(error_obj, "code", None) == 942:
                        print(f"No cleanup needed. Table {TABLE_NAME} does not exist.")
                        return 0
                    raise

            conn.commit()

        print(f"Dropped table {TABLE_NAME}. Removed {row_count} rows.")
        return 0


    if __name__ == "__main__":
        sys.exit(main())
    </copy>
    ```

2. Run cleanup:

    ```bash
    <copy>cd ~/image-search-lab
    python cleanup_db.py</copy>
    ```

3. Expected output:

    ```text
    Dropped table IMAGE_LIBRARY. Removed <N> rows.
    ```

## Learn More

- [Oracle Private AI Services Container User Guide](https://docs.oracle.com/en/database/oracle/oracle-database/26/prvai/oracle-private-ai-services-container.html)
- [Private AI Services Container API Reference](https://docs.oracle.com/en/database/oracle/oracle-database/26/prvai/private-ai-services-container-api-reference.html)
- [Oracle AI Vector Search](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/)

## Acknowledgements
- **Author** - Oracle LiveLabs Team
- **Last Updated By/Date** - Oracle LiveLabs Team, March 2026
