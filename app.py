import os
import json
from pathlib import Path
from flask import Flask, redirect, url_for, render_template_string, send_from_directory, abort, request
import datetime

APP_DIR = Path(__file__).parent.resolve()
GIF_DIR = APP_DIR / "gifs"
GIF_DIR.mkdir(exist_ok=True)
RECORD_FILE = APP_DIR / "record.json"  # File to track remaining and verified GIFs
LOG_FILE = APP_DIR / "review_log.json"  # Log file to store approvals and rejections

app = Flask(__name__, static_folder=None)

# --- Add this helper (near your other helpers) ---
def initialize_or_sync_record_file():
    # Current GIF files on disk
    all_files = sorted(p.name for p in GIF_DIR.glob("*.gif"))

    if not RECORD_FILE.exists():
        record_data = {"remaining": all_files, "verified": []}
        save_record_data(record_data)
        return

    # Load existing and compute new files
    data = load_record_data()
    remaining = data.get("remaining", [])
    verified = data.get("verified", [])

    # verified can be list of dicts like {"filename": "...", "response": "..."}
    verified_names = {
        (v.get("filename") if isinstance(v, dict) else v) for v in verified
    }

    known = set(remaining) | verified_names
    new_files = [f for f in all_files if f not in known]

    if new_files:
        # Append new files to remaining (preserve existing order)
        data["remaining"].extend(new_files)
        # Deduplicate while keeping order
        data["remaining"] = list(dict.fromkeys(data["remaining"]))
        save_record_data(data)

# Load the record data
def load_record_data():
    with open(RECORD_FILE, "r") as f:
        return json.load(f)

# Save the updated record data
def save_record_data(data):
    with open(RECORD_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Log review to the log file (in JSON format)
def log_verified(filename, response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"filename": filename, "response": response, "timestamp": timestamp}
    if LOG_FILE.exists():
        with open(LOG_FILE, "r+") as f:
            log_data = json.load(f)
            log_data.append(log_entry)
            f.seek(0)
            json.dump(log_data, f, indent=4)
    else:
        with open(LOG_FILE, "w") as f:
            json.dump([log_entry], f, indent=4)


@app.route("/")
def index():
    record_data = load_record_data()
    if record_data["remaining"]:
        return redirect(url_for("view_local", filename=record_data["remaining"][0]))
    return render_template_string("<p>No GIFs remaining to review.</p>")

@app.route("/gifs/<filename>")
def serve_gif(filename):
    if not filename.endswith(".gif"):
        abort(404)
    return send_from_directory(GIF_DIR, filename, mimetype="image/gif", as_attachment=False)

@app.route("/view/local/<filename>")
def view_local(filename):
    initialize_or_sync_record_file()  # Ensure the record is synced when viewing a GIF
    if not filename.endswith(".gif"):
        abort(404)
    
    record_data = load_record_data()
    if filename not in record_data["remaining"]:
        return redirect(url_for("index"))
    
    # Serve the GIF
    gif_path = url_for("serve_gif", filename=filename)
    current_index = record_data["remaining"].index(filename)
    prev_index = (current_index - 1) % len(record_data["remaining"])
    next_index = (current_index + 1) % len(record_data["remaining"])
    
    return render_template_string(
        PAGE,
        gif_names=record_data["remaining"],
        show_viewer=True,
        src=gif_path, src_type="local", src_label=filename,
        current_index=current_index,
        prev_name=record_data["remaining"][prev_index], next_name=record_data["remaining"][next_index]
    )

@app.route("/log_review/<filename>", methods=["POST"])
def log_review(filename):
    response = request.form.get("response")
    
    if response not in ["approved", "rejected"]:
        return "Invalid response", 400

    # Load current record data
    record_data = load_record_data()
    
    # Update the remaining and verified lists
    record_data["remaining"].remove(filename)
    record_data["verified"].append({"filename": filename, "response": response})

    # Save the updated record data
    save_record_data(record_data)
    
    # Log the review result
    log_verified(filename, response)
    
    # Redirect to the next file
    if record_data["remaining"]:
        next_file = record_data["remaining"][0]
        return redirect(url_for("view_local", filename=next_file))
    
    return "No more GIFs to review", 200

# HTML template for the viewer page
PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GIF Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
    header { margin-bottom: 1rem; display:flex; justify-content:space-between; align-items:center; gap: 1rem; flex-wrap: wrap; }
    h1 { margin: 0; }
    .controls { display:flex; gap:.5rem; align-items:center; }
    button { padding: .5rem .9rem; border-radius: 10px; border: 1px solid #888; cursor: pointer; background: transparent; }
    button[disabled] { opacity: .4; cursor: not-allowed; }
    .viewer { max-width: min(1200px, 95vw); margin-top: 1rem; }
    img { max-width: 100%; height: auto; display: block; }
    .note { font-size: 0.9rem; opacity: .85; }
    code { background: rgba(127,127,127,.15); padding: .1rem .3rem; border-radius: 6px; }
    footer { margin-top: 2rem; font-size: .9rem; opacity: .7; }
  </style>
  <meta http-equiv="Cache-Control" content="no-transform">
</head>
<body>
<header>
  <div>
    <h1>GIF Viewer</h1>
    <p class="note">Use the ◀/▶ buttons or your keyboard arrows to navigate.</p>
  </div>
  <div class="controls">
    {% if gif_names %}
      <button id="prevBtn" onclick="location.href='{{ url_for('view_local', filename=prev_name) }}'">◀ Prev</button>
      <div class="muted">{{ current_index+1 }} / {{ gif_names|length }}</div>
      <button id="nextBtn" onclick="location.href='{{ url_for('view_local', filename=next_name) }}'">Next ▶</button>
    {% endif %}
  </div>
</header>

{% if show_viewer %}
  <section class="viewer">
    <p class="muted">Showing: <code>{{ src_label }}</code></p>
    <img id="gifImg" src="{{ src }}" alt="Animated GIF" />
    <form method="POST" action="{{ url_for('log_review', filename=src_label) }}">
      <button type="submit" name="response" value="approved">Approve</button>
      <button type="submit" name="response" value="rejected">Reject</button>
    </form>
  </section>
{% endif %}

<footer>
  <p>GIFs auto-play in browsers if they’re truly animated GIFs. Very large files may decode slowly depending on your machine.</p>
</footer>

<script>
  // Keyboard navigation: Left/Right arrows
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') {
      {% if gif_names %} window.location.href = "{{ url_for('view_local', filename=prev_name) }}"; {% endif %}
    } else if (e.key === 'ArrowRight') {
      {% if gif_names %} window.location.href = "{{ url_for('view_local', filename=next_name) }}"; {% endif %}
    }
  });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    # Manually call the function to sync the record file before running the app
    initialize_or_sync_record_file()
    
    # Now run the Flask app
    app.run(debug=True)
