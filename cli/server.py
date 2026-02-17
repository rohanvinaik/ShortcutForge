#!/usr/bin/env python3
"""
ShortcutForge Web Server: Idiot-proof web UI for generating Apple Shortcuts.

Start with:
    python scripts/server.py
    # Then open http://localhost:8000

Routes:
    GET  /                  → Single-page HTML form
    POST /generate          → SSE stream of stage updates + final result
    POST /compile           → Direct DSL compilation (no LLM)
    GET  /download/{name}   → Download a generated .shortcut file
    GET  /health            → Health check
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "output")

app = FastAPI(title="ShortcutForge", version="0.1.0")

# ── Request Models ─────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str
    model: str = "claude-sonnet-4-20250514"
    max_retries: int = 3
    sign: bool = True
    auto_import: bool = False
    engine: str = "claude"
    model_path: str | None = None
    adapter_path: str | None = None
    use_grammar: bool = True  # True = allow grammar retry fallback
    chat_template: str = "llama3"
    # Phase 3: creative/architecture/scenario controls
    creative_mode: str = "pragmatic"
    candidate_count: int = 1
    strategy: str = "auto"


class CompileRequest(BaseModel):
    dsl_text: str
    sign: bool = True
    auto_import: bool = False


class UploadRequest(BaseModel):
    shortcut_base64: str  # Base64-encoded .shortcut file


class EditRequest(BaseModel):
    shortcut_base64: str  # Base64-encoded .shortcut file
    modification: str     # LLM modification instruction
    sign: bool = True
    auto_import: bool = False
    engine: str = "claude"
    model: str = "claude-sonnet-4-20250514"
    model_path: str | None = None
    adapter_path: str | None = None
    use_grammar: bool = True  # True = allow grammar retry fallback
    chat_template: str = "llama3"


# ── Routes ─────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


@app.get("/health")
async def health():
    return {"status": "ok", "service": "shortcutforge"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate a shortcut from a natural language prompt.
    Returns a Server-Sent Events stream with stage updates and final result.
    """

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        def on_stage(stage_result):
            # Called from sync thread — put into async queue
            queue.put_nowait(stage_result)

        def run_generation():
            from orchestrator import Orchestrator, LocalBackend, GenerationResult

            try:
                backend = None
                if req.engine == "local" and req.model_path:
                    backend = LocalBackend(
                        model_path=req.model_path,
                        adapter_path=req.adapter_path,
                        use_grammar=False,
                        never_grammar=not req.use_grammar,
                        chat_template=req.chat_template,
                    )
                orch = Orchestrator(backend=backend)
                return orch.generate(
                    req.prompt,
                    model=req.model,
                    max_retries=req.max_retries,
                    output_dir=OUTPUT_DIR,
                    sign=req.sign,
                    auto_import=req.auto_import,
                    on_stage_update=on_stage,
                    candidate_count=req.candidate_count,
                    creative_mode=req.creative_mode,
                    implementation_strategy=req.strategy,
                )
            except Exception as e:
                result = GenerationResult()
                result.errors = [str(e)]
                return result

        # Run synchronous generation in a thread
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, run_generation)

        # Stream stage updates as they arrive
        while not task.done():
            try:
                stage = await asyncio.wait_for(queue.get(), timeout=0.2)
                yield f"data: {json.dumps({'type': 'stage', **asdict(stage)})}\n\n"
            except asyncio.TimeoutError:
                continue

        # Drain remaining stage events
        while not queue.empty():
            stage = queue.get_nowait()
            yield f"data: {json.dumps({'type': 'stage', **asdict(stage)})}\n\n"

        # Get the final result
        result = await task
        result_dict = asdict(result)
        result_dict["type"] = "result"

        # Make paths relative for download URLs
        if result.shortcut_path:
            result_dict["download_filename"] = os.path.basename(result.shortcut_path)
        if result.signed_path:
            result_dict["download_filename"] = os.path.basename(result.signed_path)

        yield f"data: {json.dumps(result_dict)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/compile")
async def compile_dsl(req: CompileRequest):
    """Compile DSL text directly (no LLM)."""

    def run_compile():
        from orchestrator import Orchestrator, GenerationResult

        try:
            orch = Orchestrator()
        except ValueError:
            # No API key — use direct pipeline for compilation
            pass

        # Direct pipeline
        from dsl_parser import parse_dsl
        from dsl_validator import validate_ir
        from dsl_bridge import compile_ir

        result = GenerationResult()
        result.dsl_text = req.dsl_text

        try:
            ir = parse_dsl(req.dsl_text)
            result.name = ir.name
        except Exception as e:
            result.errors = [f"Parse error: {e}"]
            return result

        validation = validate_ir(ir)
        result.warnings = [str(w) for w in validation.warnings]
        if validation.errors:
            result.errors = [
                f"Line {e.line_number}: [{e.category}] {e.message}"
                for e in validation.errors
            ]
            return result

        try:
            shortcut = compile_ir(ir)
        except Exception as e:
            result.errors = [f"Compile error: {e}"]
            return result

        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            delivery = shortcut.deliver(
                output_dir=OUTPUT_DIR,
                sign=req.sign,
                auto_import=req.auto_import,
            )
            result.shortcut_path = delivery.get("unsigned")
            result.signed_path = delivery.get("signed")
            result.imported = delivery.get("imported", False)
            result.success = True
        except Exception as e:
            result.errors = [f"Delivery error: {e}"]
            return result

        return result

    result = await asyncio.get_event_loop().run_in_executor(None, run_compile)
    result_dict = asdict(result)
    if result.signed_path:
        result_dict["download_filename"] = os.path.basename(result.signed_path)
    elif result.shortcut_path:
        result_dict["download_filename"] = os.path.basename(result.shortcut_path)
    return JSONResponse(result_dict)


@app.post("/upload")
async def upload_shortcut(req: UploadRequest):
    """Upload a .shortcut file and return its DSL text.

    Accepts a base64-encoded .shortcut file, decompiles it to DSL,
    and returns the DSL text for viewing/editing.
    """
    import base64
    import tempfile

    def run_decompile():
        from plist_to_dsl import shortcut_file_to_dsl_safe

        # Decode and write to temp file
        try:
            data = base64.b64decode(req.shortcut_base64)
        except Exception as e:
            return {"success": False, "error": f"Invalid base64: {e}"}

        with tempfile.NamedTemporaryFile(suffix=".shortcut", delete=False) as f:
            f.write(data)
            temp_path = f.name

        try:
            dsl, error = shortcut_file_to_dsl_safe(temp_path)
            if error:
                return {"success": False, "error": f"Decompile error: {error}"}
            return {"success": True, "dsl_text": dsl}
        finally:
            os.unlink(temp_path)

    result = await asyncio.get_event_loop().run_in_executor(None, run_decompile)
    return JSONResponse(result)


@app.post("/edit")
async def edit_shortcut(req: EditRequest):
    """Upload + modify a .shortcut file, return new .shortcut.

    Accepts a base64-encoded .shortcut file and a modification instruction.
    Decompiles to DSL, sends to LLM with the modification, then compiles
    the result and returns the new shortcut.
    """
    import base64
    import tempfile

    def run_edit():
        from plist_to_dsl import shortcut_file_to_dsl_safe
        from orchestrator import Orchestrator, LocalBackend, GenerationResult

        # Decode and decompile
        try:
            data = base64.b64decode(req.shortcut_base64)
        except Exception as e:
            result = GenerationResult()
            result.errors = [f"Invalid base64: {e}"]
            return result

        with tempfile.NamedTemporaryFile(suffix=".shortcut", delete=False) as f:
            f.write(data)
            temp_path = f.name

        try:
            dsl, error = shortcut_file_to_dsl_safe(temp_path)
        finally:
            os.unlink(temp_path)

        if error:
            result = GenerationResult()
            result.errors = [f"Decompile error: {error}"]
            return result

        # Build modification prompt
        modify_prompt = (
            f"Here is an existing Apple Shortcut in DSL format:\n\n"
            f"```\n{dsl}\n```\n\n"
            f"Please modify this shortcut according to the following instruction:\n"
            f"{req.modification}\n\n"
            f"Output the complete modified shortcut in ShortcutDSL format."
        )

        # Build backend
        backend = None
        if req.engine == "local":
            if not req.model_path:
                result = GenerationResult()
                result.errors = ["model_path required for local engine"]
                return result
            backend = LocalBackend(
                model_path=req.model_path,
                adapter_path=req.adapter_path,
                use_grammar=False,
                never_grammar=not req.use_grammar,
                chat_template=req.chat_template,
            )

        try:
            orch = Orchestrator(backend=backend)
        except ValueError as e:
            result = GenerationResult()
            result.errors = [str(e)]
            return result

        return orch.generate(
            modify_prompt,
            model=req.model,
            output_dir=OUTPUT_DIR,
            sign=req.sign,
            auto_import=req.auto_import,
        )

    result = await asyncio.get_event_loop().run_in_executor(None, run_edit)
    result_dict = asdict(result)
    if result.signed_path:
        result_dict["download_filename"] = os.path.basename(result.signed_path)
    elif result.shortcut_path:
        result_dict["download_filename"] = os.path.basename(result.shortcut_path)
    return JSONResponse(result_dict)


@app.get("/download/{filename}")
async def download(filename: str):
    """Download a generated .shortcut file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(
        filepath,
        media_type="application/octet-stream",
        filename=filename,
    )


# ── Single-Page HTML UI ───────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ShortcutForge</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 720px; margin: 40px auto; padding: 0 20px;
    color: #1d1d1f; background: #fbfbfd;
  }
  h1 { font-size: 28px; margin-bottom: 8px; }
  .subtitle { color: #86868b; font-size: 15px; margin-bottom: 24px; }
  textarea {
    width: 100%; height: 100px; padding: 12px; border: 1px solid #d2d2d7;
    border-radius: 8px; font-size: 15px; font-family: inherit;
    resize: vertical; outline: none; transition: border-color 0.2s;
  }
  textarea:focus { border-color: #0071e3; }
  button {
    background: #0071e3; color: white; border: none; padding: 10px 24px;
    border-radius: 8px; font-size: 15px; cursor: pointer; margin-top: 12px;
    transition: background 0.2s;
  }
  button:hover { background: #0077ed; }
  button:disabled { background: #86868b; cursor: not-allowed; }

  .pipeline { display: flex; gap: 8px; margin: 20px 0; flex-wrap: wrap; }
  .stage {
    display: flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 20px;
    background: #f5f5f7; color: #86868b; font-size: 13px;
    transition: all 0.3s;
  }
  .stage.running { background: #e8f0fe; color: #0071e3; }
  .stage.success { background: #e8f8e8; color: #1d7d1d; }
  .stage.failed { background: #fee8e8; color: #d70015; }
  .stage .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #d2d2d7; transition: background 0.3s;
  }
  .stage.running .dot { background: #0071e3; animation: pulse 1s infinite; }
  .stage.success .dot { background: #1d7d1d; }
  .stage.failed .dot { background: #d70015; }

  @keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.3; }
  }

  .result { margin-top: 20px; padding: 16px; border-radius: 8px; display: none; }
  .result.success { background: #e8f8e8; border: 1px solid #b8e8b8; }
  .result.error { background: #fee8e8; border: 1px solid #f8b8b8; }
  .result a { color: #0071e3; text-decoration: none; font-weight: 500; }
  .result a:hover { text-decoration: underline; }

  .dsl-viewer {
    margin-top: 16px; display: none;
  }
  .dsl-toggle {
    background: none; color: #0071e3; padding: 0; margin: 0;
    font-size: 13px; cursor: pointer; border: none;
  }
  .dsl-toggle:hover { text-decoration: underline; }
  .dsl-code {
    display: none; background: #1d1d1f; color: #f5f5f7;
    padding: 16px; border-radius: 8px; margin-top: 8px;
    font-family: "SF Mono", Monaco, monospace; font-size: 13px;
    white-space: pre-wrap; overflow-x: auto; max-height: 400px;
    overflow-y: auto;
  }

  .error-list { list-style: none; padding: 0; }
  .error-list li { padding: 4px 0; font-size: 14px; }
  .error-list li::before { content: "\\2717 "; color: #d70015; }

  .warnings { margin-top: 12px; font-size: 13px; color: #86868b; }
</style>
</head>
<body>

<h1>ShortcutForge</h1>
<p class="subtitle">Describe what you want and get a working Apple Shortcut.</p>

<textarea id="prompt" placeholder="Example: Set a 5-minute timer and notify me when it's done"></textarea>

<div style="margin-top: 12px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
  <label style="font-size: 13px; color: #86868b;">Engine:
    <select id="engine" onchange="toggleLocalOpts()" style="padding: 4px 8px; border-radius: 6px; border: 1px solid #d2d2d7; font-size: 13px;">
      <option value="claude" selected>Claude API</option>
      <option value="local">Local Model</option>
    </select>
  </label>
  <span id="local-opts" style="display:none; gap: 8px; align-items: center;">
    <input id="model-path" type="text" placeholder="Model path" style="padding: 4px 8px; border-radius: 6px; border: 1px solid #d2d2d7; font-size: 13px; width: 260px;">
    <input id="adapter-path" type="text" placeholder="Adapter path (optional)" style="padding: 4px 8px; border-radius: 6px; border: 1px solid #d2d2d7; font-size: 13px; width: 200px;">
    <label style="font-size: 13px; color: #86868b;"><input id="use-grammar" type="checkbox" checked> Grammar</label>
  </span>
</div>

<button id="generate" onclick="doGenerate()" style="margin-top: 12px;">Generate Shortcut</button>

<div class="pipeline" id="pipeline">
  <div class="stage" id="s-generating"><span class="dot"></span>Generating</div>
  <div class="stage" id="s-parsing"><span class="dot"></span>Parsing</div>
  <div class="stage" id="s-validating"><span class="dot"></span>Validating</div>
  <div class="stage" id="s-compiling"><span class="dot"></span>Compiling</div>
  <div class="stage" id="s-delivering"><span class="dot"></span>Delivering</div>
</div>

<div class="result" id="result"></div>

<div class="dsl-viewer" id="dsl-viewer">
  <button class="dsl-toggle" onclick="toggleDsl()">Show generated DSL</button>
  <pre class="dsl-code" id="dsl-code"></pre>
</div>

<script>
function toggleLocalOpts() {
  const engine = document.getElementById('engine').value;
  const opts = document.getElementById('local-opts');
  opts.style.display = engine === 'local' ? 'flex' : 'none';
}

function resetUI() {
  document.querySelectorAll('.stage').forEach(s => {
    s.className = 'stage';
  });
  document.getElementById('result').style.display = 'none';
  document.getElementById('result').className = 'result';
  document.getElementById('dsl-viewer').style.display = 'none';
  document.getElementById('dsl-code').style.display = 'none';
}

function doGenerate() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;

  resetUI();
  const btn = document.getElementById('generate');
  btn.disabled = true;
  btn.textContent = 'Generating...';

  const engine = document.getElementById('engine').value;
  const payload = { prompt: prompt, engine: engine };
  if (engine === 'local') {
    payload.model_path = document.getElementById('model-path').value;
    payload.adapter_path = document.getElementById('adapter-path').value || null;
    payload.use_grammar = document.getElementById('use-grammar').checked;
  }
  const body = JSON.stringify(payload);

  fetch('/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body
  }).then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    function processChunk({ done, value }) {
      if (done) {
        btn.disabled = false;
        btn.textContent = 'Generate Shortcut';
        return;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'stage') {
              updateStage(data);
            } else if (data.type === 'result') {
              showResult(data);
            }
          } catch (e) {
            console.error('Parse error:', e, line);
          }
        }
      }

      reader.read().then(processChunk);
    }

    reader.read().then(processChunk);
  }).catch(err => {
    showError('Network error: ' + err.message);
    btn.disabled = false;
    btn.textContent = 'Generate Shortcut';
  });
}

function updateStage(data) {
  const el = document.getElementById('s-' + data.stage);
  if (el) {
    el.className = 'stage ' + data.status;
  }
}

function showResult(data) {
  const resultEl = document.getElementById('result');
  const btn = document.getElementById('generate');
  btn.disabled = false;
  btn.textContent = 'Generate Shortcut';

  if (data.success) {
    resultEl.className = 'result success';
    let html = '<strong>Shortcut generated successfully!</strong><br><br>';
    if (data.download_filename) {
      html += '<a href="/download/' + encodeURIComponent(data.download_filename) +
              '" download>Download ' + data.download_filename + '</a><br>';
    }
    if (data.imported) {
      html += 'Imported into Shortcuts.app<br>';
    }
    html += '<span style="color:#86868b;font-size:13px;">' +
            data.attempts + ' attempt(s)</span>';

    if (data.warnings && data.warnings.length > 0) {
      html += '<div class="warnings">Warnings: ' +
              data.warnings.map(w => w).join(', ') + '</div>';
    }
    resultEl.innerHTML = html;
  } else {
    resultEl.className = 'result error';
    let html = '<strong>Generation failed</strong>';
    if (data.errors && data.errors.length > 0) {
      html += '<ul class="error-list">';
      data.errors.forEach(e => { html += '<li>' + escapeHtml(e) + '</li>'; });
      html += '</ul>';
    }
    resultEl.innerHTML = html;
  }
  resultEl.style.display = 'block';

  // Show DSL viewer if we have DSL text
  if (data.dsl_text) {
    document.getElementById('dsl-viewer').style.display = 'block';
    document.getElementById('dsl-code').textContent = data.dsl_text;
  }
}

function showError(msg) {
  const resultEl = document.getElementById('result');
  resultEl.className = 'result error';
  resultEl.innerHTML = '<strong>Error:</strong> ' + escapeHtml(msg);
  resultEl.style.display = 'block';
}

function toggleDsl() {
  const code = document.getElementById('dsl-code');
  const btn = code.previousElementSibling || document.querySelector('.dsl-toggle');
  if (code.style.display === 'none' || code.style.display === '') {
    code.style.display = 'block';
    document.querySelector('.dsl-toggle').textContent = 'Hide generated DSL';
  } else {
    code.style.display = 'none';
    document.querySelector('.dsl-toggle').textContent = 'Show generated DSL';
  }
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}
</script>

</body>
</html>
"""


# ── Startup ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print(f"\n  ShortcutForge server starting on http://localhost:{port}")
    print(f"  API docs at http://localhost:{port}/docs")
    print(f"  Output directory: {OUTPUT_DIR}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
