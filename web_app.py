#!/usr/bin/env python3
"""Local-only web UI for the offline M4A transcriber."""

from __future__ import annotations

import subprocess
import sys
import threading
import uuid
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request, send_file
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
JOB_DIR = BASE_DIR / ".web_jobs"
JOB_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()

PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>本地录音转文字</title>
  <style>
    :root { color-scheme: light; --ink:#18212f; --muted:#637083; --line:#dbe2ea; --blue:#1769e0; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:linear-gradient(135deg,#eef5ff,#f9fbfd 52%,#eefaf5); }
    main { width:min(720px,calc(100% - 32px)); margin:52px auto; background:#fff; border:1px solid rgba(219,226,234,.8); border-radius:22px; padding:34px; box-shadow:0 20px 60px rgba(40,65,95,.12); }
    h1 { margin:0 0 8px; font-size:30px; }
    .sub { color:var(--muted); margin:0 0 28px; }
    label { display:block; font-weight:650; margin:18px 0 8px; }
    input,select,button { width:100%; font:inherit; border-radius:12px; }
    input,select { border:1px solid var(--line); padding:12px 14px; background:#fff; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    button { margin-top:24px; border:0; padding:14px; color:#fff; background:var(--blue); font-weight:700; cursor:pointer; }
    button:disabled { opacity:.55; cursor:wait; }
    #status { display:none; margin-top:24px; padding:16px; border-radius:14px; background:#f5f8fc; white-space:pre-wrap; line-height:1.55; }
    .note { margin-top:24px; color:var(--muted); font-size:14px; }
    a { color:var(--blue); font-weight:700; }
    @media(max-width:580px){ main{margin:20px auto;padding:24px}.grid{grid-template-columns:1fr} }
  </style>
</head>
<body><main>
  <h1>本地录音转文字</h1>
  <p class="sub">中英混合识别 · 自动区分讲话人 · 音频不上传</p>
  <form id="form">
    <label for="audio">选择 M4A 录音</label>
    <input id="audio" name="audio" type="file" accept=".m4a,audio/mp4" required>
    <div class="grid">
      <div><label for="speakers">讲话人数（可留空）</label><input id="speakers" name="speakers" type="number" min="1" placeholder="例如 2"></div>
      <div><label for="model">识别模型</label><select id="model" name="model"><option value="small">small（推荐）</option><option value="medium">medium（更准确、更慢）</option><option value="large-v3">large-v3（最慢）</option></select></div>
    </div>
    <button id="submit" type="submit">开始转写</button>
  </form>
  <div id="status"></div>
  <p class="note">网页只在本机运行。首次使用前，请在启动网页的终端设置 <code>HF_TOKEN</code>。</p>
</main>
<script>
const form=document.querySelector('#form'), statusBox=document.querySelector('#status'), button=document.querySelector('#submit');
function show(text){statusBox.style.display='block';statusBox.textContent=text;}
async function poll(id){
  const response=await fetch('/status/'+id), data=await response.json();
  show(data.message || '正在处理……');
  if(data.state==='done'){statusBox.innerHTML='转写完成。<a href="/download/'+id+'">下载 TXT</a>';button.disabled=false;button.textContent='开始转写';return;}
  if(data.state==='error'){button.disabled=false;button.textContent='重新尝试';return;}
  setTimeout(()=>poll(id),1500);
}
form.addEventListener('submit',async event=>{
  event.preventDefault();button.disabled=true;button.textContent='正在上传到本机……';show('正在将文件复制到本机工作目录……');
  try{const response=await fetch('/start',{method:'POST',body:new FormData(form)}),data=await response.json();if(!response.ok)throw new Error(data.error||'启动失败');button.textContent='正在转写……';poll(data.job_id);}catch(error){show('错误：'+error.message);button.disabled=false;button.textContent='重新尝试';}
});
</script></body></html>
"""


def update_job(job_id: str, **values: Any) -> None:
    with jobs_lock:
        jobs[job_id].update(values)


def run_job(job_id: str, source: Path, output: Path, model: str, speakers: int | None) -> None:
    command = [
        sys.executable,
        str(BASE_DIR / "transcribe_m4a.py"),
        str(source),
        "--output", str(output),
        "--model", model,
    ]
    if speakers is not None:
        command.extend(["--speakers", str(speakers)])
    if not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")):
        command.append("--offline")
    update_job(job_id, state="running", message="模型正在本机转写，长录音可能需要较久……")
    try:
        process = subprocess.Popen(
            command,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        log: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.strip()
            if clean:
                log.append(clean)
                update_job(job_id, message="\n".join(log[-6:]))
        return_code = process.wait()
        if return_code != 0 or not output.is_file():
            raise RuntimeError("\n".join(log[-10:]) or "转写进程失败")
        update_job(job_id, state="done", message="转写完成", output=str(output))
    except Exception as exc:
        update_job(job_id, state="error", message=f"错误：{exc}")


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/start")
def start():
    upload = request.files.get("audio")
    if upload is None or not upload.filename:
        return jsonify(error="请选择 M4A 文件"), 400
    if Path(upload.filename).suffix.lower() != ".m4a":
        return jsonify(error="只支持 .m4a 文件"), 400
    model = request.form.get("model", "small")
    if model not in {"small", "medium", "large-v3"}:
        return jsonify(error="模型选项无效"), 400
    speakers_text = request.form.get("speakers", "").strip()
    try:
        speakers = int(speakers_text) if speakers_text else None
        if speakers is not None and speakers < 1:
            raise ValueError
    except ValueError:
        return jsonify(error="讲话人数必须是正整数"), 400

    job_id = uuid.uuid4().hex
    folder = JOB_DIR / job_id
    folder.mkdir()
    original_name = secure_filename(upload.filename) or "recording.m4a"
    source = folder / original_name
    output = folder / f"{Path(original_name).stem}.txt"
    upload.save(source)
    with jobs_lock:
        jobs[job_id] = {"state": "queued", "message": "等待开始……"}
    threading.Thread(
        target=run_job,
        args=(job_id, source, output, model, speakers),
        daemon=True,
    ).start()
    return jsonify(job_id=job_id)


@app.get("/status/<job_id>")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify(state="error", message="找不到任务"), 404
        return jsonify(state=job["state"], message=job["message"])


@app.get("/download/<job_id>")
def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        path = Path(job["output"]) if job and job.get("state") == "done" else None
    if path is None or not path.is_file():
        return "文件尚未生成", 404
    return send_file(path, as_attachment=True, download_name=path.name)


if __name__ == "__main__":
    print("请在浏览器打开：http://127.0.0.1:7860")
    app.run(host="127.0.0.1", port=7860, debug=False)
