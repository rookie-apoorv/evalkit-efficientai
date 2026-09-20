import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # CS6013 — submission eval

        Give it a **GitHub submission URL** and a **HuggingFace URL**. Everything
        else — roll number, week, compression target, submission number — is parsed
        out of those two links and cross-checked, the same way the graders do it.

        Mirrors `eval.sh` using the graders' own scripts (`measure_checkpoint_bits.py`,
        `ensure_visual_zero.py`, `evaluation/run_eval.py`, `configs/eval_config.yaml`).
        Dropped: Slack, the submissions CSV, the batch loop, the resume logic.

        Attach a GPU from the notebook-specs button before running anything.
        """
    )
    return


@app.cell
def _(mo):
    evalkit_url = mo.ui.text(
        placeholder="https://github.com/<you>/cs6013-evalkit.git",
        label="Evalkit repo (this repo)",
        full_width=True,
    )
    github_url = mo.ui.text(
        placeholder="https://github.com/<you>/CS6013/tree/main/<roll>/Week05/Compression_40/Submission02",
        label="Submission GitHub URL",
        full_width=True,
    )
    hf_url = mo.ui.text(
        placeholder="https://huggingface.co/<you>/<roll>-Week05-Compression40-Submission02",
        label="Submission HuggingFace URL",
        full_width=True,
    )
    gh_token = mo.ui.text(
        placeholder="needed if either GitHub repo is private",
        label="GitHub token", kind="password",
    )
    hf_token = mo.ui.text(
        placeholder="needed only for a private HF repo",
        label="HF token", kind="password",
    )
    base_model_id = mo.ui.text(value="Qwen/Qwen3.5-4B", label="Base model (baseline arm)")
    suite_pick = mo.ui.multiselect(
        options=["smoke", "gsm8k", "math500", "amc", "aime"],
        value=["smoke", "gsm8k", "math500", "amc", "aime"],
        label="Suites to run",
    )

    mo.vstack([
        mo.md("### Inputs"),
        evalkit_url,
        github_url,
        hf_url,
        mo.hstack([gh_token, hf_token], justify="start"),
        mo.hstack([base_model_id, suite_pick], justify="start"),
    ])
    return (
        base_model_id,
        evalkit_url,
        gh_token,
        github_url,
        hf_token,
        hf_url,
        suite_pick,
    )


@app.cell
def _():
    import json
    import os
    import re
    import shutil
    import socket
    import subprocess
    import sys
    import time
    import urllib.request
    from datetime import datetime, timezone
    from pathlib import Path

    WORK = Path("/root/work") if Path("/root").exists() else Path.home() / "work"
    KIT_DIR = WORK / "evalkit"
    REPO_DIR = WORK / "submission_repo"
    COMP_DIR = WORK / "compressed_model"
    DEC_DIR = WORK / "decompressed_model"
    BASE_DIR = WORK / "base_model"
    OUT_DIR = WORK / "eval_outputs"
    for _d in (WORK, OUT_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    # The graders' serving and sampling parameters. Constants, not knobs:
    # changing any of them changes what the accuracy number means.
    MAX_NEW_TOKENS = 32000
    MAX_MODEL_LEN = 1000 + MAX_NEW_TOKENS
    MAX_CONCURRENCY = 30
    SERVED_MODEL_NAME = "qwen-3.5-4b"
    GPU_MEM_UTIL = "0.35"
    MAX_NUM_SEQS = "50"
    ORIGINAL_TEXT_GB = 8.0585

    VLLM = {"proc": None, "port": None, "model": None}

    def run_streaming(cmd, cwd=None, env=None, quiet_prefixes=()):
        merged = {**os.environ, **(env or {})}
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=merged,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:
            if not any(line.startswith(p) for p in quiet_prefixes):
                print(line, end="")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"failed ({proc.returncode}): {' '.join(map(str, cmd))}")

    def capture(cmd, cwd=None):
        return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                              capture_output=True, text=True, check=True).stdout

    def auth_url(url, token):
        if not token:
            return url
        return url.replace("https://", f"https://oauth2:{token}@", 1)

    def free_port():
        s = socket.socket(); s.bind(("", 0)); p = s.getsockname()[1]; s.close()
        return p

    def dir_gb(path):
        path = Path(path)
        if not path.exists():
            return 0.0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30

    def disk_free_gb():
        return shutil.disk_usage(WORK).free / 2**30

    return (
        BASE_DIR,
        COMP_DIR,
        DEC_DIR,
        GPU_MEM_UTIL,
        KIT_DIR,
        MAX_CONCURRENCY,
        MAX_MODEL_LEN,
        MAX_NEW_TOKENS,
        MAX_NUM_SEQS,
        ORIGINAL_TEXT_GB,
        OUT_DIR,
        Path,
        REPO_DIR,
        SERVED_MODEL_NAME,
        VLLM,
        WORK,
        auth_url,
        capture,
        datetime,
        dir_gb,
        disk_free_gb,
        free_port,
        json,
        os,
        re,
        run_streaming,
        shutil,
        socket,
        subprocess,
        sys,
        time,
        timezone,
        urllib,
    )


@app.cell
def _(github_url, hf_url, re):
    def parse_submission(gh: str, hf: str) -> dict:
        """Derive identity from the two links and cross-check them.

        The HuggingFace repo name is the authoritative source for roll / week /
        target / submission, because its format is fully specified. The GitHub
        path is then checked against it. A mismatch is an automatic zero on the
        grading node, which aborts the entire batch before any model runs.
        """
        gh = (gh or "").strip().rstrip("/")
        hf = (hf or "").strip().rstrip("/")
        info = {"ok": False, "errors": [], "warnings": []}
        if not gh or not hf:
            info["errors"].append("Both URLs are required.")
            return info

        hm = re.match(r"^https?://huggingface\.co/([^/]+)/([^/]+)/?$", hf, re.I)
        if not hm:
            info["errors"].append("HF URL must be `https://huggingface.co/<user>/<repo>`.")
            return info
        info["hf_id"] = f"{hm.group(1)}/{hm.group(2)}"

        nm = re.fullmatch(r"^(.+)-(Week\d{2})-Compression-?(\d+)-(Submission\d{2})$", hm.group(2))
        if not nm:
            info["errors"].append(
                f"HF repo name `{hm.group(2)}` must be "
                "`<roll>-WeekNN-Compression<target>-SubmissionNN` "
                "(`Compression40` or `Compression-40`, never `Compression_40`; "
                "two-digit week and submission)."
            )
            return info
        info.update(roll=nm.group(1), week=nm.group(2),
                    target=nm.group(3), submission=nm.group(4))

        gm = re.match(r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/tree/([^/]+)/(.+)$", gh, re.I)
        if not gm:
            info["errors"].append(
                "GitHub URL must be `https://github.com/<user>/CS6013/tree/<branch>/"
                f"{info['roll']}/{info['week']}/Compression_{info['target']}/{info['submission']}`."
            )
            return info
        info.update(gh_owner=gm.group(1), gh_repo=gm.group(2),
                    gh_branch=gm.group(3), gh_subdir=gm.group(4).strip("/"))
        if info["gh_repo"] != "CS6013":
            info["errors"].append(f"GitHub repo must be exactly `CS6013`, got `{info['gh_repo']}`.")

        pm = re.match(r"^([^/]+)/(Week\d{2})/(Compression_[^/]+)/(Submission\d{2})/?$", info["gh_subdir"])
        if not pm:
            info["errors"].append(
                f"GitHub path must be `<roll>/WeekNN/Compression_<target>/SubmissionNN` "
                f"— note the **underscore** in `Compression_{info['target']}`, which the "
                f"HF name must NOT have — got `{info['gh_subdir']}`."
            )
        else:
            if pm.group(1).lower() != info["roll"].lower():
                info["errors"].append(f"roll: GitHub `{pm.group(1)}` vs HF `{info['roll']}`.")
            if pm.group(2) != info["week"]:
                info["errors"].append(f"week: GitHub `{pm.group(2)}` vs HF `{info['week']}`.")
            if pm.group(3) != f"Compression_{info['target']}":
                info["errors"].append(
                    f"target: GitHub `{pm.group(3)}` vs expected `Compression_{info['target']}`.")
            if pm.group(4) != info["submission"]:
                info["errors"].append(
                    f"submission: GitHub `{pm.group(4)}` vs HF `{info['submission']}`.")

        info["ok"] = not info["errors"]
        return info

    SUB = parse_submission(github_url.value, hf_url.value)
    return SUB, parse_submission


@app.cell
def _(SUB, mo):
    def _report():
        if SUB.get("errors") and not SUB.get("roll"):
            return "### Identity\n\n" + "\n".join(f"- {e}" for e in SUB["errors"])
        head = (
            f"### Identity\n\n"
            f"| | |\n|---|---|\n"
            f"| roll | `{SUB.get('roll')}` |\n"
            f"| week | `{SUB.get('week')}` |\n"
            f"| target | `{SUB.get('target')}%` |\n"
            f"| submission | `{SUB.get('submission')}` |\n"
            f"| HF repo | `{SUB.get('hf_id')}` |\n"
            f"| GitHub path | `{SUB.get('gh_subdir')}` |\n\n"
        )
        if SUB["ok"]:
            return head + "Format check **passes**. The graders abort the whole batch on a mismatch here."
        return head + "**Format check FAILS — automatic zero:**\n\n" + \
            "\n".join(f"- {e}" for e in SUB["errors"])

    mo.md(_report())
    return


@app.cell
def _(WORK, disk_free_gb, mo, subprocess, sys):
    def _env():
        out = [f"- Python `{sys.version.split()[0]}`"]
        try:
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True).strip()
            out.append(f"- GPU: **{gpu}**")
        except Exception:
            out.append("- **No GPU.** Attach one from the notebook-specs button.")
        try:
            kb = int(subprocess.check_output(["grep", "MemTotal", "/proc/meminfo"]).split()[1])
            out.append(f"- RAM: {kb / 1024**2:.0f} GiB")
        except Exception:
            pass
        free = disk_free_gb()
        out.append(f"- Disk free at `{WORK}`: **{free:.0f} GiB**")
        out.append("")
        out.append(
            "Needs roughly: eval venv 15, compressed 3, decompressed 8, base model 9 "
            "→ **~35 GiB** with the baseline arm. The baseline cell frees the "
            "decompressed model first; if disk is tight, run the compressed arm, "
            "record the numbers, then do the baseline."
        )
        return "\n".join(out)

    mo.md("### Environment\n" + _env())
    return


@app.cell
def _(mo):
    setup_btn = mo.ui.run_button(label="Clone evalkit + sync environment (~10 min)")
    mo.vstack([
        mo.md(
            "### 1. Evalkit\n"
            "Clones this repo and runs `uv sync --extra cuda129` against the graders' "
            "`pyproject.toml`, pinning vLLM, torch and transformers to their exact versions."
        ),
        setup_btn,
    ])
    return (setup_btn,)


@app.cell
def _(KIT_DIR, auth_url, evalkit_url, gh_token, mo, run_streaming, setup_btn, shutil, sys):
    mo.stop(not setup_btn.value, mo.md("*Not set up.*"))
    mo.stop(not evalkit_url.value, mo.md("**Fill in the evalkit repo URL.**"))

    shutil.rmtree(KIT_DIR, ignore_errors=True)
    run_streaming(["git", "clone", "--depth", "1",
                   auth_url(evalkit_url.value.strip(), gh_token.value), str(KIT_DIR)])

    for _need in ("pyproject.toml", "configs/eval_config.yaml", "evaluation/run_eval.py",
                  "measure_checkpoint_bits.py", "ensure_visual_zero.py", "tools/eval_suites.py"):
        assert (KIT_DIR / _need).exists(), f"evalkit is missing {_need}"

    run_streaming([sys.executable, "-m", "pip", "install", "-q", "uv", "datasets", "pyyaml"])
    print("\nsyncing the graders' environment (slow part) ...")
    run_streaming(["uv", "sync", "--extra", "cuda129"], cwd=KIT_DIR)

    EVAL_PY = KIT_DIR / ".venv" / "bin" / "python"
    assert EVAL_PY.exists(), "uv sync created no .venv"
    print(f"\neval interpreter: {EVAL_PY}")
    return (EVAL_PY,)


@app.cell
def _(EVAL_PY, KIT_DIR, mo):
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("eval_suites", str(KIT_DIR / "tools" / "eval_suites.py"))
    ES = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(ES)

    from datasets import load_dataset as _ld

    SUITES = {}
    _lines = []
    for _name in ES.SUITE_NAMES:
        _p = KIT_DIR / "suites" / f"CS6013_{_name}"
        if not _p.exists():
            _lines.append(f"| `{_name}` | — | **missing** |")
            continue
        _ds = _ld(str(_p), split="train")
        SUITES[_name] = (str(_p), len(_ds))
        _lines.append(f"| `{_name}` | {len(_ds)} | `{_ds[0]['answer']}` |")

    mo.md(
        "### 2. Benchmark suites (committed, frozen)\n\n"
        "| suite | problems | first gold answer |\n|---|---|---|\n"
        + "\n".join(_lines)
        + "\n\nRebuild only with `python tools/build_suites.py`, and note it in the "
        "commit message — results either side of that commit are not comparable."
    )
    return ES, SUITES


@app.cell
def _(mo):
    clone_btn = mo.ui.run_button(label="Sparse-clone the submission")
    clone_btn
    return (clone_btn,)


@app.cell
def _(REPO_DIR, SUB, auth_url, clone_btn, gh_token, mo, run_streaming, shutil):
    mo.stop(not clone_btn.value, mo.md("*Not cloned.*"))
    mo.stop(not SUB.get("gh_subdir"), mo.md("**Fix the URLs first.**"))

    _url = f"https://github.com/{SUB['gh_owner']}/{SUB['gh_repo']}.git"
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    # Same sparse strategy as eval.sh: only the submission folder is fetched.
    run_streaming(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                   "--no-checkout", "--branch", SUB["gh_branch"],
                   auth_url(_url, gh_token.value), str(REPO_DIR)])
    run_streaming(["git", "-C", str(REPO_DIR), "sparse-checkout", "set", "--cone", SUB["gh_subdir"]])
    if not (REPO_DIR / SUB["gh_subdir"]).exists():
        run_streaming(["git", "-C", str(REPO_DIR), "checkout", SUB["gh_branch"]])

    SUB_DIR = REPO_DIR / SUB["gh_subdir"]
    assert SUB_DIR.is_dir(), f"sparse checkout produced no {SUB_DIR}"

    _req = ["compress.py", "decompress.py", "pyproject.toml", "README.md",
            "compression/__init__.py", "decompression/__init__.py"]
    _missing = [f for f in _req if not (SUB_DIR / f).is_file()]
    _missing += [f"{d}/" for d in ("compression", "decompression") if not (SUB_DIR / d).is_dir()]
    print(f"submission: {SUB_DIR}\n")
    if _missing:
        raise RuntimeError(f"LAYOUT INCOMPLETE — graders fail at 0_format. Missing: {_missing}")
    print("layout OK: " + ", ".join(_req))
    return (SUB_DIR,)


@app.cell
def _(mo):
    download_btn = mo.ui.run_button(label="Download + measure + zero visual")
    mo.vstack([
        mo.md(
            "### 3. Download, size, visual zeroing\n"
            "`size_frac = compressed_text_GB / 8.0585`, where **only non-visual "
            "tensors count** — the vision tower is free, `mtp` is not. Then "
            "`ensure_visual_zero.py` rewrites the checkpoint in place, so "
            "`decompress.py` runs against the zeroed version exactly as on the "
            "grading node."
        ),
        download_btn,
    ])
    return (download_btn,)


@app.cell
def _(
    COMP_DIR,
    EVAL_PY,
    KIT_DIR,
    ORIGINAL_TEXT_GB,
    SUB,
    capture,
    dir_gb,
    download_btn,
    hf_token,
    json,
    mo,
    os,
    run_streaming,
    shutil,
):
    mo.stop(not download_btn.value, mo.md("*Not downloaded.*"))

    if hf_token.value:
        os.environ["HF_TOKEN"] = hf_token.value
    from huggingface_hub import snapshot_download as _snap

    shutil.rmtree(COMP_DIR, ignore_errors=True)
    _snap(SUB["hf_id"], local_dir=str(COMP_DIR))
    print(f"{SUB['hf_id']} -> {dir_gb(COMP_DIR):.3f} GiB on disk\n")

    _banned = [f.name for f in COMP_DIR.iterdir()
               if f.is_file() and f.suffix.lower() in {".md", ".py", ".ipynb", ".log"}]
    if _banned:
        print(f"WARNING: files the handout bans from the HF checkpoint: {_banned}\n")

    _m = json.loads(capture([EVAL_PY, str(KIT_DIR / "measure_checkpoint_bits.py"),
                             "--json", str(COMP_DIR)]))
    SIZE_FRAC = _m["text_gb"] / ORIGINAL_TEXT_GB
    print(f"total   {_m['total_gb']:.4f} GiB")
    print(f"visual  {_m['visual_gb']:.4f} GiB  (excluded from size_frac)")
    print(f"text    {_m['text_gb']:.4f} GiB  (language_model + mtp)")
    print(f"\nsize_frac = {SIZE_FRAC:.4f}   target {int(SUB['target']) / 100:.2f}")
    print("WITHIN target" if SIZE_FRAC <= int(SUB["target"]) / 100
          else "*** OVER TARGET — this submission fails ***")

    print("\nzeroing visual tensors in place ...")
    run_streaming([EVAL_PY, str(KIT_DIR / "ensure_visual_zero.py"), str(COMP_DIR)])
    return (SIZE_FRAC,)


@app.cell
def _(mo):
    decompress_btn = mo.ui.run_button(label="uv sync submission + decompress")
    mo.vstack([
        mo.md(
            "### 4. Decompress\n"
            "Builds a venv from **the submission's** `pyproject.toml` and runs "
            "`decompress.py` with it — the graders' `3_venv` and `4_decompress` steps. "
            "`--model_name` is the literal string `Qwen-3.5-4B`, which is not a "
            "resolvable HF id: a decompressor that tries to download it dies here."
        ),
        decompress_btn,
    ])
    return (decompress_btn,)


@app.cell
def _(COMP_DIR, DEC_DIR, SUB_DIR, decompress_btn, dir_gb, mo, run_streaming, shutil):
    mo.stop(not decompress_btn.value, mo.md("*Not decompressed.*"))

    run_streaming(["uv", "sync"], cwd=SUB_DIR)
    _py = SUB_DIR / ".venv" / "bin" / "python"
    if not _py.exists():
        raise RuntimeError("uv sync produced no .venv in the submission dir")

    shutil.rmtree(DEC_DIR, ignore_errors=True)
    DEC_DIR.mkdir(parents=True)
    run_streaming([str(_py), "decompress.py",
                   "--model_name", "Qwen-3.5-4B",
                   "--checkpoint_path", str(COMP_DIR),
                   "--output_path", str(DEC_DIR)], cwd=SUB_DIR)
    if not ((DEC_DIR / "config.json").exists() or any(DEC_DIR.glob("*.safetensors"))):
        raise RuntimeError("decompress.py exited 0 but wrote no checkpoint")
    print(f"\nrestored: {dir_gb(DEC_DIR):.3f} GiB")
    return


@app.cell
def _(
    GPU_MEM_UTIL,
    KIT_DIR,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    SERVED_MODEL_NAME,
    VLLM,
    free_port,
    os,
    subprocess,
    time,
    urllib,
):
    def start_vllm(model_path, log_path):
        if VLLM["proc"] and VLLM["proc"].poll() is None:
            if VLLM["model"] == str(model_path):
                print(f"reusing server on port {VLLM['port']}")
                return VLLM["port"]
            stop_vllm()

        port = free_port()
        env = {**os.environ,
               "PATH": f"{KIT_DIR / '.venv' / 'bin'}:{os.environ['PATH']}",
               "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        # Flags copied verbatim from eval.sh. --language-model-only skips the
        # vision tower; the modest memory/seq caps exist because Qwen3.5's
        # GDN/Mamba cache blocks must accommodate max_num_seqs.
        cmd = [str(KIT_DIR / ".venv" / "bin" / "python"),
               "-m", "vllm.entrypoints.openai.api_server",
               "--model", str(model_path),
               "--served-model-name", SERVED_MODEL_NAME,
               "--tensor-parallel-size", "1",
               "--dtype", "bfloat16",
               "--max-model-len", str(MAX_MODEL_LEN),
               "--language-model-only",
               "--port", str(port),
               "--gpu-memory-utilization", GPU_MEM_UTIL,
               "--max-num-seqs", MAX_NUM_SEQS]
        log = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        VLLM.update(proc=proc, port=port, model=str(model_path))
        print(f"vLLM PID={proc.pid} port={port}  log={log_path}")

        deadline = time.time() + 1800
        while time.time() < deadline:
            if proc.poll() is not None:
                print(open(log_path).read()[-3000:])
                raise RuntimeError("vLLM exited before becoming ready")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                print("server ready")
                return port
            except Exception:
                time.sleep(10)
        raise RuntimeError("vLLM did not become ready within 30 min")

    def stop_vllm():
        proc = VLLM.get("proc")
        if proc and proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), 15)
            try:
                proc.wait(timeout=60)
            except Exception:
                os.killpg(os.getpgid(proc.pid), 9)
            print("vLLM stopped")
        VLLM.update(proc=None, port=None, model=None)

    return start_vllm, stop_vllm


@app.cell
def _(
    ES,
    EVAL_PY,
    KIT_DIR,
    MAX_CONCURRENCY,
    MAX_NEW_TOKENS,
    OUT_DIR,
    SUITES,
    run_streaming,
    suite_pick,
    time,
):
    def run_suites(port, tag):
        import yaml

        base_cfg = yaml.safe_load((KIT_DIR / "configs" / "eval_config.yaml").read_text())
        rows = []
        for name in ES.SUITE_NAMES:
            if name not in suite_pick.value or name not in SUITES:
                continue
            path, n = SUITES[name]
            out_json = OUT_DIR / f"{tag}_{name}.json"
            cfg = dict(base_cfg)
            cfg["output"] = str(out_json)
            cfg["vllm_base_url"] = f"http://localhost:{port}/v1"
            tmp = OUT_DIR / f"cfg_{tag}_{name}.yaml"
            tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))

            print(f"\n{'=' * 70}\n[{tag}] {name} — {n} problems\n{'=' * 70}", flush=True)
            t0 = time.time()
            run_streaming(
                [EVAL_PY, "evaluation/run_eval.py", "--config", str(tmp),
                 "--dataset", path, "--port", str(port), "--limit", str(n),
                 "--max-new-tokens", str(MAX_NEW_TOKENS),
                 "--max-concurrency", str(MAX_CONCURRENCY)],
                cwd=KIT_DIR,
                quiet_prefixes=("Submitted request", "DEBUG ROW:", "{'problem_idx'"),
            )
            s = ES.score_output(out_json)
            s["suite"] = name
            s["minutes"] = (time.time() - t0) / 60
            rows.append(s)
            print(f"  accuracy={s['accuracy']:.3f}  parse={s['parse_rate']:.3f}  "
                  f"no-boxed={s['truncation_rate']:.3f}  ({s['minutes']:.1f} min)")
        return rows

    return (run_suites,)


@app.cell
def _(mo):
    eval_btn = mo.ui.run_button(label="Serve + evaluate the compressed model")
    mo.vstack([
        mo.md(
            "### 5. Evaluate\n"
            "Their config: temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5, "
            "seed 42, thinking enabled, 32k token budget. Budget 30–120 min depending "
            "on how long `aime` reasons."
        ),
        eval_btn,
    ])
    return (eval_btn,)


@app.cell
def _(DEC_DIR, eval_btn, mo, run_suites, start_vllm):
    mo.stop(not eval_btn.value, mo.md("*Not evaluated.*"))
    _port = start_vllm(DEC_DIR, "/root/work/vllm_compressed.log")
    COMPRESSED_ROWS = run_suites(_port, "compressed")
    return (COMPRESSED_ROWS,)


@app.cell
def _(COMPRESSED_ROWS, ES, SIZE_FRAC, SUB, mo):
    mo.md(
        f"### Results — {SUB['roll']} {SUB['week']} {SUB['submission']}\n\n"
        f"`size_frac = {SIZE_FRAC:.4f}`\n\n"
        + ES.format_results_table(COMPRESSED_ROWS)
        + "\n`no-\\boxed` is the share of answers with no boxed expression at all — "
        "usually the token budget ran out mid-reasoning. If it climbs on `aime` while "
        "accuracy holds elsewhere, the problem is reasoning length, not arithmetic."
    )
    return


@app.cell
def _(mo):
    baseline_btn = mo.ui.run_button(label="Free disk, download original, run baseline")
    mo.vstack([
        mo.md(
            "### 6. Baseline\n"
            "Stops the server, deletes the decompressed model for room, downloads the "
            "original bf16 checkpoint and reruns the same suites. The **delta** is the "
            "number for your report — absolute accuracy on public sets says much less, "
            "since `gsm8k` and `math500` are almost certainly in Qwen's training data. "
            "Contamination shifts both arms equally, so the difference stays honest."
        ),
        baseline_btn,
    ])
    return (baseline_btn,)


@app.cell
def _(
    BASE_DIR,
    DEC_DIR,
    base_model_id,
    baseline_btn,
    disk_free_gb,
    mo,
    run_suites,
    shutil,
    start_vllm,
    stop_vllm,
):
    mo.stop(not baseline_btn.value, mo.md("*Baseline not run.*"))
    stop_vllm()
    shutil.rmtree(DEC_DIR, ignore_errors=True)
    print(f"freed decompressed model; {disk_free_gb():.0f} GiB free")

    from huggingface_hub import snapshot_download as _snap2

    _snap2(base_model_id.value, local_dir=str(BASE_DIR),
           ignore_patterns=["*.pth", "*.bin", "*.msgpack", "*.h5"])
    BASELINE_ROWS = run_suites(start_vllm(BASE_DIR, "/root/work/vllm_baseline.log"), "baseline")
    return (BASELINE_ROWS,)


@app.cell
def _(BASELINE_ROWS, COMPRESSED_ROWS, ES, SIZE_FRAC, mo):
    mo.md(
        f"### Compressed vs baseline\n\n`size_frac = {SIZE_FRAC:.4f}`\n\n"
        + ES.format_comparison_table(COMPRESSED_ROWS, BASELINE_ROWS)
        + "\nRead the ladder, not the average. Damage confined to `amc` and `aime` "
        "points at the quantization grid being too coarse for long dependency chains; "
        "damage reaching `gsm8k` means something is structurally wrong."
    )
    return


@app.cell
def _(mo):
    save_btn = mo.ui.run_button(label="Save this run to the evalkit results/")
    mo.vstack([
        mo.md(
            "### 7. Record the run\n"
            "Writes a JSON and a markdown summary into the cloned evalkit. Commit them "
            "so accuracy-versus-`size_frac` across submissions stays in one place."
        ),
        save_btn,
    ])
    return (save_btn,)


@app.cell
def _(COMPRESSED_ROWS, ES, KIT_DIR, SIZE_FRAC, SUB, datetime, json, mo, save_btn, timezone):
    mo.stop(not save_btn.value, mo.md("*Not saved.*"))

    # Looked up dynamically on purpose: naming BASELINE_ROWS directly would make
    # marimo treat it as a dependency, so this cell could not run until the
    # baseline arm had. Saving a compressed-only run has to stay possible.
    _base = globals().get("BASELINE_ROWS")

    _stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _tag = f"{SUB['roll']}-{SUB['week']}-{SUB['submission']}-{_stamp}"
    _dir = KIT_DIR / "results"
    _dir.mkdir(parents=True, exist_ok=True)

    _payload = {
        "submission": {k: SUB.get(k) for k in
                       ("roll", "week", "target", "submission", "hf_id", "gh_subdir")},
        "size_frac": SIZE_FRAC,
        "compressed": COMPRESSED_ROWS,
        "baseline": _base,
        "timestamp": _stamp,
    }
    (_dir / f"{_tag}.json").write_text(json.dumps(_payload, indent=2))

    _md = (f"# {SUB['roll']} {SUB['week']} {SUB['submission']}\n\n"
           f"- HF: `{SUB['hf_id']}`\n- size_frac: **{SIZE_FRAC:.4f}** "
           f"(target {int(SUB['target']) / 100:.2f})\n\n"
           + ES.format_results_table(COMPRESSED_ROWS))
    if _base:
        _md += "\n## vs baseline\n\n" + ES.format_comparison_table(COMPRESSED_ROWS, _base)
    (_dir / f"{_tag}.md").write_text(_md)

    mo.md(
        f"Saved `results/{_tag}.json` and `.md`.\n\n"
        "To keep them, from a terminal in the evalkit clone:\n\n"
        f"```bash\ncd {KIT_DIR}\ngit add results/ && git commit -m 'eval {_tag}' && git push\n```"
    )
    return


@app.cell
def _(mo):
    cleanup_btn = mo.ui.run_button(label="Stop vLLM and free everything")
    cleanup_btn
    return (cleanup_btn,)


@app.cell
def _(BASE_DIR, COMP_DIR, DEC_DIR, cleanup_btn, disk_free_gb, mo, shutil, stop_vllm):
    mo.stop(not cleanup_btn.value, mo.md("*Artifacts kept.*"))
    stop_vllm()
    for _d in (DEC_DIR, BASE_DIR, COMP_DIR):
        shutil.rmtree(_d, ignore_errors=True)
    mo.md(f"Cleaned. {disk_free_gb():.0f} GiB free.")
    return
