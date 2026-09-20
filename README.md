# cs6013-evalkit

Local reproduction of the CS6013 grading pipeline. Give the molab notebook a
**GitHub submission URL** and a **HuggingFace URL**; it returns `size_frac` and
per-suite accuracy using the graders' own scripts.

## Layout

```
cs6013-evalkit/
├── README.md
├── .gitignore
├── pyproject.toml                  # copied from the TA bundle (uv sync --extra cuda129)
├── configs/eval_config.yaml        # copied, unmodified
├── evaluation/                     # copied, unmodified
│   ├── common.py
│   ├── run_eval.py
│   └── random_src/normalise.py
├── measure_checkpoint_bits.py      # copied, unmodified
├── ensure_visual_zero.py           # copied, unmodified
├── suites/                         # committed benchmark data (frozen)
│   ├── CS6013_smoke/  CS6013_gsm8k/  CS6013_math500/
│   └── CS6013_amc/    CS6013_aime/
├── tools/
│   ├── eval_suites.py              # suite specs, answer adapters, scoring
│   └── build_suites.py             # one-time builder for suites/
├── notebooks/
│   └── run_eval.py                 # the molab notebook
└── results/                        # one .json + .md per eval run
```

## One-time setup

**1. Copy the graders' scripts in.** From the unzipped bundle:

```bash
KIT=/path/to/CS6013Fall_ProjectEval
cp "$KIT/pyproject.toml" "$KIT/measure_checkpoint_bits.py" "$KIT/ensure_visual_zero.py" .
cp -r "$KIT/configs" "$KIT/evaluation" .
```

Deliberately **not** copied: `environment.sh` (contains a live GitHub token),
`slack.py`, `Week05.csv`, `run_scripts/`, `eval.sh`, `datasets/`. The batch
runner and its Slack/CSV plumbing exist to grade a class; the notebook replaces
them for a single submission.

**2. Build the suites** (needs network, once):

```bash
pip install datasets
python tools/build_suites.py
git add suites/ && git commit -m "Build eval suites"
```

**3. Push**, ideally to a **private** repo. It contains course material, and the
honour code is about not sharing your own work — a private repo keeps both
concerns clear.

## Running an eval

Open `notebooks/run_eval.py` in molab, attach a GPU, and fill in:

- evalkit repo URL (this repo)
- submission GitHub URL — must include `/tree/<branch>/<roll>/WeekNN/Compression_<t>/SubmissionNN`
- submission HuggingFace URL

Roll, week, target and submission number are parsed from the HF repo name and
cross-checked against the GitHub path — the same consistency check that aborts
the graders' batch. Then work down the buttons.

A GitHub token is needed only if a repo is private; an HF token only for a
private model repo.

## The benchmark suites

A difficulty ladder, scored separately, because quantization damage is not
uniform: a 4-bit model usually still nails grade-school arithmetic while
collapsing on ten-step chains, and one aggregate number hides exactly that.

| suite | n | source | role |
|---|---|---|---|
| `smoke` | 8 | hand-written | ~2 min; is anything catastrophically broken |
| `gsm8k` | 100 | `openai/gsm8k` | easy word problems; the floor |
| `math500` | 150 | `HuggingFaceH4/MATH-500` | broad topics, every expert gets traffic |
| `amc` | 83 | `AI-MO/aimo-validation-amc` | competition; where quantization bites |
| `aime` | 90 | `AI-MO/aimo-validation-aime` + `MathArena/aime_2025` | hardest; long reasoning |

`gsm8k` and `math500` are near-certainly in Qwen's training data. That is fine:
these measure *compressed vs baseline on identical inputs*, and contamination
shifts both arms equally. Quote the deltas, not the absolute numbers.

Alongside accuracy and parse rate, each suite reports a **`no-\boxed` rate** —
answers with no boxed expression at all, which almost always means the 32k token
budget ran out mid-reasoning. That is a different failure from answering wrongly
and needs a different fix, so the two are never merged into one number.

## Scoring notes

- `size_frac = compressed_text_GB / 8.0585`, counting **only non-visual**
  tensors. The vision tower is free; `mtp` is not.
- `ensure_visual_zero.py` rewrites the downloaded checkpoint before
  `decompress.py` runs, so the decompressor must tolerate zeroed visual weights.
- vLLM is served with the graders' flags, including `--language-model-only`,
  `--max-model-len 33000`, `--gpu-memory-utilization 0.35`, `--max-num-seqs 50`.
- Sampling: temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5, seed 42,
  thinking enabled. These are constants, not knobs — changing any of them changes
  what the accuracy number means.

## Rebuilding the suites

Only when deliberately changing the benchmark, and say so in the commit message:
results either side of that commit are not comparable.

```bash
python tools/build_suites.py --verify        # check what is committed
python tools/build_suites.py --only aime     # rebuild one
```
