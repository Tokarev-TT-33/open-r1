#!/usr/bin/env bash
set -euo pipefail

# Apply the local beam text fix to the lighteval vLLM backend in the current workspace/venv.
# This is needed because the bug lives in site-packages and can be lost when recreating the venv.
#
# Usage:
#   bash scripts/apply_lighteval_beam_fix.sh
#

TARGET="/root/open-r1/openr1/lib/python3.11/site-packages/lighteval/models/vllm/vllm_model.py"

if [[ ! -f "$TARGET" ]]; then
  echo "[ERROR] target file not found: $TARGET"
  exit 1
fi

python - <<'PY'
from pathlib import Path
import re

path = Path("/root/open-r1/openr1/lib/python3.11/site-packages/lighteval/models/vllm/vllm_model.py")
txt = path.read_text(encoding="utf-8")

old = re.compile(r"\n\s*text = seq\.text\s*\n\s*if text is None:\s*\n\s*text = self\.tokenizer\.decode\(gen_token_ids, skip_special_tokens=True\)\s*\n")
if not old.search(txt):
    print("[INFO] Patch pattern not found; either already patched or upstream changed. No-op.")
    raise SystemExit(0)

replacement = """
                    # IMPORTANT:
                    # vLLM BeamSearchSequence.text may include the *full* decoded sequence (prompt + generation).
                    # Downstream metrics (e.g. MATH-500) often extract the "first match" from the completion text;
                    # if the prompt is included, they may accidentally extract math from the *question* and mark
                    # almost everything wrong. Therefore we ensure `text` contains only the generated suffix.
                    text = None
                    if getattr(seq, "text", None):
                        full_text = seq.text
                        # Best-effort strip of prompt prefix if present.
                        try:
                            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                            if full_text.startswith(prompt_text):
                                text = full_text[len(prompt_text) :].lstrip()
                            else:
                                text = None
                        except Exception:
                            text = None
                    if text is None:
                        text = self.tokenizer.decode(gen_token_ids, skip_special_tokens=True)
"""

txt2 = old.sub("\n" + replacement + "\n", txt, count=1)
path.write_text(txt2, encoding="utf-8")
print("[OK] Applied beam text fix to:", path)
PY


