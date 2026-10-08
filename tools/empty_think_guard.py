"""tools/empty_think_guard.py — the fork's template guard, in one place (Minefield trap 04/25).

A pack's `tokenizer/chat_template.jinja` is extracted from the GGUF's own metadata
(`tools/strata_tokenizer.py`), and the server renders the PACK's copy in preference to `serve/chat_template.jinja`
(`server.py`: `ChatTemplate(tpl if tpl.exists() else ROOT / "serve/chat_template.jinja")`).  So editing the repo's
template alone never reaches a deployment: the same guard has to be applied where the template is written, and to
packs that were built before the fix.

What it does: a prior assistant turn whose reasoning is empty (thinking was off for it, or the client did not
resend it) used to render an empty `<think></think>` block.  That nudges the model to skip its reasoning on later
turns and makes two equivalent histories render differently (a conversation-cache miss).  The guard writes the
wrapper only when there is reasoning to preserve; `preserve_empty_think: true` restores the checkpoint's
rendering exactly.

`apply()` is a targeted string replacement, not a rewrite: if the model's template does not have the pattern this
fixes (another architecture's template), it changes nothing and says so, rather than guessing.
"""
from __future__ import annotations

MARKER = "Minefield 04/25"
IDEMPOTENT_GUARD = "{%- if reasoning_content or (preserve_empty_think is defined and preserve_empty_think) %}"

OLD = """        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}"""

NEW = """        {#- Minefield 04/25: a prior assistant turn whose reasoning is empty (thinking was off for it, or the
            client did not resend it) used to render an empty <think></think> block.  That nudges the model to
            skip its reasoning on later turns and makes two equivalent histories render differently (a cache
            miss), so the wrapper is now written only when there is reasoning to preserve.
            "preserve_empty_think": true - in the config or in the request's chat_template_kwargs - restores
            the pack's rendering exactly. -#}
        {%- if reasoning_content or (preserve_empty_think is defined and preserve_empty_think) %}
            {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}"""


def apply(text: str) -> tuple[str, str]:
    """(the guarded text, what happened): "applied", "already guarded" or "pattern not found"."""
    if MARKER in text or IDEMPOTENT_GUARD in text:
        return text, "already guarded"
    if OLD not in text:
        return text, "pattern not found"
    return text.replace(OLD, NEW, 1), "applied"


def revert(text: str) -> tuple[str, str]:
    """Undo: (the checkpoint's text, what happened).  Only the guarded block this module inserted is removed."""
    if MARKER not in text or NEW not in text:
        return text, "nothing to revert"
    return text.replace(NEW, OLD, 1), "reverted"


def main():                                    # a tiny CLI, for a pack or for any template file
    import argparse
    import pathlib
    import sys
    ap = argparse.ArgumentParser(description="apply (or --undo) the empty-think guard to a chat template")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rc = 0
    for p in a.paths:
        path = pathlib.Path(p)
        if not path.exists():
            print(f"{path}: not found"); rc = 1; continue
        old = path.read_text(encoding="utf-8")
        new, what = (revert(old) if a.undo else apply(old))
        if new != old and not a.dry_run:
            path.write_text(new, encoding="utf-8", newline="\n")
        print(f"{path}: {what}" + ("" if new == old else (" (dry run)" if a.dry_run else "")))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
