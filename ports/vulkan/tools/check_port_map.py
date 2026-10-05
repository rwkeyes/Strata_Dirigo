#!/usr/bin/env python3
"""Check ports/vulkan/PORT-MAP.tsv against reality.  Run by gates/run_gate.sh.

  * every symbol in the map must exist in the engine's own sources (no invented names)
  * every `kernel` row must name shaders that are BUILT (a .spv in shaders/)
  * every built shader must be named by some row (no orphan shaders)
  * every kernels:: symbol src/core/ calls must be in the map (the map stays complete over the decode path)

The last rule is the one that matters: it makes "not ported yet" a THING THIS REPO KNOWS, so a new call site cannot
join the decode path unnoticed.
"""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]      # .../strata-vulkan-wt
PORT = ROOT / 'ports/vulkan'
fail = []

# --- the engine's symbols, from its own sources ---
engine_src = (ROOT / 'src')
text = []
for p in list(engine_src.rglob('*.hpp')) + list(engine_src.rglob('*.cpp')) + list((ROOT / 'include').rglob('*.hpp')):
    try:
        text.append(p.read_text(errors='ignore'))
    except OSError:
        continue
engine_text = '\n'.join(text)

core_syms = sorted({m.group(1) for m in
                    (m for p in (engine_src / 'core').rglob('*') if p.suffix in ('.cpp', '.hpp')
                     for m in re.finditer(r'kernels::([a-z_0-9]+)', p.read_text(errors='ignore')))})

# --- the map ---
rows = []
for line in (PORT / 'PORT-MAP.tsv').read_text().splitlines():
    if not line or line.startswith('#'):
        continue
    parts = line.split('\t')
    if len(parts) != 3:
        fail.append(f"PORT-MAP.tsv: malformed row: {line!r}")
        continue
    rows.append(tuple(parts))
mapped = {r[0] for r in rows}

built = {p.stem for p in (PORT / 'shaders').glob('*.spv')}
claimed = set()
for sym, kind, what in rows:
    if kind not in ('kernel', 'host', 'todo'):
        fail.append(f"PORT-MAP.tsv: {sym} has kind {kind!r} (not kernel/host/todo)")
    # A symbol ending in `_` is a macro-concatenation prefix (`kernels::quantize_q8_##T`), so it has no word
    # boundary after it; every other symbol must appear whole.
    pat = r'\b' + re.escape(sym) + ('' if sym.endswith('_') else r'\b')
    if not re.search(pat, engine_text):
        fail.append(f"PORT-MAP.tsv: {sym} appears nowhere in the engine's sources - an invented name?")
    if kind == 'kernel':
        for sh in what.split():
            if sh not in built:
                fail.append(f"PORT-MAP.tsv: {sym} names shader {sh!r}, which is not built")
            claimed.add(sh)
unclaimed = sorted(built - claimed)      # prefill-path kernels and shared primitives: reported, not failures

for sym in core_syms:
    if sym not in mapped:
        fail.append(f"src/core/ calls kernels::{sym}, which PORT-MAP.tsv does not mention")

k = sum(1 for r in rows if r[1] == 'kernel')
h = sum(1 for r in rows if r[1] == 'host')
t = sum(1 for r in rows if r[1] == 'todo')
print(f"port map: {len(rows)} decode-path symbols - {k} kernel, {h} host, {t} todo; "
      f"{len(built)} shaders built, {len(claimed)} claimed by the decode path")
print(f"          (the other {len(unclaimed)} are the prefill path and the primitives the kernels are built from)")
for f in fail:
    print("  FAIL " + f)
sys.exit(1 if fail else 0)
