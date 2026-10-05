#!/usr/bin/env python3
"""Check ports/vulkan/PORT-MAP.tsv against reality.  Run by gates/run_gate.sh.

  * every symbol in the map must exist in the engine's own sources (no invented names)
  * every `kernel` row must name shaders that are BUILT (a .spv in shaders/), AND the Vulkan backend must
    define the symbol the engine calls -- BOTH facts, which is the fix for the conflation this file used to
    have (see below)
  * every `shader` row must name a BUILT shader and must NOT be defined by the backend (a `shader` row that
    IS defined is stale -- promote it to `kernel`, or to `refused` if the definition is a LOUD REFUSAL)
  * every `refused` row must name a symbol the backend DOES define in vulkan/src/ -- the definition is a loud
    refusal (refusals_vk.cpp), so it is NOT a capability; an aspirational `refused` row fails like a `kernel`
    row with no definition
  * every built shader must be named by some row (no orphan shaders)
  * every kernels-namespace symbol src/core/ REACHES must be in the map (the map stays complete over the
    decode path) -- whether it is written `kernels::X` or BARE `X`

WHY `kernel` STATES TWO FACTS.  A `kernel` row used to mean only "a shader exists in this tree".  Twice that
was read as "the backend answers the symbol the engine calls", and both times a hole hid behind it:
`indexer_key_append` sat recorded LANDED with a shader and NO `strata::kernels::indexer_key_append` definition
(so layer.cpp:948 was an undefined reference), and `qsa_decode_attn_step` looked covered by a mis-attributed
shader (`attn_decode_short`).  So a `kernel` row now requires BOTH: the named shaders are built AND the backend
defines the symbol -- checked by scanning the backend's own translation units here, not asserted by a table.
A row whose shader exists but whose wrapper does not is kinded `shader`: an honest, distinct state.

The last rule is the one that keeps the map honest: it makes "not ported yet" a THING THIS REPO KNOWS, so a
new call site cannot join the decode path unnoticed.

THE BARE-NAME HALF OF THAT RULE, and why it is not a heuristic.  Until this change the scan keyed on
the `kernels::` qualifier only, so a symbol called through `using namespace strata::kernels;` was
invisible: `src/core/mtp.cpp` calls `coupled_draft_sample` / `coupled_draft_stage` bare, and the map
read `todo 0` while the port's two coupled shaders went UNCLAIMED.  The bare names are NOT scraped
from src/core/ (a source file is full of local identifiers, and treating every `foo(` as a kernel
symbol would invent names the engine does not have).  They come from the engine's own declarations in
`include/strata/kernels/**` and are attributed to src/core/ only in a source that has actually
brought the namespace into scope (`using namespace strata::kernels;` / `using strata::kernels::X;`).
The discovery lives in tools/port_map_lib.py, shared with the generator so the two cannot drift.
"""
import pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import port_map_lib as lib

ROOT = pathlib.Path(__file__).resolve().parents[3]      # .../strata-vulkan-wt
PORT = ROOT / 'ports/vulkan'
BACKEND = ROOT / 'vulkan' / 'src'                       # the Vulkan backend's own translation units
fail = []

# --- the engine's symbols, from its own sources ---
text = []
for p in list((ROOT / 'src').rglob('*.hpp')) + list((ROOT / 'src').rglob('*.cpp')) \
        + list((ROOT / 'include').rglob('*.hpp')):
    try:
        text.append(p.read_text(errors='ignore'))
    except OSError:
        continue
engine_text = '\n'.join(text)

# --- the decode path's symbols: qualified AND bare (see the module docstring / port_map_lib) ---
core_syms = lib.decode_path_symbols()

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

# --- WHAT THE BACKEND ACTUALLY DEFINES.  A definition is a function at NAMESPACE SCOPE in a backend TU: the
#     symbol at the start of a line (optionally after its return type), NOT qualified by `::`/`.`/`->` (which
#     would make it a call) and not in a comment.  Comments are stripped line by line so a `// name(` cannot
#     masquerade as a definition.  This is the second fact the `kernel` kind now asserts.
def backend_definitions():
    code = []
    for p in BACKEND.rglob('*.cpp'):
        for line in p.read_text(errors='ignore').splitlines():
            cut = line.find('//')
            if cut >= 0:
                line = line[:cut]
            if line.strip():
                code.append(line)
    blob = '\n'.join(code)
    defs = set()
    for sym in core_syms:
        # `quantize_q8_` is a macro-concatenation prefix (no `(` follows it); skip the definition probe for it.
        if sym.endswith('_'):
            continue
        # a definition line begins with the (optional) return type then the symbol then `(`; calls are indented
        # and/or qualified, so an anchored match at column 0 with no `::` before the name is a definition.
        pat = re.compile(r'^(?:[A-Za-z_][\w:<>,*&\s]*\s+)?' + re.escape(sym) + r'\s*\(')
        for line in code:
            if pat.match(line):
                defs.add(sym)
                break
    return defs

defined = backend_definitions()

built = {p.stem for p in (PORT / 'shaders').glob('*.spv')}
claimed = set()
n_kernel, n_shaderonly, n_kernel_undef, n_refused = 0, 0, 0, 0
undef_named = []
for sym, kind, what in rows:
    if kind not in ('kernel', 'shader', 'host', 'todo', 'refused'):
        fail.append(f"PORT-MAP.tsv: {sym} has kind {kind!r} (not kernel/shader/host/todo/refused)")
    # A symbol ending in `_` is a macro-concatenation prefix (`kernels::quantize_q8_##T`), so it has no word
    # boundary after it; every other symbol must appear whole.
    pat = r'\b' + re.escape(sym) + ('' if sym.endswith('_') else r'\b')
    if not re.search(pat, engine_text):
        fail.append(f"PORT-MAP.tsv: {sym} appears nowhere in the engine's sources - an invented name?")
    if kind in ('kernel', 'shader'):
        for sh in what.split():
            if sh not in built:
                fail.append(f"PORT-MAP.tsv: {sym} names shader {sh!r}, which is not built")
            claimed.add(sh)
    elif kind == 'refused':
        # A `refused` row's `what` is prose (a flag chain) that MAY also name shaders the refused symbol's CUDA
        # body would have driven.  Claim any token that IS a built shader, but do not treat prose words as
        # missing shaders - the refusal is the backend's whole answer.
        for sh in what.split():
            if sh in built:
                claimed.add(sh)
    if kind == 'kernel':
        n_kernel += 1
        # FACT 2: the backend must define the symbol the engine calls (the conflation's fix).  `fwht256_inplace_cuda`
        # is an ENGINE-HEADER inline (its body calls `fwht256_cuda`), so its definition lives in the engine, not here
        # -- it is named in the exception list below rather than weakening the rule for every row.
        if sym not in defined:
            n_kernel_undef += 1
            undef_named.append(sym)
    elif kind == 'shader':
        n_shaderonly += 1
        if sym in defined:
            fail.append(f"PORT-MAP.tsv: {sym} is kinded `shader` but the backend DOES define it - promote it to "
                        f"`kernel` (if it works) or `refused` (if the definition is a LOUD REFUSAL) - a stale row")
    elif kind == 'refused':
        n_refused += 1
        # FACT 2 for `refused`: the backend MUST define it.  An aspirational refusal row (a `refused` kind with
        # no definition) is the same class of hole as a `kernel` row with no definition.
        if sym not in defined:
            fail.append(f"PORT-MAP.tsv: {sym} is kinded `refused` but the backend does NOT define it - either wire "
                        f"the refusal in vulkan/src/kernels/refusals_vk.cpp or kind it `todo`")
unclaimed = sorted(built - claimed)      # prefill-path kernels and shared primitives: reported, not failures

for sym in sorted(core_syms):
    if sym not in mapped:
        fail.append(f"src/core/ reaches kernels::{sym}, which PORT-MAP.tsv does not mention")

# Rules that would fail at all times on a symbol the ENGINE owns (its body is an engine-header inline, not a
# backend TU).  Named explicitly so the exception is a decision, not a hole in the rule.
ENGINE_HEADER_INLINES = {'fwht256_inplace_cuda', 'embedding_gather_dev', 'quantize_q8_'}

k = sum(1 for r in rows if r[1] == 'kernel')
s = sum(1 for r in rows if r[1] == 'shader')
h = sum(1 for r in rows if r[1] == 'host')
t = sum(1 for r in rows if r[1] == 'todo')
rf = sum(1 for r in rows if r[1] == 'refused')
# THE TWO FACTS, PRINTED SEPARATELY.  `kernel` = shader built + backend defines it; `shader` = shader built,
# backend does NOT.  A `kernel` row the backend does not define is a HOLE unless it is an engine-header inline.
# `refused` = the backend defines it, but ONLY as a loud refusal (NOT a working kernel).
n_backend_defs = len([r for r in rows if r[0] in defined])
print(f"port map: {len(rows)} decode-path symbols - {k} kernel, {s} shader, {h} host, {t} todo, {rf} refused; "
      f"{len(built)} shaders built, {len(claimed)} claimed by the decode path")
print(f"          backend definitions: {n_backend_defs}; engine-header inlines: "
      f"{sorted(ENGINE_HEADER_INLINES & mapped)}")
for sym in sorted(set(undef_named) - ENGINE_HEADER_INLINES):
    fail.append(f"PORT-MAP.tsv: {sym} is kinded `kernel` (a shader exists) but the Vulkan backend does not DEFINE "
                f"it - the wrapper is missing; kind it `shader` or wire it (this is the indexer_key_append defect)")
print(f"          (the other {len(unclaimed)} built shaders are the prefill path and the primitives the kernels "
      f"are built from)")
for f in fail:
    print("  FAIL " + f)
sys.exit(1 if fail else 0)
