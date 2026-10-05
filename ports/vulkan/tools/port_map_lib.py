#!/usr/bin/env python3
"""Shared symbol discovery for the port map's checker and generator, so the two cannot drift.

THE RULE (and why it is not a heuristic that invents symbols):

  A DECODE-PATH SYMBOL is an identifier that `src/core/` references to the engine's kernels
  namespace.  Today `check_port_map.py` only saw the ones written with the `kernels::` qualifier,
  so a symbol called BARE -- which `using namespace strata::kernels;` makes legal, and which
  `src/core/mtp.cpp` does for the coupled-draft entry points -- was invisible, and the map could
  read `todo 0` while such a symbol was unported.

  The bare names are NOT scraped from `src/core/` itself: a source file is full of local
  identifiers, and treating every `foo(` in it as a kernel symbol would invent names the engine
  does not have.  They come from the ENGINE'S OWN DECLARATIONS -- the namespace-scope functions
  declared in `include/strata/kernels/**` -- and are attributed to `src/core/` only in a source
  that has actually brought the namespace into scope (`using namespace strata::kernels;` or a
  `using strata::kernels::X;` declaration).  So every symbol the map is asked about is a real
  engine symbol, and it is a symbol the decode path reached.

The residual assumption is stated rather than hidden: a bare name is attributed to the kernels
namespace when a namespace-scope function of that name is declared in a kernel header and the
source has the namespace in scope.  A LOCAL function of the same name in `src/core/` would
shadow it; no such shadow exists today, and if one appeared the map would show a spurious row
(a reportable drift), never a missing kernel.
"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[3]      # .../strata-vulkan-wt
PORT = ROOT / 'ports/vulkan'

# C++ keywords and control words a `name(` scan must never mistake for a declared function.
_KEYWORDS = {
    'for', 'if', 'while', 'switch', 'return', 'sizeof', 'catch', 'do', 'else', 'new', 'delete',
    'alignof', 'decltype', 'static_assert', 'noexcept', 'operator', 'and', 'or', 'not', 'defined',
    'throw', 'typedef', 'using', 'namespace', 'template', 'typename', 'constexpr', 'static', 'if',
}


def _strip(src: str) -> str:
    """Comments and string literals out, so a name inside either is not a reference."""
    src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
    src = re.sub(r'//[^\n]*', ' ', src)
    src = re.sub(r'"(?:\\.|[^"\\])*"', '""', src)
    return '\n'.join(l for l in src.splitlines() if not l.lstrip().startswith('#'))


def _region(src: str, opener: str):
    """The text inside the first namespace block the opener matches, brace-balanced."""
    m = re.search(opener, src)
    if not m:
        return None
    i, depth = m.end(), 1
    while i < len(src) and depth:
        if src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
        i += 1
    return src[m.end():i - 1]


def _depth0(region: str) -> str:
    """Only brace-depth-0 text: a function's parameters, never its body or a struct's members."""
    out, d = [], 0
    for ch in region:
        if ch == '{':
            d += 1
        elif ch == '}':
            d -= 1
        out.append(ch if d == 0 else (';' if ch == ';' else ' '))
    return ''.join(out)


def kernel_header_functions():
    """Namespace-scope function names declared in namespace strata::kernels, by header."""
    names = {}
    for p in (ROOT / 'include/strata/kernels').rglob('*.hpp'):
        s = _strip(p.read_text(errors='ignore'))
        for opener in (r'namespace\s+strata::kernels\b[^{]*\{',
                       r'namespace\s+strata\s*\{[^{]*\{[^}]*namespace\s+kernels\b[^{]*\{'):
            region = _region(s, opener)
            if region is None:
                continue
            for m in re.finditer(r'(?<![\w:.])([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', _depth0(region)):
                n = m.group(1)
                if n not in _KEYWORDS:
                    names.setdefault(n, str(p.relative_to(ROOT)))
    return names


def core_sources():
    for p in sorted((ROOT / 'src/core').rglob('*')):
        if p.suffix in ('.cpp', '.hpp'):
            yield p


def qualified_symbols():
    """Every `kernels::X` src/core/ writes."""
    syms = set()
    for p in core_sources():
        syms |= set(re.findall(r'kernels::([a-z_][a-z0-9_]*)', _strip(p.read_text(errors='ignore'))))
    return syms


def unqualified_symbols(names):
    """Bare references to a declared kernel-header function, in a source with the namespace in scope."""
    found = {}
    for p in core_sources():
        text = _strip(p.read_text(errors='ignore'))
        if not re.search(r'using\s+namespace\s+strata::kernels\s*;|using\s+strata::kernels::', text):
            continue
        for name in names:
            if re.search(r'(?<![\w:.])' + re.escape(name) + r'\s*\(', text):
                found.setdefault(name, set()).add(str(p.relative_to(ROOT)))
    return found


def decode_path_symbols():
    """The union: every kernels-namespace symbol src/core/ reaches, qualified or bare."""
    return qualified_symbols() | set(unqualified_symbols(kernel_header_functions()))
