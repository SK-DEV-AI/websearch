"""MathML → LaTeX recovery (donsetch extract/math.rs port).

MediaWiki/technical pages carry formulas as MathML with the original
LaTeX in `alttext` or `<annotation encoding="application/x-tex">`.
Trafilatura flattens `<math>` away, gutting the page's core info.
This module recovers the formula — true LaTeX when present, compact
token-efficient linearization otherwise — and rewrites `<math>`
elements in the HTML before extraction: block (`$$..$$`) when
display="block", inline (` $..$ `) otherwise.
"""

import re

_MATH_TAG = re.compile(r"<math\b[^>]*>.*?</math>", re.S)
_ATTR = re.compile(r'(\w+)(?:="([^"]*)")?')
_SCRIPTED = {
    "msup", "msub", "msubsup", "munder", "mover",
    "munderover", "mroot", "mmultiscripts",
}
_MATHML = _SCRIPTED | {
    "math", "semantics", "annotation", "annotation-xml", "mrow",
    "mi", "mn", "mo", "ms", "mtext", "mspace", "mpadded",
    "mstyle", "maction", "menclose", "mphantom", "mfrac", "msqrt",
    "mtable", "mtr", "mtd",
}


def _children(el):
    return [c for c in el if isinstance(c.tag, str)]


def _text_of(el):
    parts = []
    for node in el.itertext():
        parts.append(node)
    return "".join(parts)


def _strip_displaystyle(s):
    t = s.strip()
    if t.startswith("{\\displaystyle "):
        # Strip exactly ONE closing brace — trim would eat inner
        # braces ("W_{Q}}" → "W_{Q").
        rest = t[len("{\\displaystyle "):]
        if rest.endswith("}"):
            rest = rest[:-1]
        t = rest.strip()
    elif t.startswith("\\displaystyle "):
        t = t[len("\\displaystyle "):].strip()
    return t


def _serialize(el):
    tag = el.tag
    if tag in _SCRIPTED:
        kids = _children(el)
        base = _serialize(kids[0]) if kids else ""
        sub = sup = None
        if tag in ("msub", "msubsup", "munder", "munderover", "mmultiscripts") and len(kids) > 1:
            sub = _serialize(kids[1])
        if tag in ("msup", "msubsup", "mover", "munderover", "mroot", "mmultiscripts") and len(kids) > 1:
            sup = _serialize(kids[-1])
        out = base
        if sub:
            out += "_{%s}" % sub
        if sup:
            out += "^{%s}" % sup
        return out
    if tag == "mfrac":
        kids = _children(el)
        if len(kids) == 2:
            return "(%s)/(%s)" % (_serialize(kids[0]), _serialize(kids[1]))
        return " ".join(_serialize(k) for k in kids)
    if tag == "msqrt":
        return "sqrt(%s)" % "".join(_serialize(k) for k in _children(el))
    if tag == "mtable":
        rows = []
        for tr in [c for c in el if c.tag == "mtr"][:12]:
            cells = [_serialize(td) for td in [c for c in _children(tr) if c.tag == "mtd"][:12]]
            if cells:
                rows.append(", ".join(cells))
        if rows:
            return "(%s)" % "; ".join(rows)
        return " ".join(_text_of(el).split())
    if tag in ("annotation", "annotation-xml"):
        return ""
    if tag == "mspace":
        return " "
    # Containers and tokens: concat text + serialized math children.
    out = []
    if el.text and el.text.strip():
        out.append(el.text.strip())
    for child in _children(el):
        if child.tag in _MATHML:
            out.append(_serialize(child))
        if child.tail and child.tail.strip():
            out.append(child.tail.strip())
    return "".join(out)


def latex(fragment_html):
    """Best LaTeX for a `<math>` fragment: alttext → annotation → serialized."""
    try:
        from lxml import etree
        el = etree.fromstring(fragment_html)
    except Exception:
        from lxml import html as lh
        el = lh.fromstring(fragment_html)
        if el is None:
            return ""
    alt = el.get("alttext")
    if alt and alt.strip():
        return _strip_displaystyle(alt)
    for enc in ("application/x-tex", "application/latex"):
        for ann in el.iter():
            if ann.tag == "annotation" and (ann.get("encoding") or "").lower() in (enc,):
                t = _text_of(ann).strip()
                if t:
                    return t
    return _serialize(el).strip()


def _unhide_formulas(out):
    """MediaWiki hides the MathML wrapper (display:none) behind an img; the
    recovered $$..$$ / $..$ text inherits the hiding and trafilatura prunes
    it. Un-hide any element that now contains a formula marker."""
    if not out or ("$$" not in out and " $" not in out):
        return out
    try:
        from lxml import html as lh
        doc = lh.fromstring(out)
    except Exception:
        return out
    for el in doc.iter():
        st = el.get("style") or ""
        if "display" not in st:
            continue
        text = "".join(el.itertext())
        if "$$" not in text and " $" not in text:
            continue
        st2 = re.sub(r"display\s*:\s*none\s*;?", "", st).strip("; ")
        if st2:
            el.set("style", st2)
        else:
            del el.attrib["style"]
    return lh.tostring(doc, encoding="unicode")


def transform(html):
    """Rewrite `<math>..</math>` elements → `$$..$$` (block) / ` $..$ ` (inline)."""
    if "<math" not in html:
        return html

    def _repl(m):
        frag = m.group(0)
        attrs = dict(_ATTR.findall(frag[: frag.index(">")]))
        disp = (attrs.get("display") or "").lower()
        l = latex(frag)
        if not l:
            return ""
        if disp == "block" or "display:block" in (attrs.get("style") or ""):
            return "\n$$\n%s\n$$\n" % l
        return " $%s$ " % l

    return _unhide_formulas(_MATH_TAG.sub(_repl, html))


if __name__ == "__main__":
    # selfcheck
    cases = [
        (r'<math alttext="{\displaystyle W_{Q}}"><mi>x</mi></math>', "W_{Q}"),
        (r'<math alttext="{\displaystyle W_{Q}+W_{K}}"><mi>x</mi></math>', "W_{Q}+W_{K}"),
        (r'<math alttext="\mathrm{Attention}(Q,K,V)">x</math>', r"\mathrm{Attention}(Q,K,V)"),
    ]
    for frag, want in cases:
        got = latex(frag)
        assert got == want, "latex(%r) = %r, want %r" % (frag[:40], got, want)

    ann = ('<math><semantics><mrow><mi>x</mi></mrow>'
           '<annotation encoding="application/x-tex">x^2 + 1</annotation></semantics></math>')
    assert latex(ann) == "x^2 + 1", latex(ann)

    sub_sup = "<math><mrow><msubsup><mi>W</mi><mi>Q</mi><mi>T</mi></msubsup></mrow></math>"
    l = latex(sub_sup)
    assert "W" in l and "_{Q}" in l and "^{T}" in l, l

    frac = ("<math><mfrac><mrow><mi>Q</mi><msup><mi>K</mi><mi>T</mi></msup></mrow>"
            "<msqrt><msub><mi>d</mi><mi>k</mi></msub></msqrt></mfrac></math>")
    l = latex(frac)
    assert ")/(" in l and "sqrt(" in l and "^{T}" in l, l

    mat = ("<math><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr>"
           "<mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr></mtable></math>")
    assert latex(mat) == "(1, 2; 3, 4)", latex(mat)

    no_leak = ('<math><semantics><mrow><mi>z</mi></mrow>'
               '<annotation encoding="application/x-tex">z</annotation></semantics></math>')
    assert latex(no_leak) == "z", latex(no_leak)

    html = ("<p>Attention is <math display=\"block\"><mfrac><mi>a</mi><mi>b</mi></mfrac></math>"
            " and <math alttext=\"E=mc^2\"><mi>x</mi></math> here.</p>")
    out = transform(html)
    assert "$$\n(a)/(b)\n$$" in out, out
    assert " $E=mc^2$ " in out, out
    assert "<math" not in out

    mw = ("<span class=\"mwe-math-mathml-inline mwe-math-mathml-a11y\" "
          "style=\"display: none;\"><math alttext=\"x^2\"><mi>x</mi></math></span>")
    out = transform(mw)
    assert " $x^2$ " in out, out
    assert "display: none" not in out, out

    print("MATHML-OK")
