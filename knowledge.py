"""
Course knowledge engine for the Logistics Management (LM) tutor.
Pure Python (no Streamlit, no LLM): reads PDF / DOCX / XLSX / TXT, splits them into chunks,
ranks chunks with BM25 search, and "X-rays" saved Excel Solver models into objective / variables / constraints.
"""
from __future__ import annotations

import io
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 1400
SUPPORTED = {".pdf", ".docx", ".xlsx", ".txt", ".md", ".csv"}


@dataclass
class Chunk:
    source: str
    loc: str
    text: str
    kind: str


# ----------------------------------------------------------------------------
# Chunking helpers
# ----------------------------------------------------------------------------
def _split_long(text: str, max_chars: int) -> list:
    parts, cur = [], ""
    for piece in re.split(r"(?<=[.!?\n])\s+", text):
        if cur and len(cur) + len(piece) > max_chars:
            parts.append(cur)
            cur = ""
        cur += (" " if cur else "") + piece
    if cur:
        parts.append(cur)
    return [p[:max_chars * 2] for p in parts]


def _pack(units: list, source: str, kind: str, max_chars: int = MAX_CHARS) -> list:
    """units = [((prefix, number), text)]; consecutive units are merged into chunks of ~max_chars."""
    chunks, buf, size, first, last = [], [], 0, None, None

    def flush():
        nonlocal buf, size, first, last
        if buf:
            if first == last:
                label = f"{first[0]} {first[1]}"
            elif first[0] == last[0]:
                label = f"{first[0]} {first[1]}-{last[1]}"
            else:
                label = f"{first[0]} {first[1]} to {last[0]} {last[1]}"
            chunks.append(Chunk(source, label, "\n".join(buf), kind))
        buf, size, first, last = [], 0, None, None

    for loc, text in units:
        text = (text or "").strip()
        if not text:
            continue
        for piece in (_split_long(text, max_chars) if len(text) > max_chars else [text]):
            if buf and size + len(piece) > max_chars:
                flush()
            if first is None:
                first = loc
            last = loc
            buf.append(piece)
            size += len(piece) + 1
    flush()
    return chunks


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    s = str(v).replace("\n", " ")
    return s if len(s) < 220 else s[:217] + "..."


# ----------------------------------------------------------------------------
# File readers
# ----------------------------------------------------------------------------
def _read_pdf(data: bytes, name: str) -> list:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    units = []
    for i, page in enumerate(reader.pages, 1):
        try:
            units.append((("page", i), page.extract_text() or ""))
        except Exception:
            continue
    return _pack(units, name, "pdf")


def _read_docx(data: bytes, name: str) -> list:
    import docx
    d = docx.Document(io.BytesIO(data))
    units = [(("paragraph", i), p.text) for i, p in enumerate(d.paragraphs, 1)]
    for t, table in enumerate(d.tables, 1):
        for r, row in enumerate(table.rows, 1):
            units.append(((f"table {t} row", r), " | ".join(c.text.strip() for c in row.cells)))
    return _pack(units, name, "docx")


def _read_text(data: bytes, name: str) -> list:
    text = data.decode("utf-8", errors="ignore")
    return _pack([(("line", i), ln) for i, ln in enumerate(text.splitlines(), 1)], name, "text")


def _read_xlsx(data: bytes, name: str) -> list:
    from openpyxl import load_workbook
    wf = load_workbook(io.BytesIO(data))
    wv = load_workbook(io.BytesIO(data), data_only=True)
    chunks = [Chunk(name, "workbook overview",
                    "Workbook sheets: " + "; ".join(f"'{ws.title}' ({ws.dimensions})" for ws in wf), "xlsx")]
    for ws in wf:
        vs = wv[ws.title]
        units, headings = [], []
        for row in ws.iter_rows():
            cells = []
            for c in row:
                if c.value is None:
                    continue
                if isinstance(c.value, str) and not c.value.startswith("=") and len(headings) < 40 and c.value not in headings:
                    headings.append(c.value.strip()[:60])
                if isinstance(c.value, str) and c.value.startswith("="):
                    cells.append(f"{c.coordinate}={_fmt(vs[c.coordinate].value)} [formula {c.value}]")
                else:
                    cells.append(f"{c.coordinate}: {_fmt(c.value)}")
            if cells:
                units.append(((f"sheet '{ws.title}' row", row[0].row), " | ".join(cells)))
        chunks.append(Chunk(name, f"sheet '{ws.title}' overview",
                            f"Sheet '{ws.title}' ({ws.dimensions}). Labels on this sheet: " + "; ".join(headings), "xlsx"))
        chunks += _pack(units, name, "xlsx")
    for m in extract_solver_models(data):
        chunks.append(Chunk(name, f"Solver model on sheet '{m['sheet']}'", solver_report_text(m), "solver"))
    return chunks


def ingest_file(name: str, data: bytes) -> list:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return _read_pdf(data, name)
    if ext == ".docx":
        return _read_docx(data, name)
    if ext == ".xlsx":
        return _read_xlsx(data, name)
    if ext in (".txt", ".md", ".csv"):
        return _read_text(data, name)
    return []


# ----------------------------------------------------------------------------
# BM25 search
# ----------------------------------------------------------------------------
STOP = set("a an the of to in on for and or is are was were be been it this that these those with as at by from how what why "
           "which who whom do does did can could should would will i me my we our you your explain tell show give about "
           "please using use used between into than then so if not no yes also".split())

ALIASES = {  # question keyword -> filename fragments to boost
    "parcelxpress": ["parcelxpress"], "phantom": ["parcelxpress"], "absorption": ["parcelxpress"],
    "flat rate": ["parcelxpress"], "zone pricing": ["parcelxpress"], "overhead": ["parcelxpress"],
    "blinkit": ["blinkit"], "dark store": ["blinkit", "dark store", "vizquick"], "vizquick": ["vizquick", "dark store"],
    "rider": ["blinkit"], "zenith": ["warehouse"], "warehouse": ["warehouse"], "space": ["warehouse"],
    "3pl": ["3pl", "lsp"], "lsp": ["lsp", "3pl"], "agrifresh": ["3pl", "lsp"], "provider": ["3pl", "lsp"],
    "rail": ["rail"], "road": ["rail"], "vidarbha": ["rail"], "transportation": ["transport"],
    "transport": ["transport"], "slides": ["slides"], "session": ["slides"],
}


def tokenize(text: str) -> list:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOP or len(t) < 2:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class KnowledgeBase:
    def __init__(self, chunks: list, files: dict | None = None):
        self.chunks = chunks
        self.files = files or {}          # filename -> bytes
        self.docs = [tokenize(c.source + " " + c.loc + " " + c.text) for c in chunks]
        self.N = len(chunks)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def merged_with(self, other: "KnowledgeBase") -> "KnowledgeBase":
        return KnowledgeBase(self.chunks + other.chunks, {**self.files, **other.files})

    def _boost_sources(self, query: str) -> set:
        q = query.lower()
        frags = set()
        for key, vals in ALIASES.items():
            if key in q:
                frags.update(vals)
        return frags

    def search(self, query: str, k: int = 6, max_chars: int = 9000) -> list:
        q = tokenize(query)
        if not q:
            return []
        boost = self._boost_sources(query)
        k1, b = 1.5, 0.75
        scored = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avgdl))
            if s > 0:
                src = self.chunks[i].source.lower()
                if boost and any(fr in src for fr in boost):
                    s *= 1.6
                elif boost and "slides" not in src:
                    s *= 0.7
                if self.chunks[i].kind == "solver":
                    s *= 1.15
                scored.append((s, i))
        scored.sort(reverse=True)
        picked, used = [], 0
        for s, i in scored:
            c = self.chunks[i]
            if len(picked) >= k or used + len(c.text) > max_chars:
                continue
            picked.append(c)
            used += len(c.text)
        return picked

    def summary(self) -> list:
        by = defaultdict(lambda: [0, 0, ""])
        for c in self.chunks:
            by[c.source][0] += 1
            by[c.source][1] += len(c.text)
            by[c.source][2] = c.kind if c.kind != "solver" else by[c.source][2] or "xlsx"
        return [{"File": s, "Type": Path(s).suffix.lstrip(".").upper(), "Chunks": v[0], "Characters": v[1]}
                for s, v in sorted(by.items())]


def build_kb_from_dir(folder: str) -> KnowledgeBase:
    chunks, files = [], {}
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() in SUPPORTED and not p.name.startswith("~$"):
            data = p.read_bytes()
            files[p.name] = data
            try:
                chunks += ingest_file(p.name, data)
            except Exception as e:  # one bad file must not break the app
                chunks.append(Chunk(p.name, "read error", f"Could not read this file: {e}", "error"))
    return KnowledgeBase(chunks, files)


def build_kb_from_uploads(uploads: list) -> KnowledgeBase:
    """uploads = [(filename, bytes)]"""
    chunks, files = [], {}
    for name, data in uploads:
        files[name] = data
        try:
            chunks += ingest_file(name, data)
        except Exception as e:
            chunks.append(Chunk(name, "read error", f"Could not read this file: {e}", "error"))
    return KnowledgeBase(chunks, files)


# ----------------------------------------------------------------------------
# Excel Solver X-ray (reads the model Solver saved inside the workbook)
# ----------------------------------------------------------------------------
REL = {1: "<=", 2: "=", 3: ">=", 4: "must be integer", 5: "must be binary (0 or 1)", 6: "all different"}
ENGINE = {1: "GRG Nonlinear", 2: "Simplex LP (linear)", 3: "Evolutionary"}
SENSE = {1: "Maximize", 2: "Minimize", 3: "Reach a target value"}


def _parse_ref(txt: str):
    """'Sheet1!$E$4:$E$13' -> ('Sheet1', 'E4:E13'); constants return None."""
    txt = (txt or "").strip()
    if "!" not in txt:
        return None
    sheet, rng = txt.rsplit("!", 1)
    return sheet.strip("'"), rng.replace("$", "")


def _cells(wb, ref_txt: str):
    ref = _parse_ref(ref_txt)
    if not ref or ref[0] not in wb.sheetnames:
        return None, []
    ws, rng = wb[ref[0]], ref[1]
    got = ws[rng]
    if not isinstance(got, tuple):
        got = ((got,),)
    elif got and not isinstance(got[0], tuple):
        got = (got,)
    return ws, [c for row in got for c in row]


def _is_label(v) -> bool:
    return isinstance(v, str) and len(v.strip()) > 1 and not v.startswith("=") and re.search(r"[A-Za-z]", v) is not None


def _label(ws, cell, prefer_header: bool = False) -> str:
    """Text label for a cell: row label to the left and column header above (order depends on prefer_header)."""
    left = head = ""
    for col in range(cell.column - 1, 0, -1):
        v = ws.cell(row=cell.row, column=col).value
        if _is_label(v):
            left = v.strip()
            break
    for row in range(cell.row - 1, 0, -1):
        v = ws.cell(row=row, column=cell.column).value
        if _is_label(v):
            head = v.strip()
            break
    if prefer_header:
        return head or left
    return left or head


def _range_label(ws, cells) -> str:
    """Single cell or row vector -> label to the left; column vector -> header above; block -> row labels."""
    if not cells:
        return ""
    if len({c.column for c in cells}) == 1 and len(cells) > 1:
        return _label(ws, cells[0], prefer_header=True)
    return _label(ws, cells[0])


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def extract_solver_models(data: bytes) -> list:
    from openpyxl import load_workbook
    wf = load_workbook(io.BytesIO(data))
    wv = load_workbook(io.BytesIO(data), data_only=True)
    models = []
    for ws in wf:
        names = {k: v.attr_text for k, v in ws.defined_names.items() if k.startswith("solver_")}
        if "solver_opt" not in names and "solver_adj" not in names:
            continue
        g = lambda key, d=None: names.get(key, d)
        to_int = lambda s: int(float(str(s).strip('"'))) if str(s).strip('"').replace(".", "", 1).lstrip("-").isdigit() else None
        sense = to_int(g("solver_typ", "2"))
        m = {"sheet": ws.title, "engine": ENGINE.get(to_int(g("solver_eng", "1")), "unknown"),
             "sense": SENSE.get(sense, "Minimize"), "target_value": g("solver_val") if sense == 3 else None,
             "non_negative": g("solver_neg", "1") == "1", "objective": None, "changing": None, "constraints": []}
        sheet_vals = wv[ws.title]
        # objective
        oc_ws, oc = _cells(wv, g("solver_opt", ""))
        if oc:
            c = oc[0]
            m["objective"] = {"ref": g("solver_opt").replace("$", ""), "label": _label(oc_ws, c), "value": c.value}
        # decision variables
        ac_ws, ac = _cells(wv, g("solver_adj", ""))
        if ac:
            labels = []
            for c in ac[:60]:
                lab = _label(ac_ws, c)
                if lab and lab not in labels:
                    labels.append(lab)
            m["changing"] = {"ref": g("solver_adj").replace("$", ""), "n": len(ac),
                             "labels": labels[:12], "values": [c.value for c in ac[:12]]}
        # constraints
        n = to_int(g("solver_num", "0")) or 0
        for i in range(1, n + 1):
            rel = to_int(g(f"solver_rel{i}"))
            lhs_txt, rhs_txt = g(f"solver_lhs{i}", ""), g(f"solver_rhs{i}", "")
            l_ws, lc = _cells(wv, lhs_txt)
            r_ws, rc = _cells(wv, rhs_txt)
            lv = [_num(c.value) for c in lc]
            rv = [_num(c.value) for c in rc] if rc else ([_num(rhs_txt)] if _num(rhs_txt) is not None else [])
            con = {"lhs": lhs_txt.replace("$", ""), "rel": REL.get(rel, "?"), "rhs": rhs_txt.replace("$", "").strip('"'),
                   "lhs_label": _range_label(l_ws, lc), "n": len(lc), "binding": None, "min_slack": None}
            if rel in (1, 2, 3) and lv and rv and all(x is not None for x in lv + rv) and (len(rv) == 1 or len(rv) == len(lv)):
                rvb = rv * len(lv) if len(rv) == 1 else rv
                slack = [(r - l) if rel == 1 else (l - r) if rel == 3 else abs(r - l) for l, r in zip(lv, rvb)]
                con["binding"] = sum(1 for s in slack if abs(s) < 1e-6)
                con["min_slack"] = min(slack)
            m["constraints"].append(con)
        models.append(m)
    return models


def solver_report_text(m: dict) -> str:
    out = [f"EXCEL SOLVER MODEL SAVED ON SHEET '{m['sheet']}'",
           f"Algorithm: {m['engine']}. Goal: {m['sense']}" + (f" (target {m['target_value']})" if m["target_value"] else "") + ".",
           f"Variables forced non-negative: {'yes' if m['non_negative'] else 'no'}."]
    if m["objective"]:
        o = m["objective"]
        out.append(f"Objective (target cell {o['ref']}, label '{o['label']}'): current value {_fmt(o['value'])}.")
    if m["changing"]:
        c = m["changing"]
        out.append(f"Decision variables (changing cells {c['ref']}, {c['n']} cells; labels: {', '.join(c['labels']) or 'n/a'}): "
                   f"first values {[_fmt(v) for v in c['values']]}.")
    for i, k in enumerate(m["constraints"], 1):
        tail = ""
        if k["binding"] is not None:
            tail = f" Binding (no slack) on {k['binding']} of {k['n']} cells; smallest slack {_fmt(k['min_slack'])}."
        out.append(f"Constraint {i}: {k['lhs']} ({k['lhs_label'] or 'cells'}) {k['rel']} {k['rhs'] if k['rel'][0] in '<=>' else ''}.{tail}".replace(" .", "."))
    return "\n".join(out)
