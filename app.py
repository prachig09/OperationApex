"""AegisDeal (Operation Apex) - unified Gradio + FastAPI app.

Run:  python app.py   ->  UI at http://localhost:7860/ui   API at /api/*
The existing FastAPI `app` from backend/server.py is reused unchanged; Gradio is
mounted on it, so one process / one port serves both.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import httpx
import pandas as pd
import uvicorn
from fastapi.responses import HTMLResponse, RedirectResponse

import backend.server as srv
from backend.server import app as api_app

PORT = int(os.environ.get("APEX_PORT", "7860"))
BASE = f"http://127.0.0.1:{PORT}"
TIMEOUT = float(os.environ.get("APEX_HTTP_TIMEOUT", "900"))
OUT_DIR = Path(os.environ.get("APEX_DATA_DIR", "apex_data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

STARTERS = [
    "Why is the bid 34.5 plus a 10 earn-out?",
    "Show Northgate's revenue share from the CSV",
    "Was anything sent outside?",
]
SAMPLE_PII = (
    "Contact Rahul Mehta (CFO, Northgate Logistics) at rahul.mehta@northgate-logistics.com "
    "or +91 98220 11234. SSN 123-45-6789, card 4111 1111 1111 1111. "
    "Wire to IBAN DE89 3704 0044 0532 0130 00. Change of control clause applies on 51% transfer."
)


# --------------------------------------------------------------------------- helpers
def api(method: str, path: str, json_body: Any = None, files: Any = None,
        params: Optional[dict] = None) -> Tuple[bool, Any]:
    try:
        r = httpx.request(method, BASE + path, json=json_body, files=files, params=params, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        return False, {"error": f"{type(exc).__name__}: {exc}"}
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        body = {"raw": r.text[:4000]}
    return r.is_success, body


def model_fields(model: Any) -> set:
    f = getattr(model, "model_fields", None) or getattr(model, "__fields__", None) or {}
    return set(f.keys())


def model_schema(model: Any) -> Any:
    try:
        return model.model_json_schema()
    except Exception:  # noqa: BLE001
        try:
            return model.schema()
        except Exception:  # noqa: BLE001
            return {}


def fit(model: Any, spec: Dict[str, Tuple[Any, Tuple[str, ...]]]) -> Dict[str, Any]:
    """Map logical keys onto the field names the Pydantic model really declares."""
    names = model_fields(model)
    payload: Dict[str, Any] = {}
    for canon, (value, aliases) in spec.items():
        if value is None:
            continue
        for cand in (canon, *aliases):
            if cand in names:
                payload[cand] = value
                break
        else:
            if not names:
                payload[canon] = value
    return payload


def find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() == key:
                return v
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def find_like(obj: Any, frag: str) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if frag in str(k).lower():
                return v
        for v in obj.values():
            r = find_like(v, frag)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_like(v, frag)
            if r is not None:
                return r
    return None


def as_json(obj: Any) -> Any:
    val = json.loads(json.dumps(obj, default=str))
    if isinstance(val, (dict, list)):
        return val
    return {"value": val} if val is not None else {}


def err_with_schema(body: Any, model: Any = None) -> Dict[str, Any]:
    out = {"backend_error": body}
    if model is not None:
        out["expected_schema"] = model_schema(model)
    return out


def runtime() -> Any:
    return srv.rt()


def find_instance(cls_name: str) -> Any:
    cls = getattr(srv, cls_name, None)
    if cls is None:
        return None
    try:
        for v in vars(runtime()).values():
            if isinstance(v, cls):
                return v
    except Exception:  # noqa: BLE001
        pass
    return None


# --------------------------------------------------------------------------- Tab 1
def fetch_financials() -> pd.DataFrame:
    ok, body = api("GET", "/api/financials")
    if ok and isinstance(body, list) and body:
        return pd.DataFrame(body)
    return pd.DataFrame({"info": ["No demo_financials.csv yet - click 'Generate synthetic data room'."]})


def do_generate() -> Tuple[Any, pd.DataFrame, Optional[str]]:
    ok, body = api("POST", "/api/generate-demo")
    df = fetch_financials()
    csv_path = None
    if ok:
        ok2, _ = api("GET", "/api/demo-financials")
        if ok2 or True:
            try:
                r = httpx.get(BASE + "/api/demo-financials", timeout=60)
                if r.is_success:
                    csv_path = str(OUT_DIR / "demo_financials_download.csv")
                    Path(csv_path).write_bytes(r.content)
            except Exception:  # noqa: BLE001
                pass
    return as_json(body), df, csv_path


def do_upload(files: Optional[List[str]]) -> Tuple[Any, pd.DataFrame]:
    if not files:
        return {"error": "Choose at least one PDF or CSV first."}, fetch_financials()
    results = []
    for f in files:
        p = Path(f if isinstance(f, str) else f.name)
        with p.open("rb") as fh:
            ok, body = api("POST", "/api/upload", files={"file": (p.name, fh.read())})
        results.append({"file": p.name, "ok": ok, "response": body})
    return as_json(results), fetch_financials()


def do_ingest() -> Any:
    ok, body = api("POST", "/api/ingest", json_body=fit(srv.IngestRequest, {}))
    if not ok:
        return as_json(err_with_schema(body, srv.IngestRequest))
    return as_json(body)


def list_docs() -> Any:
    ok, body = api("GET", "/api/documents")
    return as_json(body)


def pii_scrub(text: str) -> Tuple[str, Any]:
    proxy = find_instance("AirGappedProxy")
    if proxy is None:
        return "AirGappedProxy instance not found on Runtime.", {}
    for name in ("scrub", "anonymize", "redact", "sanitize", "scrub_text", "anonymize_text", "mask", "process"):
        fn = getattr(proxy, name, None)
        if callable(fn):
            try:
                res = fn(text)
            except Exception as exc:  # noqa: BLE001
                return f"{name}() raised {type(exc).__name__}: {exc}", {}
            meta: Any = {}
            if isinstance(res, tuple):
                res, meta = res[0], (res[1] if len(res) > 1 else {})
            if not isinstance(res, str):
                for attr in ("text", "scrubbed", "anonymized", "clean", "output"):
                    if hasattr(res, attr):
                        meta = {k: v for k, v in vars(res).items() if k != attr} if hasattr(res, "__dict__") else {}
                        res = getattr(res, attr)
                        break
            return str(res), as_json(meta)
    public = [m for m in dir(proxy) if not m.startswith("_")]
    return f"No scrub method matched. Public methods: {public}", {}


def egress_log() -> Tuple[Any, pd.DataFrame, str]:
    ok, health = api("GET", "/api/health")
    rows: List[Dict[str, Any]] = []
    ledger = find_instance("EgressLedger")
    if ledger is not None:
        for attr in ("entries", "records", "events", "rows", "log", "items"):
            seq = getattr(ledger, attr, None)
            seq = seq() if callable(seq) else seq
            if isinstance(seq, (list, tuple)) and seq:
                for e in seq:
                    if isinstance(e, dict):
                        rows.append(as_json(e))
                    elif hasattr(e, "__dict__"):
                        rows.append(as_json(vars(e)))
                    else:
                        rows.append({"entry": str(e)})
                break
    df = pd.DataFrame(rows) if rows else pd.DataFrame({"egress_ledger": ["No outbound payload entries recorded yet."]})
    ok2, chat = api("POST", "/api/chat", json_body=fit(srv.ChatRequest, {
        "message": ("Was anything sent outside?", ("query", "prompt", "text")),
        "session_id": ("audit-panel", ("session", "thread_id")),
    }))
    answer = chat.get("answer", json.dumps(chat, default=str)) if isinstance(chat, dict) else str(chat)
    return as_json(health.get("airgap", health) if isinstance(health, dict) else health), df, answer


# --------------------------------------------------------------------------- Tab 2
CLAIM_KEYS = ("claim", "finding", "text", "summary", "value", "statement", "description")
QUOTE_KEYS = ("verbatim_quote", "quote", "evidence_quote", "evidence", "source_quote", "span")


def fid_of(row: Dict[str, Any]) -> str:
    return str(row.get("finding_id") or row.get("id") or "")


def verdict_of(row: Dict[str, Any]) -> str:
    blob = json.dumps(row, default=str).upper()
    for v in ("RETRACTED", "PARTIAL", "SUPPORTED", "FAITHFUL", "VERIFIED"):
        if f'"{v}"' in blob:
            return v
    return "UNKNOWN"


def get_findings() -> List[Dict[str, Any]]:
    ok, body = api("GET", "/api/findings")
    return body if ok and isinstance(body, list) else []


def item_of(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        it = srv.effective_item(row)
        return it if isinstance(it, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def pick(d: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Optional[str], str]:
    for k in keys:
        if k in d and isinstance(d[k], str):
            return k, d[k]
    return None, ""


def findings_view() -> Tuple[pd.DataFrame, Any, str]:
    rows = get_findings()
    table = []
    choices = []
    for r in rows:
        v = verdict_of(r)
        status = r.get("hitl_status", "")
        _, claim = pick(item_of(r), CLAIM_KEYS)
        table.append({"finding_id": fid_of(r), "verdict": v, "hitl_status": status, "claim": claim[:140]})
        if v in ("PARTIAL", "RETRACTED"):
            choices.append((f"{fid_of(r)} | {v} | {status}", fid_of(r)))
    df = pd.DataFrame(table) if table else pd.DataFrame({"info": ["No findings yet - run the audit."]})
    note = f"{len(rows)} findings, {len(choices)} flagged PARTIAL/RETRACTED for human review."
    return df, gr.update(choices=choices, value=(choices[0][1] if choices else None)), note


def run_audit() -> Tuple[Any, Any, Any, pd.DataFrame, Any, str]:
    ok, body = api("POST", "/api/audit/run", json_body=fit(srv.AuditRequest, {}))
    if not ok:
        e = as_json(err_with_schema(body, srv.AuditRequest))
        return e, {}, {}, pd.DataFrame(), gr.update(), "Audit failed."
    triad = find_like(body, "triad")
    drift = find_like(body, "drift")
    df, dd, note = findings_view()
    return as_json(body), as_json(triad or {"note": "no 'triad' key in audit response"}), \
        as_json(drift or {"note": "no 'drift' key in audit response"}), df, dd, note


def load_finding(fid: Optional[str]) -> Tuple[str, str, Any]:
    for r in get_findings():
        if fid_of(r) == fid:
            it = item_of(r)
            return pick(it, CLAIM_KEYS)[1], pick(it, QUOTE_KEYS)[1], as_json(r)
    return "", "", {}


def resolve(kind: str, fid: Optional[str], claim: str, quote: str, reviewer: str, note: str):
    if not fid:
        return {"error": "Select a PARTIAL/RETRACTED finding first."}, {}, gr.update(), ""
    row = next((r for r in get_findings() if fid_of(r) == fid), None)
    if row is None:
        return {"error": f"finding {fid} not found"}, {}, gr.update(), ""
    base = item_of(row)
    ck, orig_claim = pick(base, CLAIM_KEYS)
    qk, orig_quote = pick(base, QUOTE_KEYS)
    claim, quote = (claim or "").strip(), (quote or "").strip()
    edited = bool(claim) and claim != orig_claim.strip()
    if kind == "approve" and edited:
        if not quote:
            return {"error": "Override rejected: provide the VERBATIM source quote that supports your edited claim."}, \
                {}, gr.update(), ""
        action = "override"
    elif kind == "approve":
        action = "approve"
    else:
        action = "reject"

    edited_item = copy.deepcopy(base)
    if edited:
        edited_item[ck or "claim"] = claim
        edited_item[qk or "verbatim_quote"] = quote
    spec = {
        "finding_id": (fid, ("id",)),
        "action": (action, ("decision", "resolution", "status")),
        "reviewer": (reviewer or "analyst", ("reviewer_id", "user")),
        "note": (note or "", ("comment", "reason", "notes")),
        "edited_item": (edited_item if edited else None, ("item", "override_item", "corrected_item", "human_item", "output")),
        "edited_claim": (claim if edited else None, ("claim", "new_claim", "override_claim", "claim_text", "edited_text")),
        "verbatim_quote": (quote if edited else None, ("quote", "evidence_quote", "edited_quote", "quote_text")),
    }
    before = api("GET", "/api/dpo/stats")[1]
    ok, body = api("POST", "/api/hitl/resolve", json_body=fit(srv.HitlRequest, spec))
    after = api("GET", "/api/dpo/stats")[1]
    df, dd, summary = findings_view()
    if not ok:
        return as_json(err_with_schema(body, srv.HitlRequest)), as_json({"dpo_before": before, "dpo_after": after}), dd, summary
    return as_json(body), as_json({"dpo_before": before, "dpo_after": after, "jsonl": str(OUT_DIR / "dpo_preferences.jsonl")}), dd, summary


def show_schemas() -> Any:
    return {n: model_schema(getattr(srv, n)) for n in
            ("IngestRequest", "AuditRequest", "HitlRequest", "NegotiateRequest", "DpoStepRequest", "ChatRequest")
            if hasattr(srv, n)}


# --------------------------------------------------------------------------- Tab 3
def quant_view() -> Tuple[str, Any, Any]:
    ok, q = api("GET", "/api/quant", params={"demo": "true"})
    ok2, d = api("GET", "/api/dpo/stats")
    lines = ["### 4-bit Q4_K_M vs FP16"]
    fp16 = find_like(q, "fp16")
    q4 = find_like(q, "q4")
    lines.append(f"- FP16: `{json.dumps(fp16, default=str)}`" if fp16 is not None else "- FP16: see raw JSON")
    lines.append(f"- Q4_K_M: `{json.dumps(q4, default=str)}`" if q4 is not None else "- Q4_K_M: see raw JSON")
    pairs = find_like(d, "pair") if ok2 else None
    lines.append(f"- Live DPO preference pairs: **{pairs if pairs is not None else 'see stats JSON'}**")
    return "\n".join(lines), as_json(q), as_json(d)


def dpo_step() -> Tuple[Any, Any]:
    ok, body = api("POST", "/api/dpo/step", json_body=fit(srv.DpoStepRequest, {}))
    stats = api("GET", "/api/dpo/stats")[1]
    if not ok:
        return as_json(err_with_schema(body, srv.DpoStepRequest)), as_json(stats)
    return as_json(body), as_json(stats)


def _flat(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _flat(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _flat(v, f"{prefix}{i}.")
    else:
        out.append((prefix.rstrip("."), obj))
    return out


def negotiate() -> Tuple[Any, str, Optional[str]]:
    ok, body = api("POST", "/api/negotiate", json_body=fit(srv.NegotiateRequest, {}))
    if not ok:
        return as_json(err_with_schema(body, srv.NegotiateRequest)), "Negotiation failed - see JSON.", None
    memo = find_like(body, "memo")
    if isinstance(memo, str) and len(memo) > 80:
        text = memo
    else:
        def g(k: str) -> str:
            v = find_key(body, k)
            return f"${v}M" if isinstance(v, (int, float)) else (str(v) if v is not None else "see table")
        text = "\n".join([
            "# Investment Recommendation Memo - Project Apex",
            f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} - air-gapped, human-verified findings only_",
            "",
            "## Recommendation",
            f"Proceed at a structured bid: **{g('upfront_musd')} upfront** plus an **earn-out** "
            f"(base price {g('base_price_musd')}, EBITDA {g('ebitda_musd')}, "
            f"earn-out exercise probability {find_key(body, 'exercise_probability')}).",
            "",
            "## Negotiation result (raw fields)",
            "| Field | Value |", "|---|---|",
            *[f"| {k} | {str(v).replace('|', '/')} |" for k, v in _flat(body)],
            "",
            "## Basis",
            "Terms derive from HITL-approved CHANGE_OF_CONTROL findings; all numbers trace to the data room.",
        ])
    path = OUT_DIR / "Investment_Recommendation_Memo.md"
    path.write_text(text, encoding="utf-8")
    return as_json(body), text, str(path)


# --------------------------------------------------------------------------- Tab 4
def chat_fn(message: str, history: List[Dict[str, str]], session: str):
    history = list(history or [])
    if not message or not message.strip():
        return "", history
    ok, body = api("POST", "/api/chat", json_body=fit(srv.ChatRequest, {
        "message": (message, ("query", "prompt", "text")),
        "session_id": (session, ("session", "thread_id")),
    }))
    if ok and isinstance(body, dict):
        tools = ", ".join(body.get("tools_called", []) or []) or "none"
        reply = f"{body.get('answer', json.dumps(body, default=str))}\n\n`tools: {tools} | redactions: {json.dumps(body.get('redactions', {}))}`"
    else:
        reply = f"Backend error: {json.dumps(as_json(err_with_schema(body, srv.ChatRequest)))}"
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return "", history


def badge() -> str:
    ok, h = api("GET", "/api/health")
    if not ok or not isinstance(h, dict):
        return "**Backend unreachable** - " + json.dumps(h, default=str)
    air = (h.get("airgap") or {}).get("status")
    m = h.get("model") or {}
    return (f"**Air-gap:** {air} | **Model:** {m.get('name')} "
            f"({'live' if m.get('live') else 'rule fallback'}) | **DB rows:** {json.dumps(h.get('db'))}")


# --------------------------------------------------------------------------- UI
with gr.Blocks(title="AegisDeal - Operation Apex", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AegisDeal - Operation Apex\nAir-gapped M&A due diligence: RAG-verified findings, human-in-the-loop, DPO-aligned local LLM.")
    status = gr.Markdown()
    demo.load(badge, None, status)

    with gr.Tabs():
        # ---------------- Tab 1
        with gr.Tab("1 - Data Room & Air-Gap"):
            with gr.Row():
                gen_btn = gr.Button("Generate synthetic data room", variant="primary")
                ingest_btn = gr.Button("Ingest data room")
                docs_btn = gr.Button("List documents")
            uploads = gr.File(label="Upload PDFs / CSVs", file_count="multiple", file_types=[".pdf", ".csv"])
            up_btn = gr.Button("Upload files")
            api_out = gr.JSON(label="Backend response")
            csv_dl = gr.File(label="demo_financials.csv (download)")
            fin_df = gr.Dataframe(value=fetch_financials, label="demo_financials.csv preview", interactive=False, wrap=True)
            gr.Markdown("### PII Scrubbing Inspector (AirGappedProxy)")
            with gr.Row():
                pii_in = gr.Textbox(label="Raw text (leaves nothing the proxy hasn't scrubbed)", value=SAMPLE_PII, lines=6)
                pii_out = gr.Textbox(label="Scrubbed output sent to any external call", lines=6)
            pii_meta = gr.JSON(label="Redaction metadata")
            pii_btn = gr.Button("Scrub")
            gr.Markdown("### Network Egress Audit Log")
            eg_btn = gr.Button("Refresh egress audit")
            with gr.Row():
                eg_status = gr.JSON(label="Air-gap status")
                eg_answer = gr.Textbox(label="Copilot egress report", lines=6)
            eg_df = gr.Dataframe(label="Outbound payload ledger", interactive=False)

            gen_btn.click(do_generate, None, [api_out, fin_df, csv_dl])
            up_btn.click(do_upload, uploads, [api_out, fin_df])
            ingest_btn.click(do_ingest, None, api_out)
            docs_btn.click(list_docs, None, api_out)
            pii_btn.click(pii_scrub, pii_in, [pii_out, pii_meta])
            eg_btn.click(egress_log, None, [eg_status, eg_df, eg_answer])

        # ---------------- Tab 2
        with gr.Tab("2 - RAG Triad, Drift & HITL"):
            audit_btn = gr.Button("Run RAG extraction + Triad + Drift monitor", variant="primary")
            audit_out = gr.JSON(label="Audit response")
            with gr.Row():
                triad_out = gr.JSON(label="RAG Triad (context relevance / groundedness / answer relevance)")
                drift_out = gr.JSON(label="Semantic drift monitor")
            find_df = gr.Dataframe(label="Findings", interactive=False, wrap=True)
            find_note = gr.Markdown()
            gr.Markdown("### HITL Control Panel")
            with gr.Row():
                finding_dd = gr.Dropdown(label="PARTIAL / RETRACTED finding", choices=[])
                refresh_btn = gr.Button("Refresh findings")
            claim_tb = gr.Textbox(label="Claim (edit to override)", lines=3)
            quote_tb = gr.Textbox(label="Verbatim supporting quote (required when overriding; backend verifies it appears in source)", lines=2)
            with gr.Row():
                reviewer_tb = gr.Textbox(label="Reviewer", value="analyst")
                note_tb = gr.Textbox(label="Note")
            with gr.Row():
                approve_btn = gr.Button("Approve", variant="primary")
                reject_btn = gr.Button("Reject", variant="stop")
            raw_row = gr.JSON(label="Selected finding (raw)")
            hitl_out = gr.JSON(label="HITL result")
            dpo_out = gr.JSON(label="DPO pair store (before/after)")
            with gr.Accordion("Backend request schemas (debug)", open=False):
                schema_btn = gr.Button("Show schemas")
                schema_out = gr.JSON()

            audit_btn.click(run_audit, None, [audit_out, triad_out, drift_out, find_df, finding_dd, find_note])
            refresh_btn.click(findings_view, None, [find_df, finding_dd, find_note])
            finding_dd.change(load_finding, finding_dd, [claim_tb, quote_tb, raw_row])
            approve_btn.click(lambda *a: resolve("approve", *a), [finding_dd, claim_tb, quote_tb, reviewer_tb, note_tb],
                              [hitl_out, dpo_out, finding_dd, find_note])
            reject_btn.click(lambda *a: resolve("reject", *a), [finding_dd, claim_tb, quote_tb, reviewer_tb, note_tb],
                             [hitl_out, dpo_out, finding_dd, find_note])
            schema_btn.click(show_schemas, None, schema_out)

        # ---------------- Tab 3
        with gr.Tab("3 - Quantization, DPO & Valuation"):
            quant_btn = gr.Button("Load quantization + DPO stats")
            quant_md = gr.Markdown()
            with gr.Row():
                quant_json = gr.JSON(label="/api/quant?demo=true")
                dpo_stats_json = gr.JSON(label="/api/dpo/stats")
            dpo_btn = gr.Button("Run live DPO Bradley-Terry gradient step", variant="primary")
            dpo_step_out = gr.JSON(label="DPO step result")
            gr.Markdown("### Multi-Agent Negotiation Loop")
            neg_btn = gr.Button("Run negotiation ($50.0M ask -> $34.5M upfront + $10.0M earn-out)", variant="primary")
            neg_json = gr.JSON(label="Negotiation result")
            memo_md = gr.Markdown(label="Memo")
            memo_file = gr.File(label="Investment_Recommendation_Memo.md")

            quant_btn.click(quant_view, None, [quant_md, quant_json, dpo_stats_json])
            dpo_btn.click(dpo_step, None, [dpo_step_out, dpo_stats_json])
            neg_btn.click(negotiate, None, [neg_json, memo_md, memo_file])

        # ---------------- Tab 4
        with gr.Tab("4 - Due Diligence Copilot"):
            session = gr.State(lambda: "ui-" + uuid.uuid4().hex[:8])
            chatbot = gr.Chatbot(type="messages", height=480, label="AegisDeal Copilot")
            with gr.Row():
                starter_btns = [gr.Button(p) for p in STARTERS]
            msg = gr.Textbox(placeholder="Ask about the deal, CSV, audit findings or egress...", show_label=False)
            msg.submit(chat_fn, [msg, chatbot, session], [msg, chatbot])
            for b, p in zip(starter_btns, STARTERS):
                b.click(lambda h, s, _p=p: chat_fn(_p, h, s), [chatbot, session], [msg, chatbot])


# --------------------------------------------------------------------------- mount
# Keep the original HTML console at /legacy; "/" redirects to the Gradio UI.
api_app.router.routes = [r for r in api_app.router.routes if getattr(r, "path", None) != "/"]
api_app.add_api_route("/legacy", srv.index, methods=["GET"], response_class=HTMLResponse, include_in_schema=False)
api_app.add_api_route("/", lambda: RedirectResponse("/ui"), methods=["GET"], include_in_schema=False)
app = gr.mount_gradio_app(api_app, demo.queue(), path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("APEX_HOST", "0.0.0.0"), port=PORT, log_level="info")