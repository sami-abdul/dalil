/* Dalil reviewer screen. Vanilla JS, no build step, no network calls except to
 * this app's own server.
 * Add ?mock=1 to the URL to run against web/fixtures/mock.json instead of the API.
 */
(function () {
  "use strict";

  const MOCK = new URLSearchParams(location.search).get("mock") === "1";
  const REVIEWER_KEY = "dalil.reviewer";
  const POLL_MS = 1500;      // a question that is not changing
  const LIVE_POLL_MS = 350;  // a question being answered right now

  // ---------- small DOM helpers ----------
  const $ = (id) => document.getElementById(id);

  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined || v === false) continue;
        if (k === "class") el.className = v;
        else if (k === "text") el.textContent = v;
        else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
        else el.setAttribute(k, v === true ? "" : String(v));
      }
    }
    return add(el, children);
  }

  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
  // Like el.append, but skips null/false so conditional children never render as "null".
  function add(el, ...kids) {
    for (const c of kids.flat(Infinity)) {
      if (c === null || c === undefined || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  // Text whose language varies: let the browser pick the direction from the
  // first strong character, and mark Arabic so it gets an Arabic font and size.
  const FIRST_STRONG = /[A-Za-z\u00C0-\u024F\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/;
  const ARABIC = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/;
  function isArabic(s) {
    const m = String(s || "").match(FIRST_STRONG);
    return !!m && ARABIC.test(m[0]);
  }
  function tx(s, attrs) { return Object.assign({ dir: "auto", lang: isArabic(s) ? "ar" : null }, attrs || {}); }
  // A paragraph made of isolated sentence spans cannot use dir="auto": isolates
  // do not count when the browser picks the direction. Take it from the first sentence,
  // so citation numbers land after the sentence in Arabic too.
  function paraDir(s) { return isArabic(s) ? "rtl" : "ltr"; }

  const timeFmt = new Intl.DateTimeFormat("en-GB", {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "Asia/Dubai",
  });
  const clockFmt = new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "Asia/Dubai",
  });
  function fmtTime(iso) { const d = new Date(iso); return isNaN(d) ? (iso || "") : timeFmt.format(d); }
  function fmtClock(iso) {
    const d = new Date(iso);
    if (isNaN(d)) return iso || "";
    return clockFmt.format(d) + "." + String(Math.floor(d.getMilliseconds() / 100));
  }
  function ms(iso) { const t = Date.parse(iso); return Number.isFinite(t) ? t : null; }
  function fmtDur(msv) {
    if (msv === null || msv === undefined || !Number.isFinite(msv) || msv < 0) return "";
    if (msv < 10) return "under 0.01 s";
    const s = msv / 1000;
    if (s < 1) return s.toFixed(2) + " s";
    if (s < 60) return s.toFixed(1) + " s";
    const m = Math.floor(s / 60);
    return m + " min " + Math.round(s - m * 60) + " s";
  }
  function plural(n, one, many) { return n + " " + (n === 1 ? one : (many || one + "s")); }
  function num(n) { return typeof n === "number" ? n.toLocaleString("en-GB") : String(n ?? ""); }

  // ---------- labels ----------
  const STATUS = {
    screening: { label: "Answering", cls: "pill-working" },
    awaiting_review: { label: "Awaiting review", cls: "pill-awaiting" },
    held: { label: "Held", cls: "pill-held" },
    approved: { label: "Approved and sent", cls: "pill-approved" },
    rejected: { label: "Rejected, not sent", cls: "pill-rejected" },
  };
  function statusPill(status) {
    const s = STATUS[status] || { label: "Answering", cls: "pill-working" };
    return h("span", { class: "pill " + s.cls }, s.label);
  }

  const CHECK = {
    pass: "Every sentence passed the checker",
    pass_with_cuts: "Checked, some sentences cut",
    held: "Held: too much was cut to send",
  };

  // dalil/safety.py names what it found as "personal_data:<kind>".
  const PERSONAL = {
    emirates_id: "an Emirates ID number",
    passport: "a passport number",
    phone: "a phone number",
    email: "an email address",
    national_insurance: "a National Insurance number",
    nhs_number: "an NHS number",
  };
  function flagCode(flag) {
    return String(typeof flag === "string" ? flag : (flag && (flag.kind || flag.code || flag.type)) || "").toLowerCase();
  }
  function isPersonal(flag) { return /pii|personal|redact/.test(flagCode(flag)); }
  function personalWhat(flag) {
    const kind = flagCode(flag).replace(/^personal_data[:_]?/, "");
    return PERSONAL[kind] || "personal details";
  }
  function joinWords(list) {
    return list.length < 2 ? (list[0] || "") : list.slice(0, -1).join(", ") + " and " + list[list.length - 1];
  }

  function flagInfo(flag, all) {
    const detail = typeof flag === "object" && flag ? (flag.detail || flag.message || "") : "";
    const c = flagCode(flag);
    if (isPersonal(flag)) {
      const what = joinWords((all || [flag]).filter(isPersonal).map(personalWhat));
      return { cls: "flag-privacy", title: "Personal data removed",
        body: detail || ("Dalil replaced " + what + " with a placeholder before anything was saved. Only the placeholder is stored.") };
    }
    if (/evasion|evade|circumvent|loophole/.test(c)) {
      return { cls: "flag-evasion", title: "Asks how to get around a rule",
        body: detail || "Dalil does not help anyone avoid a law or a government process, so it searched nothing and drafted a fixed refusal." };
    }
    if (/vague/.test(c)) {
      return { cls: "flag-other", title: "No clear subject",
        body: detail || "The question does not say which service it is about, so Dalil asks for that instead of guessing." };
    }
    return { cls: "flag-other", title: "Flagged at screening", body: detail || c.replace(/[_:]/g, " ") };
  }

  // ---------- pipeline stages, from the audit trail ----------
  // Y.3172 nodes as docs/itu/report-draft.md section 4 maps them.
  const NODES = {
    PP: "Preprocessor", M: "Model", P: "Policy", D: "Distributor", SINK: "Sink",
  };
  const STAGES = [
    { key: "screen", name: "Screener", node: "PP", kinds: ["screened"] },
    { key: "search", name: "Searcher", node: "M", kinds: ["recalled", "retrieved", "data_looked_up"] },
    { key: "write", name: "Writer", node: "M", kinds: ["drafted"] },
    { key: "check", name: "Checker", node: "P", kinds: ["verified"] },
    { key: "review", name: "Human reviewer", node: "P", kinds: ["approved", "rejected"] },
    { key: "send", name: "Sent", node: "D", kinds: ["delivered"] },
  ];

  function lastOf(audit, kinds) {
    let found = null;
    for (const a of audit) if (kinds.includes(a.kind)) found = a;
    return found;
  }
  function firstOf(audit, kind) { return audit.find((a) => a.kind === kind) || null; }
  const obj = (d) => (d && typeof d === "object" && !Array.isArray(d) ? d : {});

  function drafterLabel(d) {
    const who = String(d.drafter || "");
    if (who === "fixed refusal") return "Fixed reply, no model";
    if (/LLM/i.test(who)) return "Nemotron wrote " + plural(Number(d.sentences) || 0, "sentence");
    if (/Extractive/i.test(who)) return d.fallback_reason ? "Model unavailable, quoted sources instead" : "Quoted " + plural(Number(d.sentences) || 0, "source sentence");
    return who ? who + ", " + plural(Number(d.sentences) || 0, "sentence") : "";
  }

  function searchSummary(audit) {
    const r = firstOf(audit, "retrieved");
    if (!r) return "";
    const d = obj(r.detail);
    if (d.skipped === "rule_evasion") return "Not run: a request to get around the rules";
    if (d.skipped === "vague_question") return "Not run: no clear subject";
    const rec = obj((firstOf(audit, "recalled") || {}).detail);
    const left = (rec.passages_left_out || []).length;
    let s = typeof d.hits === "number" ? plural(d.hits, "passage") + " found" : "Passages found";
    if (left) s += ", " + left + " left out after a rejection";
    return s;
  }

  // Each stage: state is waiting | running | done | skipped | stopped | attention.
  function computeStages(r) {
    const audit = r.audit || [];
    const failed = firstOf(audit, "failed");
    const inProgress = !r.check && !failed && (r.status === "screening" || r.status === "working" || !r.status);
    let prevEnd = ms(r.created_at);
    let runningAssigned = false;
    const out = [];
    for (const st of STAGES) {
      const ev = lastOf(audit, st.kinds);
      const s = { key: st.key, name: st.name, node: st.node, state: "waiting", text: "Waiting", dur: null, since: null };
      if (st.key === "screen" && ev) {
        s.state = "done";
        const flags = (obj(ev.detail).flags) || r.flags || [];
        s.text = flags.some((f) => /personal/.test(f)) ? "Personal data removed"
          : flags.includes("rule_evasion") ? "Rule-dodging request found"
          : flags.includes("vague_question") ? "No clear subject"
          : "Nothing to remove";
      } else if (st.key === "search" && ev) {
        const d = obj((firstOf(audit, "retrieved") || {}).detail);
        s.state = d.skipped ? "skipped" : "done";
        s.text = searchSummary(audit);
        if (!firstOf(audit, "retrieved") && inProgress) { s.state = "running"; s.text = "Searching"; s.since = prevEnd; }
      } else if (st.key === "write" && ev) {
        s.state = "done";
        s.text = drafterLabel(obj(ev.detail));
      } else if (st.key === "check" && ev) {
        const d = obj(ev.detail);
        s.state = d.status === "held" ? "attention" : "done";
        s.text = d.status === "held" ? "Held: too much cut"
          : (Number(d.kept) || 0) + " kept, " + (Number(d.cut) || 0) + " cut";
      } else if (st.key === "review" && ev) {
        const d = obj(ev.detail);
        s.state = ev.kind === "approved" ? "done" : "attention";
        s.text = (ev.kind === "approved" ? "Approved" : "Rejected") + (d.reviewer ? " by " + d.reviewer : "");
        const q = firstOf(audit, "queued");
        if (q) prevEnd = ms(q.at);
      } else if (st.key === "send" && ev) {
        s.state = "done";
        s.text = "Sent to the resident";
      }

      if (ev && (s.state !== "running")) {
        const end = ms(ev.at);
        s.dur = prevEnd !== null && end !== null ? end - prevEnd : null;
        prevEnd = end;
      } else if (!ev) {
        // Not reached yet. Say why, from what the trail does show.
        if (failed) {
          if (!runningAssigned) { s.state = "stopped"; s.text = "Stopped on an error"; runningAssigned = true; }
          else { s.state = "skipped"; s.text = "Not reached"; }
        } else if (inProgress && !runningAssigned) {
          s.state = "running"; s.text = st.key === "write" ? "Writing" : st.key === "check" ? "Checking" : st.key === "search" ? "Searching" : "Running";
          s.since = prevEnd; runningAssigned = true;
        } else if (st.key === "review") {
          if (r.check && (r.status === "awaiting_review" || r.status === "held")) s.state = "current";
          s.text = r.status === "held" ? "Can only be rejected" : r.check ? "Waiting for a decision" : "Waiting";
        } else if (st.key === "send") {
          s.state = r.status === "rejected" ? "skipped" : r.status === "held" ? "skipped" : "waiting";
          s.text = r.status === "rejected" || r.status === "held" ? "Not sent" : "Waiting for approval";
        }
      }
      if (s.state === "running") runningAssigned = true;
      out.push(s);
    }
    return out;
  }

  function renderStages(r) {
    const ol = clear($("stages"));
    for (const s of computeStages(r)) {
      const time = s.state === "running"
        ? h("span", { class: "stage-time", "data-since": s.since === null ? null : String(s.since) }, s.since === null ? "" : fmtDur(Date.now() - s.since))
        : s.dur !== null ? h("span", { class: "stage-time" }, fmtDur(s.dur)) : null;
      ol.append(h("li", { class: "stage stage-" + s.state, "data-stage": s.key },
        h("span", { class: "stage-name" }, s.name),
        h("span", { class: "stage-node" }, h("abbr", { title: "ITU-T Y.3172 " + NODES[s.node] + " node" }, s.node), " " + NODES[s.node]),
        h("span", { class: "stage-state" },
          h("span", { class: "visually-hidden" }, STATE_WORD[s.state] + ": "),
          h("span", { class: "stage-text" }, s.text),
          time ? [" ", time] : null)));
    }
    syncTicker();
  }
  const STATE_WORD = { waiting: "Not started", running: "Running", done: "Done", skipped: "Skipped", stopped: "Stopped", attention: "Done", current: "Waiting for a person" };

  let ticker = null;
  function syncTicker() {
    const live = document.querySelector(".stage-time[data-since]");
    if (live && !ticker) {
      ticker = setInterval(() => {
        const els = document.querySelectorAll(".stage-time[data-since]");
        if (!els.length) { clearInterval(ticker); ticker = null; return; }
        els.forEach((el) => { el.textContent = fmtDur(Date.now() - Number(el.dataset.since)); });
      }, 100);
    }
  }

  // ---------- audit trail in plain words ----------
  function auditRow(a, r) {
    const d = obj(a.detail);
    const str = typeof a.detail === "string" ? a.detail : "";
    switch (a.kind) {
      case "screened": {
        const flags = d.flags || [];
        const personal = flags.filter(isPersonal).map(personalWhat);
        let text = "Checked the question for personal data and for requests to get around the rules.";
        if (personal.length) text = "Replaced " + joinWords(personal) + " with a placeholder before saving.";
        if (flags.includes("rule_evasion")) text = "Found a request to get around the rules.";
        if (flags.includes("vague_question")) text += " The question has no clear subject.";
        if (!flags.length && !str) text = "Nothing to remove, nothing to refuse.";
        return { part: "Screener", node: "PP", text: str || text };
      }
      case "recalled":
        if (d.available === false) return { part: "Review memory", text: "The review memory was not available, so no past reviews were used." };
        return { part: "Review memory", text: (d.similar_reviews && d.similar_reviews.length
          ? "Found " + plural(d.similar_reviews.length, "similar past review") + "."
          : "No similar past reviews.") + ((d.passages_left_out || []).length
          ? " Left out " + plural(d.passages_left_out.length, "passage") + " a reviewer rejected before." : "") };
      case "retrieved":
        if (d.skipped === "rule_evasion") return { part: "Searcher", node: "M", text: "Searched nothing: the question asks how to get around the rules." };
        if (d.skipped === "vague_question") return { part: "Searcher", node: "M", text: "Searched nothing: the question has no clear subject." };
        return { part: "Searcher", node: "M", text: str || ("Found " + plural(Number(d.hits) || 0, "passage") + " by keywords and by meaning.") };
      case "data_looked_up": {
        const notes = (d.notes || []).join(" ");
        if (/skipped|unreachable|not configured/i.test(notes)) {
          return { part: "Searcher", node: "M", text: "Open-data lookup skipped: the data service is not connected. The answer uses the documents only." };
        }
        const rows = (d.rows || []).reduce((x, y) => x + (Number(y) || 0), 0);
        const where = d.source === "local" ? " in the local database" : d.source === "zetaris" ? " through Zetaris" : "";
        return { part: "Searcher", node: "M", text: "Queried " + plural((d.tables || []).length, "open-data table") + where + ", " + plural(rows, "row") + " returned." };
      }
      case "drafted": {
        const who = String(d.drafter || "");
        let text = str;
        if (!text) {
          if (who === "fixed refusal") text = "Used Dalil's fixed reply. No model was involved.";
          else if (/LLM/i.test(who)) text = "NVIDIA Nemotron drafted " + plural(Number(d.sentences) || 0, "sentence") + ", each pointing at a source.";
          else if (/Extractive/i.test(who)) text = (d.fallback_reason ? "The model did not answer (" + d.fallback_reason + "), so Dalil quoted " : "Dalil quoted ") + plural(Number(d.sentences) || 0, "source sentence") + " instead.";
          else text = plural(Number(d.sentences) || 0, "sentence") + " drafted by " + who + ".";
        }
        return { part: "Writer", node: "M", text };
      }
      case "verified": {
        const cut = ((r.check && r.check.sentences) || []).filter((s) => !s.kept);
        const restated = Number(d.data_restated) || 0;
        const text = (str || (d.status === "held"
          ? "Kept " + (Number(d.kept) || 0) + ", cut " + (Number(d.cut) || 0) + ". Too much was cut to send, so the answer is held."
          : "Tested every quote and number. Kept " + (Number(d.kept) || 0) + ", cut " + (Number(d.cut) || 0) + "."))
          + (restated ? " Dalil wrote " + plural(restated, "sentence") + " from its open-data rule, because the model did not state the value; the query was rerun like any other." : "");
        return { part: "Checker", node: "P", text, cut };
      }
      case "queued":
        return { part: "Review queue", node: "P", text: str || (d.status === "held"
          ? "Held for a person. It can only be rejected."
          : "Waiting for a person to approve or reject it.") };
      case "approved":
      case "rejected":
        return { part: "Human reviewer", node: "P",
          text: str || ((a.kind === "approved" ? "Approved" : "Rejected") + (d.reviewer ? " by " + d.reviewer : "") + "."),
          note: d.note || "" };
      case "delivered":
        return { part: "Distributor", node: "D", text: str || "Sent to the resident. Only the kept sentences went out." };
      case "memory_written":
        return { part: "Review memory", node: "SINK", text: d.available === false
          ? "The review memory was not available. The decision is kept in the audit trail."
          : "Decision kept in the review memory with " + plural(Number(d.passages) || 0, "cited passage") + ", so a similar question can learn from it." };
      case "failed":
        return { part: "Dalil", text: "Stopped on an error (" + (d.error || "unknown") + "). The answer is held and nothing will be sent." };
      default:
        return { part: String(a.kind).replace(/_/g, " "), text: str || "" };
    }
  }

  // ---------- API (real or mock) ----------
  async function http(method, path, body) {
    const res = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      let msg = res.status + " " + res.statusText;
      try { const j = await res.json(); if (j && j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (e) { /* not json */ }
      throw new Error(msg);
    }
    return res.json();
  }

  const realApi = {
    health: () => http("GET", "/api/health"),
    list: () => http("GET", "/api/questions"),
    get: (id) => http("GET", "/api/questions/" + encodeURIComponent(id)),
    ask: (question) => http("POST", "/api/questions", { question }),
    review: (id, body) => http("POST", "/api/questions/" + encodeURIComponent(id) + "/review", body),
    page: (doc, page) => http("GET", "/api/documents/" + encodeURIComponent(doc) + "/pages/" + encodeURIComponent(page)),
    fileUrl: (doc) => "/api/documents/" + encodeURIComponent(doc) + "/file",
    knowledgeBases: () => http("GET", "/api/knowledge-bases"),
    switchKnowledgeBase: (id) => http("POST", "/api/knowledge-base", { id }),
  };

  function mockApi() {
    let fx = null;
    const ready = fetch("fixtures/mock.json").then((r) => r.json()).then((j) => { fx = j; });
    const copy = (o) => JSON.parse(JSON.stringify(o));
    const wait = (t) => new Promise((r) => setTimeout(r, t));
    const pending = new Map(); // id -> time it becomes ready
    let seq = 200;

    function summary(q) {
      const s = (q.check && q.check.sentences) || [];
      return {
        id: q.id, question: q.question, status: q.status, created_at: q.created_at,
        verifier_status: q.check ? q.check.status : null,
        kept: s.filter((x) => x.kept).length, cut: s.filter((x) => !x.kept).length,
      };
    }
    function find(id) {
      const q = fx.questions.find((x) => x.id === id);
      if (!q) throw new Error("404 Not Found");
      return q;
    }
    function view(q) {
      const r = copy(q);
      if (pending.has(q.id) && Date.now() < pending.get(q.id)) {
        r.check = null; r.status = "screening";
        r.audit = r.audit.filter((a) => Date.parse(a.at) <= Date.now() - 1500);
      }
      return r;
    }
    return {
      health: async () => { await ready; return copy(fx.health); },
      list: async () => { await ready; await wait(80); return fx.questions.map(view).map(summary); },
      get: async (id) => { await ready; await wait(80); return view(find(id)); },
      ask: async (question) => {
        await ready; await wait(200);
        const now = new Date();
        const t = copy(fx.new_question_template);
        const q = Object.assign(t, { id: "q-0" + (++seq), question, created_at: now.toISOString() });
        q.audit = q.audit.map((a, i) => Object.assign({ at: new Date(now.getTime() + 1500 + i * 700).toISOString() }, a));
        fx.questions.unshift(q);
        pending.set(q.id, Date.now() + 1500 + q.audit.length * 700);
        await wait(1500 + q.audit.length * 700);
        return { id: q.id, status: q.status };
      },
      review: async (id, body) => {
        await ready; await wait(150);
        const q = find(id);
        if (q.review) throw new Error("This answer already has a decision.");
        if (body.decision === "approve" && q.status !== "awaiting_review") throw new Error("Only answers awaiting review can be approved.");
        if (body.decision === "reject" && !["awaiting_review", "held"].includes(q.status)) throw new Error("This answer cannot be rejected now.");
        const at = new Date().toISOString();
        q.review = { decision: body.decision, reviewer: body.reviewer, note: body.note || "", at };
        q.status = body.decision === "approve" ? "approved" : "rejected";
        q.audit.push({ at, kind: q.status, detail: { reviewer: body.reviewer, note: body.note || "" } });
        if (body.decision === "approve") q.audit.push({ at, kind: "delivered", detail: { reviewer: body.reviewer } });
        return view(q);
      },
      page: async (doc, page) => {
        await ready; await wait(60);
        const p = fx.pages[doc + "/" + page];
        if (!p) throw new Error("404 Page not found");
        return copy(p);
      },
      fileUrl: () => null,
      knowledgeBases: async () => {
        await ready;
        const kb = fx.health.knowledge_base || { id: "sample", name: "Sample knowledge base", documents: null, passages: fx.health.passages };
        return [Object.assign({ active: true, loaded: true }, kb)];
      },
      switchKnowledgeBase: async () => { throw new Error("Sample data has one knowledge base."); },
    };
  }

  const api = MOCK ? mockApi() : realApi;

  // ---------- state ----------
  const state = {
    list: [],
    selectedId: null,
    record: null,
    cites: [],        // numbered citations for the selected record
    activeCite: null, // number
    pages: new Map(),
    pollTimer: null,
    kbSwitches: 0,    // counts knowledge-base switches, so an answer from before one is not shown after it
    busy: false,
    asking: null,
    sourceToken: 0,
  };

  function notify(msg) {
    const n = $("notice");
    if (!msg) { n.hidden = true; n.textContent = ""; return; }
    n.textContent = msg;
    n.hidden = false;
  }

  // ---------- health ----------
  function kbLine(hl) {
    const kb = hl.knowledge_base;
    if (kb && typeof kb === "object") {
      const bits = [];
      if (typeof kb.documents === "number") bits.push(num(kb.documents) + " documents");
      if (typeof kb.passages === "number") bits.push(num(kb.passages) + " passages");
      return { label: kb.name || "Knowledge base", detail: bits.join(", ") };
    }
    return { label: "Knowledge base", detail: typeof hl.passages === "number" ? num(hl.passages) + " passages" : "" };
  }

  function dataLine(hl) {
    const ds = hl.data_source;
    if (ds && typeof ds === "object") {
      // The server's label reads "Open data: local database"; show the part after the colon as the detail.
      const full = String(ds.label || ds.kind || "Open data");
      const at = full.indexOf(":");
      const label = at > 0 ? full.slice(0, at).trim() : "Open data";
      const what = at > 0 ? full.slice(at + 1).trim() : full;
      const n = Array.isArray(ds.tables) ? ds.tables.length : typeof ds.tables === "number" ? ds.tables : null;
      return ds.connected
        ? { on: true, label, detail: what + (n ? ", " + plural(n, "table") : "") }
        : { on: false, label, detail: (what || "not configured") + ", answers use the documents only" };
    }
    return hl.zetaris
      ? { on: true, label: "Open data", detail: "connected" }
      : { on: false, label: "Open data", detail: "not connected, answers use the documents only" };
  }

  async function loadHealth() {
    const el = $("health");
    clear(el);
    if (MOCK) el.append(h("span", { class: "badge badge-mock" }, "Sample data (mock mode)"));
    try {
      const hl = await api.health();
      const kb = kbLine(hl);
      const ds = dataLine(hl);
      el.append(
        h("p", { class: "health-line" }, h("span", { class: "health-key" }, kb.label), kb.detail ? h("span", { class: "health-val" }, kb.detail) : null),
        h("p", { class: "health-line" },
          h("span", { class: "dot " + (ds.on ? "dot-on" : "dot-off"), "aria-hidden": "true" }),
          h("span", { class: "health-key" }, ds.label), h("span", { class: "health-val" }, ds.detail)));
      // Each knowledge base names its own country in the ask box.
      const kbInfo = hl.knowledge_base;
      if (kbInfo && kbInfo.ask_label) $("ask-label").textContent = kbInfo.ask_label;
      if (kbInfo && kbInfo.example_question) $("ask-input").placeholder = kbInfo.example_question;
    } catch (e) {
      el.append(h("p", { class: "health-line health-warn" }, "Dalil's server on this computer is not answering. Start it, then reload this page."));
    }
  }

  // ---------- knowledge base ----------
  // The header line under the menu gives the active one's counts, so the menu shows names only.
  function kbLabel(k) { return k.name || k.id; }

  async function loadKnowledgeBases() {
    const sel = $("kb-select");
    try {
      const list = await api.knowledgeBases();
      clear(sel);
      for (const k of list) sel.append(h("option", { value: k.id, selected: k.active }, kbLabel(k)));
      sel.disabled = list.length < 2;
    } catch (e) {
      sel.disabled = true;
    }
  }

  async function switchKnowledgeBase(e) {
    const sel = e.target;
    const id = sel.value;
    sel.disabled = true;
    notify(null);
    const el = clear($("health"));
    el.append(h("p", { class: "health-line" }, h("span", { class: "dot dot-off", "aria-hidden": "true" }),
      h("span", { class: "health-key" }, "Switching knowledge base")));
    try {
      await api.switchKnowledgeBase(id);
      // Questions belong to the knowledge base that answered them: start from its own queue.
      // A question still being answered on the old one finishes there, and is not shown here.
      state.kbSwitches++;
      state.asking = null;
      stopPoll();
      state.selectedId = null;
      state.record = null;
      state.cites = [];
      state.activeCite = null;
      history.replaceState(null, "", location.pathname + location.search);
      renderRecord();
      renderSourceEmpty();
      clear($("searched"));
      await Promise.all([loadHealth(), loadQueue()]);
    } catch (err) {
      notify("Could not switch the knowledge base. " + err.message);
      loadHealth();
    }
    await loadKnowledgeBases();
  }

  // ---------- queue ----------
  async function loadQueue() {
    try {
      state.list = await api.list();
      renderQueue();
    } catch (e) {
      notify("The review queue could not be loaded: " + e.message + ". Check that Dalil is running, then reload.");
    }
  }

  function renderQueue() {
    const ul = clear($("queue-list"));
    const waiting = state.list.filter((q) => q.status === "awaiting_review" || q.status === "held").length;
    $("queue-count").textContent = waiting ? waiting + " to review" : "Nothing waiting";
    if (!state.list.length) {
      ul.append(h("li", { class: "queue-empty" }, "No questions yet. Ask one above."));
      return;
    }
    for (const q of state.list) {
      const selected = q.id === state.selectedId;
      const counts = q.verifier_status
        ? h("span", { class: "counts" },
            h("span", null, q.kept + " kept"),
            q.cut ? h("span", { class: "counts-cut" }, q.cut + " cut") : null)
        : h("span", { class: "counts" }, "Answering");
      const btn = h("button", {
        type: "button", class: "queue-item" + (selected ? " is-selected" : ""),
        "aria-current": selected ? "true" : null,
        onclick: () => select(q.id, true),
      },
        h("span", { class: "queue-top" }, statusPill(q.status), h("time", { datetime: q.created_at }, fmtTime(q.created_at))),
        h("span", tx(q.question, { class: "queue-q" }), q.question),
        counts);
      ul.append(h("li", null, btn));
    }
  }

  // ---------- selection ----------
  async function select(id, focusPanel) {
    if (state.selectedId !== id) { state.activeCite = null; setTab("tab-answer"); }
    state.selectedId = id;
    renderQueue();
    stopPoll();
    history.replaceState(null, "", location.pathname + location.search + "#" + encodeURIComponent(id));
    try {
      const rec = await api.get(id);
      if (state.selectedId !== id) return;
      state.record = rec;
      renderRecord();
      if (focusPanel) $("answer-panel").focus({ preventScroll: true });
      if (!rec.check) startPoll(id);
    } catch (e) {
      notify("Question " + id + " could not be loaded: " + e.message + ".");
    }
  }

  function startPoll(id) {
    stopPoll();
    state.pollTimer = setTimeout(async () => {
      if (state.selectedId !== id) return;
      try {
        const rec = await api.get(id);
        if (state.selectedId !== id) return;
        const changed = !state.record || (rec.audit || []).length !== (state.record.audit || []).length || !!rec.check !== !!state.record.check;
        state.record = rec;
        if (changed) renderRecord();
        if (rec.check || rec.status === "held") { loadQueue(); }
        else startPoll(id);
      } catch (e) { startPoll(id); }
    }, state.record && !state.record.check ? LIVE_POLL_MS : POLL_MS);
  }
  function stopPoll() { if (state.pollTimer) clearTimeout(state.pollTimer); state.pollTimer = null; }

  // ---------- record ----------
  function numberCitations(sentences) {
    const cites = [];
    const byKey = new Map();
    const order = sentences.filter((s) => s.kept).concat(sentences.filter((s) => !s.kept));
    for (const s of order) {
      s._nums = [];
      for (const c of s.citations || []) {
        const key = c.table
          ? "t|" + c.table + "|" + c.sql + "|" + c.value
          : "d|" + c.doc_id + "|" + c.page + "|" + c.page_start + "|" + c.page_end + "|" + (c.quote || "");
        let n = byKey.get(key);
        if (!n) {
          n = cites.length + 1;
          byKey.set(key, n);
          cites.push(Object.assign({ n }, c));
        } else if (c.ok === false) {
          cites[n - 1].ok = false;
        }
        s._nums.push(n);
      }
    }
    return cites;
  }

  function citeTitle(c) { return c.table ? (c.title || "Open-data table") : (c.title || c.doc_id); }

  function chip(n) {
    const c = state.cites[n - 1];
    const isData = !!c.table;
    const where = isData ? "open data, " + citeTitle(c) : citeTitle(c) + ", page " + c.page;
    const label = "Source " + n + ": " + where + (c.ok === false ? ". This citation did not pass the checker." : "");
    return h("button", {
      type: "button",
      class: "chip" + (isData ? " chip-data" : "") + (c.ok === false ? " chip-bad" : "") + (state.activeCite === n ? " is-active" : ""),
      "aria-label": label, title: label, "data-n": n,
      "aria-pressed": state.activeCite === n ? "true" : "false",
      onclick: () => openCitation(n),
    }, String(n));
  }
  function chips(nums) { return nums && nums.length ? h("span", { class: "chips", dir: "ltr", lang: "en" }, nums.map(chip)) : null; }

  // A sentence with its citation numbers. The numbers stay on the line of the
  // sentence's last word instead of wrapping onto a line of their own.
  function sentenceEl(text, nums, cls, strike) {
    const t = String(text || "");
    const i = t.trimEnd().lastIndexOf(" ");
    const head = i > 0 ? t.slice(0, i + 1) : "";
    const tail = i > 0 ? t.slice(i + 1) : t;
    const words = (x) => (strike ? h("s", null, x) : x);
    const c = chips(nums);
    return h("span", tx(t, { class: cls }),
      strike ? h("span", { class: "visually-hidden" }, "Cut sentence: ") : null,
      head ? words(head) : null,
      c ? h("span", { class: "keep" }, words(tail), "\u00a0", c) : words(tail));
  }

  function renderRecord() {
    const r = state.record;
    $("answer-empty").hidden = !!r;
    $("answer-body").hidden = !r;
    if (!r) { $("review-form").hidden = true; return; }

    $("q-id").textContent = r.id;
    $("q-time").textContent = fmtTime(r.created_at);
    $("q-time").setAttribute("datetime", r.created_at || "");
    $("q-flags").textContent = (r.flags || []).length ? (r.flags || []).join(", ") : "none";
    $("q-check").textContent = r.check ? r.check.status : "not checked yet";
    const qt = $("q-text");
    qt.textContent = r.question;
    if (isArabic(r.question)) qt.setAttribute("lang", "ar"); else qt.removeAttribute("lang");

    const meta = clear($("q-meta"));
    meta.append(statusPill(r.status));
    if (r.check) meta.append(h("span", { class: "check-status check-" + r.check.status }, CHECK[r.check.status] || r.check.status));
    const queued = firstOf(r.audit || [], "queued");
    if (queued && ms(r.created_at) !== null) {
      meta.append(h("span", { class: "answered-in" }, "Answered in " + fmtDur(ms(queued.at) - ms(r.created_at))));
    }
    meta.append(h("span", { class: "asked-at" }, "Asked " + fmtTime(r.created_at)));

    const flags = clear($("flags"));
    // One notice for all personal data found, then one per other flag.
    const shown = (r.flags || []).filter((f, i, all) => !isPersonal(f) || all.findIndex(isPersonal) === i);
    for (const f of shown) {
      const fi = flagInfo(f, r.flags);
      flags.append(h("div", { class: "flag " + fi.cls, role: "note" },
        h("strong", null, fi.title), h("span", null, fi.body)));
    }

    const sentences = (r.check && r.check.sentences) || [];
    state.cites = numberCitations(sentences);
    $("working").hidden = !!r.check || !!firstOf(r.audit || [], "failed");
    renderStages(r);
    renderStateNote(r);
    renderDraft(sentences);
    renderCuts(sentences);
    renderAudit(r);
    renderSearched(r);
    renderReview(r);
    if (state.activeCite && state.cites[state.activeCite - 1]) openCitation(state.activeCite, true);
    else renderSourceEmpty();
  }

  function renderStateNote(r) {
    const box = clear($("state-note"));
    const failed = firstOf(r.audit || [], "failed");
    if (failed) {
      box.append(h("div", { class: "note note-warn", role: "note" },
        h("strong", null, "Dalil stopped before it finished this answer."),
        h("span", null, " Nothing will be sent. The answer is held for a person, and the audit trail shows where it stopped.")));
    } else if (r.status === "held" && !r.review) {
      box.append(h("div", { class: "note note-warn", role: "note" },
        h("strong", null, "Held."),
        h("span", null, " The checker cut too much of this draft to send it. A reviewer can only reject it.")));
    }
  }

  function renderDraft(sentences) {
    const box = clear($("draft"));
    const kept = sentences.filter((s) => s.kept);
    const failed = firstOf(state.record.audit || [], "failed");
    // While Dalil is still answering, the working note above says what is happening.
    box.closest("section").hidden = !state.record.check && !failed;
    if (!state.record.check) {
      box.append(h("p", { class: "muted" }, "There is no answer to send."));
      return;
    }
    if (!kept.length) {
      const ans = state.record.check.answer;
      box.append(h("p", tx(ans, { class: "muted" }), ans || "No sentence passed the checker, so there is nothing to send."));
      return;
    }
    let para = null;
    const asksDetail = (state.record.flags || []).includes("vague_question");
    let callout = null; // { kind, p } so a refusal and its follow-up share one box
    for (const s of kept) {
      if (s.kind === "refusal" || s.kind === "caveat") {
        para = null;
        const isRef = s.kind === "refusal";
        if (!callout || callout.kind !== s.kind) {
          const p = h("p", { dir: paraDir(s.text) });
          box.append(h("div", { class: "callout " + (isRef ? "callout-refusal" : "callout-caveat") },
            h("span", { class: "callout-label" }, isRef ? (asksDetail ? "Dalil asks for detail" : "Dalil declines") : "Note for the resident"), p));
          callout = { kind: s.kind, p };
        }
        add(callout.p, sentenceEl(s.text, s._nums, "sentence"), " ");
        continue;
      }
      callout = null;
      if (!para) { para = h("p", { class: "answer-para", dir: paraDir(s.text) }); box.append(para); }
      const bad = (s.citations || []).some((c) => c.ok === false);
      add(para, sentenceEl(s.text, s._nums, "sentence" + (bad ? " sentence-warn" : "")), " ");
      if (s.restated) {
        add(para, h("span", { class: "restated-tag", title: "The model did not state this value, so Dalil wrote it from its open-data rule. The checker reran the query." },
          "Written by Dalil from the open data"), " ");
      }
    }
  }

  function cutReasons(s) {
    return (s.reasons && s.reasons.length) ? s.reasons : (s.citations || []).filter((c) => c.ok === false).map((c) => c.reason).filter(Boolean);
  }

  function renderCuts(sentences) {
    const cut = sentences.filter((s) => !s.kept);
    const sec = $("cut-section");
    sec.hidden = !cut.length;
    const ol = clear($("cut-list"));
    $("cut-heading").textContent = "Cut by the checker (" + cut.length + ")";
    for (const s of cut) {
      const reasons = cutReasons(s);
      ol.append(h("li", { class: "cut-item" },
        h("p", { class: "cut-text", dir: paraDir(s.text) },
          sentenceEl(s.text, s._nums, "sentence", true)),
        h("p", { class: "cut-reason" },
          h("span", { class: "cut-reason-label" }, "Why it was cut: "),
          reasons.length ? reasons.join("; ") : "it did not pass the checker")));
    }
  }

  function renderAudit(r) {
    const audit = r.audit || [];
    const ol = clear($("audit-list"));
    if (!audit.length) ol.append(h("li", { class: "muted" }, "No steps recorded yet."));
    let prev = ms(r.created_at);
    for (const a of audit) {
      const row = auditRow(a, r);
      const t = ms(a.at);
      const delta = prev !== null && t !== null ? t - prev : null;
      prev = t ?? prev;
      ol.append(h("li", { class: "audit-item audit-" + a.kind },
        h("div", { class: "audit-when" },
          h("time", { datetime: a.at, class: "audit-time" }, fmtClock(a.at)),
          delta !== null && delta >= 0 ? h("span", { class: "audit-delta" }, delta < 10 ? "+<0.01 s" : "+" + fmtDur(delta)) : null),
        h("div", { class: "audit-body" },
          h("p", { class: "audit-part" }, row.part,
            row.node ? h("span", { class: "audit-node" }, h("abbr", { title: "ITU-T Y.3172 " + NODES[row.node] + " node" }, row.node)) : null),
          h("p", { class: "audit-text" }, row.text),
          row.note ? h("p", tx(row.note, { class: "audit-note" }), "Note: " + row.note) : null,
          row.cut && row.cut.length ? h("ul", { class: "audit-cuts" }, row.cut.map((s) => h("li", null,
            h("s", tx(s.text), s.text),
            h("span", { class: "audit-cut-why" }, " Cut because: " + (cutReasons(s).join("; ") || "it did not pass the checker") + ".")))) : null)));
    }
    $("audit-raw").textContent = JSON.stringify(r, (k, v) => (k.startsWith("_") ? undefined : v), 2);
  }

  // ---------- evidence ----------
  function renderSourceEmpty() {
    const v = clear($("source-view"));
    const any = state.cites.length > 0;
    v.append(h("div", { class: "source-empty" },
      h("h3", { class: "panel-title" }, "Source"),
      h("p", null, any
        ? "Select a numbered citation in the answer to see the exact words it quotes, highlighted on the official page."
        : state.record && state.record.check
          ? "This answer cites no source."
          : "Sources appear here once the answer is checked.")));
  }

  function markActiveChips() {
    document.querySelectorAll(".chip").forEach((el) => {
      const on = Number(el.dataset.n) === state.activeCite;
      el.classList.toggle("is-active", on);
      el.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  async function getPage(doc, page) {
    const key = doc + "/" + page;
    if (!state.pages.has(key)) state.pages.set(key, api.page(doc, page));
    try { return await state.pages.get(key); } catch (e) { state.pages.delete(key); throw e; }
  }

  function citeStatus(c) {
    return c.ok === false
      ? h("p", { class: "cite-status cite-bad" }, "This citation did not pass the checker" + (c.reason ? ": " + c.reason : "."))
      : h("p", { class: "cite-status cite-ok" }, c.table ? "The checker reran this query and got the same value." : "The checker found these exact words on this page.");
  }

  async function openCitation(n, quiet) {
    const c = state.cites[n - 1];
    if (!c) return;
    state.activeCite = n;
    markActiveChips();
    if (c.table) return renderDataCitation(c, quiet);
    return renderDocCitation(c, quiet, { start: c.page_start, end: c.page_end, quote: c.quote, label: "Source " + n, cite: c });
  }

  async function renderDocCitation(c, quiet, sel) {
    const token = ++state.sourceToken;
    const v = clear($("source-view"));
    const heading = h("h3", { class: "panel-title", tabindex: "-1" }, sel.label);
    v.append(heading, h("p", { class: "muted" }, "Loading the page..."));
    if (!quiet) heading.focus({ preventScroll: true });
    let p;
    try {
      p = await getPage(c.doc_id, c.page);
    } catch (e) {
      if (token !== state.sourceToken) return;
      clear(v).append(heading, h("p", { class: "note note-warn" }, "Page " + c.page + " of " + (c.title || c.doc_id) + " could not be loaded: " + e.message + "."));
      return;
    }
    if (token !== state.sourceToken) return; // a newer click won
    const text = p.text || "";
    let s = Number(sel.start), e = Number(sel.end);
    const valid = Number.isFinite(s) && Number.isFinite(e) && s >= 0 && e > s && e <= text.length;
    if (!valid && sel.quote) {
      const i = text.indexOf(sel.quote);
      if (i >= 0) { s = i; e = i + sel.quote.length; }
    }
    const ok = Number.isFinite(s) && Number.isFinite(e) && s >= 0 && e > s && e <= text.length;

    const pageBox = h("div", tx(text, { class: "page-text", tabindex: "0", "aria-label": "Text of page " + p.page }));
    let mark = null;
    if (ok) {
      mark = h("mark", { class: "quote-mark" }, text.slice(s, e));
      pageBox.append(text.slice(0, s), mark, text.slice(e));
    } else {
      pageBox.append(text);
    }

    const fileUrl = api.fileUrl(c.doc_id);
    const links = h("p", { class: "source-links" },
      p.source_url ? h("a", { href: p.source_url, target: "_blank", rel: "noopener noreferrer" }, "Open the official page") : null,
      fileUrl ? h("a", { href: fileUrl, target: "_blank", rel: "noopener" }, "Open the original file") : null);
    const title = p.title || c.title || c.doc_id;

    add(clear(v),
      heading,
      h("p", tx(title, { class: "source-title" }), title),
      h("p", { class: "source-meta" }, (p.publisher ? p.publisher + ", " : "") + "page " + p.page),
      links,
      sel.cite ? citeStatus(sel.cite) : null,
      sel.quote && !ok ? h("p", { class: "cite-status cite-bad" }, "The quoted words could not be found on this page.") : null,
      pageBox);
    // Rebuilding the panel took the heading out of the page for a moment; put focus back.
    if (!quiet) heading.focus({ preventScroll: true });
    if (mark) {
      requestAnimationFrame(() => {
        // Scroll the page box (not the window) so the highlight sits near the top third.
        const top = mark.offsetTop - pageBox.offsetTop - pageBox.clientHeight / 3;
        pageBox.scrollTop = Math.max(0, top);
        if (!quiet) { const ev = v.closest(".evidence"); if (ev) ev.scrollTop = 0; }
      });
    }
  }

  function renderDataCitation(c, quiet) {
    state.sourceToken++;
    const v = clear($("source-view"));
    const heading = h("h3", { class: "panel-title", tabindex: "-1" }, "Source " + c.n);
    const d = (state.record.data || []).find((x) => x.table === c.table && x.sql === c.sql) ||
              (state.record.data || []).find((x) => x.table === c.table);
    v.append(heading,
      h("p", tx(citeTitle(c), { class: "source-title" }), citeTitle(c)),
      // source_label says where the lookup ran, for example "queried from the local database".
      h("p", { class: "source-meta" }, "Open data, " + (c.source_label || (d && d.source_label) || "queried through Zetaris")),
      c.source_url ? h("p", { class: "source-links" }, h("a", { href: c.source_url, target: "_blank", rel: "noopener noreferrer" }, "Open the dataset page")) : null,
      citeStatus(c),
      h("p", { class: "data-value" }, "Cited value: ", h("mark", { class: "quote-mark" }, String(c.value))));
    if (d && d.rows) v.append(h("p", { class: "sub-title" }, "Rows returned (" + d.rows.length + ")"), rowsTable(d.rows, c.value));
    v.append(h("details", { class: "tech" }, h("summary", null, "Show the query Dalil ran"),
      h("p", { class: "tech-line" }, "Table ", h("code", null, c.table)),
      h("pre", { class: "sql", tabindex: "0", "aria-label": "SQL query" }, h("code", null, c.sql || (d && d.sql) || "")),
      d && d.note ? h("p", { class: "muted small" }, d.note) : null));
    if (!quiet) heading.focus({ preventScroll: true });
    const hit = v.querySelector("td.is-cited");
    if (hit) requestAnimationFrame(() => hit.scrollIntoView({ block: "nearest" }));
  }

  function sameValue(a, b) {
    if (a === null || a === undefined || b === null || b === undefined) return false;
    const sa = String(a).trim(), sb = String(b).trim();
    if (sa === sb) return true;
    const na = Number(sa.replace(/,/g, "")), nb = Number(sb.replace(/,/g, ""));
    return sa !== "" && sb !== "" && Number.isFinite(na) && Number.isFinite(nb) && na === nb;
  }

  function rowsTable(rows, value) {
    const cols = [];
    for (const r of rows) for (const k of Object.keys(r)) if (!cols.includes(k)) cols.push(k);
    const shown = rows.slice(0, 50);
    return h("div", { class: "rows-wrap", tabindex: "0", role: "region", "aria-label": "Rows returned" },
      h("table", { class: "rows" },
        h("thead", null, h("tr", null, cols.map((k) => h("th", { scope: "col" }, k.replace(/_/g, " "))))),
        h("tbody", null, shown.map((r) => h("tr", null, cols.map((k) => {
          const cited = sameValue(r[k], value);
          return h("td", { class: cited ? "is-cited" : null, dir: "auto" },
            cited ? h("mark", { class: "quote-mark" }, fmtCell(r[k])) : fmtCell(r[k]),
            cited ? h("span", { class: "visually-hidden" }, " (cited value)") : null);
        }))))),
      rows.length > shown.length ? h("p", { class: "muted small" }, "Showing 50 of " + rows.length + " rows.") : null);
  }
  function fmtCell(v) { return v === null || v === undefined ? "" : typeof v === "object" ? JSON.stringify(v) : String(v); }

  function renderSearched(r) {
    const box = clear($("searched"));
    const hits = r.retrieval || [];
    const data = r.data || [];
    const retrieved = firstOf(r.audit || [], "retrieved");
    const skipped = retrieved && obj(retrieved.detail).skipped;
    if (!r.check && !hits.length && !skipped) { box.append(h("p", { class: "muted" }, retrieved ? "Search done. Results appear here once the answer is checked." : "Searching the passages...")); return; }
    if (skipped) { box.append(h("p", { class: "muted" }, "Nothing was searched for this question.")); return; }
    if (!hits.length && !data.length) { box.append(h("p", { class: "muted" }, "Dalil found nothing relevant in the documents or the data tables.")); return; }
    if (hits.length) {
      // A law's scope article can be added ahead of the ranked results (score 0); say so rather than call it a match.
      const added = hits.filter((x) => x.score === 0).length;
      box.append(h("p", { class: "sub-title" }, "Passages the writer was given (" + hits.length + ")" +
        (added ? ": " + plural(added, "scope article") + " added, then the best matches" : ", best match first")));
      const ol = h("ol", { class: "hits" });
      hits.forEach((x, i) => {
        const cited = state.cites.some((c) => c.passage_id && c.passage_id === x.passage_id);
        const title = x.title || x.doc_id;
        ol.append(h("li", null, h("button", {
          type: "button", class: "hit",
          "aria-label": "Search result " + (i + 1) + ": " + title + ", page " + x.page + (cited ? ", cited in the answer" : ""),
          onclick: () => openHit(x, i),
        },
          h("span", { class: "hit-top" },
            h("span", { class: "hit-rank", "aria-hidden": "true" }, String(i + 1)),
            h("span", { class: "hit-title", dir: paraDir(title) }, h("bdi", tx(title), title), h("span", { class: "hit-page" }, ", page " + x.page)),
            cited ? h("span", { class: "tag-cited" }, "Cited") : null),
          x.score === 0 ? h("span", { class: "hit-added" }, "Added: the law's own scope article") : null,
          x.heading ? h("span", tx(x.heading, { class: "hit-heading" }), x.heading) : null,
          h("span", tx(x.text, { class: "hit-text" }), x.text))));
      });
      box.append(ol);
      box.append(h("details", { class: "tech" }, h("summary", null, "Show search scores and passage ids"),
        h("div", { class: "rows-wrap" }, h("table", { class: "rows" },
          h("thead", null, h("tr", null, h("th", { scope: "col" }, "Rank"), h("th", { scope: "col" }, "Score"), h("th", { scope: "col" }, "Passage id"))),
          h("tbody", null, hits.map((x, i) => h("tr", null, h("td", null, String(i + 1)), h("td", { class: "num" }, fmtScore(x.score)), h("td", null, h("code", null, x.passage_id)))))))));
    }
    if (data.length) {
      box.append(h("p", { class: "sub-title" }, "Open-data queries (" + data.length + ")"));
      const ul = h("ul", { class: "lookups" });
      for (const d of data) {
        const cite = state.cites.find((c) => c.table === d.table);
        ul.append(h("li", { class: "lookup" },
          h("p", { class: "lookup-top" }, cite ? citeTitle(cite) : "Open-data table", " ", h("span", { class: "muted" }, plural(d.rows ? d.rows.length : 0, "row"))),
          d.note ? h("p", { class: "muted small" }, /skipped|unreachable/i.test(d.note) ? "Not run: the data service is not connected." : d.note) : null,
          h("details", { class: "tech" }, h("summary", null, "Show the query and rows"),
            h("p", { class: "tech-line" }, "Table ", h("code", null, d.table)),
            h("pre", { class: "sql" }, h("code", null, d.sql || "")),
            d.rows && d.rows.length ? rowsTable(d.rows, null) : null)));
      }
      box.append(ul);
    }
  }
  function fmtScore(s) { return typeof s === "number" ? s.toFixed(2) : String(s ?? ""); }

  function openHit(x, i) {
    state.activeCite = null;
    markActiveChips();
    const n = (state.record.retrieval || []).length;
    renderDocCitation({ doc_id: x.doc_id, page: x.page, title: x.title }, false,
      { start: x.start, end: x.end, quote: x.text, label: "Search result " + (i + 1) + " of " + n });
  }

  // ---------- review ----------
  function renderReview(r) {
    const form = $("review-form");
    form.hidden = false;
    const done = $("review-done");
    const fields = $("review-fields");
    const rule = $("review-rule");
    const approve = $("btn-approve");
    const reject = $("btn-reject");

    if (r.review) {
      const ok = r.review.decision === "approve";
      add(clear(done),
        h("span", { class: "pill " + (ok ? "pill-approved" : "pill-rejected") }, ok ? "Approved and sent" : "Rejected, not sent"),
        h("span", null, " by ", h("strong", null, r.review.reviewer || "unknown"), ", " + fmtTime(r.review.at)),
        r.review.note ? h("p", tx(r.review.note, { class: "review-note" }), "Note: " + r.review.note) : null);
      done.hidden = false;
      fields.hidden = true;
      approve.disabled = true; reject.disabled = true; $("review-form").classList.add("is-decided");
      rule.textContent = ok ? "The kept sentences were sent to the resident. A decision cannot be changed." : "Nothing was sent. The note goes back to Dalil for similar questions. A decision cannot be changed.";
      return;
    }
    done.hidden = true; $("review-form").classList.remove("is-decided");
    fields.hidden = false;
    const canApprove = r.status === "awaiting_review" && !!r.check;
    const canReject = (r.status === "awaiting_review" || r.status === "held") && !!r.check;
    approve.disabled = !canApprove || state.busy;
    reject.disabled = !canReject || state.busy;
    if (!r.check && firstOf(r.audit || [], "failed")) rule.textContent = "Dalil stopped before it finished, so there is nothing to decide.";
    else if (!r.check) rule.textContent = "You can decide once Dalil has finished checking the draft.";
    else if (r.status === "held") rule.textContent = "A held answer cannot be approved. Reject it with a note.";
    else rule.textContent = "Approving sends only the sentences under Answer to send. Cut sentences are never sent. A rejection note goes back to Dalil.";
  }

  async function decide(decision) {
    const r = state.record;
    if (!r || state.busy) return;
    const nameEl = $("reviewer");
    const reviewer = nameEl.value.trim();
    if (!reviewer) {
      nameEl.setAttribute("aria-invalid", "true");
      nameEl.focus();
      notify("Enter your name before you approve or reject.");
      return;
    }
    nameEl.removeAttribute("aria-invalid");
    try { localStorage.setItem(REVIEWER_KEY, reviewer); } catch (e) { /* storage blocked */ }
    if (decision === "approve" && r.status !== "awaiting_review") { notify("Only an answer awaiting review can be approved."); return; }
    state.busy = true;
    renderReview(r);
    try {
      const updated = await api.review(r.id, { decision, reviewer, note: $("note").value.trim() });
      notify(null);
      $("note").value = "";
      state.record = updated && updated.id ? updated : await api.get(r.id);
      state.busy = false;
      renderRecord();
      await loadQueue();
      $("review-done").setAttribute("tabindex", "-1");
      $("review-done").focus();
      // The review memory writes its own audit step a moment after the decision.
      setTimeout(async () => {
        if (state.selectedId !== r.id) return;
        try { state.record = await api.get(r.id); if (state.selectedId === r.id) renderRecord(); } catch (e) { /* keep what is shown */ }
      }, 800);
    } catch (e) {
      state.busy = false;
      renderReview(state.record);
      notify("The decision was not saved: " + e.message + ".");
    }
  }

  // ---------- ask ----------
  // The answer takes seconds. The question is in the store from its first step,
  // so watch the queue for it and show its steps as the audit trail records them.
  async function watchForNew(asking) {
    while (state.asking === asking && !asking.found) {
      await new Promise((r) => setTimeout(r, LIVE_POLL_MS));
      if (state.asking !== asking) return;
      try {
        const list = await api.list();
        const fresh = list.find((q) => !asking.known.has(q.id));
        if (fresh && !asking.found && state.asking === asking) {
          asking.found = fresh.id;
          state.list = list;
          renderQueue();
          select(fresh.id, false);
        }
      } catch (e) { /* the POST reports any real failure */ }
    }
  }

  async function ask(ev) {
    ev.preventDefault();
    const input = $("ask-input");
    const question = input.value.trim();
    if (!question) { input.focus(); return; }
    const btn = $("ask-submit");
    btn.disabled = true;
    btn.textContent = "Asking...";
    const asking = { known: new Set(state.list.map((q) => q.id)), found: null };
    state.asking = asking;
    // Until the question's first step is recorded there is nothing real to show.
    stopPoll();
    state.selectedId = null;
    state.record = null;
    renderQueue();
    $("answer-body").hidden = true;
    $("review-form").hidden = true;
    clear($("answer-empty")).append(
      h("h2", { id: "answer-heading" }, "Sending the question to Dalil"),
      h("p", null, "Its steps appear here as soon as the screener has read it."));
    $("answer-empty").hidden = false;
    watchForNew(asking);
    const switches = state.kbSwitches;
    try {
      const res = await api.ask(question);
      if (state.kbSwitches !== switches) { await loadQueue(); return; }  // switched while it was answered
      state.asking = null;
      input.value = "";
      notify(null);
      await loadQueue();
      await select(res.id, true);
    } catch (e) {
      state.asking = null;
      if (!state.selectedId) {
        clear($("answer-empty")).append(
          h("h2", { id: "answer-heading" }, "The question was not answered"),
          h("p", null, "Check the message above, then ask again or pick a question from the review queue."));
      }
      notify("Dalil could not take the question: " + e.message + ".");
    } finally {
      state.asking = null;
      btn.disabled = false;
      btn.textContent = "Ask Dalil";
    }
  }

  // ---------- tabs ----------
  function setTab(which) {
    const tabs = [["tab-answer", "view-answer"], ["tab-audit", "view-audit"]];
    for (const [t, v] of tabs) {
      const on = t === which;
      $(t).setAttribute("aria-selected", on ? "true" : "false");
      $(t).tabIndex = on ? 0 : -1;
      $(v).hidden = !on;
    }
  }

  function initTabs() {
    $("tab-answer").addEventListener("click", () => setTab("tab-answer"));
    $("tab-audit").addEventListener("click", () => setTab("tab-audit"));
    document.querySelector(".tabs").addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft" && e.key !== "Home" && e.key !== "End") return;
      e.preventDefault();
      const next = $("tab-answer").getAttribute("aria-selected") === "true" ? "tab-audit" : "tab-answer";
      const target = e.key === "Home" ? "tab-answer" : e.key === "End" ? "tab-audit" : next;
      setTab(target);
      $(target).focus();
    });
  }

  // ---------- boot ----------
  async function boot() {
    try { $("reviewer").value = localStorage.getItem(REVIEWER_KEY) || ""; } catch (e) { /* storage blocked */ }
    $("reviewer").addEventListener("change", (e) => {
      const v = e.target.value.trim();
      try { if (v) localStorage.setItem(REVIEWER_KEY, v); } catch (err) { /* storage blocked */ }
    });
    $("ask-form").addEventListener("submit", ask);
    $("btn-approve").addEventListener("click", () => decide("approve"));
    $("btn-reject").addEventListener("click", () => decide("reject"));
    $("review-form").addEventListener("submit", (e) => e.preventDefault());
    initTabs();
    if (MOCK) document.title = "Dalil reviewer (sample data)";
    $("kb-select").addEventListener("change", switchKnowledgeBase);
    loadKnowledgeBases();
    loadHealth();
    await loadQueue();
    const fromHash = decodeURIComponent(location.hash.slice(1));
    const first = state.list.find((q) => q.id === fromHash) ||
                  state.list.find((q) => q.status === "awaiting_review") || state.list[0];
    if (first) select(first.id, false);
    // Keep the queue fresh when other reviewers or new questions change it.
    setInterval(() => { if (!document.hidden && !state.asking) loadQueue(); }, 10000);
  }

  boot();
})();
