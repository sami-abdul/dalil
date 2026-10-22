/* Dalil reviewer screen. Vanilla JS, no build step.
 * Add ?mock=1 to the URL to run against web/fixtures/mock.json instead of the API.
 */
(function () {
  "use strict";

  const MOCK = new URLSearchParams(location.search).get("mock") === "1";
  const REVIEWER_KEY = "dalil.reviewer";
  const POLL_MS = 1500;

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
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
  // Like el.append, but skips null/false so conditional children never render as "null".
  function add(el, ...kids) {
    for (const c of kids.flat()) {
      if (c === null || c === undefined || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  const timeFmt = new Intl.DateTimeFormat("en-GB", {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "Asia/Dubai",
  });
  const clockFmt = new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "Asia/Dubai",
  });
  function fmtTime(iso) { const d = new Date(iso); return isNaN(d) ? (iso || "") : timeFmt.format(d); }
  function fmtClock(iso) { const d = new Date(iso); return isNaN(d) ? (iso || "") : clockFmt.format(d); }

  // ---------- labels ----------
  const STATUS = {
    awaiting_review: { label: "Awaiting review", cls: "pill-awaiting" },
    held: { label: "Held", cls: "pill-held" },
    approved: { label: "Approved", cls: "pill-approved" },
    rejected: { label: "Rejected", cls: "pill-rejected" },
  };
  function statusPill(status) {
    const s = STATUS[status] || { label: "Working", cls: "pill-working" };
    return h("span", { class: "pill " + s.cls }, s.label);
  }

  const CHECK = {
    pass: "Every quote matched its source",
    pass_with_cuts: "Checked, some sentences cut",
    held: "Held for a person: not safe to send as drafted",
  };

  const AUDIT = {
    screened: "Question screened",
    retrieved: "Sources searched",
    drafted: "Answer drafted",
    verified: "Quotes checked",
    queued: "Sent for review",
    approved: "Approved",
    rejected: "Rejected",
    delivered: "Delivered to the resident",
  };

  function flagInfo(flag) {
    const code = typeof flag === "string" ? flag : (flag && (flag.kind || flag.code || flag.type)) || "";
    const detail = typeof flag === "object" && flag ? (flag.detail || flag.message || "") : "";
    const c = String(code).toLowerCase();
    if (/pii|personal|redact/.test(c)) {
      return { cls: "flag-privacy", title: "Personal data removed",
        body: detail || "Dalil removed personal details from this question before searching. They are not stored with the answer." };
    }
    if (/evasion|evade|circumvent|loophole/.test(c)) {
      return { cls: "flag-evasion", title: "Asks how to get around a rule",
        body: detail || "Dalil will not help anyone avoid a law or a government process. This answer is held and can only be rejected." };
    }
    return { cls: "flag-other", title: "Flagged at screening", body: detail || String(code).replace(/_/g, " ") };
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
  };

  function mockApi() {
    let fx = null;
    const ready = fetch("fixtures/mock.json").then((r) => r.json()).then((j) => { fx = j; });
    const copy = (o) => JSON.parse(JSON.stringify(o));
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
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
        r.check = null; r.status = "working";
        r.audit = r.audit.slice(0, 1);
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
        const iso = now.toISOString();
        const t = copy(fx.new_question_template);
        const q = Object.assign(t, { id: "q-0" + (++seq), question, created_at: iso });
        q.audit = q.audit.map((a, i) => Object.assign({ at: new Date(now.getTime() + i * 700).toISOString() }, a));
        fx.questions.unshift(q);
        pending.set(q.id, Date.now() + 2500);
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
        q.audit.push({ at, kind: q.status, detail: body.reviewer + (body.note ? ": " + body.note : "") });
        if (body.decision === "approve") q.audit.push({ at, kind: "delivered", detail: "Answer sent to the resident" });
        return view(q);
      },
      page: async (doc, page) => {
        await ready; await wait(60);
        const p = fx.pages[doc + "/" + page];
        if (!p) throw new Error("404 Page not found");
        return copy(p);
      },
      fileUrl: () => null,
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
    busy: false,
    sourceToken: 0,
  };

  function notify(msg) {
    const n = $("notice");
    if (!msg) { n.hidden = true; n.textContent = ""; return; }
    n.textContent = msg;
    n.hidden = false;
  }

  // ---------- health ----------
  async function loadHealth() {
    const el = $("health");
    clear(el);
    if (MOCK) el.append(h("span", { class: "badge badge-mock" }, "Sample data (mock mode)"));
    try {
      const hl = await api.health();
      const ok = hl.ok && hl.zetaris;
      el.append(h("span", null, h("span", { class: "dot " + (ok ? "dot-ok" : "dot-bad"), "aria-hidden": "true" }),
        (hl.zetaris ? "Zetaris connected" : "Zetaris not connected") +
          (typeof hl.passages === "number" ? ", " + hl.passages.toLocaleString("en-GB") + " passages" : "")));
    } catch (e) {
      el.append(h("span", null, h("span", { class: "dot dot-bad", "aria-hidden": "true" }), "Server not reachable"));
    }
  }

  // ---------- queue ----------
  async function loadQueue() {
    try {
      state.list = await api.list();
      renderQueue();
    } catch (e) {
      notify("Could not load the review queue. " + e.message);
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
        : h("span", { class: "counts" }, "Checking");
      const btn = h("button", {
        type: "button", class: "queue-item" + (selected ? " is-selected" : ""),
        "aria-current": selected ? "true" : null,
        onclick: () => select(q.id, true),
      },
        h("span", { class: "queue-top" }, statusPill(q.status), h("time", { datetime: q.created_at }, fmtTime(q.created_at))),
        h("span", { class: "queue-q", dir: "auto" }, q.question),
        counts);
      ul.append(h("li", null, btn));
    }
  }

  // ---------- selection ----------
  async function select(id, focusPanel) {
    state.selectedId = id;
    state.activeCite = null;
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
      notify("Could not load question " + id + ". " + e.message);
    }
  }

  function startPoll(id) {
    stopPoll();
    state.pollTimer = setTimeout(async () => {
      if (state.selectedId !== id) return;
      try {
        const rec = await api.get(id);
        if (state.selectedId !== id) return;
        if (rec.check) {
          state.record = rec;
          renderRecord();
          loadQueue();
        } else {
          startPoll(id);
        }
      } catch (e) { startPoll(id); }
    }, POLL_MS);
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

  function chip(n) {
    const c = state.cites[n - 1];
    const isData = !!c.table;
    const where = isData ? "data table " + c.table : (c.title || c.doc_id) + ", page " + c.page;
    const label = "Source " + n + ": " + where + (c.ok === false ? ". This quote did not match." : "");
    return h("button", {
      type: "button",
      class: "chip" + (isData ? " chip-data" : "") + (c.ok === false ? " chip-bad" : "") + (state.activeCite === n ? " is-active" : ""),
      "aria-label": label, title: label, "data-n": n,
      "aria-pressed": state.activeCite === n ? "true" : "false",
      onclick: () => openCitation(n),
    }, String(n));
  }

  function renderRecord() {
    const r = state.record;
    $("answer-empty").hidden = !!r;
    $("answer-body").hidden = !r;
    if (!r) { $("review-form").hidden = true; return; }

    $("q-id").textContent = r.id;
    $("q-time").textContent = fmtTime(r.created_at);
    $("q-time").setAttribute("datetime", r.created_at || "");
    $("q-text").textContent = r.question;

    const meta = clear($("q-meta"));
    meta.append(statusPill(r.status));
    if (r.check) {
      meta.append(h("span", { class: "check-status check-" + r.check.status }, CHECK[r.check.status] || r.check.status));
    }

    const flags = clear($("flags"));
    for (const f of r.flags || []) {
      const fi = flagInfo(f);
      flags.append(h("div", { class: "flag " + fi.cls, role: "note" },
        h("strong", null, fi.title), h("span", null, fi.body)));
    }

    const sentences = (r.check && r.check.sentences) || [];
    state.cites = numberCitations(sentences);
    $("working").hidden = !!r.check;
    renderDraft(sentences);
    renderCuts(sentences);
    renderAudit(r.audit || []);
    renderSearched(r);
    renderReview(r);
    if (state.activeCite && state.cites[state.activeCite - 1]) openCitation(state.activeCite, true);
    else renderSourceEmpty();
  }

  function renderDraft(sentences) {
    const box = clear($("draft"));
    const kept = sentences.filter((s) => s.kept);
    if (!state.record.check) {
      box.append(h("p", { class: "muted" }, "The draft will appear here once every quote has been checked."));
      return;
    }
    if (!kept.length) {
      const ans = state.record.check.answer;
      box.append(h("p", { class: "muted", dir: "auto" }, ans || "No sentence passed the check, so there is nothing to send."));
      return;
    }
    let para = null;
    for (const s of kept) {
      const chips = s._nums.length ? h("span", { class: "chips" }, s._nums.map(chip)) : null;
      if (s.kind === "refusal" || s.kind === "caveat") {
        para = null;
        const isRef = s.kind === "refusal";
        box.append(h("div", { class: "callout " + (isRef ? "callout-refusal" : "callout-caveat") },
          h("span", { class: "callout-label" }, isRef ? "Dalil declines" : "Note for the resident"),
          h("p", { dir: "auto" }, h("span", { class: "sentence" }, s.text), chips ? "\u00a0" : null, chips)));
        continue;
      }
      if (!para) { para = h("p", { class: "answer-para", dir: "auto" }); box.append(para); }
      const bad = (s.citations || []).some((c) => c.ok === false);
      add(para, h("span", { class: "sentence" + (bad ? " sentence-warn" : "") }, s.text), chips ? "\u00a0" : null, chips, " ");
    }
  }

  function renderCuts(sentences) {
    const cut = sentences.filter((s) => !s.kept);
    const sec = $("cut-section");
    sec.hidden = !cut.length;
    const ol = clear($("cut-list"));
    $("cut-heading").textContent = "Cut by the checker (" + cut.length + ")";
    for (const s of cut) {
      const reasons = (s.reasons && s.reasons.length) ? s.reasons : (s.citations || []).filter((c) => c.ok === false).map((c) => c.reason).filter(Boolean);
      ol.append(h("li", { class: "cut-item" },
        h("p", { class: "cut-text", dir: "auto" },
          h("s", null, h("span", { class: "visually-hidden" }, "Cut sentence: "), s.text), " ",
          s._nums.length ? h("span", { class: "chips" }, s._nums.map(chip)) : null),
        h("p", { class: "cut-reason" },
          h("span", { class: "cut-reason-label" }, "Why it was cut: "),
          reasons.length ? reasons.join("; ") : "it did not pass the check")));
    }
  }

  function auditDetail(d) {
    if (d === null || d === undefined || d === "") return null;
    if (typeof d === "string") return h("p", { class: "audit-detail", dir: "auto" }, d);
    if (Array.isArray(d)) return h("p", { class: "audit-detail" }, d.join(", "));
    const dl = h("dl", { class: "audit-kv" });
    for (const [k, v] of Object.entries(d)) {
      dl.append(h("dt", null, k.replace(/_/g, " ")), h("dd", null, typeof v === "object" ? JSON.stringify(v) : String(v)));
    }
    return dl;
  }

  function renderAudit(audit) {
    const ol = clear($("audit-list"));
    if (!audit.length) { ol.append(h("li", { class: "muted" }, "No events yet.")); return; }
    for (const a of audit) {
      ol.append(h("li", { class: "audit-item audit-" + a.kind },
        h("time", { datetime: a.at, class: "audit-time" }, fmtClock(a.at)),
        h("div", { class: "audit-body" },
          h("p", { class: "audit-kind" }, AUDIT[a.kind] || String(a.kind).replace(/_/g, " ")),
          auditDetail(a.detail))));
    }
  }

  // ---------- evidence ----------
  function renderSourceEmpty() {
    const v = clear($("source-view"));
    v.append(h("div", { class: "source-empty" },
      h("h3", { class: "panel-title" }, "Source"),
      h("p", null, "Select a numbered citation in the answer to see the exact words it quotes.")));
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
      ? h("p", { class: "cite-status cite-bad" }, "This quote did not match the source" + (c.reason ? ": " + c.reason : "."))
      : h("p", { class: "cite-status cite-ok" }, c.table ? "The checker reran this lookup and got the same value." : "The checker found these exact words on this page.");
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
      clear(v).append(heading, h("p", { class: "cite-status cite-bad" }, "Could not load page " + c.page + " of " + (c.title || c.doc_id) + ". " + e.message));
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

    const pageBox = h("div", { class: "page-text", dir: "auto", tabindex: "0", "aria-label": "Text of page " + p.page });
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

    add(clear(v),
      heading,
      h("p", { class: "source-title", dir: "auto" }, p.title || c.title || c.doc_id),
      h("p", { class: "source-meta" }, (p.publisher ? p.publisher + ", " : "") + "page " + p.page),
      links,
      sel.cite ? citeStatus(sel.cite) : null,
      sel.quote && !ok ? h("p", { class: "cite-status cite-bad" }, "The quoted words could not be located on this page.") : null,
      pageBox);
    if (mark) {
      requestAnimationFrame(() => {
        // Scroll the page box (not the window) so the highlight sits near the top third.
        const top = mark.offsetTop - pageBox.offsetTop - pageBox.clientHeight / 3;
        pageBox.scrollTop = Math.max(0, top);
        if (!quiet) v.scrollIntoView({ block: "nearest" });
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
      h("p", { class: "source-title" }, c.title || "Open-data table"),
      h("p", { class: "source-meta" }, "Table ", h("code", null, c.table), ", queried through Zetaris"),
      citeStatus(c),
      h("p", { class: "data-value" }, "Cited value: ", h("mark", { class: "quote-mark" }, String(c.value))),
      h("p", { class: "sub-title" }, "Query Dalil ran"),
      h("pre", { class: "sql", tabindex: "0", "aria-label": "SQL query" }, h("code", null, c.sql || (d && d.sql) || "")));
    if (d && d.rows) v.append(h("p", { class: "sub-title" }, "Rows returned (" + d.rows.length + ")"), rowsTable(d.rows, c.value));
    if (d && d.note) v.append(h("p", { class: "muted small" }, d.note));
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
        h("thead", null, h("tr", null, cols.map((k) => h("th", { scope: "col" }, k)))),
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
    if (!r.check && !hits.length) { box.append(h("p", { class: "muted" }, "Searching...")); return; }
    if (!hits.length && !data.length) { box.append(h("p", { class: "muted" }, "Dalil found nothing relevant in the documents or the data tables.")); return; }
    if (hits.length) {
      box.append(h("p", { class: "sub-title" }, "Document passages (" + hits.length + ")"));
      const ol = h("ol", { class: "hits" });
      hits.forEach((x, i) => {
        const cited = state.cites.some((c) => c.passage_id && c.passage_id === x.passage_id);
        ol.append(h("li", null, h("button", {
          type: "button", class: "hit",
          "aria-label": "Passage from " + (x.title || x.doc_id) + ", page " + x.page + ", match score " + fmtScore(x.score) + (cited ? ", cited in the answer" : ""),
          onclick: () => openHit(x, i),
        },
          h("span", { class: "hit-top" },
            h("span", { class: "score", title: "Match score" }, fmtScore(x.score)),
            h("span", { class: "hit-title", dir: "auto" }, (x.title || x.doc_id) + ", p. " + x.page),
            cited ? h("span", { class: "tag-cited" }, "Cited") : null),
          x.heading ? h("span", { class: "hit-heading", dir: "auto" }, x.heading) : null,
          h("span", { class: "hit-text", dir: "auto" }, x.text))));
      });
      box.append(ol);
    }
    if (data.length) {
      box.append(h("p", { class: "sub-title" }, "Data lookups (" + data.length + ")"));
      const ul = h("ul", { class: "lookups" });
      for (const d of data) {
        ul.append(h("li", { class: "lookup" },
          h("p", { class: "lookup-top" }, h("code", null, d.table), " ", h("span", { class: "muted" }, (d.rows ? d.rows.length : 0) + " rows")),
          h("details", null, h("summary", null, "Show query and rows"),
            h("pre", { class: "sql" }, h("code", null, d.sql || "")),
            d.rows && d.rows.length ? rowsTable(d.rows, null) : null,
            d.note ? h("p", { class: "muted small" }, d.note) : null)));
      }
      box.append(ul);
    }
  }
  function fmtScore(s) { return typeof s === "number" ? s.toFixed(2) : String(s ?? ""); }

  function openHit(x, i) {
    state.activeCite = null;
    markActiveChips();
    renderDocCitation({ doc_id: x.doc_id, page: x.page, title: x.title }, false,
      { start: x.start, end: x.end, quote: x.text, label: "Search result " + (i + 1) });
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
        h("span", { class: "pill " + (ok ? "pill-approved" : "pill-rejected") }, ok ? "Approved" : "Rejected"),
        h("span", null, " by ", h("strong", null, r.review.reviewer || "unknown"), ", " + fmtTime(r.review.at)),
        r.review.note ? h("p", { class: "review-note", dir: "auto" }, "Note: " + r.review.note) : null);
      done.hidden = false;
      fields.hidden = true;
      approve.disabled = true; reject.disabled = true;
      rule.textContent = ok ? "This answer has been sent. Decisions cannot be changed." : "This answer was not sent. Decisions cannot be changed.";
      return;
    }
    done.hidden = true;
    fields.hidden = false;
    const canApprove = r.status === "awaiting_review" && !!r.check;
    const canReject = (r.status === "awaiting_review" || r.status === "held") && !!r.check;
    approve.disabled = !canApprove || state.busy;
    reject.disabled = !canReject || state.busy;
    if (!r.check) rule.textContent = "You can decide once Dalil has finished checking the draft.";
    else if (r.status === "held") rule.textContent = "Held answers cannot be approved. Reject it with a note to send it back.";
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
    localStorage.setItem(REVIEWER_KEY, reviewer);
    if (decision === "approve" && r.status !== "awaiting_review") { notify("Only answers awaiting review can be approved."); return; }
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
    } catch (e) {
      state.busy = false;
      renderReview(state.record);
      notify("The decision was not saved. " + e.message);
    }
  }

  // ---------- ask ----------
  async function ask(ev) {
    ev.preventDefault();
    const input = $("ask-input");
    const question = input.value.trim();
    if (!question) { input.focus(); return; }
    const btn = $("ask-submit");
    btn.disabled = true;
    btn.textContent = "Asking...";
    try {
      const res = await api.ask(question);
      input.value = "";
      notify(null);
      await loadQueue();
      await select(res.id, true);
    } catch (e) {
      notify("Dalil could not take the question. " + e.message);
    } finally {
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
    $("reviewer").value = localStorage.getItem(REVIEWER_KEY) || "";
    $("reviewer").addEventListener("change", (e) => {
      const v = e.target.value.trim();
      if (v) localStorage.setItem(REVIEWER_KEY, v);
    });
    $("ask-form").addEventListener("submit", ask);
    $("btn-approve").addEventListener("click", () => decide("approve"));
    $("btn-reject").addEventListener("click", () => decide("reject"));
    $("review-form").addEventListener("submit", (e) => e.preventDefault());
    initTabs();
    if (MOCK) document.title = "Dalil reviewer (sample data)";
    loadHealth();
    await loadQueue();
    const fromHash = decodeURIComponent(location.hash.slice(1));
    const first = state.list.find((q) => q.id === fromHash) ||
                  state.list.find((q) => q.status === "awaiting_review") || state.list[0];
    if (first) select(first.id, false);
    // Keep the queue fresh when other reviewers or new questions change it.
    setInterval(() => { if (!document.hidden) loadQueue(); }, 10000);
  }

  boot();
})();
