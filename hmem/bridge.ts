// Dalil's bridge to H-MEM, Meterless's memory engine with provenance and a
// trust ledger. It keeps two kinds of memory about reviewed answers:
//
//   approved   an answer a person approved, with the passages it cited.
//              Recorded as a memory with "helpful" feedback.
//   rejected   a reviewer's correction: the question, their note and the
//              passages the rejected answer cited.
//
// /recall returns the reviewed memories H-MEM holds, each with its question,
// note and cited passages as separate fields. Dalil decides in Python which of
// them are about a similar question (dalil/hmem.py), because the reference
// engine's hash embedder cannot tell a rewording from an unrelated question.
//
// Every write lands in H-MEM's append-only trust ledger (ledger.jsonl).
// Listens on 127.0.0.1 only. Run with:
//
//   node --experimental-transform-types --no-warnings hmem/bridge.ts
//
// Env: DALIL_HMEM_PORT (default 8788), DALIL_HMEM_DIR (default build/hmem).

import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { HMEM } from "./reference/src/index.ts";
import type { MemoryRecord } from "./reference/src/index.ts";

const HOST = "127.0.0.1";
const PORT = Number(process.env.DALIL_HMEM_PORT ?? 8788);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DIR = process.env.DALIL_HMEM_DIR ?? path.join(ROOT, "build", "hmem");
const MAX_BODY = 64 * 1024;

const hmem = new HMEM({ persistDir: DIR, actor: "dalil" });

type Json = Record<string, unknown>;

class BadRequest extends Error {}

const QUESTION_PREFIX = "Reviewed question: ";
const NOTE_PREFIX = "Reviewer note: ";

function str(v: unknown, max = 4000): string {
  return typeof v === "string" ? v.slice(0, max) : "";
}

// One line, so the stored content splits back into its fields unambiguously.
function line(v: unknown, max = 4000): string {
  return str(v, max).replace(/\s+/g, " ").trim();
}

function strList(v: unknown, max = 50): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string").slice(0, max) : [];
}

function field(content: string, prefix: string): string {
  const found = content.split("\n").find((l) => l.startsWith(prefix));
  return found ? found.slice(prefix.length) : "";
}

function view(m: MemoryRecord): Json {
  return {
    id: m.id,
    kind: m.tags.includes("rejected") ? "rejected" : m.tags.includes("approved") ? "approved" : "other",
    question: field(m.content, QUESTION_PREFIX),
    note: field(m.content, NOTE_PREFIX),
    content: m.content,
    confidence: m.confidence,
    source: m.source,
    passage_ids: m.provenance.refs ?? [],
    tags: m.tags,
    timestamp: m.timestamp,
  };
}

function remember(kind: "approved" | "rejected", body: Json): Json {
  const qid = line(body.question_id, 64);
  const question = line(body.question);
  const reviewer = line(body.reviewer, 200) || "reviewer";
  const passageIds = strList(body.passage_ids);
  if (!qid || !question) throw new BadRequest("question_id and question are required");
  const content = [
    `${kind === "approved" ? "Approved" : "Rejected"} answer`,
    `${QUESTION_PREFIX}${question}`,
    kind === "approved" ? `Answer: ${line(body.answer)}` : `${NOTE_PREFIX}${line(body.note) || "(no note)"}`,
  ].join("\n");
  const record = hmem.add({
    content,
    type: "factual",
    tags: [kind, `question:${qid}`],
    confidence: kind === "approved" ? 0.7 : 0.9,
    source: `dalil:question:${qid}`,
    provenance: {
      origin: kind === "approved" ? "manual" : "user_correction",
      learnedAt: Date.now(),
      label: `${kind} by ${reviewer}`,
      refs: passageIds,
    },
  });
  if (kind === "approved") hmem.feedback(record.id, "helpful");
  return { memory: view(hmem.store.get(record.id) ?? record) };
}

function recall(q: string, topN: number): Json {
  // H-MEM's query is the read path, so each recall is in the trust ledger.
  const result = hmem.query(q, { topN, threshold: 0 });
  return {
    memories: result.memories.map((r) => ({ ...view(r.memory), relevance: r.relevance })),
    trace: result.trace,
  };
}

function send(res: http.ServerResponse, status: number, body: unknown): void {
  const data = JSON.stringify(body);
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(data) });
  res.end(data);
}

async function readJson(req: http.IncomingMessage): Promise<Json> {
  // A JSON content type forces a CORS preflight that this server never
  // answers, so a web page cannot post here with a "simple" request.
  if (!(req.headers["content-type"] ?? "").startsWith("application/json")) {
    throw new BadRequest("content-type must be application/json");
  }
  let size = 0;
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY) throw new BadRequest("body too large");
    chunks.push(chunk as Buffer);
  }
  const parsed = JSON.parse(Buffer.concat(chunks).toString("utf-8") || "{}");
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) throw new BadRequest("expected a JSON object");
  return parsed as Json;
}

const server = http.createServer(async (req, res) => {
  // Same reasoning as Dalil's API: answer only requests addressed to this machine.
  const host = (req.headers.host ?? "").split(":")[0];
  if (host !== HOST && host !== "localhost") return send(res, 400, { error: "invalid host" });
  // Dalil calls this from Python, which sends no Origin. A browser always does.
  if (req.headers.origin) return send(res, 403, { error: "browser requests are not accepted" });
  const url = new URL(req.url ?? "/", `http://${HOST}`);
  try {
    if (req.method === "GET" && url.pathname === "/health") {
      return send(res, 200, { ok: true, memories: hmem.store.all().length, ledger: hmem.ledger.all().length });
    }
    if (req.method === "POST" && (url.pathname === "/approved" || url.pathname === "/rejected")) {
      return send(res, 200, remember(url.pathname.slice(1) as "approved" | "rejected", await readJson(req)));
    }
    if (req.method === "GET" && url.pathname === "/recall") {
      const topN = Math.min(Math.max(Number(url.searchParams.get("top") ?? 50) || 50, 1), 200);
      return send(res, 200, recall(str(url.searchParams.get("q")), topN));
    }
    if (req.method === "GET" && url.pathname === "/ledger") {
      const limit = Math.min(Math.max(Number(url.searchParams.get("limit") ?? 50) || 50, 1), 500);
      const memoryId = url.searchParams.get("memory_id");
      const entries = memoryId ? hmem.ledger.history(memoryId) : hmem.ledger.all().slice(-limit);
      return send(res, 200, { entries, stats: hmem.ledger.stats() });
    }
    return send(res, 404, { error: "not found" });
  } catch (e) {
    if (e instanceof BadRequest || e instanceof SyntaxError) return send(res, 400, { error: e.message });
    // Anything else, such as a failed write to state.json, is the bridge's fault.
    console.error(e);
    return send(res, 500, { error: "memory write or read failed" });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`dalil h-mem bridge on http://${HOST}:${PORT}, state in ${DIR}`);
});
