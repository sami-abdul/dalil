// Dalil's bridge to H-MEM, Meterless's memory engine with provenance and a
// trust ledger. It keeps two kinds of memory about reviewed answers:
//
//   approved   an answer a person approved, with the passages it cited.
//              Recorded as a memory with "helpful" feedback.
//   rejected   a reviewer's correction: the question, their note and the
//              passages the rejected answer cited. Before drafting, Dalil
//              recalls rejections for similar questions and keeps those
//              passages out, so the same mistake is not offered twice.
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

function str(v: unknown, max = 4000): string {
  return typeof v === "string" ? v.slice(0, max) : "";
}

function strList(v: unknown, max = 50): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string").slice(0, max) : [];
}

function view(m: MemoryRecord): Json {
  return {
    id: m.id,
    kind: m.tags.includes("rejected") ? "rejected" : m.tags.includes("approved") ? "approved" : "other",
    content: m.content,
    confidence: m.confidence,
    source: m.source,
    passage_ids: m.provenance.refs ?? [],
    tags: m.tags,
    timestamp: m.timestamp,
  };
}

function remember(kind: "approved" | "rejected", body: Json): Json {
  const qid = str(body.question_id, 64);
  const question = str(body.question);
  const reviewer = str(body.reviewer, 200) || "reviewer";
  const passageIds = strList(body.passage_ids);
  if (!qid || !question) throw new Error("question_id and question are required");
  const content =
    kind === "approved"
      ? `Approved answer to: ${question}\n${str(body.answer)}`
      : `Rejected answer to: ${question}\nReviewer note: ${str(body.note) || "(no note)"}`;
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
  const result = hmem.query(q, { topN, threshold: 0.05 });
  const memories = result.memories.map((r) => ({ ...view(r.memory), relevance: r.relevance }));
  const rejected = memories.filter((m) => m.kind === "rejected");
  return {
    memories,
    avoid_passage_ids: [...new Set(rejected.flatMap((m) => m.passage_ids as string[]))],
    reviewer_notes: rejected.map((m) => String(m.content).split("\nReviewer note: ")[1] ?? ""),
    trace: result.trace,
  };
}

function send(res: http.ServerResponse, status: number, body: unknown): void {
  const data = JSON.stringify(body);
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(data) });
  res.end(data);
}

async function readJson(req: http.IncomingMessage): Promise<Json> {
  let size = 0;
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY) throw new Error("body too large");
    chunks.push(chunk as Buffer);
  }
  const parsed = JSON.parse(Buffer.concat(chunks).toString("utf-8") || "{}");
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) throw new Error("expected a JSON object");
  return parsed as Json;
}

const server = http.createServer(async (req, res) => {
  // Same reasoning as Dalil's API: answer only requests addressed to this machine.
  const host = (req.headers.host ?? "").split(":")[0];
  if (host !== HOST && host !== "localhost") return send(res, 400, { error: "invalid host" });
  const url = new URL(req.url ?? "/", `http://${HOST}`);
  try {
    if (req.method === "GET" && url.pathname === "/health") {
      return send(res, 200, { ok: true, memories: hmem.store.all().length, ledger: hmem.ledger.all().length });
    }
    if (req.method === "POST" && (url.pathname === "/approved" || url.pathname === "/rejected")) {
      return send(res, 200, remember(url.pathname.slice(1) as "approved" | "rejected", await readJson(req)));
    }
    if (req.method === "GET" && url.pathname === "/recall") {
      const topN = Math.min(Math.max(Number(url.searchParams.get("top") ?? 5) || 5, 1), 20);
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
    return send(res, 400, { error: e instanceof Error ? e.message : "bad request" });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`dalil h-mem bridge on http://${HOST}:${PORT}, state in ${DIR}`);
});
