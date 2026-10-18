// The page's own upload checks and the Space reply validation, without a browser or the Space.
import assert from "node:assert/strict";
import { test } from "node:test";
import { quotaLeft, refusal, validChunks } from "../src/documents/uploads.ts";

const limits = { per_day: 3, max_mb: 10, max_pages: 5, max_chunks: 2000, types: [".pdf", ".md", ".txt"] };
const fresh = { key: "quota" as const, window_started_at: null, used: 0 };
const chunk = { id: "a_0", doc_hash: "a", chunk_index: 0, page: 1, kind: "text", content: "Page text." };

test("a file of the wrong type, too large, empty or already present is refused before it is sent", () => {
  assert.match(refusal({ name: "x.docx", size: 10 }, [], fresh, limits)!, /not a PDF/);
  assert.match(refusal({ name: "x.pdf", size: 11 * 1024 * 1024 }, [], fresh, limits)!, /larger than 10 MB/);
  assert.match(refusal({ name: "x.pdf", size: 0 }, [], fresh, limits)!, /empty/);
  assert.match(refusal({ name: "Report.PDF", size: 10 }, ["report.pdf"], fresh, limits)!, /already here/);
  assert.equal(refusal({ name: "x.pdf", size: 10 }, [], fresh, limits), null);
});

test("the daily quota counts uploads in a 24 hour window and resets after it", () => {
  const now = Date.parse("2026-10-18T12:00:00Z");
  const used = { key: "quota" as const, window_started_at: "2026-10-18T09:00:00Z", used: 3 };
  assert.equal(quotaLeft(used, limits, now).left, 0);
  assert.match(refusal({ name: "x.pdf", size: 10 }, [], used, limits, now)!, /used today's 3 uploads/);
  assert.equal(quotaLeft(used, limits, now + 24 * 3600 * 1000).left, 3);
});

test("only chunks with the shape the index expects are kept from a Space reply", () => {
  const malformed = [
    null, "text", { ...chunk, id: 7 }, { ...chunk, content: "" }, { ...chunk, content: "x".repeat(20_001) },
    { ...chunk, page: "1" }, { ...chunk, page: -1 }, { ...chunk, chunk_index: 1.5 }, { ...chunk, kind: null },
    { ...chunk, doc_hash: 3 },
  ];
  const textFile = { ...chunk, id: "b_0", page: null, doc_hash: null };
  assert.deepEqual(validChunks([chunk, textFile, ...malformed]), [chunk, textFile]);
});
