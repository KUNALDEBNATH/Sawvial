// Runs the REAL frontend scripts (script.js, i18n.js, sawvial-api.js, chat.js) inside jsdom with a
// mocked fetch and a mocked SpeechRecognition. Requires:  npm install jsdom  (dev-only, optional)
//   node tests/frontend/dom_test.js
const fs = require("fs"), path = require("path"), assert = require("assert");
const { JSDOM } = require(process.env.JSDOM_PATH || "jsdom");
const WEB = path.join(__dirname, "..", "..", "sawvial");

function makeWindow({ fetchImpl, recognition }) {
  const html = fs.readFileSync(path.join(WEB, "ask.html"), "utf8").replace(/<script[^>]*src=[^>]*><\/script>/g, "");
  const dom = new JSDOM(html, { url: "http://localhost:8000/ask.html", runScripts: "dangerously", pretendToBeVisual: true });
  const w = dom.window;
  w.localStorage.setItem("sawvialLanguage", "en");
  w.fetch = fetchImpl;
  w.scrollTo = () => {}; w.HTMLElement.prototype.scrollIntoView = () => {};
  w.matchMedia = w.matchMedia || (() => ({ matches: false, addEventListener() {}, addListener() {} }));
  if (recognition) w.webkitSpeechRecognition = recognition;
  // real <script> elements share one global lexical scope, exactly like the browser
  for (const f of ["script.js", "i18n.js", "sawvial-api.js", "chat.js"]) {
    const el = w.document.createElement("script");
    el.textContent = fs.readFileSync(path.join(WEB, "js", f), "utf8");
    w.document.body.appendChild(el);
  }
  return w;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const okJson = (obj) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(obj) });
const questionsSent = (calls) => calls.filter((c) => /\/api\/ask/.test(c.url)).map((c) => JSON.parse(c.opts.body).question);

(async () => {
  let passed = 0;
  const test = async (name, fn) => { await fn(); passed++; console.log("PASS", name); };

  // 1. speech recognition error -> NO question is submitted, text box stays usable
  await test("mic error submits nothing and creates no user message", async () => {
    const calls = [];
    class SR { start() { setTimeout(() => { this.onerror && this.onerror({ error: "no-speech" }); this.onend && this.onend(); }, 5); } stop() {} }
    const w = makeWindow({ fetchImpl: (url, opts) => { calls.push({ url, opts }); return okJson({ status: "ok", speech_local: [] }); }, recognition: SR });
    await sleep(30);
    w.document.getElementById("mic-btn").click();
    await sleep(80);
    assert.deepStrictEqual(questionsSent(calls), [], "no question may be sent after a recognition failure");
    assert.ok(!w.document.querySelector(".msg.user"), "no user message may be created");
  });

  // 2. empty transcript (onend with nothing heard) -> nothing submitted
  await test("empty transcript submits nothing", async () => {
    const calls = [];
    class SR { start() { setTimeout(() => this.onend && this.onend(), 5); } stop() {} }
    const w = makeWindow({ fetchImpl: (u, o) => { calls.push({ url: u, opts: o }); return okJson({}); }, recognition: SR });
    await sleep(30); w.document.getElementById("mic-btn").click(); await sleep(60);
    assert.deepStrictEqual(questionsSent(calls), []);
  });

  // 3. recognised speech reaches the same /api/ask pipeline with the exact transcript
  await test("recognised speech is sent verbatim to /api/ask", async () => {
    const calls = [];
    class SR { start() { setTimeout(() => { this.onresult({ resultIndex: 0, results: [Object.assign([{ transcript: "how to irrigate groundnut" }], { isFinal: true })] }); this.onend(); }, 5); } stop() {} }
    const w = makeWindow({ fetchImpl: (u, o) => { calls.push({ url: u, opts: o }); return okJson({ answer: "x", grounded: false, sources: [] }); }, recognition: SR });
    await sleep(30); w.document.getElementById("mic-btn").click(); await sleep(150);
    assert.deepStrictEqual(questionsSent(calls), ["how to irrigate groundnut"]);
  });

  // 4. API failure -> error text + Retry button; Retry re-sends the SAME question once more
  await test("failed request offers Retry and re-sends the same question", async () => {
    const calls = []; let n = 0;
    const w = makeWindow({ fetchImpl: (u, o) => {
      calls.push({ url: u, opts: o });
      if (!/\/api\/ask/.test(u)) return okJson({ speech_local: [] });
      return ++n === 1 ? Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({ detail: "boom" }) })
                       : okJson({ answer: "Recovered answer.", grounded: false, sources: [] });
    } });
    await sleep(30);
    const input = w.document.getElementById("input");
    input.value = "soil moisture question"; input.dispatchEvent(new w.Event("input", { bubbles: true }));
    w.document.getElementById("send-btn").click();
    await sleep(150);
    const retry = w.document.querySelector("button.retry");
    assert.ok(retry, "Retry button must appear after a failed request");
    retry.click(); await sleep(150);
    assert.deepStrictEqual(questionsSent(calls), ["soil moisture question", "soil moisture question"]);
    assert.strictEqual(w.document.querySelectorAll(".msg.user").length, 1, "retry must not duplicate the user bubble");
  });

  // 5. sources: DOI/http links are clickable, javascript: URLs never become links, missing metadata not invented
  await test("source links are safe and metadata is not invented", async () => {
    const resp = { answer: "A grounded answer sentence about drip irrigation.", grounded: true, answer_language: "en",
      grounding: { notes: ["single_source"] },
      sources: [
        { id: "SWM_000001", cited: true, title: "Drip study", authors: ["A. Rao", "B. Das"], year: 2022, journal: "Irrig. Sci.", doi: "10.1000/abc.123", url: "https://example.org/p.pdf", page: 4 },
        { id: "SWM_000002", cited: true, title: "Evil", url: "javascript:alert(1)", doi: "not-a-doi" },
        { id: "SWM_000003", cited: true, title: "No metadata" } ] };
    const w2 = makeWindow({ fetchImpl: (u, o) => /\/api\/ask/.test(u) ? okJson(resp) : okJson({ speech_local: [] }) });
    await sleep(30);
    const input = w2.document.getElementById("input");
    input.value = "drip irrigation?"; input.dispatchEvent(new w2.Event("input", { bubbles: true }));
    w2.document.getElementById("send-btn").click();
    await sleep(2500);
    const links = [...w2.document.querySelectorAll(".sources a")].map((a) => a.href);
    assert.ok(links.includes("https://doi.org/10.1000/abc.123"), "DOI link missing: " + links);
    assert.ok(links.includes("https://example.org/p.pdf"), "URL link missing");
    assert.ok(!links.some((h) => h.startsWith("javascript:")), "unsafe URL became a link");
    assert.ok([...w2.document.querySelectorAll(".sources a")].every((a) => a.rel.includes("noopener")));
    const txt = w2.document.querySelector(".sources").textContent;
    assert.ok(/A\. Rao, B\. Das \(2022\)/.test(txt) && /p\.4/.test(txt));
    assert.ok(/no online link available/.test(txt), "missing links must be stated, not invented");
    assert.ok(/single source/i.test(w2.document.querySelector(".notes").textContent));
  });

  console.log(`\n${passed} frontend DOM tests passed`);
})().catch((e) => { console.error("FAIL:", e.message); process.exit(1); });
