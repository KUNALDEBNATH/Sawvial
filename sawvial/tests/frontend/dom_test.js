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
const okStream = (finalObj, stages = []) => Promise.resolve({ ok: true, status: 200, body: null, headers: { get: () => "" },
  text: () => Promise.resolve([...stages.map((st) => JSON.stringify({ stage: st })), JSON.stringify({ final: finalObj })].join("\n") + "\n") });
const isAsk = (u) => /\/api\/ask/.test(u);
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
    const w = makeWindow({ fetchImpl: (u, o) => { calls.push({ url: u, opts: o }); return isAsk(u) ? okStream({ answer: "x", grounded: false, sources: [], text_language: "en" }) : okJson({ speech_local: [] }); }, recognition: SR });
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
                       : okStream({ answer: "Recovered answer.", grounded: false, sources: [], text_language: "en" });
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
    const resp = { answer: "A grounded answer sentence about drip irrigation.", grounded: true, text_language: "en",
      grounding: { notes: ["single_source"] },
      sources: [
        { id: "SWM_000001", cited: true, title: "Drip study", authors: ["A. Rao", "B. Das"], year: 2022, journal: "Irrig. Sci.", doi: "10.1000/abc.123", url: "https://example.org/p.pdf", page: 4 },
        { id: "SWM_000002", cited: true, title: "Evil", url: "javascript:alert(1)", doi: "not-a-doi" },
        { id: "SWM_000003", cited: true, title: "No metadata" } ] };
    const w2 = makeWindow({ fetchImpl: (u, o) => isAsk(u) ? okStream(resp) : okJson({ speech_local: [] }) });
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

  // ------------------------------------------------------------------ helpers for the speech / language tests
  const TA = "தக்காளிக்கு வடிகால் வசதியுள்ள மணல் கலந்த களிமண் நல்லது.";
  function speechEnv(w, { voices = [], local = [] } = {}) {
    const spoken = [], fetched = [];
    w.speechSynthesis = { getVoices: () => voices, speak: (u) => spoken.push(u), cancel() {}, addEventListener() {}, removeEventListener() {} };
    w.SpeechSynthesisUtterance = function (text) { this.text = text; };
    w.AudioContext = class { constructor() { this.currentTime = 0; this.destination = {}; } resume() { return Promise.resolve(); } close() { return Promise.resolve(); }
      createBuffer() { return { copyToChannel() {}, duration: 0.1 }; } createBufferSource() { return { connect() {}, start() {}, stop() {}, addEventListener() {} }; } };
    w.SAWVIAL_CAPS = { speech_local: local };
    return { spoken, fetched };
  }
  async function askOnce(w, q) {
    const input = w.document.getElementById("input");
    input.value = q; input.dispatchEvent(new w.Event("input", { bubbles: true }));
    w.document.getElementById("send-btn").click();
    await sleep(2500);
  }
  const tamilAnswer = { answer: TA, grounded: true, text_language: "ta", answer_language: "ta", input_language: "en", warnings: [], sources: [] };

  // 6. progress stages are shown while the server works, and the final answer replaces them
  await test("UI shows the real stages (local -> online -> reading -> generating) while waiting", async () => {
    let release; const gate = new Promise((r) => (release = r));
    const enc = new TextEncoder();
    const chunks = [{ stage: "local_search" }, { stage: "online_search" }, { stage: "reading", n: 2 }, { stage: "generating" }];
    const w = makeWindow({ fetchImpl: (u) => {
      if (!isAsk(u)) return okJson({ speech_local: [] });
      let i = 0;
      return Promise.resolve({ ok: true, status: 200, headers: { get: () => "" }, body: { getReader: () => ({ read: async () => {
        if (i < chunks.length) { await sleep(60); return { done: false, value: enc.encode(JSON.stringify(chunks[i++]) + "\n") }; }
        if (i === chunks.length) { i++; await gate; return { done: false, value: enc.encode(JSON.stringify({ final: { answer: "Done.", grounded: false, text_language: "en", sources: [] } }) + "\n") }; }
        return { done: true };
      } }) } });
    } });
    await sleep(30);
    const input = w.document.getElementById("input"); input.value = "soil for tomato"; input.dispatchEvent(new w.Event("input", { bubbles: true }));
    w.document.getElementById("send-btn").click();
    await sleep(500);
    const label = w.document.querySelector(".stage").textContent;
    assert.ok(/Writing the answer/.test(label), "last streamed stage should be visible, got: " + label);
    release(); await sleep(400);
    assert.ok(!w.document.querySelector(".stage"), "stage label must go away when the answer arrives");
  });

  // 7. Tamil answer + local Tamil voice available -> the ACTUAL Tamil text goes to the speech API with code "ta"
  await test("Tamil answer is sent to local speech synthesis with language 'ta' and the Tamil text", async () => {
    const calls = [];
    const w = makeWindow({ fetchImpl: (u, o) => { calls.push({ u, o }); return isAsk(u) ? okStream(tamilAnswer) : (/speech/.test(u) ? Promise.resolve({ ok: false, status: 503, json: () => Promise.resolve({ detail: "x" }) }) : okJson({ speech_local: ["ta"] })); } });
    await sleep(40); speechEnv(w, { local: ["ta"] });
    await askOnce(w, "Which type of soil is good for growing tomatoes?");
    w.document.querySelector("[data-speak]").click(); await sleep(200);
    const sp = calls.filter((c) => /\/api\/speech\/stream/.test(c.u)).map((c) => JSON.parse(c.o.body));
    assert.strictEqual(sp.length, 1); assert.strictEqual(sp[0].language, "ta"); assert.strictEqual(sp[0].text, TA);
  });

  // 8. no local model -> browser Tamil voice speaks the Tamil text; 9. no voice at all -> clear message, text stays usable
  await test("falls back to the browser Tamil voice (no request to the local speech API)", async () => {
    const calls = [];
    const w = makeWindow({ fetchImpl: (u, o) => { calls.push(u); return isAsk(u) ? okStream(tamilAnswer) : okJson({ speech_local: [] }); } });
    await sleep(40);
    const env = speechEnv(w, { voices: [{ lang: "ta-IN", name: "Tamil Voice", localService: true }], local: [] });
    await askOnce(w, "tomato soil"); w.document.querySelector("[data-speak]").click(); await sleep(300);
    assert.strictEqual(env.spoken.length, 1); assert.strictEqual(env.spoken[0].text, TA);
    assert.strictEqual(env.spoken[0].lang, "ta-IN"); assert.strictEqual(env.spoken[0].voice.lang, "ta-IN");
    assert.ok(!calls.some((u) => /\/api\/speech\/stream/.test(u)), "local speech API must not be called when the model is not cached");
  });
  await test("no local model and no browser voice -> clear message, answer text still shown", async () => {
    const w = makeWindow({ fetchImpl: (u) => (isAsk(u) ? okStream(tamilAnswer) : okJson({ speech_local: [] })) });
    await sleep(40); const env = speechEnv(w, { voices: [], local: [] });
    await askOnce(w, "tomato soil"); w.document.querySelector("[data-speak]").click(); await sleep(1700);
    assert.strictEqual(env.spoken.length, 0, "must not speak with a wrong voice");
    assert.ok(/Tamil|ta/.test(w.document.getElementById("toast").textContent) && w.document.getElementById("toast").textContent.length > 10);
    assert.ok(w.document.querySelector(".msg.bot .text").textContent.includes("தக்காளி"));
  });

  // 10. translation was unavailable: English text is labelled English, a visible banner appears, and speech uses English
  await test("English text is never spoken with the Tamil voice and a visible banner explains the fallback", async () => {
    const res = { answer: "Tomato grows well in loamy, well drained soil.", grounded: true, text_language: "en", answer_language: "ta",
                  warnings: ["translation_unavailable"], grounding: { notes: [] }, sources: [] };
    const w = makeWindow({ fetchImpl: (u) => (isAsk(u) ? okStream(res) : okJson({ speech_local: ["ta"] })) });
    await sleep(40); const env = speechEnv(w, { voices: [{ lang: "en-IN", name: "English" }, { lang: "ta-IN", name: "Tamil" }], local: [] });
    await askOnce(w, "Which type of soil is good for growing tomatoes?");
    assert.ok(w.document.querySelector(".banner"), "a visible banner must explain that the answer is not in Tamil");
    assert.strictEqual(w.document.querySelector(".msg.bot .text").getAttribute("lang"), "en");
    w.document.querySelector("[data-speak]").click(); await sleep(300);
    assert.strictEqual(env.spoken.length, 1); assert.strictEqual(env.spoken[0].lang, "en-IN"); assert.strictEqual(env.spoken[0].voice.lang, "en-IN");
  });

  // 11. each message is spoken in ITS OWN language (not the previous message's)
  await test("a later message does not reuse the language of an earlier one", async () => {
    let n = 0;
    const answers = [{ answer: "Loam soil is good.", grounded: false, text_language: "en", sources: [] }, tamilAnswer];
    const w = makeWindow({ fetchImpl: (u) => (isAsk(u) ? okStream(answers[n++]) : okJson({ speech_local: [] })) });
    await sleep(40);
    const env = speechEnv(w, { voices: [{ lang: "en-IN", name: "E" }, { lang: "ta-IN", name: "T" }], local: [] });
    await askOnce(w, "first question about soil"); await askOnce(w, "second question about soil");
    const btns = w.document.querySelectorAll("[data-speak]");
    assert.strictEqual(btns.length, 2);
    btns[0].click(); await sleep(300); w.speechSynthesis.cancel(); btns[1].click(); await sleep(300);
    assert.deepStrictEqual(env.spoken.map((u) => u.lang), ["en-IN", "ta-IN"]);
  });

  // 12. the request tells the server the ANSWER language and lets it detect the question language
  await test("request carries answer_language and input_language=auto", async () => {
    const calls = [];
    const w = makeWindow({ fetchImpl: (u, o) => { calls.push({ u, o }); return isAsk(u) ? okStream({ answer: "x", grounded: false, text_language: "ta", sources: [] }) : okJson({ speech_local: [] }); } });
    await sleep(40); w.localStorage.setItem("sawvialLanguage", "ta");
    await askOnce(w, "Which type of soil is good for growing tomatoes?");
    const body = JSON.parse(calls.find((c) => isAsk(c.u)).o.body);
    assert.strictEqual(body.answer_language, "ta"); assert.strictEqual(body.input_language, "auto");
    assert.ok(/\/api\/ask\/stream$/.test(calls.find((c) => isAsk(c.u)).u));
  });

  // 13. research result: kind + origin badges, link to the real page, differing values shown
  await test("web sources show kind/origin badges and differing values", async () => {
    const res = { answer: "Tomato prefers well drained loam.", grounded: true, text_language: "en", warnings: [],
      grounding: { notes: ["web_unreviewed"] },
      research: { status: "used", comparison: { differences: [{ quantity: "pH", values: [{ title: "TNAU page", low: 6, high: 7, unit: "" }, { title: "ICAR paper", low: 5.5, high: 6.8, unit: "" }] }] } },
      sources: [{ id: "SWR_000001", cited: true, title: "Tomato - soil", url: "http://agritech.tnau.ac.in/t.html", origin: "web", source_type: "official_guidance" }] };
    const w = makeWindow({ fetchImpl: (u) => (isAsk(u) ? okStream(res) : okJson({ speech_local: [] })) });
    await sleep(40); await askOnce(w, "tomato soil");
    const src = w.document.querySelector(".sources").textContent;
    assert.ok(/Official guidance · live web/.test(src), src);
    assert.ok([...w.document.querySelectorAll(".sources a")].some((a) => a.href === "http://agritech.tnau.ac.in/t.html"));
    assert.ok(/expert/.test(w.document.querySelector(".notes").textContent));
    assert.ok(/pH/.test(w.document.querySelector(".differing").textContent) && /5\.5–6\.8/.test(w.document.querySelector(".differing").textContent));
  });

  console.log(`\n${passed} frontend DOM tests passed`);
})().catch((e) => { console.error("FAIL:", e.message); process.exit(1); });
