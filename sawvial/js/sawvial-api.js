/* SAWVIAL API client. Loaded by ask.html before chat.js. */
(function () {
  // Same origin when the page is served by api.py (any port, e.g. 8000 or 8001).
  // Only fall back to api.py on :8000 when the page is opened as file:// or from a
  // separate static dev server (Live Server, http.server, Vite, ...).
  // To force a different API address, set window.SAWVIAL_API_BASE before this file loads.
  const STATIC_DEV_PORTS = ["3000", "4173", "5173", "5500", "5501", "8080", "8081", "8888"];
  const BASE =
    window.SAWVIAL_API_BASE ||
    (location.protocol === "file:" || STATIC_DEV_PORTS.indexOf(location.port) !== -1
      ? "http://localhost:8000"
      : "");

  // Resolves to the final answer object:
  //   {answer, grounded, input_language, answer_language (requested), text_language (language the text is REALLY in),
  //    translation, warnings, grounding, research, evidence_origin, sources:[{id,title,authors,year,journal,doi,doi_url,
  //    url,page,origin,source_type,trust_tier,score,cited,snippet}], latency_ms}
  // onStage({stage: "local_search"|"online_search"|"reading"|"generating"|"translating_in"|"translating", ...}) is called while
  // the server works. Rejects with err.status / err.detail.
  const requests = Object.create(null);

  function apiError(status, detail) {
    const err = new Error("SAWVIAL API error " + status + (detail ? ": " + detail : ""));
    err.status = status;
    err.detail = detail || "";
    return err;
  }

  window.sawvialAsk = async function (question, language, history, requestId, onStage) {
    const ctrl = new AbortController();
    requestId = requestId || String(Date.now()) + Math.random().toString(36).slice(2);
    requests[requestId] = ctrl;
    const timer = setTimeout(function () { ctrl.abort(); }, 300000); // first request + online research can be slow
    const payload = JSON.stringify({
      question: question,
      language: language || "en",          // legacy field = the language the user wants the ANSWER in
      answer_language: language || "en",
      input_language: "auto",              // the server detects what language the question is written in
      history: history || [],
      request_id: requestId,
    });
    try {
      let r = await fetch(BASE + "/api/ask/stream", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: payload, signal: ctrl.signal });
      if (r.status === 404 || r.status === 405) {   // older server without streaming
        r = await fetch(BASE + "/api/ask", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: payload, signal: ctrl.signal });
        if (!r.ok) throw apiError(r.status, await readDetail(r));
        return await r.json();
      }
      if (!r.ok) throw apiError(r.status, await readDetail(r));
      let final = null;
      const handle = function (line) {
        if (!line.trim()) return;
        const ev = JSON.parse(line);
        if (ev.stage) { if (onStage) { try { onStage(ev); } catch (e) { console.error(e); } } }
        else if (ev.final) final = ev.final;
        else if (ev.error) throw apiError(ev.error.status || 500, ev.error.detail);
      };
      if (r.body && r.body.getReader) {
        const reader = r.body.getReader();
        const dec = new TextDecoder("utf-8");
        let buf = "";
        while (true) {
          const part = await reader.read();
          if (part.done) break;
          buf += dec.decode(part.value, { stream: true });
          let nl;
          while ((nl = buf.indexOf("\n")) >= 0) { handle(buf.slice(0, nl)); buf = buf.slice(nl + 1); }
        }
        buf += dec.decode();
        handle(buf);
      } else {
        (await r.text()).split("\n").forEach(handle);
      }
      if (!final) throw apiError(502, "the server closed the connection without an answer");
      return final;
    } finally {
      clearTimeout(timer);
      delete requests[requestId];
    }
  };

  async function readDetail(r) {
    try { return (await r.json()).detail || ""; } catch (e) { return ""; }
  }

  window.sawvialCancel = async function (requestId) {
    try {
      await fetch(BASE + "/api/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: requestId }),
        keepalive: true,
      });
    } catch (err) {
      console.warn("Could not cancel SAWVIAL request:", err);
    } finally {
      if (requests[requestId]) requests[requestId].abort();
    }
  };

  window.sawvialSpeak = async function (text, language, requestId, signal, onAudioChunk) {
    const r = await fetch(BASE + "/api/speech/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        language: language,
        request_id: requestId,
      }),
      signal: signal,
    });
    if (!r.ok) {
      let detail = "Speech generation failed";
      try {
        const error = await r.json();
        if (error.detail) detail = error.detail;
      } catch (err) {
        console.error("Could not read speech error response:", err);
      }
      throw new Error(detail);
    }
    if (r.status === 204) return null;
    if (!r.body || !r.body.getReader) {
      throw new Error("This browser cannot play streamed speech");
    }
    const sampleRate = Number(r.headers.get("X-Audio-Sample-Rate")) || 44100;
    const reader = r.body.getReader();
    let receivedBytes = 0;
    while (true) {
      const result = await reader.read();
      if (result.done) break;
      receivedBytes += result.value.byteLength;
      onAudioChunk(result.value, sampleRate);
    }
    if (receivedBytes === 0) throw new Error("Speech stream returned no audio");
  };

  window.sawvialHealth = async function () {
    const r = await fetch(BASE + "/api/health");
    return r.json();
  };

  // Server capabilities (e.g. which languages have a LOCAL voice). Missing/failed = browser voices only.
  window.SAWVIAL_CAPS = { speech_local: [] };
  window.sawvialHealth().then(function (h) { window.SAWVIAL_CAPS = h; }).catch(function () {});
})();
