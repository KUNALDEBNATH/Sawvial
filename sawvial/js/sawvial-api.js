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

  // Resolves to {answer, grounded, language, sources:[{id,title,page,score,cited,snippet}], latency_ms}
  const requests = Object.create(null);

  window.sawvialAsk = async function (question, language, history, requestId) {
    const ctrl = new AbortController();
    requestId = requestId || String(Date.now()) + Math.random().toString(36).slice(2);
    requests[requestId] = ctrl;
    const timer = setTimeout(function () { ctrl.abort(); }, 180000); // first request can be slow
    try {
      const r = await fetch(BASE + "/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question,
          language: language || "en",
          history: history || [],
          request_id: requestId,
        }),
        signal: ctrl.signal,
      });
      if (!r.ok) throw new Error("SAWVIAL API error " + r.status);
      return await r.json();
    } finally {
      clearTimeout(timer);
      delete requests[requestId];
    }
  };

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
})();
