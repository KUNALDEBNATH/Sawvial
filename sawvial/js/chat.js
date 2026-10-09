/* =========================================================
   SAWVIAL — chat.js
   Conversational ask page. Depends on script.js for the
   language list, demo questions and demo answers.

   `respond()` calls the SAWVIAL API (js/sawvial-api.js -> api.py).
   `startListening()` still uses the browser's speech recognition.
   ========================================================= */

(function () {
  "use strict";

  /* ---------- element refs ---------- */
  const el = {
    side: document.getElementById("side"),
    scrim: document.getElementById("scrim"),
    menuBtn: document.getElementById("menu-btn"),
    closeSide: document.getElementById("close-side"),
    newChat: document.getElementById("new-chat"),
    search: document.getElementById("history-search"),
    history: document.getElementById("history"),
    title: document.getElementById("chat-title"),
    thread: document.getElementById("thread"),
    composer: document.getElementById("composer"),
    input: document.getElementById("input"),
    mic: document.getElementById("mic-btn"),
    send: document.getElementById("send-btn"),
    recLabel: document.getElementById("rec-label"),
    toast: document.getElementById("toast"),
  };

  const STORE = "sawvialChats";
  const ACTIVE = "sawvialActiveChat";

  const SPEECH_LOCALE = {
    hi: "hi-IN", bn: "bn-IN", ta: "ta-IN", te: "te-IN",
    kn: "kn-IN", mr: "mr-IN", en: "en-IN",
  };
  const SPEECH_LANGUAGE_NAME = {
    hi: "Hindi", bn: "Bengali", ta: "Tamil", te: "Telugu",
    kn: "Kannada", mr: "Marathi", en: "English",
  };

  /* Keywords used to route a typed question to a demo answer.
     Replace with your real intent detection / retrieval. */
  const TOPIC_HINTS = {
    water: ["water", "irrigat", "पानी", "सिंचाई", "நீர", "பாய்ச்ச", "నీరు", "ನೀರು", "पाणी", "জল", "সেচ"],
    soil: ["soil", "manure", "compost", "fertil", "yellow", "मिट्टी", "खाद", "மண்", "உரம்", "నేల", "ఎరువు", "ಮಣ್ಣು", "माती", "মাটি", "সার"],
    pest: ["pest", "insect", "worm", "disease", "spray", "कीट", "कीड़े", "பூச்சி", "పురుగు", "ಕೀಟ", "किड", "পোকা"],
    crop: ["crop", "rice", "paddy", "sow", "seed", "harvest", "फसल", "धान", "பயிர", "நெல", "పంట", "వరి", "ಬೆಳೆ", "ಭತ್ತ", "पीक", "भात", "ফসল", "ধান"],
  };

  let chats = [];
  let activeId = null;
  let busy = false;
  let recognition = null;
  let activeRequestId = null;
  let activeHolder = null;
  let activeTypewriterStop = null;
  let activeSpeech = null;
  let speechAudio = null;

  /* ---------- storage ---------- */
  function load() {
    try {
      chats = JSON.parse(localStorage.getItem(STORE)) || [];
    } catch (e) {
      chats = [];
    }
    activeId = localStorage.getItem(ACTIVE);
  }

  function save() {
    try {
      localStorage.setItem(STORE, JSON.stringify(chats.slice(0, 60)));
      if (activeId) localStorage.setItem(ACTIVE, activeId);
      else localStorage.removeItem(ACTIVE);
    } catch (e) {
      /* storage full or blocked — chat still works in memory */
    }
  }

  function current() {
    return chats.find((c) => c.id === activeId) || null;
  }

  function newChat() {
    stopSpeech();
    activeId = null;
    save();
    render();
    el.input.focus();
    closeSidebar();
  }

  function ensureChat(firstMessage) {
    if (current()) return current();
    const chat = {
      id: "c" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      title: firstMessage.length > 46 ? firstMessage.slice(0, 46).trim() + "…" : firstMessage,
      createdAt: Date.now(),
      lang: getSelectedLanguage() || "en",
      messages: [],
    };
    chats.unshift(chat);
    activeId = chat.id;
    return chat;
  }

  /* ---------- helpers ---------- */
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function toast(text) {
    el.toast.textContent = text;
    el.toast.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(function () {
      el.toast.classList.remove("show");
    }, 2200);
  }

  function detectTopic(text) {
    const low = text.toLowerCase();
    const keys = ["pest", "water", "soil", "crop"];
    for (let i = 0; i < keys.length; i++) {
      const hits = TOPIC_HINTS[keys[i]];
      for (let j = 0; j < hits.length; j++) {
        if (low.indexOf(hits[j].toLowerCase()) !== -1) return keys[i];
      }
    }
    return "crop";
  }

  function scrollDown() {
    el.thread.scrollTop = el.thread.scrollHeight;
  }

  /* ---------- sidebar ---------- */
  function openSidebar() {
    el.side.classList.add("open");
    el.scrim.classList.add("show");
  }

  function closeSidebar() {
    el.side.classList.remove("open");
    el.scrim.classList.remove("show");
  }

  function renderHistory() {
    const q = (el.search.value || "").trim().toLowerCase();
    const list = chats.filter(function (c) {
      return !q || c.title.toLowerCase().indexOf(q) !== -1;
    });

    if (!list.length) {
      el.history.innerHTML =
        '<p class="history-empty">' +
        (q ? t("chat.noMatch") : t("chat.historyEmpty")) +
        "</p>";
      return;
    }

    const dayMs = 86400000;
    const today = [];
    const week = [];
    const older = [];
    const now = Date.now();
    list.forEach(function (c) {
      const age = now - c.createdAt;
      if (age < dayMs) today.push(c);
      else if (age < dayMs * 7) week.push(c);
      else older.push(c);
    });

    let html = "";
    [[t("chat.today"), today], [t("chat.week"), week], [t("chat.earlier"), older]].forEach(function (grp) {
      if (!grp[1].length) return;
      html += '<div class="history-group">' + grp[0] + "</div>";
      grp[1].forEach(function (c) {
        html +=
          '<div class="history-item' + (c.id === activeId ? " active" : "") +
          '" role="button" tabindex="0" data-open="' + c.id + '">' +
          '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" style="flex:none;opacity:.55"><path d="M21 12a8 8 0 0 1-11.6 7.1L3 21l1.9-6.4A8 8 0 1 1 21 12Z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>' +
          "<span>" + escapeHtml(c.title) + "</span>" +
          '<button class="history-del" data-del="' + c.id + '" aria-label="' + t("chat.delete") + '">' +
          '<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
          "</button></div>";
      });
    });
    el.history.innerHTML = html;
  }

  /* ---------- sources (from the API) ---------- */
  function sourcesHtml(sources) {
    if (!sources || !sources.length) return "";
    const items = sources.map(function (s) {
      const title = s.title ? escapeHtml(s.title) : "";
      const page = s.page ? " · p." + escapeHtml(s.page) : "";
      return '<li><span class="src-id">' + escapeHtml(s.id) + "</span> " + title + page + "</li>";
    }).join("");
    return (
      '<details class="sources"><summary>' + escapeHtml(t("chat.sources")) +
      " (" + sources.length + ")</summary><ul>" + items + "</ul></details>"
    );
  }

  function pickSources(list) {
    list = list || [];
    const cited = list.filter(function (s) { return s.cited; });
    return (cited.length ? cited : list).slice(0, 4).map(function (s) {
      return { id: s.id, title: s.title, page: s.page };
    });
  }

  (function injectSourceStyles() {
    const st = document.createElement("style");
    st.textContent =
      ".sources{margin-top:10px;font-size:13px;opacity:.85}" +
      ".sources summary{cursor:pointer;font-weight:600}" +
      ".sources ul{margin:6px 0 0;padding-left:18px}" +
      ".sources li{margin:3px 0;line-height:1.35}" +
      ".src-id{font-family:ui-monospace,Consolas,monospace;font-size:11px;opacity:.7}";
    document.head.appendChild(st);
  })();

  /* ---------- thread ---------- */
  function botAvatar() {
    return (
      '<span class="avatar" aria-hidden="true">' +
      '<svg viewBox="0 0 24 24" fill="none"><path d="M12 21.5C12 21.5 4.5 16.6 4.5 10.3C4.5 5.9 7.9 2.5 12 2.5C16.1 2.5 19.5 5.9 19.5 10.3C19.5 16.6 12 21.5 12 21.5Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M12 17.5V7.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>' +
      "</span>"
    );
  }

  function emptyState() {
    const code = getSelectedLanguage() || "en";
    const picks = [
      { topic: "water", icon: "💧" },
      { topic: "soil", icon: "🪨" },
      { topic: "crop", icon: "🌱" },
      { topic: "pest", icon: "🐛" },
    ];
    let cards = "";
    picks.forEach(function (p) {
      const q = (demoSampleQuestions[p.topic] && (demoSampleQuestions[p.topic][code] || demoSampleQuestions[p.topic].en)) || "";
      cards +=
        '<button data-ask="' + escapeHtml(q) + '" data-topic="' + p.topic + '">' +
        '<span class="chip-ico">' + p.icon + "</span>" +
        "<span>" + escapeHtml(q) + "<small>" + t("topic." + p.topic) + "</small></span>" +
        "</button>";
    });

    return (
      '<div class="empty">' +
      '<div class="empty-mark"><svg viewBox="0 0 24 24" fill="none"><path d="M12 21.5C12 21.5 4.5 16.6 4.5 10.3C4.5 5.9 7.9 2.5 12 2.5C16.1 2.5 19.5 5.9 19.5 10.3C19.5 16.6 12 21.5 12 21.5Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="M12 17.5V7.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M12 11C12 11 8.8 10.4 8.8 7.6" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><path d="M12 14.2C12 14.2 15.2 13.6 15.2 10.8" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg></div>' +
      "<h1>" + escapeHtml(t("chat.emptyTitle")) + "</h1>" +
      "<p>" + escapeHtml(t("chat.emptySub")) + "</p>" +
      '<div class="suggest">' + cards + "</div>" +
      "</div>"
    );
  }

  function messageHtml(m, index) {
    if (m.role === "user") {
      return '<div class="msg user"><div class="bubble">' + escapeHtml(m.text) + "</div></div>";
    }
    const meta = topicLabels[m.topic] || topicLabels.crop;
    const label = t("topic." + (topicLabels[m.topic] ? m.topic : "crop"));
    return (
      '<div class="msg bot" data-i="' + index + '">' +
      botAvatar() +
      '<div class="body">' +
      '<div class="meta">SAWVIAL <span class="tag">' + meta.icon + " " + label + "</span></div>" +
      '<div class="text">' + escapeHtml(m.text) + "</div>" +
      sourcesHtml(m.sources) +
      '<div class="actions">' +
      '<button class="act" data-speak="' + index + '"><svg viewBox="0 0 24 24" fill="none"><path d="M11 5 6 9H3v6h3l5 4V5Z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/><path d="M16 9a4 4 0 0 1 0 6M19 6a8 8 0 0 1 0 12" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg><span data-speak-label>' + escapeHtml(t("chat.listen")) + '</span></button>' +
      '<button class="act" data-copy="' + index + '"><svg viewBox="0 0 24 24" fill="none"><rect x="9" y="9" width="11" height="11" rx="2.5" stroke="currentColor" stroke-width="1.7"/><path d="M15 5.5A2.5 2.5 0 0 0 12.5 4h-6A2.5 2.5 0 0 0 4 6.5v6A2.5 2.5 0 0 0 5.5 15" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>' + t("chat.copy") + '</button>' +
      '<button class="act" data-rate="up"><svg viewBox="0 0 24 24" fill="none"><path d="M7 20V10l4.5-7 .8.4a2.5 2.5 0 0 1 1.3 2.8L13 10h4.8a2.2 2.2 0 0 1 2.1 2.8l-1.6 6A2.2 2.2 0 0 1 16.2 20H7Zm0 0H4V10h3" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg></button>' +
      '<button class="act" data-rate="down"><svg viewBox="0 0 24 24" fill="none"><path d="M17 4v10l-4.5 7-.8-.4a2.5 2.5 0 0 1-1.3-2.8L11 14H6.2a2.2 2.2 0 0 1-2.1-2.8l1.6-6A2.2 2.2 0 0 1 7.8 4H17Zm0 0h3v10h-3" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg></button>' +
      "</div></div></div>"
    );
  }

  function render() {
    const chat = current();
    el.title.textContent = chat ? chat.title : t("chat.new");

    if (!chat || !chat.messages.length) {
      el.thread.innerHTML = emptyState();
    } else {
      let html = '<div class="thread-inner">';
      chat.messages.forEach(function (m, i) {
        html += messageHtml(m, i);
      });
      html += "</div>";
      el.thread.innerHTML = html;
      scrollDown();
    }
    renderHistory();
  }

  /* ---------- sending ---------- */
  function ask(text, forcedTopic) {
    text = (text || "").trim();
    if (!text || busy) return;
    stopSpeech();

    if (!getSelectedLanguage()) {
      sessionStorage.setItem("sawvialPending", text);
      window.location.href = "language.html?redirect=ask.html";
      return;
    }

    const chat = ensureChat(text);
    const history = chat.messages.slice(-24).map(function (message) {
      return {
        role: message.role === "bot" ? "assistant" : "user",
        content: message.text,
      };
    });
    chat.messages.push({ role: "user", text: text });
    save();
    render();

    el.input.value = "";
    autosize();
    updateSend();

    respond(text, forcedTopic || detectTopic(text), history);
  }

  /* Calls the SAWVIAL API (api.py). No canned fallback: if the server is
     unreachable we say so instead of showing a made-up answer. */
  function respond(question, topic, history) {
    busy = true;
    activeRequestId = makeRequestId();
    const requestId = activeRequestId;
    updateSend();

    const code = getSelectedLanguage() || "en";
    const chatRef = current();

    // typing indicator
    const inner = el.thread.querySelector(".thread-inner");
    const holder = document.createElement("div");
    holder.className = "msg bot";
    holder.innerHTML =
      botAvatar() +
      '<div class="body"><div class="meta">SAWVIAL</div>' +
      '<div class="dots"><i></i><i></i><i></i></div></div>';
    activeHolder = holder;
    inner.appendChild(holder);
    scrollDown();

    const call =
      typeof window.sawvialAsk === "function"
        ? window.sawvialAsk(question, code, history, requestId)
        : Promise.reject(new Error("js/sawvial-api.js is not loaded"));

    call.then(function (res) {
      if (requestId !== activeRequestId) return;
      const answer = ((res && res.answer) || "").trim();
      if (!answer) throw new Error("empty answer");
      const sources = res.grounded ? pickSources(res.sources) : [];

      const meta = topicLabels[topic] || topicLabels.crop;
      holder.querySelector(".meta").innerHTML =
        'SAWVIAL <span class="tag">' + meta.icon + " " + t("topic." + (topicLabels[topic] ? topic : "crop")) + "</span>";
      const textEl = document.createElement("div");
      textEl.className = "text";
      holder.querySelector(".dots").replaceWith(textEl);

      typeOut(textEl, answer, function (displayedText) {
        if (chatRef && displayedText.trim()) {
          chatRef.messages.push({ role: "bot", text: displayedText, topic: topic, sources: sources });
        }
        save();
        busy = false;
        activeRequestId = null;
        activeHolder = null;
        activeTypewriterStop = null;
        updateSend();
        if (chatRef && chatRef.id === activeId) render();
        else renderHistory();
      });
    }).catch(function (err) {
      if (requestId !== activeRequestId) return;
      activeRequestId = null;
      activeHolder = null;
      console.error("SAWVIAL API error:", err);
      const textEl = document.createElement("div");
      textEl.className = "text";
      textEl.textContent = t("chat.error");
      const dots = holder.querySelector(".dots");
      if (dots) dots.replaceWith(textEl);
      busy = false;
      updateSend();
      scrollDown();
    });
  }

  function typeOut(node, text, done) {
    const words = text.split(" ");
    let i = 0;
    node.innerHTML = '<span class="caret"></span>';
    function stop() {
      clearInterval(timer);
      activeTypewriterStop = null;
      node.textContent = words.slice(0, i).join(" ");
      done(node.textContent);
    }
    activeTypewriterStop = stop;
    const timer = setInterval(function () {
      i += 2;
      node.textContent = words.slice(0, i).join(" ");
      if (i < words.length) {
        node.insertAdjacentHTML("beforeend", ' <span class="caret"></span>');
      }
      scrollDown();
      if (i >= words.length) {
        clearInterval(timer);
        activeTypewriterStop = null;
        node.textContent = text;
        done(text);
      }
    }, 45);
  }

  function makeRequestId() {
    return window.crypto && typeof window.crypto.randomUUID === "function"
      ? window.crypto.randomUUID()
      : "ask-" + Date.now().toString(36) + Math.random().toString(36).slice(2);
  }

  function stopGeneration() {
    if (!busy) return;
    const requestId = activeRequestId;
    activeRequestId = null;
    busy = false;
    updateSend();
    if (requestId && typeof window.sawvialCancel === "function") {
      window.sawvialCancel(requestId).catch(function (err) {
        console.error("Could not cancel SAWVIAL generation:", err);
      });
    }
    if (activeTypewriterStop) {
      activeTypewriterStop();
    } else if (activeHolder) {
      activeHolder.remove();
    }
    activeHolder = null;
    activeTypewriterStop = null;
    scrollDown();
  }

  /* ---------- speech in ---------- */
  function startListening() {
    if (busy) return;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const code = getSelectedLanguage() || "en";

    if (!getSelectedLanguage()) {
      window.location.href = "language.html?redirect=ask.html";
      return;
    }

    el.composer.classList.add("recording");
    el.mic.classList.add("rec");
    el.recLabel.textContent = t("chat.listening");

    if (SR) {
      try {
        recognition = new SR();
        recognition.lang = SPEECH_LOCALE[code] || "en-IN";
        recognition.interimResults = true;
        recognition.continuous = false;

        let finalText = "";
        recognition.onresult = function (ev) {
          let interim = "";
          for (let i = ev.resultIndex; i < ev.results.length; i++) {
            if (ev.results[i].isFinal) finalText += ev.results[i][0].transcript;
            else interim += ev.results[i][0].transcript;
          }
          el.recLabel.textContent = (finalText + interim) || t("chat.listening");
        };
        recognition.onerror = function () {
          stopListening();
          fallbackTranscript();
        };
        recognition.onend = function () {
          stopListening();
          if (finalText.trim()) ask(finalText.trim());
          else fallbackTranscript();
        };
        recognition.start();
        return;
      } catch (e) {
        recognition = null;
      }
    }

    // No speech API available — simulate so the flow can still be demoed.
    setTimeout(function () {
      stopListening();
      fallbackTranscript();
    }, 2200);
  }

  function fallbackTranscript() {
    const code = getSelectedLanguage() || "en";
    const keys = Object.keys(demoSampleQuestions);
    const topic = keys[Math.floor(Math.random() * keys.length)];
    const q = demoSampleQuestions[topic][code] || demoSampleQuestions[topic].en;
    ask(q, topic);
  }

  function stopListening() {
    el.composer.classList.remove("recording");
    el.mic.classList.remove("rec");
    el.recLabel.textContent = t("chat.listening");
    if (recognition) {
      try { recognition.onend = null; recognition.stop(); } catch (e) {}
      recognition = null;
    }
  }

  /* ---------- speech out ---------- */
  function matchingSpeechVoice(code) {
    const targetLocale = SPEECH_LOCALE[code] || SPEECH_LOCALE.en;
    const targetLanguage = targetLocale.split("-")[0].toLowerCase();
    const voices = window.speechSynthesis.getVoices()
      .filter(function (voice) {
        return voice.lang && voice.lang.replace(/_/g, "-").toLowerCase().split("-")[0] === targetLanguage;
      })
      .map(function (voice) {
        const locale = voice.lang.replace(/_/g, "-").toLowerCase();
        let score = locale === targetLocale.toLowerCase() ? 100 : 50;
        if (/\bnatural\b/i.test(voice.name)) score += 10;
        if (voice.localService) score += 1;
        return { voice: voice, score: score };
      })
      .sort(function (a, b) { return b.score - a.score; });
    return voices.length ? voices[0].voice : null;
  }

  function resolveSpeechVoice(code) {
    const current = matchingSpeechVoice(code);
    if (current) return Promise.resolve(current);

    return new Promise(function (resolve) {
      let settled = false;
      function finish() {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        window.speechSynthesis.removeEventListener("voiceschanged", onVoicesChanged);
        resolve(matchingSpeechVoice(code));
      }
      function onVoicesChanged() {
        if (matchingSpeechVoice(code)) finish();
      }
      const timer = setTimeout(finish, 1200);
      window.speechSynthesis.addEventListener("voiceschanged", onVoicesChanged);
    });
  }

  function speak(text, btn) {
    if (activeSpeech && activeSpeech.button === btn) {
      stopSpeech();
      return;
    }
    stopSpeech();
    const code = getSelectedLanguage() || "en";
    if (["bn", "mr", "en"].indexOf(code) !== -1 && window.sawvialSpeak) {
      speakWithLocalVoice(text, code, btn);
      return;
    }
    if (!("speechSynthesis" in window)) {
      toast(t("chat.noAudio"));
      return;
    }
    const u = new SpeechSynthesisUtterance(text);
    u.lang = SPEECH_LOCALE[code] || "en-IN";
    u.rate = 0.95;
    btn.classList.add("on");
    const label = btn.querySelector("[data-speak-label]");
    if (label) label.textContent = t("chat.stopListening");
    activeSpeech = { utterance: u, button: btn };
    u.onend = u.onerror = function () {
      if (activeSpeech && activeSpeech.utterance === u) stopSpeech();
    };
    resolveSpeechVoice(code).then(function (voice) {
      if (!activeSpeech || activeSpeech.utterance !== u) return;
      if (!voice) {
        stopSpeech();
        toast(t("chat.noLanguageVoice").replace("{language}", SPEECH_LANGUAGE_NAME[code] || "English"));
        return;
      }
      u.voice = voice;
      window.speechSynthesis.speak(u);
    }).catch(function (err) {
      if (activeSpeech && activeSpeech.utterance === u) stopSpeech();
      console.error("Could not select a speech voice:", err);
    });
  }

  function speakWithLocalVoice(text, code, btn) {
    const requestId = "speech-" + Date.now() + Math.random().toString(36).slice(2);
    const controller = new AbortController();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
      toast(t("chat.noAudio"));
      return;
    }
    let audioContext;
    try {
      audioContext = new AudioContextClass();
      audioContext.resume().catch(function (err) {
        console.error("Could not start audio playback:", err);
      });
    } catch (err) {
      console.error("Could not initialize audio playback:", err);
      toast(t("chat.noAudio"));
      return;
    }
    const status = document.createElement("span");
    status.className = "speech-status";
    status.setAttribute("role", "status");
    status.textContent = t("chat.generatingVoice");
    btn.parentElement.appendChild(status);
    const speech = {
      requestId: requestId,
      controller: controller,
      button: btn,
      status: status,
      audioContext: audioContext,
      nextAudioTime: audioContext.currentTime,
      pendingBytes: new Uint8Array(0),
      receivedAudio: false,
    };
    activeSpeech = speech;
    btn.classList.add("on");
    const label = btn.querySelector("[data-speak-label]");
    if (label) label.textContent = t("chat.stopListening");

    function playChunk(bytes, sampleRate) {
      const combined = new Uint8Array(speech.pendingBytes.length + bytes.length);
      combined.set(speech.pendingBytes);
      combined.set(bytes, speech.pendingBytes.length);
      const alignedLength = combined.length - (combined.length % 4);
      speech.pendingBytes = combined.slice(alignedLength);
      if (!alignedLength) return;

      const samples = new Float32Array(combined.slice(0, alignedLength).buffer);
      const buffer = audioContext.createBuffer(1, samples.length, sampleRate);
      buffer.copyToChannel(samples, 0);
      const source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.destination);
      const startAt = Math.max(speech.nextAudioTime, audioContext.currentTime + 0.04);
      source.start(startAt);
      speech.nextAudioTime = startAt + buffer.duration;
      speech.receivedAudio = true;
      speech.status.textContent = t("chat.audioPlaying");
    }

    window.sawvialSpeak(text, code, requestId, controller.signal, playChunk).then(function () {
      speech.controller = null;
      if (activeSpeech !== speech) return;
      if (speech.pendingBytes.length || !speech.receivedAudio) {
        throw new Error("Speech stream ended with incomplete audio");
      }
      speechAudio = speech;
      const playbackDelay = Math.max(0, (speech.nextAudioTime - audioContext.currentTime) * 1000);
      setTimeout(function () {
        if (speechAudio === speech) {
          status.textContent = t("chat.audioReady");
        }
        if (activeSpeech === speech) {
          activeSpeech = null;
          btn.classList.remove("on");
          if (label) label.textContent = t("chat.listen");
        }
      }, playbackDelay);
    }).catch(function (err) {
      if (activeSpeech !== speech || err.name === "AbortError") return;
      stopSpeech();
      console.error("Could not synthesize speech:", err);
      toast(t("chat.speechError"));
    });
  }

  function stopSpeech() {
    const speech = activeSpeech || speechAudio;
    activeSpeech = null;
    speechAudio = null;
    if (speech && speech.controller) {
      speech.controller.abort();
      if (window.sawvialCancel) {
        window.sawvialCancel(speech.requestId).catch(function (err) {
          console.error("Could not cancel speech generation:", err);
        });
      }
    }
    if (speech && speech.audioContext) {
      speech.audioContext.close().catch(function (err) {
        console.error("Could not close speech playback:", err);
      });
    }
    if (speech && speech.status) speech.status.remove();
    if (speech && speech.button) {
      speech.button.classList.remove("on");
      const label = speech.button.querySelector("[data-speak-label]");
      if (label) label.textContent = t("chat.listen");
    }
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  }

  /* ---------- composer plumbing ---------- */
  function autosize() {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(el.input.scrollHeight, 168) + "px";
  }

  function updateSend() {
    if (busy) {
      el.send.disabled = false;
      el.send.setAttribute("aria-label", t("chat.stop"));
      el.send.title = t("chat.stop");
      el.send.innerHTML = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="6.5" y="6.5" width="11" height="11" rx="2" fill="currentColor"/></svg>';
      el.send.classList.add("stop-generating");
      return;
    }
    el.send.disabled = !el.input.value.trim();
    el.send.setAttribute("aria-label", t("chat.send"));
    el.send.title = t("chat.send");
    el.send.innerHTML = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 19V5M6 11l6-6 6 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    el.send.classList.remove("stop-generating");
  }

  /* ---------- events ---------- */
  el.menuBtn.addEventListener("click", openSidebar);
  el.scrim.addEventListener("click", closeSidebar);
  if (el.closeSide) el.closeSide.addEventListener("click", closeSidebar);
  el.newChat.addEventListener("click", newChat);
  el.search.addEventListener("input", renderHistory);

  el.history.addEventListener("click", function (ev) {
    const del = ev.target.closest("[data-del]");
    if (del) {
      ev.stopPropagation();
      const id = del.getAttribute("data-del");
      chats = chats.filter(function (c) { return c.id !== id; });
      if (activeId === id) activeId = null;
      save();
      render();
      return;
    }
    const open = ev.target.closest("[data-open]");
    if (open) {
      stopSpeech();
      activeId = open.getAttribute("data-open");
      save();
      render();
      closeSidebar();
    }
  });

  el.thread.addEventListener("click", function (ev) {
    const sug = ev.target.closest("[data-ask]");
    if (sug) {
      ask(sug.getAttribute("data-ask"), sug.getAttribute("data-topic"));
      return;
    }
    const speakBtn = ev.target.closest("[data-speak]");
    if (speakBtn) {
      const chat = current();
      const m = chat && chat.messages[+speakBtn.getAttribute("data-speak")];
      if (m) speak(m.text, speakBtn);
      return;
    }
    const copyBtn = ev.target.closest("[data-copy]");
    if (copyBtn) {
      const chat = current();
      const m = chat && chat.messages[+copyBtn.getAttribute("data-copy")];
      if (m && navigator.clipboard) {
        navigator.clipboard.writeText(m.text).then(function () {
          toast(t("chat.copied"));
        });
      }
      return;
    }
    const rate = ev.target.closest("[data-rate]");
    if (rate) {
      rate.parentElement.querySelectorAll("[data-rate]").forEach(function (b) {
        b.classList.remove("on");
      });
      rate.classList.add("on");
      toast(t("chat.thanks"));
    }
  });

  el.input.addEventListener("input", function () {
    autosize();
    updateSend();
  });

  el.input.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      ask(el.input.value);
    }
  });

  el.send.addEventListener("click", function () {
    if (busy) stopGeneration();
    else ask(el.input.value);
  });

  el.mic.addEventListener("click", function () {
    if (el.composer.classList.contains("recording")) {
      stopListening();
    } else {
      startListening();
    }
  });
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) stopSpeech();
  });

  /* ---------- boot ---------- */
  load();
  if (typeof applyI18n === "function") applyI18n();
  render();

  // Topic deep link from the landing page (ask.html?topic=water)
  const topicParam = new URLSearchParams(window.location.search).get("topic");
  const pending = sessionStorage.getItem("sawvialPending");
  if (pending) {
    sessionStorage.removeItem("sawvialPending");
    setTimeout(function () { ask(pending); }, 200);
  } else if (topicParam && demoSampleQuestions[topicParam] && getSelectedLanguage()) {
    const code = getSelectedLanguage();
    el.input.value = demoSampleQuestions[topicParam][code] || demoSampleQuestions[topicParam].en;
    autosize();
    updateSend();
    el.input.focus();
  }

  updateSend();
})();
