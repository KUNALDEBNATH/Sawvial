# SAWVIAL — Static Frontend

A static HTML/CSS/JS frontend for SAWVIAL, a voice-first, multilingual
agricultural assistant. No build step, no dependencies, no backend.

## File structure

```
sawvial/
├── index.html          Landing page (photo-led marketing page)
├── ask.html            The assistant — chat UI with sidebar history
├── language.html       Language selection
├── help.html           How-to-use guide + FAQ
├── css/landing.css     Styles for index / language / help
├── css/chat.css        Styles for the chat UI
├── js/i18n.js          Every UI string in all 7 languages + applyI18n()/t()
├── js/script.js        Shared data (languages, demo Q&A) + language helpers
├── js/chat.js          All chat behaviour: history, answers, mic, playback
└── assets/             logo/, images/, icons/ — empty by design, see below
```

## How the pages connect

```
index.html ──► ask.html ──► language.html (only if no language chosen yet)
                  ▲                │
                  └────────────────┘
```

There is no longer a separate "listening" or "answer" page. Questions and
answers appear in the same thread, one below the other, like a chat.

- Selected language → `localStorage.sawvialLanguage`
- Chat history → `localStorage.sawvialChats` (array of conversations)
- Currently open chat → `localStorage.sawvialActiveChat`

## How to run

Open `index.html` in any browser, or serve the folder with any static
server (`python3 -m http.server`). A server is recommended — the
microphone permission and Google Fonts behave better over `http://`
than over `file://`.

## Interface language

Picking a language does not just change the answers — the entire
interface changes with it. Every visible string lives in `js/i18n.js`
as `{ en, hi, bn, ta, te, kn, mr }`.

- In HTML, mark an element with `data-i18n="key"` (or `data-i18n-html`,
  `data-i18n-ph` for placeholders, `data-i18n-aria` for labels).
  `applyI18n()` runs on every page load and fills them in.
- In JS, call `t("key")` for anything rendered dynamically.
- `<html lang>` is set to the chosen code, so browser hyphenation and
  screen readers follow along.
- On `language.html` the page re-renders live as you tap each card, so a
  farmer can see their own script before committing.

To add a string: add the key to `I18N` with all seven values, then
reference it. To add a language: add its code to every entry, to
`languages` in `js/script.js`, and to `SPEECH_LOCALE` in `js/chat.js`.

## What's real and what's simulated

- **Speech-to-text** — `startListening()` in `js/chat.js` uses the browser's
  Web Speech API (`webkitSpeechRecognition`) when available, with the locale
  set from the chosen language. If the browser has no speech API, or
  recognition errors out, it falls back to inserting a demo question so the
  flow can still be shown. Swap that fallback for your own STT endpoint.
- **Text-to-speech** — the "Listen" button under each answer uses
  `speechSynthesis` with the matching Indian locale. Real, not simulated.
  Replace with your own TTS audio if you want better voices.
- **Answers** — `respond()` in `js/chat.js` picks from the static
  `demoAnswers` object in `js/script.js` using keyword matching
  (`TOPIC_HINTS`). This is the one place to change: call your backend,
  await the reply, and pass the text into the same `typeOut()` renderer.
- **Photography** — the landing page hotlinks Unsplash images. Download
  them into `assets/images/` and switch the `src` attributes if you need
  the site to work offline.

Everything else — layout, history, persistence, language selection,
keyboard handling, responsive behaviour, accessibility — is real.

## Deploying

This is a static site — no build command needed.

- **Render**: New Static Site → connect the repo → Build Command: empty
  → Publish Directory: `.`
- **Any other static host** (Netlify, GitHub Pages, Vercel): same idea —
  point it at the repo root, no build step.
