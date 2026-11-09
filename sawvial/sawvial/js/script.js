/* =========================================================
   SAWVIAL — script.js
   Shared data + language helpers. Loaded by every page.
   The chat behaviour itself lives in js/chat.js.
   ========================================================= */

/* ---------- 1. DATA: LANGUAGES ---------- */
const languages = {
  hi: { name: "Hindi", nativeName: "हिन्दी" },
  bn: { name: "Bengali", nativeName: "বাংলা" },
  ta: { name: "Tamil", nativeName: "தமிழ்" },
  te: { name: "Telugu", nativeName: "తెలుగు" },
  kn: { name: "Kannada", nativeName: "ಕನ್ನಡ" },
  mr: { name: "Marathi", nativeName: "मराठी" },
  en: { name: "English", nativeName: "English" },
};

/* ---------- 2. DATA: SUGGESTED QUESTIONS (shown as tappable cards; only sent when the user taps one) ---------- */
const demoSampleQuestions = {
  crop: {
    en: "When should I water my rice crop?",
    hi: "मुझे अपनी धान की फसल को कब पानी देना चाहिए?",
    ta: "என் நெல் பயிருக்கு எப்போது நீர் பாய்ச்ச வேண்டும்?",
    te: "నా వరి పంటకు ఎప్పుడు నీరు పెట్టాలి?",
    kn: "ನನ್ನ ಭತ್ತದ ಬೆಳೆಗೆ ಯಾವಾಗ ನೀರು ಹಾಕಬೇಕು?",
    mr: "माझ्या भात पिकाला मी पाणी कधी द्यावे?",
    bn: "আমার ধান ফসলে কখন জল দেওয়া উচিত?",
  },
  water: {
    en: "How much water does my field need this week?",
    hi: "इस सप्ताह मेरे खेत को कितने पानी की आवश्यकता है?",
    ta: "இந்த வாரம் என் வயலுக்கு எவ்வளவு நீர் தேவை?",
    te: "ఈ వారం నా పొలానికి ఎంత నీరు అవసరం?",
    kn: "ಈ ವಾರ ನನ್ನ ಹೊಲಕ್ಕೆ ಎಷ್ಟು ನೀರು ಬೇಕು?",
    mr: "या आठवड्यात माझ्या शेताला किती पाणी लागेल?",
    bn: "এই সপ্তাহে আমার জমিতে কতটা জল দরকার?",
  },
  soil: {
    en: "How can I improve my soil?",
    hi: "मैं अपनी मिट्टी को कैसे बेहतर बना सकता हूँ?",
    ta: "என் மண்ணை நான் எப்படி மேம்படுத்தலாம்?",
    te: "నా నేలను నేను ఎలా మెరుగుపరచగలను?",
    kn: "ನನ್ನ ಮಣ್ಣನ್ನು ನಾನು ಹೇಗೆ ಸುಧಾರಿಸಬಹುದು?",
    mr: "मी माझी माती कशी सुधारू शकतो?",
    bn: "আমি কীভাবে আমার মাটির উন্নতি করতে পারি?",
  },
  pest: {
    en: "How can I control pests on my crop?",
    hi: "मैं अपनी फसल में कीटों को कैसे नियंत्रित करूँ?",
    ta: "என் பயிரில் பூச்சிகளை நான் எப்படி கட்டுப்படுத்துவது?",
    te: "నా పంటలో పురుగులను ఎలా నియంత్రించాలి?",
    kn: "ನನ್ನ ಬೆಳೆಯಲ್ಲಿ ಕೀಟಗಳನ್ನು ಹೇಗೆ ನಿಯಂತ್ರಿಸುವುದು?",
    mr: "माझ्या पिकावरील किडींवर मी नियंत्रण कसे ठेवू?",
    bn: "আমি কীভাবে আমার ফসলে পোকামাকড় নিয়ন্ত্রণ করব?",
  },
};

const topicLabels = {
  crop: { en: "Crops", icon: "🌱" },
  water: { en: "Water", icon: "💧" },
  soil: { en: "Soil", icon: "🪨" },
  pest: { en: "Pests", icon: "🐛" },
};

/* ---------- 4. LANGUAGE HELPERS ---------- */
function getSelectedLanguage() {
  return localStorage.getItem("sawvialLanguage");
}

function saveLanguage(code) {
  localStorage.setItem("sawvialLanguage", code);
}

function getLanguageDisplay(code) {
  const lang = languages[code];
  if (!lang) return null;
  return lang.nativeName + (lang.nativeName !== lang.name ? " (" + lang.name + ")" : "");
}

/* ---------- 5. LANGUAGE HELPERS (UI) ---------- */
function reflectSelectedLanguage() {
  const code = getSelectedLanguage();
  document.querySelectorAll("[data-current-language]").forEach((el) => {
    el.textContent = code ? getLanguageDisplay(code) : "Choose language";
  });
}

/* ---------- 6. LANGUAGE SELECTOR PAGE ---------- */
function initLanguageSelector() {
  const grid = document.querySelector(".language-grid");
  if (!grid) return;

  const continueBtn = document.querySelector("#language-continue");
  let selected = getSelectedLanguage();

  function refresh() {
    grid.querySelectorAll(".language-card").forEach((card) => {
      card.setAttribute("aria-pressed", card.dataset.lang === selected ? "true" : "false");
    });
    if (continueBtn) continueBtn.disabled = !selected;
  }

  grid.querySelectorAll(".language-card").forEach((card) => {
    card.addEventListener("click", () => {
      selected = card.dataset.lang;
      window.__i18nPreview = selected;
      if (typeof applyI18n === "function") applyI18n();
      refresh();
    });
    card.addEventListener("dblclick", () => {
      if (continueBtn) continueBtn.click();
    });
  });

  refresh();

  if (continueBtn) {
    continueBtn.addEventListener("click", () => {
      if (!selected) return;
      saveLanguage(selected);
      const params = new URLSearchParams(window.location.search);
      const target = params.get("redirect") || "";
      // only allow plain local page names like "ask.html" (blocks open redirects)
      window.location.href = /^[\w-]+\.html$/.test(target) ? target : "ask.html";
    });
  }
}

/* ---------- 7. INIT ---------- */
document.addEventListener("DOMContentLoaded", () => {
  if (typeof applyI18n === "function") applyI18n();
  reflectSelectedLanguage();
  initLanguageSelector();
});
