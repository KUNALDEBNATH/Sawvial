import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "sawvial"


def js(name):
    return (WEB / "js" / name).read_text(encoding="utf-8")


def test_no_demo_question_fallback_in_speech_code():
    chat = js("chat.js")
    assert "fallbackTranscript" not in chat
    assert "res.answer_language" not in chat          # answer_language is the REQUESTED language; text_language is the real one
    assert "demoAnswers" not in js("script.js") and "demoAnswers" not in chat
    # nothing may pick a question at random
    assert not re.search(r"Math\.random\(\)[^;\n]*(Sample|question)", chat, re.I)


def test_js_files_parse():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    for f in (WEB / "js").glob("*.js"):
        r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f.name}: {r.stderr}"


def test_new_i18n_keys_exist_in_all_seven_languages():
    text = js("i18n.js")
    for key in ("chat.micDenied", "chat.micNoSpeech", "chat.micUnsupported", "chat.retry", "chat.noLink", "chat.noteSingle",
                "chat.stageOnline", "chat.stageReading", "chat.noteWeb", "chat.noteDiffering", "chat.kindOfficial", "chat.sourceOpen"):
        block = text[text.index(f'"{key}"'):]
        block = block[:block.index("},")]
        for lang in ("en", "hi", "bn", "ta", "te", "kn", "mr"):
            assert re.search(rf"\b{lang}: \"", block), f"{key} missing {lang}"


def test_real_dom_behaviour():
    """Mic failure submits nothing; speech reaches /api/ask; retry works; links are safe (jsdom)."""
    if not shutil.which("node"):
        pytest.skip("node not installed")
    jsdom = os.environ.get("JSDOM_PATH")
    if not jsdom:
        probe = subprocess.run(["node", "-e", "require('jsdom')"], capture_output=True, cwd=WEB.parent)
        if probe.returncode != 0:
            pytest.skip("jsdom not installed (npm install jsdom, or set JSDOM_PATH)")
    r = subprocess.run(["node", str(WEB.parent / "tests" / "frontend" / "dom_test.js")], capture_output=True, text=True,
                       env=dict(os.environ), timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "13 frontend DOM tests passed" in r.stdout
