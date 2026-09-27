#!/usr/bin/env python3
"""OpenAI relay: send a question to the OpenAI API and manage a multi-turn log.

Usage:
    python gpt_relay.py ask "你的问题"          # 新对话,发问并打印回复
    python gpt_relay.py continue "追加/反馈"    # 继续同一会话(执行结果回传)
    python gpt_relay.py history                 # 查看当前会话记录
    python gpt_relay.py reset                   # 清空会话

Key is read from one of (in order):
    1. env OPENAI_API_KEY
    2. $HOME/.openai_key
    3. ./openai_key  (in the tiger并行加速优化 directory)

The key is never printed to stdout or written into the session log.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import urllib.request

MODEL = os.environ.get("GPT_MODEL", "gpt-4o")
API = "https://api.openai.com/v1/chat/completions"
SESSION = Path(__file__).resolve().parent / ".gpt_relay_session.json"

HERE = Path(__file__).resolve().parent


def load_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    for path in (Path.home() / ".openai_key", HERE / "openai_key"):
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    raise SystemExit("未找到 API key。请 `export OPENAI_API_KEY=...` 或写入 ~/.openai_key")


def load_session() -> List[Dict[str, str]]:
    if SESSION.exists():
        return json.loads(SESSION.read_text(encoding="utf-8"))
    return []


def save_session(messages: List[Dict[str, str]]) -> None:
    SESSION.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


def call_openai(messages: List[Dict[str, str]]) -> str:
    key = load_key()
    payload = json.dumps({"model": MODEL, "messages": messages}).encode("utf-8")
    req = urllib.request.Request(
        API,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", "replace")
        raise SystemExit(f"OpenAI API error {err.code}: {body[:500]}") from err


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    if cmd == "reset":
        if SESSION.exists():
            SESSION.unlink()
        print("session cleared")
        return
    if cmd == "history":
        for i, m in enumerate(load_session(), 1):
            role = m["role"]
            text = m["content"]
            head = text[:120].replace("\n", " ")
            print(f"[{i}] {role}: {head}{'...' if len(text) > 120 else ''}")
        return

    text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else sys.stdin.read()
    if not text.strip():
        raise SystemExit("no question text provided")

    messages = load_session() if cmd == "continue" else []
    messages.append({"role": "user", "content": text.strip()})
    reply = call_openai(messages)
    messages.append({"role": "assistant", "content": reply})
    save_session(messages)
    print(reply)


if __name__ == "__main__":
    main()
