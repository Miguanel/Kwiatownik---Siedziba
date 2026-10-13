"""Stan sieci: tryb kazdego zadania, aktywna wersja i sciezki artefaktow (DATA_DIR/nn)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from app.config import settings

MODES = ("wylaczona", "cien", "aktywna")
MODE_LABELS = {
    "wylaczona": "wylaczona - siec nie pracuje",
    "cien": "cien - siec odpowiada rownolegle z LLM, decyduje LLM (pomiar zgodnosci)",
    "aktywna": "aktywna - siec decyduje sama, gdy jest pewna; reszta idzie do LLM",
}
MODE_LABELS_CHECK = {
    "wylaczona": "wylaczona - siec nie sprawdza odpowiedzi LLM",
    "cien": "cien - siec porownuje swoja odpowiedz z LLM, nic nie zmienia (pomiar zgodnosci)",
    "aktywna": "aktywna - siec uzupelnia brakujace pola, gdy jest pewna, a rozbieznosci zglasza do rozstrzygniecia",
}
_lock = threading.Lock()


def root() -> Path:
    return settings.nn_root


def model_dir(task: str, version: int) -> Path:
    return root() / "models" / task / f"v{version}"


def _state_path() -> Path:
    return root() / "state.json"


def load_state() -> dict:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(data: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def task_state(task: str) -> dict:
    st = load_state().get("tasks", {}).get(task) or {}
    return {"mode": st.get("mode", "wylaczona"), "version": st.get("version")}


def set_task_state(task: str, **fields) -> dict:
    with _lock:
        data = load_state()
        tasks = data.setdefault("tasks", {})
        cur = tasks.setdefault(task, {"mode": "wylaczona", "version": None})
        for k, v in fields.items():
            if k == "mode" and v not in MODES:
                raise ValueError(f"nieznany tryb: {v}")
            cur[k] = v
        save_state(data)
        return dict(cur)
