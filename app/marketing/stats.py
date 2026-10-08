"""Statystyki dzialu marketingu (/posts/stats): oceny pod wieloma katami, modele, tematy, czynniki, wyniki z FB."""
import json
from collections import Counter, defaultdict
from datetime import timedelta, timezone
from statistics import mean

from sqlmodel import Session, select

from app.marketing import metrics
from app.marketing.review import CRITERIA, latest_reviews, text_hash
from app.marketing.topics import KINDS
from app.models import FbPost


def _avg(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 1) if xs else None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = mean(xs), mean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if not sx or not sy:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 2)


def engagement(p: FbPost) -> int | None:
    """Wazone zaangazowanie: reakcje + 2x komentarze + 3x udostepnienia (komentarz i udostepnienie sa cenniejsze)."""
    if p.reactions is None and p.comments is None and p.shares is None:
        return None
    return (p.reactions or 0) + 2 * (p.comments or 0) + 3 * (p.shares or 0)


def _group(rows, key, label=lambda k: k):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = []
    for k, rs in groups.items():
        out.append({
            "key": k, "label": label(k), "n": len(rs),
            "overall": _avg(r["overall"] for r in rs), "expert": _avg(r["expert"] for r in rs),
            "code": _avg(r["code"] for r in rs),
            "publikuj": round(100 * sum(r["verdict"] == "publikuj" for r in rs) / len(rs)),
            "gwara": _avg(r["m"].get("gwara_proc") for r in rs), "fog": _avg(r["m"].get("fog") for r in rs),
            "published": sum(r["status"] == "published" for r in rs),
            "eng": _avg(r["eng"] for r in rs),
        })
    return sorted(out, key=lambda g: -(g["overall"] or 0))


def _bucket(rows, name: str, fn, order: list[str]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        b = fn(r)
        if b is not None:
            groups[b].append(r["overall"])
    items = [{"label": b, "n": len(groups[b]), "avg": _avg(groups[b])} for b in order if groups.get(b)]
    best = max((i for i in items if i["n"] >= 2), key=lambda i: i["avg"], default=None)
    return {"name": name, "items": items, "best": best["label"] if best else None}


def build(s: Session, tz=timezone.utc) -> dict:
    posts = s.exec(select(FbPost)).all()
    status = Counter(p.status for p in posts)
    live = [p for p in posts if p.status != "rejected"]
    latest = latest_reviews(s, [p.id for p in posts])
    rows = []
    for p in live:
        r = latest.get(p.id)
        if not r:
            continue
        rows.append({
            "id": p.id, "kind": p.kind or "-", "author": p.model or "-", "reviewer": r.model or "tylko kod",
            "overall": r.overall, "expert": r.expert_score, "code": r.code_score, "verdict": r.verdict,
            "status": p.status, "stale": r.text_hash != text_hash(p.text),
            "scores": json.loads(r.scores_json or "{}"), "m": json.loads(r.metrics_json or "{}"),
            "rev": json.loads(r.review_json or "{}"), "eng": engagement(p),
            "day": (p.created_at.replace(tzinfo=p.created_at.tzinfo or timezone.utc)).astimezone(tz).date(),
            "hook": (p.body or "")[:110],
        })
    n = len(rows)
    out: dict = {"n_posts": len(live), "n_reviewed": n, "status": dict(status), "rows": rows,
                 "stale": sum(r["stale"] for r in rows), "unreviewed": len(live) - n}
    if not rows:
        return out
    out["avg_overall"] = _avg(r["overall"] for r in rows)
    out["avg_expert"] = _avg(r["expert"] for r in rows)
    out["avg_code"] = _avg(r["code"] for r in rows)
    out["verdicts"] = Counter(r["verdict"] for r in rows)
    out["ready_pct"] = round(100 * out["verdicts"].get("publikuj", 0) / n)

    # kryteria eksperta (1-10) i podwyniki kodu (0-100)
    out["criteria"] = [{"key": k, "label": lab, "weight": w, "desc": d,
                        "avg": _avg(r["scores"].get(k) for r in rows),
                        "n": sum(k in r["scores"] for r in rows)} for k, (lab, w, d) in CRITERIA.items()]
    out["criteria"] = [c for c in out["criteria"] if c["n"]]
    weakest = sorted((c for c in out["criteria"] if c["avg"] is not None), key=lambda c: c["avg"])
    out["weakest"] = weakest[:2]
    out["subscores"] = [{"key": k, "label": lab, "avg": _avg(r["m"].get("podwyniki", {}).get(k) for r in rows)}
                        for k, lab in metrics.SUBSCORE_LABELS.items()]

    # problemy wykryte kodem: jaki % postow je ma
    issues = Counter(i["kod"] for r in rows for i in {x["kod"]: x for x in r["m"].get("problemy", [])}.values())
    out["issues"] = [{"kod": k, "label": metrics.ISSUE_LABELS.get(k, k), "n": c, "pct": round(100 * c / n)}
                     for k, c in issues.most_common()]
    out["risks"] = Counter(x for r in rows for x in r["rev"].get("ryzyka", [])).most_common(6)
    out["fact_errors"] = sum(len(r["rev"].get("bledy_merytoryczne", [])) for r in rows)
    out["dropped_claims"] = sum(r["rev"].get("odrzucone_zarzuty", 0) for r in rows)

    # przekroje
    out["by_kind"] = _group(rows, lambda r: r["kind"], lambda k: KINDS.get(k, k))
    out["by_author"] = _group(rows, lambda r: r["author"])
    reviewers = defaultdict(list)
    for r in rows:
        if r["expert"] is not None:
            reviewers[r["reviewer"]].append(r)
    out["by_reviewer"] = sorted(({"label": k, "n": len(v), "expert": _avg(x["expert"] for x in v),
                                  "code": _avg(x["code"] for x in v),
                                  "gap": _avg(x["expert"] - x["code"] for x in v)}
                                 for k, v in reviewers.items()), key=lambda g: -g["n"])

    # co wplywa na ocene
    def words(r):
        w = r["m"].get("slowa", 0)
        return "do 119" if w < 120 else "120–159" if w < 160 else "160–200" if w <= 200 else "ponad 200"

    def gw(r):
        g = r["m"].get("gwara_proc", 0)
        return "< 3%" if g < 3 else "3–8%" if g < 8 else "8–12%" if g <= 12 else "> 12%"
    out["factors"] = [
        _bucket(rows, "Pytanie do czytelników", lambda r: "jest" if r["m"].get("cta_rozmowa") else "brak",
                ["jest", "brak"]),
        _bucket(rows, "Akapity", lambda r: "2 i więcej" if r["m"].get("akapity", 0) >= 2 else "jeden blok",
                ["2 i więcej", "jeden blok"]),
        _bucket(rows, "Pierwsze zdanie", lambda r: "do 15 słów" if r["m"].get("haczyk_slowa", 99) <= 15
                else "dłuższe", ["do 15 słów", "dłuższe"]),
        _bucket(rows, "Długość", words, ["do 119", "120–159", "160–200", "ponad 200"]),
        _bucket(rows, "Gęstość gwary", gw, ["< 3%", "3–8%", "8–12%", "> 12%"]),
        _bucket(rows, "Czytelność (FOG-PL)", lambda r: "≤ 7" if r["m"].get("fog", 0) <= 7 else "7–10"
                if r["m"].get("fog", 0) <= 10 else "> 10", ["≤ 7", "7–10", "> 10"]),
    ]

    out["factors_any"] = any(len(f["items"]) > 1 for f in out["factors"])

    # rozklad ocen i trend
    hist = [0] * 10
    for r in rows:
        hist[min(9, int(r["overall"] // 10))] += 1
    out["hist"] = [{"label": f"{i * 10}–{i * 10 + 9 if i < 9 else 100}", "n": c} for i, c in enumerate(hist)]
    out["hist_max"] = max(hist) or 1
    days = defaultdict(list)
    for r in rows:
        days[r["day"]].append(r["overall"])
    last = max(days)
    out["trend"] = [{"day": d.strftime("%d.%m"), "avg": _avg(days[d]), "n": len(days[d])}
                    for d in sorted(days) if d >= last - timedelta(days=30)]

    # ranking
    ranked = sorted(rows, key=lambda r: -r["overall"])
    out["top"] = ranked[:5]
    out["bottom"] = [r for r in ranked[::-1] if r not in out["top"]][:5]

    # ocena vs rzeczywistosc (wyniki wpisane po publikacji)
    pub = [r for r in rows if r["eng"] is not None]
    out["n_results"] = len(pub)
    out["corr_overall"] = _pearson([r["overall"] for r in pub], [r["eng"] for r in pub])
    out["corr_code"] = _pearson([r["code"] for r in pub], [r["eng"] for r in pub])
    pred = defaultdict(list)
    for r in pub:
        pred[r["rev"].get("przewidywane_zaangazowanie") or "brak prognozy"].append(r["eng"])
    out["pred_vs_real"] = [{"label": k, "n": len(v), "avg": _avg(v)} for k, v in pred.items()]
    return out
