"""Ocena szkicow jak specjalista: metryki kodem + recenzja eksperta (LLM, INNY model niz autor posta).

Recenzja jest sprawdzana kodem: oceny 1-10, zarzuty merytoryczne tylko z cytatem, ktory naprawde jest w poscie,
werdykt nie moze byc lagodniejszy niz wynikaja z twardych regul (bledy, bezpieczenstwo, liczby spoza danych)."""
import hashlib
import json
import re

from sqlmodel import Session, col, select

from app.marketing import metrics
from app.marketing.persona import Persona
from app.models import FbPost, FbPostReview

# klucz: (etykieta, waga, co oceniac)
CRITERIA = {
    "haczyk": ("Haczyk", 1.5, "czy pierwsze 1-2 zdania zatrzymują przewijanie (ciekawość, emocja, konkret)"),
    "postac": ("Postać i gwara", 1.2, "spójność Leśnego Dziadygi: lekka, naturalna śląska godka zrozumiała dla "
                                      "każdego, charakter, humor, bez nawiązań do kopalni"),
    "czytelnosc": ("Czytelność", 1.0, "czy da się to płynnie przeczytać na telefonie, mimo gwary"),
    "merytoryka": ("Merytoryka", 2.0, "zgodność z DANYMI ŹRÓDŁOWYMI, brak zmyśleń i przekłamań"),
    "wartosc": ("Wartość dla czytelnika", 1.3, "czego czytelnik się dowie, czy to ciekawe i konkretne"),
    "zaangazowanie": ("Zaangażowanie", 1.3, "czy zachęca do reakcji, komentarza, udostępnienia; pytanie do odbiorców"),
    "marka": ("Spójność z marką", 1.0, "czy buduje Kwiatownik (cyfrowy zielnik), sensowne przejście do linku"),
    "bezpieczenstwo": ("Bezpieczeństwo", 1.5, "brak obietnic leczenia i dawek, ostrzeżenia przy trujących roślinach, "
                       "ryzyko zgłoszenia/ograniczenia zasięgu przez Facebooka"),
    "sezonowosc": ("Pora roku", 0.7, "czy pasuje do obecnej pory roku i jej nastroju"),
}
VERDICTS = ("publikuj", "popraw", "odrzuc")
ENGAGEMENT = ("niskie", "średnie", "wysokie")
CODE_WEIGHT = 0.35   # udzial metryk kodu w ocenie koncowej (reszta: ekspert)


def text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


def system_prompt() -> str:
    crit = "\n".join(f"- {k}: {d}" for k, (_, _, d) in CRITERIA.items())
    return (
        "Jesteś doświadczonym specjalistą od content marketingu na Facebooku (strony tematyczne, edukacja "
        "przyrodnicza i zielarska), redaktorem i fact-checkerem. Oceniasz szkic posta na fanpage Kwiatownika – "
        "cyfrowego zielnika. Posty pisze postać Leśny Dziadyga: stary zielarz mówiący lekką śląską godką (to zamierzone – "
        "nie poprawiaj jej na polszczyznę literacką; oceniaj, czy jest naturalna i zrozumiała bez słownika – "
        "za trudna godka albo nawiązania do kopalni/gruby to minus).\n\n"
        f"Oceń w skali 1–10 (10 = wzorcowo, 5 = przeciętnie, 1 = źle) każde kryterium:\n{crit}\n\n"
        "Bądź surowy i konkretny jak profesjonalny redaktor: oceny 9–10 tylko za rzeczy naprawdę wybitne. "
        "Merytorykę oceniaj WYŁĄCZNIE względem DANYCH ŹRÓDŁOWYCH: wszystko, czego tam nie ma, a brzmi jak fakt "
        "(liczby, właściwości, historia), to potencjalne zmyślenie. Wspomnienia i humor postaci są dozwolone.\n\n"
        "Odpowiedz WYŁĄCZNIE JSON-em:\n"
        "{\"oceny\": {\"haczyk\": 1-10, ...wszystkie kryteria},\n"
        " \"werdykt\": \"publikuj\" | \"popraw\" | \"odrzuc\",\n"
        " \"podsumowanie\": \"2–3 zdania oceny\",\n"
        " \"mocne_strony\": [\"...\"], \"slabe_strony\": [\"...\"],\n"
        " \"sugestie\": [\"konkretna, wykonalna poprawka\"],\n"
        " \"bledy_merytoryczne\": [{\"cytat\": \"dosłowny fragment posta\", \"problem\": \"co jest nie tak\"}],\n"
        " \"ryzyka\": [\"np. obietnica zdrowotna, kontrowersja\"],\n"
        " \"przewidywane_zaangazowanie\": \"niskie\" | \"średnie\" | \"wysokie\", \"uzasadnienie_zaangazowania\": \"...\",\n"
        " \"lepszy_haczyk\": \"propozycja lepszego pierwszego zdania (gwarą)\",\n"
        " \"grupa_docelowa\": \"do kogo ten post najlepiej trafi\"}\n"
        "Listy: najwyżej 5 pozycji, bez powtórzeń. Cytaty w bledy_merytoryczne muszą być dosłownie z posta."
    )


def user_prompt(post: FbPost, m: dict, source: dict | None) -> str:
    parts = [f"RODZAJ POSTA: {post.kind or '-'}; PORA: {post.season or '-'}"]
    if source:
        parts.append(f"POLECENIE DLA AUTORA: {source.get('task', '')}")
        parts.append(f"DANE ŹRÓDŁOWE:\n{source.get('data', '')}")
    else:
        parts.append("DANE ŹRÓDŁOWE: brak (post sprzed zapisywania danych) – merytorykę oceń ostrożnie, "
                     "wskaż tylko ewidentne błędy.")
    issues = "; ".join(f"{metrics.ISSUE_LABELS.get(i['kod'], i['kod'])}: {i['opis']}" for i in m["problemy"]) or "brak"
    parts.append(f"POMIARY KODEM: {m['slowa']} słów, {m['zdania']} zdań, {m['akapity']} akapitów, FOG-PL {m['fog']} "
                 f"({m['fog_opis']}), gwara {m['gwara_proc']}% słów, pytania: {m['pytania']}, emoji: {m['emoji']}. "
                 f"Uwagi automatyczne: {issues}")
    parts.append(f"\nPOST (tak, jak trafi na Facebooka):\n\"\"\"\n{post.text or post.body}\n\"\"\"")
    return "\n".join(parts)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[„”\"'«»…]", "", s or "")).strip().lower()


def _strs(v, limit: int = 5) -> list[str]:
    if isinstance(v, str):
        v = [v]
    out = []
    for x in v or []:
        x = str(x).strip()
        if x and x.lower() not in {o.lower() for o in out}:
            out.append(x[:400])
    return out[:limit]


def parse_review(text: str, post_text: str) -> tuple[dict, dict]:
    """-> (oceny {kryterium: 1-10}, recenzja). ValueError przy braku JSON-a / ocen."""
    raw = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        data = json.loads(raw)
    except ValueError:
        mt = re.search(r"\{.*\}", raw, re.S)
        if not mt:
            raise
        data = json.loads(mt.group(0))
    scores = {}
    for k in CRITERIA:
        v = (data.get("oceny") or {}).get(k)
        try:
            scores[k] = max(1, min(10, int(round(float(v)))))
        except (TypeError, ValueError):
            continue
    if len(scores) < len(CRITERIA) // 2:
        raise ValueError("recenzja bez ocen")
    body = _norm(post_text)
    errors, dropped = [], 0
    for e in data.get("bledy_merytoryczne") or []:
        if not isinstance(e, dict):
            continue
        quote = str(e.get("cytat") or "").strip()
        if quote and len(_norm(quote)) >= 4 and _norm(quote) in body:
            errors.append({"cytat": quote[:300], "problem": str(e.get("problem") or "").strip()[:400]})
        else:
            dropped += 1          # zarzut bez prawdziwego cytatu - recenzent mogl go zmyslic
    verdict = str(data.get("werdykt") or "").strip().lower().replace("ć", "c")
    eng = str(data.get("przewidywane_zaangazowanie") or "").strip().lower().replace("srednie", "średnie")
    review = {
        "werdykt_modelu": verdict if verdict in VERDICTS else None,
        "podsumowanie": str(data.get("podsumowanie") or "").strip()[:800],
        "mocne_strony": _strs(data.get("mocne_strony")),
        "slabe_strony": _strs(data.get("slabe_strony")),
        "sugestie": _strs(data.get("sugestie")),
        "bledy_merytoryczne": errors[:5],
        "odrzucone_zarzuty": dropped,
        "ryzyka": _strs(data.get("ryzyka")),
        "przewidywane_zaangazowanie": eng if eng in ENGAGEMENT else None,
        "uzasadnienie_zaangazowania": str(data.get("uzasadnienie_zaangazowania") or "").strip()[:400],
        "lepszy_haczyk": str(data.get("lepszy_haczyk") or "").strip()[:300],
        "grupa_docelowa": str(data.get("grupa_docelowa") or "").strip()[:200],
    }
    return scores, review


def expert_score(scores: dict) -> float:
    tot = sum(CRITERIA[k][1] for k in scores)
    return round(sum((v - 1) / 9 * 100 * CRITERIA[k][1] for k, v in scores.items()) / tot, 1)


def decide(overall: float, scores: dict, review: dict, m: dict) -> tuple[str, list[str]]:
    """Werdykt koncowy = surowszy z: werdykt modelu, twarde reguly. Zwraca (werdykt, powody)."""
    reasons = []
    rule = "publikuj" if overall >= 72 else "popraw" if overall >= 45 else "odrzuc"
    if overall < 72:
        reasons.append(f"ocena {overall:.0f}/100")
    for k in ("merytoryka", "bezpieczenstwo"):
        if scores.get(k, 10) <= 4:
            reasons.append(f"{CRITERIA[k][0].lower()}: {scores[k]}/10")
            rule = "odrzuc" if scores[k] <= 2 else max(rule, "popraw", key=VERDICTS.index)
    if review.get("bledy_merytoryczne"):
        reasons.append(f"błędy merytoryczne: {len(review['bledy_merytoryczne'])}")
        rule = max(rule, "popraw", key=VERDICTS.index)
    if m.get("liczby_spoza_danych"):
        reasons.append("liczby spoza danych")
        rule = max(rule, "popraw", key=VERDICTS.index)
    model_v = review.get("werdykt_modelu") or rule
    if model_v != rule and VERDICTS.index(model_v) > VERDICTS.index(rule):
        reasons.append(f"ekspert: {model_v}")
    return max(rule, model_v, key=VERDICTS.index), reasons


def _names(post: FbPost) -> list[str]:
    try:
        return [p.get("nazwa_pl") for p in json.loads(post.plants_json or "[]")[:1] if p.get("nazwa_pl")]
    except ValueError:
        return []


async def evaluate(llm, post: FbPost, persona: Persona, log=lambda m: None, label: str = "") -> FbPostReview:
    """Ocena posta (takze niezapisanego - np. wersji roboczej w petli poprawek). Nie zapisuje do bazy."""
    label = label or f"#{post.id}"
    source = json.loads(post.source_json) if post.source_json else None
    m = metrics.compute(post.body or post.text, post.text, (source or {}).get("data") if source else None,
                        _names(post), persona.min_words, persona.max_words, persona.max_emoji)
    scores, review, model = {}, {}, None
    if llm is not None:
        for attempt in (1, 2):
            try:
                res = await llm.complete(user_prompt(post, m, source), system=system_prompt(), json_mode=True,
                                         avoid={post.model} if post.model else None)
                model = f"{res.provider}:{res.model}"
                scores, review = parse_review(res.text, post.text or post.body)
                break
            except ValueError as exc:
                log(f"{label}: recenzja nieczytelna ({exc}), proba {attempt}")
            except Exception as exc:  # brak modeli itp. - zostaje sama ocena kodem
                log(f"{label}: recenzja eksperta nieudana: {type(exc).__name__}: {exc}")
                break
    exp = expert_score(scores) if scores else None
    overall = round(exp * (1 - CODE_WEIGHT) + m["wynik_kodu"] * CODE_WEIGHT, 1) if exp is not None \
        else m["wynik_kodu"]
    verdict, reasons = decide(overall, scores, review, m)
    review["powody_werdyktu"] = reasons
    if exp is None:
        review["tylko_kod"] = True
    log(f"{label}: ocena {overall:.0f}/100 (kod {m['wynik_kodu']:.0f}"
        + (f", ekspert {exp:.0f} - {model}" if exp is not None else ", bez eksperta") + f") -> {verdict}")
    return FbPostReview(post_id=post.id or 0, text_hash=text_hash(post.text), overall=overall,
                        code_score=m["wynik_kodu"], expert_score=exp, verdict=verdict,
                        metrics_json=json.dumps(m, ensure_ascii=False), scores_json=json.dumps(scores),
                        review_json=json.dumps(review, ensure_ascii=False), model=model)


async def review_post(engine, llm, post_id: int, persona: Persona, log=lambda m: None) -> int | None:
    with Session(engine) as s:
        post = s.get(FbPost, post_id)
        if not post or not (post.body or post.text):
            return None
        s.expunge(post)
    r = await evaluate(llm, post, persona, log)
    with Session(engine) as s:
        s.add(r)
        s.commit()
        return r.id


def improvement_notes(rev: FbPostReview | None) -> list[str]:
    """Raport -> lista uwag do wdrozenia, od najwazniejszych: bledy i ryzyka, najslabsze kryteria, sugestie,
    pomiary kodem, lepszy poczatek."""
    if not rev:
        return []
    r = json.loads(rev.review_json or "{}")
    mt = json.loads(rev.metrics_json or "{}")
    scores = json.loads(rev.scores_json or "{}")
    notes = [f"- [BŁĄD] „{e['cytat']}” – {e['problem']}" for e in r.get("bledy_merytoryczne", [])]
    notes += [f"- [RYZYKO] {x}" for x in r.get("ryzyka", [])]
    for k, v in sorted(scores.items(), key=lambda kv: kv[1]):
        if k in CRITERIA and v < 8:
            lab, _, desc = CRITERIA[k]
            notes.append(f"- [{lab.upper()} {v}/10 → cel min. 8] {desc}")
    notes += [f"- [SUGESTIA] {x}" for x in r.get("sugestie", [])]
    notes += [f"- [SŁABOŚĆ] {x}" for x in r.get("slabe_strony", [])]
    notes += [f"- [POMIAR] {i['opis']}" for i in mt.get("problemy", [])]
    if r.get("lepszy_haczyk"):
        notes.append(f"- [POCZĄTEK] np.: {r['lepszy_haczyk']}")
    if r.get("mocne_strony"):
        notes.append("- [ZACHOWAJ] " + "; ".join(r["mocne_strony"]))
    return notes


def latest_reviews(s: Session, post_ids: list[int] | None = None) -> dict[int, FbPostReview]:
    stmt = select(FbPostReview).order_by(col(FbPostReview.id))
    if post_ids is not None:
        stmt = stmt.where(col(FbPostReview.post_id).in_(post_ids))
    out: dict[int, FbPostReview] = {}
    for r in s.exec(stmt).all():
        out[r.post_id] = r
    return out


def needs_review(s: Session) -> list[int]:
    """Posty bez oceny albo z oceną starszej wersji tekstu (pomija odrzucone)."""
    posts = s.exec(select(FbPost).where(FbPost.status != "rejected")).all()
    latest = latest_reviews(s, [p.id for p in posts])
    return [p.id for p in posts if (p.body or p.text) and (p.id not in latest
                                                           or latest[p.id].text_hash != text_hash(p.text))]
