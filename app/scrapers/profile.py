"""Profile scraperow budowane przez LLM.

LLM NIE pisze kodu - opisuje strone deklaratywnie (regex adresow przepisow + selektory CSS pol).
Profil jest sprawdzany na prawdziwych stronach; jesli nie przejdzie walidacji, LLM dostaje liste
bledow i poprawia go (max kilka rund). Gotowy profil pozwala potem pobierac przepisy bez LLM.
"""
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString

from app.scrapers.patterns import PatternStat
from app.scrapers.recipe_data import _clean, is_complete, normalize_jsonld_recipe

log = logging.getLogger(__name__)

FIELDS = ("title", "description", "ingredients", "steps", "image", "yield", "total_time", "notes", "content")
ARTICLE_MIN_CHARS = 300   # tryb artykulu: przepis opisany proza, bez osobnych list skladnikow i krokow
FACT_MIN_CHARS = 400      # ciekawostka z profilu: minimalna dlugosc tresci artykulu
MANY_FIELDS = ("ingredients", "steps")


@dataclass
class ScraperProfile:
    recipe_url_regex: str
    listing_url_regex: str | None = None
    fact_url_regex: str | None = None                # artykuly z ciekawostkami (tytul + content), bez LLM
    fields: dict = field(default_factory=dict)       # {"ingredients": {"css": "...", "many": true, "attr": null}}
    prefer_jsonld: bool = True
    built_with: str = ""
    built_at: str = ""
    validation: dict = field(default_factory=dict)
    stats: dict = field(default_factory=lambda: {"used": 0, "failed": 0})

    # ------------------------------------------------------------ dopasowanie adresow
    def _match(self, regex: str | None, url: str) -> bool:
        if not regex:
            return False
        try:
            return re.search(regex, urlparse(url).path) is not None
        except re.error:
            return False

    def is_recipe_url(self, url: str) -> bool:
        return self._match(self.recipe_url_regex, url)

    def is_listing_url(self, url: str) -> bool:
        return self._match(self.listing_url_regex, url)

    def is_fact_url(self, url: str) -> bool:
        return self._match(self.fact_url_regex, url) and not self.is_recipe_url(url)

    def extract_fact(self, html: str) -> dict | None:
        """Ciekawostka: tytul + tresc artykulu z selektorow profilu (bez LLM)."""
        if not self.fields.get("content"):
            return None
        data = extract_css(BeautifulSoup(html, "lxml"), {k: self.fields.get(k) for k in ("title", "content")})
        if data.get("title") and len(data.get("content") or "") >= FACT_MIN_CHARS:
            return {"title": data["title"], "content": data["content"]}
        return None

    # ------------------------------------------------------------ ekstrakcja
    def extract(self, html: str, jsonld: list[dict] | None = None) -> dict | None:
        if self.prefer_jsonld and jsonld:
            for node in jsonld:
                t = node.get("@type")
                if "Recipe" in (t if isinstance(t, list) else [t]):
                    data = normalize_jsonld_recipe(node)
                    if is_complete(data):
                        return data
        if not self.fields:
            return None
        data = extract_css(BeautifulSoup(html, "lxml"), self.fields)
        if is_complete(data):
            return data
        # tryb artykulu (blogi zielarskie, Karpaty...): tytul + tresc artykulu; przepis wyciagnie LLM przy tlumaczeniu
        content = data.get("content") or ""
        if data.get("title") and len(content) >= ARTICLE_MIN_CHARS:
            return {"title": data["title"], "content": content, "article": True,
                    "ingredients": [], "steps": [], "description": data.get("description")}
        return None

    # ------------------------------------------------------------ zapis
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str | None) -> "ScraperProfile | None":
        if not raw:
            return None
        try:
            d = json.loads(raw)
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        except (ValueError, TypeError):
            return None

    @property
    def health(self) -> float:
        used, failed = self.stats.get("used", 0), self.stats.get("failed", 0)
        return 1.0 if used + failed == 0 else used / (used + failed)


def extract_css(soup: BeautifulSoup, fields: dict) -> dict:
    out: dict = {}
    for name in FIELDS:
        spec = fields.get(name)
        if not spec or not spec.get("css"):
            out[name] = [] if name in MANY_FIELDS else None
            continue
        try:
            els = soup.select(spec["css"])
        except Exception:  # niepoprawny selektor
            els = []
        attr = spec.get("attr")

        def value(el):
            return _clean(el.get(attr)) if attr else _clean(el.get_text(" "))

        if name == "content":
            lines = [_clean(line) for e in els for line in e.get_text("\n").splitlines()]
            out[name] = "\n".join(x for x in lines if x) or None
            continue
        if name in MANY_FIELDS or spec.get("many"):
            vals = [v for v in (value(e) for e in els) if v]
            out[name] = vals if name in MANY_FIELDS else " ".join(vals) or None
        else:
            out[name] = next((v for v in (value(e) for e in els) if v), None)
    return out


# ================================================================ walidacja
def regex_from_patterns(stats: dict[str, PatternStat]) -> str | None:
    """Z wyuczonych wzorcow (/recipes/{slug}) robi regex dla adresow przepisow."""
    good = [p for p, s in stats.items() if s.hits >= 2 and s.hits / (s.hits + s.misses) >= 0.6]
    if not good:
        return None
    parts = []
    for p in good:
        rx = re.escape(p.rstrip("/"))
        rx = rx.replace(r"\{slug\}", "[^/]+").replace(r"\{id\}", "[^/]+").replace(r"\{n\}", r"\d+")
        parts.append(rx)
    return "^(?:" + "|".join(sorted(parts)) + ")/?$"


def score_regex(regex: str | None, recipe_urls: list[str], other_urls: list[str]) -> dict:
    if not regex:
        return {"ok": False, "error": "brak regexu", "precision": 0, "recall": 0, "f1": 0}
    try:
        rx = re.compile(regex)
    except re.error as exc:
        return {"ok": False, "error": f"niepoprawny regex: {exc}", "precision": 0, "recall": 0, "f1": 0}
    tp = sum(1 for u in recipe_urls if rx.search(urlparse(u).path))
    fp = [u for u in other_urls if rx.search(urlparse(u).path)]
    recall = tp / len(recipe_urls) if recipe_urls else 0
    precision = tp / (tp + len(fp)) if tp + len(fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    missed = [u for u in recipe_urls if not rx.search(urlparse(u).path)]
    return {"ok": True, "precision": round(precision, 2), "recall": round(recall, 2), "f1": round(f1, 2),
            "false_positives": fp[:5], "missed": missed[:5]}


def validate(profile: ScraperProfile, samples: list[tuple[str, str, list]], recipe_urls: list[str],
             other_urls: list[str]) -> dict:
    """samples: (url, html, jsonld). Zwraca raport + liste bledow do przekazania LLM."""
    rx = score_regex(profile.recipe_url_regex, recipe_urls, other_urls)
    per_page, ok_pages, errors = [], 0, []
    for url, html, jsonld in samples:
        data = profile.extract(html, jsonld)
        css = extract_css(BeautifulSoup(html, "lxml"), profile.fields) if profile.fields else {}
        counts = {k: (len(v) if isinstance(v, list) else int(bool(v))) for k, v in css.items()}
        per_page.append({"url": url, "complete": data is not None, "css_counts": counts})
        if data is not None:
            ok_pages += 1
        else:
            missing = [k for k in ("title", "ingredients", "steps") if not counts.get(k)]
            errors.append(f"{url}: selectors returned nothing for {missing}")
    if not rx["ok"]:
        errors.append(rx["error"])
    else:
        if rx["recall"] < 0.8:
            errors.append(f"recipe_url_regex misses recipe URLs, e.g. {rx['missed']}")
        if rx["precision"] < 0.9:
            errors.append(f"recipe_url_regex matches non-recipe URLs, e.g. {rx['false_positives']}")
    coverage = ok_pages / len(samples) if samples else 0
    passed = coverage >= 0.67 and rx["ok"] and rx["recall"] >= 0.6 and (rx["precision"] >= 0.8 or not other_urls)
    return {"passed": passed, "coverage": round(coverage, 2), "regex": rx, "pages": per_page, "errors": errors}


# ================================================================ budowanie przez LLM
def html_skeleton(html: str, max_chars: int = 6000) -> str:
    """Zwiezly szkielet DOM (tagi, id, klasy, krotkie teksty) - tanie wejscie dla LLM."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "svg", "iframe", "link", "meta", "header", "footer", "nav",
                   "form", "button", "input", "select", "aside"]):
        t.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    lines: list[str] = []

    def sig(el) -> str:
        s = el.name
        if el.get("id"):
            s += "#" + el["id"]
        cls = el.get("class") or []
        if cls:
            s += "." + ".".join(cls[:3])
        return s

    def walk(el, depth: int) -> None:
        children = [c for c in el.children if not isinstance(c, NavigableString) and c.name]
        i = 0
        while i < len(children):
            c = children[i]
            same = 1
            while i + same < len(children) and sig(children[i + same]) == sig(c):
                same += 1
            for c2 in children[i:i + min(same, 3)]:
                own = " ".join(s.strip() for s in c2.find_all(string=True, recursive=False) if s.strip())
                txt = f' "{own[:70]}"' if own else ""
                lines.append("  " * min(depth, 10) + sig(c2) + txt)
                walk(c2, depth + 1)
            if same > 3:
                lines.append("  " * min(depth, 10) + f"... (+{same - 3} more {sig(c)})")
            i += same

    walk(root, 0)
    return "\n".join(lines)[:max_chars]


SYSTEM_PROMPT = """You design a web scraper profile for ONE recipe website. You do not write code. \
You return ONLY JSON with CSS selectors (BeautifulSoup/soupsieve syntax) and Python regexes:
{
  "recipe_url_regex": "regex matched with re.search against the URL PATH of recipe pages",
  "listing_url_regex": "regex for category/listing pages that link to recipes, or null",
  "fact_url_regex": "regex for informational ARTICLE pages about plants/herbs (not recipes), or null",
  "fields": {
    "title":       {"css": "...", "attr": null},
    "description": {"css": "...", "attr": null} or null,
    "ingredients": {"css": "... selects EACH ingredient element (e.g. li), not the container", "attr": null},
    "steps":       {"css": "... selects EACH step element", "attr": null},
    "image":       {"css": "...", "attr": "src or content"} or null,
    "yield":       {"css": "...", "attr": null} or null,
    "total_time":  {"css": "...", "attr": null} or null,
    "notes":       {"css": "...", "attr": null} or null,
    "content":     {"css": "... the main ARTICLE BODY element (without menus, sidebars, comments)", "attr": null}
  }
}
Rules: selectors must work on ALL sample pages; prefer stable semantic class names (e.g. from recipe plugins \
like wprm/tasty/mv) over positional selectors like nth-child; the regex must match all known recipe URLs \
and none of the non-recipe URLs. Many herbal sites write recipes as prose inside an article, without separate \
ingredient/step elements: then set "ingredients" and "steps" to null and give a precise "content" selector \
(the recipe text will be extracted from it later)."""


def validate_facts(profile: ScraperProfile, fact_samples: list[tuple[str, str, list]], fact_urls: list[str],
                   non_fact_urls: list[str]) -> dict:
    """Czy profil umie pobierac ciekawostki: regex artykulow + tytul/content na probkach."""
    if not profile.fact_url_regex or not fact_samples:
        return {"ok": False, "reason": "brak regexu artykulow lub probek"}
    rx = score_regex(profile.fact_url_regex, fact_urls, non_fact_urls)
    good = sum(1 for _, html, _ in fact_samples if profile.extract_fact(html))
    ok = rx["ok"] and rx["recall"] >= 0.6 and rx["precision"] >= 0.8 and good >= max(1, round(len(fact_samples) * 0.67))
    return {"ok": ok, "precision": rx.get("precision"), "recall": rx.get("recall"), "pages_ok": good,
            "reason": "" if ok else f"regex P={rx.get('precision')} R={rx.get('recall')}, tresc na {good}/{len(fact_samples)}"}


async def build_profile(llm, samples: list[tuple[str, str, list]], recipe_urls: list[str], other_urls: list[str],
                        patterns: dict[str, PatternStat], rounds: int = 3, log_fn=None,
                        fact_samples: list[tuple[str, str, list]] | None = None,
                        fact_urls: list[str] | None = None) -> tuple[ScraperProfile, dict]:
    """Buduje i waliduje profil. Zwraca (profil, raport); raport['passed'] mowi, czy sie udalo."""
    say = log_fn or (lambda m: None)
    pattern_rx = regex_from_patterns(patterns)
    jsonld_ok = sum(1 for _, _, j in samples if ScraperProfile("x").extract("", j)) if samples else 0
    say(f"Probki: {len(samples)} stron, JSON-LD Recipe na {jsonld_ok}; regex z wzorcow: {pattern_rx or 'brak'}")

    best: tuple[ScraperProfile, dict] | None = None
    base = ScraperProfile(recipe_url_regex=pattern_rx or "", prefer_jsonld=True)
    if pattern_rx:
        rep = validate(base, samples, recipe_urls, other_urls)
        best = (base, rep)
        if rep["passed"] and jsonld_ok == len(samples):
            say("Strona ma komplet danych JSON-LD - profil bez LLM")
            if llm is None:
                return _stamp(base, "wzorce+jsonld", rep)

    if llm is None:
        rep = best[1] if best else {"passed": False, "errors": ["brak LLM i brak wyuczonych wzorcow"]}
        return _stamp(base, "wzorce", rep)

    feedback = ""
    fact_samples, fact_urls = fact_samples or [], fact_urls or []
    non_facts = list(recipe_urls) + [u for u in other_urls if u not in fact_urls]
    sk = "\n\n".join(f"=== SAMPLE PAGE {i + 1}: {u}\n{html_skeleton(h)}" for i, (u, h, _) in enumerate(samples[:2]))
    if fact_samples:
        sk += f"\n\n=== SAMPLE ARTICLE (fact) PAGE: {fact_samples[0][0]}\n{html_skeleton(fact_samples[0][1], 3000)}"
    for rnd in range(1, rounds + 1):
        prompt = (f"KNOWN RECIPE URLS:\n" + "\n".join(recipe_urls[:15]) +
                  ("\n\nKNOWN FACT/ARTICLE URLS:\n" + "\n".join(fact_urls[:10]) if fact_urls else "") +
                  "\n\nKNOWN NON-RECIPE URLS:\n" + "\n".join(other_urls[:15]) +
                  (f"\n\nREGEX LEARNED FROM URL PATTERNS (you may reuse): {pattern_rx}" if pattern_rx else "") +
                  f"\n\n{sk}" + (f"\n\nYOUR PREVIOUS ANSWER FAILED VALIDATION:\n{feedback}\nFix it." if feedback else ""))
        try:
            res = await llm.complete(prompt, system=SYSTEM_PROMPT, json_mode=True)
            data = json.loads(res.text.strip().removeprefix("```json").removesuffix("```"))
        except Exception as exc:
            say(f"Runda {rnd}: LLM nie odpowiedzial poprawnie ({exc})")
            continue
        fields = {k: v for k, v in (data.get("fields") or {}).items() if k in FIELDS and isinstance(v, dict)}
        cand = ScraperProfile(recipe_url_regex=data.get("recipe_url_regex") or "",
                              listing_url_regex=data.get("listing_url_regex"),
                              fact_url_regex=data.get("fact_url_regex") or None, fields=fields, prefer_jsonld=True,
                              built_with=f"{res.provider}/{res.model}")
        # wybierz lepszy regex: od LLM albo z wyuczonych wzorcow
        if pattern_rx and score_regex(pattern_rx, recipe_urls, other_urls)["f1"] > \
                score_regex(cand.recipe_url_regex, recipe_urls, other_urls)["f1"]:
            cand.recipe_url_regex = pattern_rx
        rep = validate(cand, samples, recipe_urls, other_urls)
        say(f"Runda {rnd} ({cand.built_with}): pokrycie {rep['coverage']:.0%}, regex P={rep['regex'].get('precision')} "
            f"R={rep['regex'].get('recall')} -> {'OK' if rep['passed'] else 'poprawki'}")
        if best is None or (rep["passed"], rep["coverage"]) > (best[1]["passed"], best[1]["coverage"]):
            best = (cand, rep)
        if rep["passed"]:
            break
        feedback = "\n".join(rep["errors"][:10]) + "\nSelector counts per page: " + json.dumps(
            [p["css_counts"] for p in rep["pages"]])
    prof, rep = best if best else (base, {"passed": False, "errors": ["brak odpowiedzi LLM"]})
    facts = validate_facts(prof, fact_samples, fact_urls, non_facts)
    rep["facts"] = facts
    if prof.fact_url_regex and not facts["ok"]:
        say(f"Ciekawostki: profil ich nie obsluzy ({facts['reason']}) - beda rozpoznawane zwyklym sposobem")
        prof.fact_url_regex = None
    elif facts["ok"]:
        say(f"Ciekawostki: profil pobiera artykuly bez LLM (regex P={facts['precision']} R={facts['recall']})")
    return _stamp(prof, prof.built_with or "wzorce", rep)


def _stamp(p: ScraperProfile, by: str, rep: dict) -> tuple[ScraperProfile, dict]:
    p.built_with = p.built_with or by
    p.built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p.validation = {k: rep.get(k) for k in ("passed", "coverage", "regex", "errors", "facts")}
    return p, rep
