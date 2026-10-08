"""Test calego zadania 'scan' z prawdziwa baza SQLite (plik tymczasowy) i sztuczna strona."""
import asyncio

from sqlmodel import Session, SQLModel, create_engine, select

import app.worker.jobs as jobs_mod
import app.worker.scan as scan_mod
from app.models import Item, ItemKind, Job, JobStatus, SitePattern, Source
from tests.test_scraper import make_fetcher


def test_scan_job_saves_items_and_patterns(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(scan_mod, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    monkeypatch.setattr(scan_mod, "Fetcher", lambda *a, **k: make_fetcher())

    with Session(eng) as s:
        src = Source(name="Herbs", base_url="https://herbs.example/", max_pages=30)
        s.add(src)
        s.commit()
        s.refresh(src)
        job = Job(kind="scan", source_id=src.id)
        s.add(job)
        s.commit()
        s.refresh(job)
        sid, jid = src.id, job.id

    async def run():
        runner = jobs_mod.make_runner(2)
        runner.start(jid, lambda: scan_mod.run_scan(jid, sid, runner))
        await asyncio.gather(*runner.tasks.values())

    asyncio.run(run())

    with Session(eng) as s:
        job = s.get(Job, jid)
        assert job.status == JobStatus.done, job.log
        assert len(s.exec(select(Item).where(Item.kind == ItemKind.recipe)).all()) == 3
        assert s.exec(select(SitePattern).where(SitePattern.pattern == "/recipes/{slug}")).one().hits == 3
        assert s.get(Source, sid).last_scan_at is not None
        assert "Koniec" in job.log


def test_same_recipe_from_two_sites_becomes_one_with_two_sources(tmp_path, monkeypatch):
    from app.scrapers.crawler import PageResult

    eng = create_engine(f"sqlite:///{(tmp_path / 'd.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(scan_mod, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        a, b = Source(name="A", base_url="https://a.example/"), Source(name="B", base_url="https://b.example/")
        s.add(a)
        s.add(b)
        s.commit()
        job = Job(kind="scan", source_id=a.id)
        s.add(job)
        s.commit()
        aid, bid, jid = a.id, b.id, job.id

    def result(url, title, ingr):
        return PageResult(url=url, title=title, kind="recipe", confidence=1, method="jsonld", reason="", text="t",
                          structured={"title": title, "ingredients": ingr, "steps": ["x"]}, language="en",
                          pattern="/{slug}")

    runner = jobs_mod.make_runner(1)
    scan_mod.DbStore(jid, aid, runner).save_result(result(
        "https://a.example/elderflower-cordial", "Easy Elderflower Cordial Recipe",
        ["20 elderflower heads", "1 kg sugar", "2 lemons", "1.5 l water"]))
    scan_mod.DbStore(jid, bid, runner).save_result(result(
        "https://b.example/cordial", "Homemade elderflower cordial",
        ["25 elderflower heads", "1kg sugar", "3 lemons", "1.5 litres water"]))
    scan_mod.DbStore(jid, bid, runner).save_result(result(
        "https://b.example/dandelion-syrup", "Dandelion syrup", ["200 dandelion flowers", "1 kg sugar", "1 l water"]))

    with Session(eng) as s:
        items = {i.url: i for i in s.exec(select(Item)).all()}
        primary = items["https://a.example/elderflower-cordial"]
        assert items["https://b.example/cordial"].duplicate_of == primary.id
        assert items["https://b.example/dandelion-syrup"].duplicate_of is None
        assert "[duplikat]" in s.get(Job, jid).log


def test_discovery_store_dedup_and_add_as_source(tmp_path, monkeypatch):
    import app.worker.discover as disc
    from app.discovery.finder import FoundSite
    from app.models import DiscoveredSite

    eng = create_engine(f"sqlite:///{(tmp_path / 'x.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(disc, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        s.add(Source(name="Old", base_url="https://www.known.example/"))
        job = Job(kind="discover:ua")
        s.add(job)
        s.commit()
        jid = job.id

    store = disc.DbDiscoveryStore(jid, jobs_mod.make_runner(1))
    assert "known.example" in store.known_domains()
    site = FoundSite("herbs-ua.example", "ua", "https://herbs-ua.example/chai", "Чай", query="чай", status="verified",
                     score=0.9, language="uk")
    store.save_site(site)
    store.save_site(site)                     # drugi raz - ignorowane
    store.bump_site("herbs-ua.example")
    store.save_query("ua", "чай", 10, 1)
    assert store.used_queries("ua") == {"чай"} and "herbs-ua.example" in store.known_domains()

    with Session(eng) as s:
        row = s.exec(select(DiscoveredSite)).one()
        assert row.hits == 2
        rid = row.id
    src_id = disc.add_site_as_source(rid)
    with Session(eng) as s:
        src = s.get(Source, src_id)
        assert (src.name, src.base_url, src.country, src.language) == (
            "herbs-ua.example", "https://herbs-ua.example/", "ua", "uk")
        assert s.get(DiscoveredSite, rid) is None            # zniknela z listy odkrytych...
    assert "herbs-ua.example" in store.known_domains()      # ...ale nadal jest znana (jako zrodlo)


def test_translate_check_and_export(tmp_path, monkeypatch):
    """raw (EN + DE ten sam przepis, + przepis, ktory juz jest w Kwiatowniku) -> tlumaczenie -> sprawdzenie -> eksport."""
    import json as _json

    import app.worker.export as exp
    from app.config import settings
    from app.models import ItemStatus

    przepisy, plants = tmp_path / "przepisy", tmp_path / "plants"
    przepisy.mkdir()
    plants.mkdir()
    (przepisy / "przepisy_kulinarne.json").write_text(_json.dumps([{"id": "napar_z_mieta", "tytul": "Napar z mięty",
        "skladniki": [{"nazwa": "Liście mięty"}, {"nazwa": "Wrzątek"}]}]), encoding="utf-8")
    (plants / "bez_czarny.json").write_text(_json.dumps({"id": "bez_czarny", "nazwa_pl": "Bez czarny"}), encoding="utf-8")
    monkeypatch.setattr(settings, "kwiatownik_przepisy_dir", przepisy)
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    monkeypatch.setattr(settings, "export_require_approval", False)
    monkeypatch.setattr(settings, "enrich_recipes", False)

    eng = create_engine(f"sqlite:///{(tmp_path / 'e.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(exp, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        a, b = Source(name="EN site", base_url="https://en.example/"), Source(name="DE site", base_url="https://de.example/")
        s.add(a)
        s.add(b)
        s.commit()
        items = [Item(source_id=a.id, url="https://en.example/elder", kind=ItemKind.recipe, title="Elderflower cordial",
                      language="en", raw_text="..."),
                 Item(source_id=b.id, url="https://de.example/holunder", kind=ItemKind.recipe, title="Holundersirup",
                      language="de", raw_text="..."),
                 Item(source_id=a.id, url="https://en.example/mint", kind=ItemKind.recipe, title="Mint tea",
                      language="en", raw_text="...")]
        for it in items:
            s.add(it)
        job = Job(kind="export")
        s.add(job)
        s.commit()
        jid, ids = job.id, [it.id for it in items]

    elder = {"tytul": "Syrop z kwiatów czarnego bzu", "typ": "kulinarno_medyczne_syrop", "roslina_id": "bez_czarny",
             "skladniki": [{"nazwa": "Kwiaty czarnego bzu", "ilosc": "20 baldachów", "link_id": "bez_czarny"},
                           {"nazwa": "Cukier", "ilosc": "1 kg"}, {"nazwa": "Cytryna", "ilosc": "2 szt."}],
             "sposob_przygotowania": ["Zalać kwiaty wodą.", "Dodać cukier."]}
    elder_de = dict(elder, tytul="Syrop z czarnego bzu")
    mint = {"tytul": "Napar z mięty", "skladniki": [{"nazwa": "Liście mięty"}, {"nazwa": "Wrzątek"}],
            "sposob_przygotowania": ["Zalać wrzątkiem."]}

    class LLM:
        answers = [elder, elder_de, mint]

        async def complete(self, prompt, system=None, json_mode=False):
            class R:
                provider, model = "fake", "m"
            r = R()
            r.text = _json.dumps(self.answers.pop(0), ensure_ascii=False)
            return r

    runner = jobs_mod.make_runner(1)
    stats = asyncio.run(exp.run_export(jid, runner, LLM(), translate_first=10))

    with Session(eng) as s:
        en, de, mint_item = (s.get(Item, i) for i in ids)
        assert de.duplicate_of == en.id                         # ten sam przepis z niemieckiej strony -> 2. zrodlo
        assert mint_item.status == ItemStatus.exists            # juz jest w Kwiatowniku
        assert en.status == ItemStatus.exported
    files = sorted(przepisy.glob("siedziba_przepisy_*.json"))  # eksport do pliku z data (<seria>_<RRRR-MM-DD>.json)
    assert len(files) == 1
    out = _json.loads(files[0].read_text(encoding="utf-8"))
    assert len(out) == 1 and out[0]["tytul"] == "Syrop z kwiatów czarnego bzu" and out[0]["slug"] == "bez_czarny"
    assert [z["url"] for z in out[0]["zrodla"]] == ["https://en.example/elder", "https://de.example/holunder"]
    assert stats["exported"] == 1


def test_cleanup_discoveries_fixes_domains(tmp_path, monkeypatch):
    import app.worker.discover as disc
    from app.models import DiscoveredSite

    eng = create_engine(f"sqlite:///{(tmp_path / 'c.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(disc, "engine", eng)
    with Session(eng) as s:
        s.add(Source(name="x", base_url="https://agnieszka.example/"))
        s.add(DiscoveredSite(domain="co.ua", country="ua", url="https://drink.co.ua/a", status="verified"))
        s.add(DiscoveredSite(domain="agnieszka.example", country="pl", url="https://agnieszka.example/s", status="added"))
        s.add(DiscoveredSite(domain="katowice.pl", country="pl", url="https://www.herbarium.katowice.pl/x"))
        s.commit()
    assert disc.cleanup_discoveries() == 3
    with Session(eng) as s:
        assert sorted(r.domain for r in s.exec(select(DiscoveredSite)).all()) == ["drink.co.ua", "herbarium.katowice.pl"]


def test_retry_job_reuses_params_and_resumes_scan(tmp_path, monkeypatch):
    import types

    import app.worker.actions as act

    eng = create_engine(f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    for mod in (act, jobs_mod, scan_mod):
        monkeypatch.setattr(mod, "engine", eng)
    calls = []

    async def fake_scan(jid, sid, runner, llm, max_pages, resume_from):
        calls.append((sid, max_pages, resume_from))

    monkeypatch.setattr(act, "run_scan", fake_scan)
    with Session(eng) as s:
        src = Source(name="H", base_url="https://h.example/", max_pages=40)
        s.add(src)
        s.commit()
        old = Job(kind="scan", source_id=src.id, status=JobStatus.failed, params_json='{"max_pages": 25}',
                  state_json='{"frontier": [{"url": "https://h.example/a"}]}')
        s.add(old)
        s.commit()
        oid, sid = old.id, src.id

    async def main():
        state = types.SimpleNamespace(jobs=jobs_mod.make_runner(1), llm=None)
        new = act.retry_job(state, oid)
        await asyncio.gather(*state.jobs.tasks.values())
        return new

    new = asyncio.run(main())
    assert calls == [(sid, 25, oid)]                      # te same parametry + kolejka ze starego zadania
    with Session(eng) as s:
        j = s.get(Job, new)
        assert j.retry_of == oid and j.status == JobStatus.done
        assert jobs_mod.retry_depth(new) == 1


def test_suggested_tasks_groups_and_bulk_actions(tmp_path, monkeypatch):
    import types

    import app.worker.actions as act
    import app.worker.discover as disc
    import app.worker.suggestions as sug
    from app.models import DiscoveredSite, ItemStatus

    eng = create_engine(f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    for mod in (sug, act, disc, jobs_mod, scan_mod):
        monkeypatch.setattr(mod, "engine", eng)
    started = []

    async def fake_scan(jid, sid, runner, llm, max_pages, resume_from):
        started.append(sid)

    monkeypatch.setattr(act, "run_scan", fake_scan)
    with Session(eng) as s:
        a, b = Source(name="A", base_url="https://a.example/"), Source(name="B", base_url="https://b.example/")
        s.add(a)
        s.add(b)
        s.commit()
        s.add(Item(source_id=a.id, url="https://a.example/1", kind=ItemKind.recipe, title="R1"))
        s.add(Item(source_id=a.id, url="https://a.example/2", kind=ItemKind.recipe, title="R2", status=ItemStatus.translated,
                   data_json='{"tytul": "Przepis 2"}'))
        s.add(DiscoveredSite(domain="c.example", country="pl", url="https://c.example/x", status="verified", score=0.9))
        s.commit()
        aid, bid = a.id, b.id

    groups = {g.key: g for g in sug.build_groups()}
    assert {i.id for i in groups["scan_new"].items} == {str(aid), str(bid)}
    assert groups["translate"].total == 1 and groups["approve"].items[0].label == "Przepis 2"
    assert groups["build_profile"].total == 1          # zrodlo A ma 2 przepisy i brak profilu
    assert groups["add_verified"].items[0].label == "c.example"
    assert sorted(groups, key=lambda k: (groups[k].total == 0, groups[k].priority))[0] != "retry"

    async def main():
        state = types.SimpleNamespace(jobs=jobs_mod.make_runner(2), llm=None)
        msg = sug.run_task(state, "scan_new", "scan", [str(aid), str(bid)])
        await asyncio.gather(*state.jobs.tasks.values())
        return msg

    assert "2 skanow" in asyncio.run(main()) and sorted(started) == sorted([aid, bid])
    approve_id = groups["approve"].items[0].id
    sug.run_task(None, "approve", "approve", [approve_id])
    sug.run_task(None, "add_verified", "add_source", [groups["add_verified"].items[0].id])
    with Session(eng) as s:
        assert s.get(Item, int(approve_id)).status == ItemStatus.approved
        assert s.exec(select(DiscoveredSite)).all() == []
        assert any(src.name == "c.example" for src in s.exec(select(Source)).all())
    groups2 = {g.key: g for g in sug.build_groups()}
    assert groups2["export"].total == 1 and groups2["scan_new"].total == 3   # nowe zrodlo c.example + A, B


def test_monitor_live_context(tmp_path, monkeypatch):
    import types
    from datetime import datetime, timedelta, timezone

    import app.web.monitor_routes as mon
    from app.worker import activity

    eng = create_engine(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(mon, "engine", eng)
    now = datetime.now(timezone.utc)
    with Session(eng) as s:
        src = Source(name="Herbs", base_url="https://h.example/")
        s.add(src)
        s.commit()
        run = Job(kind="scan", source_id=src.id, status=JobStatus.running, progress=10, total=40,
                  started_at=now - timedelta(seconds=60), log="12:00:01 Start\n12:00:05 [recipe 1.00 jsonld] Zupa\n")
        q = Job(kind="translate", status=JobStatus.queued)
        done = Job(kind="export", status=JobStatus.done, started_at=now - timedelta(minutes=2),
                   finished_at=now - timedelta(minutes=1))
        for j in (run, q, done):
            s.add(j)
        s.commit()
        rid = run.id
    activity.set(rid, "pobieram strone 11/40", "https://h.example/x")

    req = types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace(
        llm=None, jobs=types.SimpleNamespace(max_concurrent=3))))
    ctx = mon._ctx(req)
    card = ctx["cards"][0]
    assert card["name"] == "Herbs" and card["pct"] == 25 and card["act"]["step"].startswith("pobieram")
    assert card["eta"] != "-"                                   # 60 s na 10 stron -> ok. 3 min na reszte
    assert [j.kind for j in ctx["queued"]] == ["translate"] and ctx["done_h"] == 1
    assert ctx["feed"][0]["text"].startswith("[recipe")
    activity.clear(rid)


def test_translate_skips_culinary_and_enriches(tmp_path, monkeypatch):
    """Przepis kulinarny -> pominiety; leczniczy -> przetlumaczony + opracowanie (Wu Xing) bez wymyslonego dawkowania."""
    import json as _json

    import app.worker.export as exp
    from app.config import settings
    from app.models import ItemStatus

    przepisy, plants = tmp_path / "przepisy", tmp_path / "plants"
    przepisy.mkdir()
    plants.mkdir()
    monkeypatch.setattr(settings, "kwiatownik_przepisy_dir", przepisy)
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    monkeypatch.setattr(settings, "skip_culinary", True)
    monkeypatch.setattr(settings, "enrich_recipes", True)
    monkeypatch.setattr(settings, "enrich_systems", "wu_xing,piec_filarow")    # piec_filarow juz nie istnieje
    eng = create_engine(f"sqlite:///{(tmp_path / 'c.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(exp, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        src = Source(name="S", base_url="https://s.example/")
        s.add(src)
        s.commit()
        cake = Item(source_id=src.id, url="https://s.example/cake", kind=ItemKind.recipe, title="Cake", raw_text="...")
        salve = Item(source_id=src.id, url="https://s.example/salve", kind=ItemKind.recipe, title="Calendula salve",
                     raw_text="...")
        s.add(cake)
        s.add(salve)
        job = Job(kind="translate")
        s.add(job)
        s.commit()
        jid, cake_id, salve_id = job.id, cake.id, salve.id

    answers = [
        {"tytul": "Ciasto z wiśniami", "typ": "kulinarne", "skladniki": [{"nazwa": "Mąka"}],
         "sposob_przygotowania": ["Upiec."]},
        {"tytul": "Maść nagietkowa", "typ": "medyczne_zewnetrzne_masc",
         "skladniki": [{"nazwa": "Kwiaty nagietka"}, {"nazwa": "Wosk pszczeli"}],
         "sposob_przygotowania": ["Macerować kwiaty w oleju.", "Dodać wosk."]},
        {"mechanizm_tworzenia": ["Maceracja tłuszczowa"],
         "klasyfikacja_dzialania": {"skalowanie_toksykologiczne": "Łagodne.",
                                    "matryca_wu_xing": {"zywiol_leczony": "Metal (Skóra)", "porada_matrycy": "..."},
                                    "triada_kampo": {"cel_glowny": "Sui"}},
         "skladniki": [{"nazwa": "Kwiaty nagietka", "filar": "1_bazowy", "tropizm_organowy": "Skóra"},
                       {"nazwa": "Wosk pszczeli", "filar": "9_zly"}],
         "stosowanie_i_dawkowanie": {"dawkowanie_standardowe": "3 razy dziennie", "okolicznosci_stosowania": "Otarcia"},
         "bezpieczenstwo_i_interakcje": {"ostrzezenia": "Uczulenie na astrowate."}},
    ]

    class LLM:
        async def complete(self, prompt, system=None, json_mode=False):
            class R:
                provider, model = "fake", "m"
            r = R()
            r.text = _json.dumps(answers.pop(0), ensure_ascii=False)
            return r

    stats = asyncio.run(exp.run_translate(jid, jobs_mod.make_runner(1), LLM(), item_ids=[cake_id, salve_id]))
    assert stats["culinary"] == 1 and stats["translated"] == 1
    with Session(eng) as s:
        assert s.get(Item, cake_id).status == ItemStatus.skipped
        salve_item = s.get(Item, salve_id)
        rec = _json.loads(salve_item.data_json)
    assert salve_item.status == ItemStatus.translated
    kd = rec["klasyfikacja_dzialania"]
    assert kd["matryca_wu_xing"]["zywiol_leczony"] == "Metal (Skóra)" and "triada_kampo" not in kd  # kampo wylaczone
    assert all("filar" not in s and "tropizm_organowy" not in s for s in rec["skladniki"])   # nie wymyslamy rol
    assert all("link_id" not in s for s in rec["skladniki"])                                  # bez "link_id": null
    assert rec["stosowanie_i_dawkowanie"]["dawkowanie_standardowe"] == "Źródło nie podaje dawkowania."
    assert "nie jest poradą medyczną" in rec["opracowanie"]["uwaga"]


def test_import_kwiatownik1_archive(tmp_path, monkeypatch):
    """Archiwum K1: przepisy lecznicze -> Itemy 'translated' bez LLM; ten, ktory jest w Kwiatowniku2 -> 'exists';
    drugi import niczego nie dubluje."""
    import json as _json

    import app.worker.archive_import as arch
    import app.worker.export as exp
    from app.config import settings
    from app.models import ItemStatus

    k1, przepisy, plants = tmp_path / "k1", tmp_path / "przepisy", tmp_path / "plants"
    for d in (k1, przepisy, plants):
        d.mkdir()
    (k1 / "przepisy_kulinarne_global.json").write_text(_json.dumps({"przepisy": [{"tytul": "Zupa"}]}), encoding="utf-8")
    (k1 / "przepisy_medyczne_global.json").write_text(_json.dumps({"przepisy": [
        {"tytul": "napar z mięty", "skladniki": ["Liście mięty", "Wrzątek"], "sposob_przygotowania": "Zalać."},
        {"tytul": "Maść nagietkowa", "metoda": "maść", "skladniki": ["kwiaty nagietka", "smalec"],
         "sposob_przygotowania": ["Stopić smalec.", "Dodać kwiaty."], "zrodla": ["ESCOP"]},
    ]}), encoding="utf-8")
    (przepisy / "przepisy_medyczne.json").write_text(_json.dumps([{"id": "napar_z_mieta", "tytul": "Napar z mięty",
        "skladniki": [{"nazwa": "Liście mięty"}, {"nazwa": "Wrzątek"}]}]), encoding="utf-8")
    monkeypatch.setattr(settings, "kwiatownik1_przepisy_dir", k1)
    monkeypatch.setattr(settings, "kwiatownik_przepisy_dir", przepisy)
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    eng = create_engine(f"sqlite:///{(tmp_path / 'k.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(exp, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        job = Job(kind="import_k1")
        s.add(job)
        s.commit()
        jid = job.id
    stats = asyncio.run(arch.run_import_k1(jid, jobs_mod.make_runner(1), None))
    assert stats["imported"] == 2 and stats["exists"] == 1 and stats["translated"] == 1
    with Session(eng) as s:
        salve = s.exec(select(Item).where(Item.title == "Maść nagietkowa")).one()
        assert salve.status == ItemStatus.translated
        rec = _json.loads(salve.data_json)
        assert rec["typ"] == "medyczne_zewnetrzne_masc" and rec["zrodla"] == ["ESCOP"]
    again = asyncio.run(arch.run_import_k1(jid, jobs_mod.make_runner(1), None))
    assert again["known"] == 2 and again["imported"] == 0


def test_recheck_filters_and_article_split(tmp_path, monkeypatch):
    """Stare przepisy kulinarne -> pominiete; artykul z 2 przepisami -> 2 pozycje; lista linkow -> pominieta."""
    import json as _json

    import app.worker.export as exp
    from app.config import settings
    from app.models import ItemStatus

    przepisy, plants = tmp_path / "przepisy", tmp_path / "plants"
    przepisy.mkdir()
    plants.mkdir()
    monkeypatch.setattr(settings, "kwiatownik_przepisy_dir", przepisy)
    monkeypatch.setattr(settings, "kwiatownik_plants_dir", plants)
    monkeypatch.setattr(settings, "skip_culinary", True)
    monkeypatch.setattr(settings, "enrich_recipes", False)
    eng = create_engine(f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(exp, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    with Session(eng) as s:
        src = Source(name="S", base_url="https://s.example/")
        s.add(src)
        s.commit()
        old = Item(source_id=src.id, url="https://s.example/salmon", kind=ItemKind.recipe, status=ItemStatus.approved,
                   title="Salmon", data_json=_json.dumps({"tytul": "Łosoś", "typ": "kulinarne", "skladniki": []}))
        cake = Item(source_id=src.id, url="https://s.example/cake", kind=ItemKind.recipe, title="Cherry cake",
                    raw_text="Bake the cake in the oven, dessert with cherries, cake dough. " * 5)
        art = Item(source_id=src.id, url="https://s.example/teas", kind=ItemKind.recipe, title="Teas", raw_text="...")
        lst = Item(source_id=src.id, url="https://s.example/list", kind=ItemKind.recipe, title="List", raw_text="...")
        for it in (old, cake, art, lst):
            s.add(it)
        job = Job(kind="translate")
        s.add(job)
        s.commit()
        ids = (old.id, cake.id, art.id, lst.id)
        jid = job.id
    assert exp.recheck_filters() == 2
    with Session(eng) as s:
        assert all(s.get(Item, i).status == ItemStatus.skipped for i in ids[:2])

    tea = {"tytul": "Napar z rumianku", "typ": "medyczne_wewnetrzne_napar", "skladniki": [{"nazwa": "Rumianek"}],
           "sposob_przygotowania": ["Zalać."],
           "dodatkowe_przepisy": [{"tytul": "Napar z mięty", "typ": "medyczne_wewnetrzne_napar",
                                   "skladniki": [{"nazwa": "Mięta"}], "sposob_przygotowania": ["Zalać."]}]}
    answers = [tea, {"brak_przepisu": True}]

    class LLM:
        async def complete(self, prompt, system=None, json_mode=False):
            class R:
                provider, model = "fake", "m"
            r = R()
            r.text = _json.dumps(answers.pop(0), ensure_ascii=False)
            return r

    stats = asyncio.run(exp.run_translate(jid, jobs_mod.make_runner(1), LLM(), item_ids=[ids[2], ids[3]]))
    assert stats["translated"] == 1 and stats["no_recipe"] == 1
    with Session(eng) as s:
        child = s.exec(select(Item).where(Item.url == "https://s.example/teas#przepis-2")).one()
        assert child.status == ItemStatus.translated and _json.loads(child.data_json)["tytul"] == "Napar z mięty"
        assert s.get(Item, ids[3]).status == ItemStatus.skipped


def test_phased_scan_probe_profile_harvest(tmp_path, monkeypatch):
    """Skan etapowy: 1) rozpoznanie do 2 przepisow, 2) LLM buduje profil, 3) pobieranie z profilem."""
    import json as _json

    from app.config import settings

    eng = create_engine(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    SQLModel.metadata.create_all(eng)
    monkeypatch.setattr(scan_mod, "engine", eng)
    monkeypatch.setattr(jobs_mod, "engine", eng)
    monkeypatch.setattr(scan_mod, "Fetcher", lambda *a, **k: make_fetcher())
    monkeypatch.setattr(settings, "phased_scan", True)
    monkeypatch.setattr(settings, "probe_recipes", 2)
    monkeypatch.setattr(settings, "probe_facts", 0)

    class LLM:
        profile_calls = 0

        async def complete(self, prompt, system=None, json_mode=False):
            class R:
                provider, model = "fake", "m"
            r = R()
            if "scraper profile" in (system or ""):
                LLM.profile_calls += 1
                r.text = _json.dumps({"recipe_url_regex": "^/recipes/[^/]+/$", "listing_url_regex": "^/recipes/$",
                                      "fields": {"title": {"css": "h1"}, "ingredients": {"css": "article ul li"},
                                                 "steps": {"css": "article ol li"}}})
            else:
                r.text = _json.dumps({"type": "other", "confidence": 0.9, "title": "x", "reason": "unrelated"})
            return r

    with Session(eng) as s:
        src = Source(name="Herbs", base_url="https://herbs.example/", max_pages=30)
        s.add(src)
        s.commit()
        job = Job(kind="scan", source_id=src.id)
        s.add(job)
        s.commit()
        sid, jid = src.id, job.id

    async def run():
        runner = jobs_mod.make_runner(2)
        runner.start(jid, lambda: scan_mod.run_scan(jid, sid, runner, LLM()))
        await asyncio.gather(*runner.tasks.values())

    asyncio.run(run())
    with Session(eng) as s:
        job = s.get(Job, jid)
        assert job.status == JobStatus.done, job.log
        assert "Etap 1/3" in job.log and "Rozpoznanie zakonczone" in job.log
        assert "Etap 2/3" in job.log and "Etap 3/3 - pobieranie z profilem" in job.log
        assert s.get(Source, sid).profile_status == "ready" and LLM.profile_calls >= 1
        assert len(s.exec(select(Item).where(Item.kind == ItemKind.recipe)).all()) == 3
        assert _json.loads(job.stats_json)["profile_ok"] is True
