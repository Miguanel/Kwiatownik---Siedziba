#!/usr/bin/env bash
# ============================================================
#  Powrot Siedziby z laptopa na Windows: zatrzymuje Siedzibe na laptopie i pakuje jej dane.
#  Uruchamia to uruchom_lokalnie.ps1 (kopiuje ten plik na laptopa jako ~/oddaj_dane.sh).
#  Wynik: ~/powrot_siedziba.tgz (data/) i ~/powrot_kw2.tgz (dane Kwiatownika2 czekajace na commit).
# ============================================================
set -uo pipefail
cd ~/kwiatownik/KwiatownikSiedziba || { echo "Brak ~/kwiatownik/KwiatownikSiedziba"; exit 1; }
# wstrzymanie zapisuje sie w bazie - po starcie na Windows Siedziba sama przywroci limit zadan
curl -fsS -m 10 -X POST -d akcja=start http://127.0.0.1:8000/agents/aktualizacja >/dev/null 2>&1 || true
docker compose stop || exit 1
tar -czf ~/powrot_siedziba.tgz --exclude=data/_claude_tmp data || { echo "tar data/ nieudany"; exit 1; }
cd ../Kwiatownik2 || exit 1
git -c safe.directory='*' -c core.quotepath=false status --porcelain=v1 --untracked-files=all \
    -- data/plants data/przepisy data/changelog.json | grep -v '^ D\|^D ' | cut -c4- > ~/powrot_kw2.txt || true
tar -czf ~/powrot_kw2.tgz -T ~/powrot_kw2.txt || { echo "tar Kwiatownik2 nieudany"; exit 1; }
echo "Spakowano: Siedziba $(du -h ~/powrot_siedziba.tgz | cut -f1), Kwiatownik2: $(wc -l < ~/powrot_kw2.txt) plikow"
