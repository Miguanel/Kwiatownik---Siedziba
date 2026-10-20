#!/usr/bin/env bash
# ============================================================
#  Bezpieczna aktualizacja Siedziby NA LAPTOPIE (odpowiednik aktualizuj.ps1).
#      bash serwer/aktualizuj.sh               # czekaj do 20 min na koniec dzialajacych zadan
#      bash serwer/aktualizuj.sh 0             # bez czekania
#      bash serwer/aktualizuj.sh anuluj        # cofnij wstrzymanie bez aktualizacji
#  Z Windows: .\laptop.ps1 -Aktualizuj   (najpierw git push kodu Siedziby)
#
#  1. git pull kodu Siedziby (z GitHuba) i strony Kwiatownika2 (dane Siedziby zostaja nietkniete)
#  2. .env uzupelniany o ustawienia serwera (git z koncami linii LF, GPU NVIDIA jesli jest)
#  3. Siedziba wstrzymuje NOWE zadania, czeka na dzialajace, docker compose up -d --build
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ADRES="http://127.0.0.1:8000"
CZEKAJ_MIN="${1:-20}"
SKRYPT_PRZED=$(md5sum serwer/aktualizuj.sh)

if [ "$CZEKAJ_MIN" = "anuluj" ]; then
    curl -fsS -X POST -d akcja=anuluj "$ADRES/agents/aktualizacja"; echo; exit 0
fi
[ -f .env ] || { echo "Brak .env - najpierw .\\przenies_na_laptopa.ps1 z Windows"; exit 1; }

ustaw_env() {  # ustaw_env KLUCZ WARTOSC - tylko gdy klucza jeszcze nie ma w .env
    grep -qE "^$1=" .env || { printf '\n%s=%s\n' "$1" "$2" >> .env; echo ".env: dopisano $1=$2"; }
}

# ---- 1a. kod Siedziby. Niektore sledzone pliki w data/ (np. data/nn/state.json) zmienia sama Siedziba:
#          odkladamy je na czas pull i przywracamy, zeby pull sie nie wywrocil i stan nie przepadl
echo "== git pull: Siedziba"
TMP=$(mktemp -d)
mapfile -t ZMIENIONE < <(git diff --name-only -- data/)
for f in "${ZMIENIONE[@]}"; do
    [ -f "$f" ] || continue
    mkdir -p "$TMP/$(dirname "$f")"; cp -p "$f" "$TMP/$f"; git checkout -q -- "$f"
done
git pull --ff-only || echo "!! git pull nieudany (zmiany kodu zrobione na laptopie?) - buduje z obecnego kodu"
for f in "${ZMIENIONE[@]}"; do [ -f "$TMP/$f" ] && mkdir -p "$(dirname "$f")" && cp -p "$TMP/$f" "$f"; done
rm -rf "$TMP"
chmod +x serwer/*.sh 2>/dev/null || true
# pull przyniosl nowa wersje tego skryptu - dalej ma dzialac juz nowa
if [ "$(md5sum serwer/aktualizuj.sh)" != "$SKRYPT_PRZED" ] && [ -z "${AKTUALIZUJ_PONOWNIE:-}" ]; then
    echo "== nowa wersja aktualizuj.sh - uruchamiam ja"; AKTUALIZUJ_PONOWNIE=1 exec bash serwer/aktualizuj.sh "$@"
fi

# ---- 1b. Kwiatownik2: zmiany strony z GitHuba; niezacommitowane dane Siedziby przechodza dzieki autostash.
#          Git w kontenerze (root) zapisuje w Kwiatownik2/.git, wiec pull tez idzie przez kontener, gdy dziala.
if [ -d ../Kwiatownik2/.git ]; then
    echo "== git pull: Kwiatownik2"
    G=(git -c safe.directory='*' -c core.autocrlf=input -c rebase.autoStash=true
       -c user.name="Siedziba Kwiatownika" -c user.email=siedziba-kwiatownika@users.noreply.github.com)
    if [ -n "$(docker compose ps -q --status running siedziba 2>/dev/null)" ]; then
        docker compose exec -T siedziba "${G[@]}" -C /kwiatownik2 pull --rebase -q
    else
        "${G[@]}" -C ../Kwiatownik2 pull --rebase -q
    fi || echo "!! Kwiatownik2: pull nieudany - sprawdz: cd ~/kwiatownik/Kwiatownik2 && sudo git status"
fi

# ---- 2. ustawienia serwera
ustaw_env DEPLOY_GIT_AUTOCRLF input
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi nvidia; then
    ustaw_env COMPOSE_FILE docker-compose.yml:docker-compose.gpu.yml
fi

# ---- 3. wstrzymanie zadan i przebudowa
DZIALA=1
if R=$(curl -fsS -m 10 -X POST -d akcja=start "$ADRES/agents/aktualizacja" 2>/dev/null); then
    echo "Nowe zadania wstrzymane: $R"
else
    echo "Siedziba nie odpowiada - buduje od razu."; DZIALA=0
fi
if [ "$DZIALA" = 1 ] && [ "$CZEKAJ_MIN" -gt 0 ] 2>/dev/null; then
    KONIEC=$(( $(date +%s) + CZEKAJ_MIN * 60 ))
    while [ "$(date +%s)" -lt "$KONIEC" ]; do
        N=$(curl -fsS -m 10 "$ADRES/agents/praca.json" 2>/dev/null \
            | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("dzialajace") or []))' 2>/dev/null || echo 0)
        [ "$N" = 0 ] && { echo "Wszystkie zadania skonczyly prace."; break; }
        echo "$(date +%T)  dziala zadan: $N"; sleep 30
    done
fi

# obrazy: pobieramy tylko BRAKUJACE (te przyslane z Windows: .\laptop.ps1 -KopiujObrazy zostaja), z ponowieniami -
# przy slabym laczu pobieranie potrafi zerwac sie timeoutem, a docker wznawia od juz pobranych warstw
for OBRAZ in $(docker compose config --images 2>/dev/null | grep / ); do
    docker image inspect "$OBRAZ" >/dev/null 2>&1 && continue
    for PROBA in 1 2 3 4 5 6; do
        docker pull "$OBRAZ" && break
        echo "!! Pobieranie $OBRAZ przerwane (proba $PROBA/6) - ponawiam za 20 s"; sleep 20
    done
done

if ! docker compose up -d --build; then
    echo "Budowa nieudana. Stary kontener nadal dziala - cofnij wstrzymanie: bash serwer/aktualizuj.sh anuluj"
    exit 1
fi
docker image prune -f >/dev/null 2>&1 || true
echo "Gotowe. Po starcie Siedziba sama przywroci limit zadan."
docker compose ps
