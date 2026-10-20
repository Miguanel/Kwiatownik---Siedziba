# ============================================================
#  Przeniesienie Siedziby z tego komputera na laptopa z Ubuntu Server (jednorazowo).
#  Uruchom w folderze KwiatownikSiedziba:   .\przenies_na_laptopa.ps1
#  Wczesniej raz:  .\laptop.ps1 -Konfiguruj -Adres uzytkownik@ip_laptopa   (klucz SSH + alias "siedziba")
#
#  1. sprawdza, czy kod Siedziby jest wypchniety na GitHuba (laptop klonuje go stamtad)
#  2. przygotowuje laptopa: serwer/instaluj.sh (Docker, klapa/usypianie, zapora, GPU, klony repozytoriow)
#  3. wstrzymuje Siedzibe tutaj (czeka na dzialajace zadania) i zatrzymuje kontenery
#  4. kopiuje na laptopa: .env, data\ (bazy, cache LLM, sieci), models\ (NLLB) oraz niezacommitowane
#     dane Siedziby z Kwiatownika2 (rosliny/przepisy czekajace na commit)
#  5. buduje i uruchamia Siedzibe na laptopie, czeka az panel odpowie
#  6. tutaj: docker compose down; dane Kwiatownika2 odklada do git stash (kopia), zeby git pull szedl czysto
#  Modele Ollamy laptop pobierze sam przy starcie (OLLAMA_MODELS, MERGE_MODELS).
#
#  Opcje: -Serwer siedziba    alias z ~/.ssh/config albo uzytkownik@ip
#         -CzekajMin 20       ile czekac na dzialajace zadania (0 = od razu)
#         -BezInstalacji      pomin krok 2 (laptop juz przygotowany)
#         -BezModeli          nie kopiuj models\ (laptop przygotuje NLLB sam: docker compose --profile mt run --rm nllb-init)
#         -Wymus              przenies mimo niewypchnietego kodu
#  Gdyby cos poszlo nie tak po kroku 3: docker compose start  (Siedziba wraca tutaj, nic nie jest usuniete)
# ============================================================
param([string]$Serwer = "siedziba", [int]$CzekajMin = 20, [switch]$BezInstalacji, [switch]$BezModeli,
      [switch]$Wymus)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Set-Location $PSScriptRoot
$Adres = "http://localhost:8000"
$Zdalnie = "~/kwiatownik/KwiatownikSiedziba"
function Krok($t) { Write-Host "`n==> $t" -ForegroundColor Green }
function Sprawdz($co) { if ($LASTEXITCODE -ne 0) { throw "$co nie powiodlo sie (kod $LASTEXITCODE)" } }

Krok "Polaczenie z laptopem ($Serwer)"
ssh -o ConnectTimeout=10 $Serwer "echo polaczono z: `$(hostname)"
Sprawdz "Polaczenie SSH (najpierw: .\laptop.ps1 -Konfiguruj -Adres uzytkownik@ip)"

Krok "Kod Siedziby na GitHubie"
git fetch -q
$kod = @(git status --porcelain -- . ":(exclude)data")
$niewypchniete = [int](git rev-list --count "@{u}..HEAD")
if (($kod.Count -gt 0 -or $niewypchniete -gt 0) -and -not $Wymus) {
    Write-Host "Niezacommitowane pliki: $($kod.Count), niewypchniete commity: $niewypchniete" -ForegroundColor Yellow
    $kod | Select-Object -First 15 | ForEach-Object { Write-Host "   $_" }
    throw "Laptop klonuje kod z GitHuba: zrob najpierw  git add -A; git commit -m '...'; git push  (albo -Wymus)"
}
Write-Host "OK - kod wypchniety."

if (-not $BezInstalacji) {
    Krok "Przygotowanie laptopa (serwer/instaluj.sh)"
    scp serwer/instaluj.sh "${Serwer}:instaluj_siedziba.sh"; Sprawdz "Kopiowanie instaluj.sh"
    ssh -t $Serwer "bash ~/instaluj_siedziba.sh"
    if ($LASTEXITCODE -eq 3) {
        throw "Brak Dockera i sudo na laptopie - administrator laptopa musi raz wykonac polecenie pokazane wyzej, potem uruchom ten skrypt ponownie"
    }
    if ($LASTEXITCODE -eq 2) { throw "Czesc systemowa zrobiona - uruchom ten skrypt ponownie (nowe logowanie = grupa docker)" }
    Sprawdz "Instalacja na laptopie"
}
ssh $Serwer "test -f $Zdalnie/docker-compose.yml && test -d ~/kwiatownik/Kwiatownik2/.git && test -d ~/kwiatownik/Kwiatownik"
Sprawdz "Kontrola repozytoriow na laptopie (~/kwiatownik)"

Krok "Wstrzymanie Siedziby na tym komputerze"
$dziala = $true
try {
    $r = Invoke-RestMethod -Method Post -Uri "$Adres/agents/aktualizacja" -Body @{ akcja = "start" } -TimeoutSec 15
    Write-Host "Nowe zadania wstrzymane (limit przed: $($r.limit_przed_aktualizacja)) - na laptopie wroci sam."
} catch { Write-Host "Siedziba tutaj nie odpowiada - kopiuje od razu."; $dziala = $false }
if ($dziala -and $CzekajMin -gt 0) {
    $koniec = (Get-Date).AddMinutes($CzekajMin)
    while ((Get-Date) -lt $koniec) {
        $p = Invoke-RestMethod -Uri "$Adres/agents/praca.json"
        $n = @($p.dzialajace).Count
        if ($n -eq 0) { Write-Host "Wszystkie zadania skonczyly prace."; break }
        Write-Host ("{0:HH:mm:ss}  dziala zadan: {1}" -f (Get-Date), $n)
        Start-Sleep -Seconds 30
    }
}
docker compose stop; Sprawdz "docker compose stop"

Krok "Pakowanie danych"
$tmp = Join-Path $env:TEMP "siedziba_przeniesienie"
New-Item -ItemType Directory -Force $tmp | Out-Null
$tgz = Join-Path $tmp "siedziba_dane.tgz"
$co = @(".env", "data"); if (-not $BezModeli -and (Test-Path models)) { $co += "models" }
tar -czf $tgz --exclude "data/_claude_tmp" @co; Sprawdz "tar (Siedziba)"
Write-Host ("Siedziba: {0:N0} MB" -f ((Get-Item $tgz).Length / 1MB))

# dane Siedziby w Kwiatowniku2, ktore jeszcze nie przeszly przez /commit
$kw2 = Join-Path (Split-Path $PSScriptRoot) "Kwiatownik2"
$dane = @("data/plants", "data/przepisy", "data/changelog.json")
$pliki = @(git -C $kw2 -c core.quotepath=false status --porcelain=v1 --untracked-files=all -- @dane |
           Where-Object { $_.Length -gt 3 -and $_.Substring(0, 2) -notmatch "D" } | ForEach-Object { $_.Substring(3) })
$tgz2 = Join-Path $tmp "kwiatownik2_dane.tgz"
if ($pliki.Count -gt 0) {
    $lista = Join-Path $tmp "kw2_pliki.txt"
    [IO.File]::WriteAllLines($lista, [string[]]$pliki)
    tar -czf $tgz2 -C $kw2 -T $lista; Sprawdz "tar (Kwiatownik2)"
    Write-Host "Kwiatownik2: $($pliki.Count) plikow czekajacych na commit - przejda na laptopa."
}

Krok "Kopiowanie na laptopa"
scp $tgz "${Serwer}:kwiatownik/"; Sprawdz "scp (Siedziba)"
ssh $Serwer "cd $Zdalnie && tar -xzf ../siedziba_dane.tgz && rm ../siedziba_dane.tgz && sed -i 's/\r`$//' .env"
Sprawdz "Rozpakowanie (Siedziba)"
# bez GPU (np. Quadro K3000M - Ollama jej nie obsluguje): lzejsze modele lokalne, Bielik 11B na CPU bylby za wolny
ssh $Serwer ("cd $Zdalnie && free -h | head -2 && if ! nvidia-smi >/dev/null 2>&1; then " +
    "sed -i -E 's#^MERGE_MODELS=.*#MERGE_MODELS=ollama:SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0,ollama:SpeakLeash/bielik-1.5b-v3.0-instruct:Q8_0#; " +
    "s#^OLLAMA_MODELS=.*#OLLAMA_MODELS=qwen3:4b#' .env && echo 'Ollama na CPU: w .env lzejsze modele (qwen3:4b, Bielik 4.5B/1.5B)'; fi")
if ($pliki.Count -gt 0) {
    scp $tgz2 "${Serwer}:kwiatownik/"; Sprawdz "scp (Kwiatownik2)"
    ssh $Serwer "cd ~/kwiatownik/Kwiatownik2 && tar -xzf ../kwiatownik2_dane.tgz && rm ../kwiatownik2_dane.tgz"
    Sprawdz "Rozpakowanie (Kwiatownik2)"
}

Krok "Budowa i start na laptopie (pierwszy raz kilka minut)"
ssh -t $Serwer "bash $Zdalnie/serwer/aktualizuj.sh 0"; Sprawdz "Start Siedziby na laptopie"
ssh $Serwer 'for i in $(seq 60); do curl -fs -o /dev/null http://127.0.0.1:8000/ && echo "Panel odpowiada." && exit 0; sleep 5; done; exit 1'
Sprawdz "Panel na laptopie nie odpowiada po 5 min (logi: .\laptop.ps1 -Logi). Powrot tutaj: docker compose start"

Krok "Sprzatanie na tym komputerze"
docker compose down
if ($pliki.Count -gt 0) {
    git -C $kw2 stash push -u -m "Siedziba przeniesiona na laptopa $(Get-Date -Format yyyy-MM-dd)" -- @dane
    Write-Host "Kwiatownik2: dane Siedziby odlozone do git stash (kopia; teraz pracuje na nich laptop)."
}
Remove-Item -Recurse -Force $tmp
Write-Host "`nGotowe. Siedziba dziala na laptopie. Panel: .\laptop.ps1   Aktualizacja: .\laptop.ps1 -Aktualizuj" -ForegroundColor Green
Write-Host "Kwiatownik2 na tym komputerze: przed zmianami strony zawsze  git pull  (laptop wypycha commity danych)."
