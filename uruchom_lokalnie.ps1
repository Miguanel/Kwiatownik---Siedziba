# ============================================================
#  Siedziba na TYM komputerze (Docker Desktop) + komputer nie zasypia, dopoki to okno jest otwarte.
#  Uruchom w folderze KwiatownikSiedziba:   .\uruchom_lokalnie.ps1
#
#  1. uruchamia Docker Desktop (jesli nie dziala) i czeka na niego
#  2. laptop (jesli osiagalny przez SSH): gdy dziala tam Siedziba - pyta, czy ja zatrzymac
#     i czy sciagnac jej dane (baza, cache, sieci + dane Kwiatownika2 czekajace na commit).
#     Obecne data\ zostaje jako kopia: data_przed_powrotem_<data>
#  3. Kwiatownik2: gdy danych nie bierzemy z laptopa - przywraca je z git stash zrobionego przy przenoszeniu
#  4. karta NVIDIA: dopisuje do .env COMPOSE_FILE z docker-compose.gpu.yml (Ollama na GPU)
#  5. docker compose up -d --build, czeka na panel i otwiera http://localhost:8000
#  6. blokuje usypianie Windows (ekran moze gasnac) i co minute pokazuje stan. Ctrl+C / zamkniecie okna =
#     komputer znow moze zasnac; Siedziba dziala dalej w Dockerze, ale uspiony komputer ja zatrzyma.
#
#  Opcje: -BezLaptopa   nie sprawdzaj laptopa      -Serwer siedziba   alias SSH laptopa
#         -BezCzuwania  tylko uruchom, bez blokady usypiania
# ============================================================
param([switch]$BezLaptopa, [switch]$BezCzuwania, [string]$Serwer = "siedziba")

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Set-Location $PSScriptRoot
$Adres = "http://localhost:8000"
$kw2 = Join-Path (Split-Path $PSScriptRoot) "Kwiatownik2"
function Krok($t) { Write-Host "`n==> $t" -ForegroundColor Green }
function Sprawdz($co) { if ($LASTEXITCODE -ne 0) { throw "$co nie powiodlo sie (kod $LASTEXITCODE)" } }
function DockerDziala { docker info --format "{{.ServerVersion}}" *> $null; return ($LASTEXITCODE -eq 0) }

# ---- 1. Docker Desktop
Krok "Docker Desktop"
$ErrorActionPreference = "Continue"
if (-not (DockerDziala)) {
    $exe = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $exe)) { throw "Nie znalazlem Docker Desktop ($exe) - uruchom go recznie" }
    Start-Process $exe
    Write-Host "Uruchamiam Docker Desktop..." -NoNewline
    for ($i = 0; $i -lt 60 -and -not (DockerDziala); $i++) { Start-Sleep 5; Write-Host "." -NoNewline }
    Write-Host ""
    if (-not (DockerDziala)) { throw "Docker Desktop nie wstal w 5 minut" }
}
Write-Host "Docker dziala."

# ---- 2. laptop
$zLaptopa = $false
if (-not $BezLaptopa) {
    Krok "Laptop ($Serwer)"
    $stan = ssh -o BatchMode=yes -o ConnectTimeout=5 $Serwer `
        "cd ~/kwiatownik/KwiatownikSiedziba 2>/dev/null && docker compose ps -q --status running siedziba 2>/dev/null | head -1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Laptop nieosiagalny przez SSH - pomijam (gdyby Siedziba tam dzialala, zatrzymaj ja recznie)." -ForegroundColor Yellow
    } else {
        if ($stan) {
            Write-Host "Siedziba DZIALA na laptopie. Dwie naraz robia to samo i obie pisza do Kwiatownika2." -ForegroundColor Yellow
        } else {
            Write-Host "Siedziba na laptopie nie dziala (dane moga byc tam nowsze niz tutaj)."
        }
        $odp = Read-Host "Zatrzymac Siedzibe na laptopie i sciagnac jej dane tutaj? (t = tak / n = nie)"
        if ($odp -eq "t") {
            scp -q serwer/oddaj_dane.sh "${Serwer}:oddaj_dane.sh"; Sprawdz "Kopiowanie oddaj_dane.sh"
            ssh -t $Serwer "bash ~/oddaj_dane.sh"; Sprawdz "Pakowanie danych na laptopie"
            $tmp = Join-Path $env:TEMP "siedziba_powrot"
            New-Item -ItemType Directory -Force $tmp | Out-Null
            scp "${Serwer}:powrot_siedziba.tgz" "${Serwer}:powrot_kw2.tgz" $tmp; Sprawdz "Pobieranie danych z laptopa"
            $kopia = "data_przed_powrotem_" + (Get-Date -Format "yyyy-MM-dd_HHmm")
            docker compose down *> $null
            Move-Item data $kopia -ErrorAction Stop
            Write-Host "Dotychczasowe data\ -> $kopia (kopia, mozesz ja pozniej usunac)"
            tar -xzf (Join-Path $tmp "powrot_siedziba.tgz"); Sprawdz "Rozpakowanie data\"
            tar -xzf (Join-Path $tmp "powrot_kw2.tgz") -C $kw2; Sprawdz "Rozpakowanie danych Kwiatownika2"
            Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
            ssh $Serwer "rm -f ~/powrot_siedziba.tgz ~/powrot_kw2.tgz ~/powrot_kw2.txt"
            $zLaptopa = $true
            Write-Host "Dane z laptopa sa tutaj. Siedziba na laptopie zatrzymana."
        } elseif ($stan) {
            if ((Read-Host "To chociaz zatrzymac Siedzibe na laptopie? (t/n)") -eq "t") {
                ssh $Serwer "cd ~/kwiatownik/KwiatownikSiedziba && docker compose stop"
            }
        }
    }
}

# ---- 3. Kwiatownik2: dane odlozone przy przenoszeniu na laptopa
$stash = git -C $kw2 stash list | Select-String "Siedziba przeniesiona na laptopa" | Select-Object -First 1
if ($stash) {
    $ref = ($stash.Line -split ":")[0]
    if ($zLaptopa) {
        Write-Host "Kwiatownik2: zostawiam $ref (starsza kopia; dane wziete z laptopa). Usuniecie: git stash drop $ref"
    } else {
        Krok "Kwiatownik2: przywracam dane Siedziby odlozone przy przenoszeniu ($ref)"
        git -C $kw2 update-index -q --refresh
        git -C $kw2 stash pop $ref
        if ($LASTEXITCODE -ne 0) { Write-Host "!! stash pop nieudany - sprawdz: cd ..\Kwiatownik2; git status" -ForegroundColor Yellow }
    }
}

# ---- 4. GPU
if ((Get-Command nvidia-smi -ErrorAction SilentlyContinue) -and -not (Select-String -Path .env -Pattern "^COMPOSE_FILE=" -Quiet)) {
    Add-Content -Path .env -Encoding ascii -Value "`r`n# Ollama na karcie NVIDIA (Windows: separator ;)`r`nCOMPOSE_FILE=docker-compose.yml;docker-compose.gpu.yml"
    Write-Host ".env: dopisano COMPOSE_FILE z docker-compose.gpu.yml (Ollama na GPU)"
}

# ---- 5. start
Krok "Start Siedziby (docker compose up -d --build)"
docker compose up -d --build; Sprawdz "docker compose up"
Write-Host "Czekam na panel..." -NoNewline
$ok = $false
for ($i = 0; $i -lt 60 -and -not $ok; $i++) {
    try { Invoke-WebRequest $Adres -UseBasicParsing -TimeoutSec 5 | Out-Null; $ok = $true }
    catch { Start-Sleep 5; Write-Host "." -NoNewline }
}
Write-Host ""
if (-not $ok) { throw "Panel nie odpowiada po 5 minutach - logi: docker compose logs --tail 100 siedziba" }
Write-Host "Panel dziala: $Adres"
Start-Process $Adres
if ($BezCzuwania) { exit 0 }

# ---- 6. bez usypiania
Add-Type -Namespace Win32 -Name Zasilanie -MemberDefinition `
    '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Win32.Zasilanie]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
Krok "Komputer nie zasnie, dopoki to okno jest otwarte (ekran moze gasnac). Ctrl+C konczy czuwanie."
try {
    while ($true) {
        try {
            $p = Invoke-RestMethod "$Adres/agents/praca.json" -TimeoutSec 15
            $lista = @($p.dzialajace | ForEach-Object { "#$($_.id) $($_.rodzaj)" }) -join ", "
            Write-Host ("{0:HH:mm}  zadania: {1}  {2}" -f (Get-Date), @($p.dzialajace).Count, $lista)
        } catch {
            Write-Host ("{0:HH:mm}  panel nie odpowiada (restart/aktualizacja?)" -f (Get-Date)) -ForegroundColor Yellow
        }
        [void][Win32.Zasilanie]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
        Start-Sleep -Seconds 60
    }
} finally {
    [void][Win32.Zasilanie]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-Host "Czuwanie wylaczone - komputer znow moze zasypiac. Siedziba dziala dalej w Dockerze."
}
