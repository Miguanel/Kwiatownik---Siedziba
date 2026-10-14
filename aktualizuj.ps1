# ============================================================
#  Bezpieczna aktualizacja Siedziby (ekspertyza 2026-10-14, R5)
#  Uruchamiaj w katalogu KwiatownikSiedziba:   .\aktualizuj.ps1
#
#  1. Siedziba wstrzymuje NOWE zadania (limit 0); dzialajace koncza prace.
#  2. Skrypt czeka, az skoncza (najdluzej -CzekajMin minut, domyslnie 20).
#  3. docker compose up -d --build (nowy kod).
#  4. Po starcie Siedziba sama przywraca limit zadan sprzed aktualizacji.
#  Zadania, ktore nie zdaza skonczyc, nie przepadaja: ponowienie zaczyna od punktu kontrolnego
#  (skonczone rosliny sa pomijane), a odpowiedzi modeli LLM sa w pamieci podrecznej (data/llm_cache.db).
#
#  Opcje:  -CzekajMin 0   = bez czekania (przebuduj od razu)
#          -Anuluj        = cofnij wstrzymanie bez aktualizacji
# ============================================================
param([int]$CzekajMin = 20, [switch]$Anuluj, [string]$Adres = "http://localhost:8000")

$ErrorActionPreference = "Stop"
if ($Anuluj) {
    $r = Invoke-RestMethod -Method Post -Uri "$Adres/agents/aktualizacja" -Body @{ akcja = "anuluj" }
    Write-Host "Wstrzymanie cofniete - limit zadan: $($r.limit)"
    exit 0
}

$dziala = $true
try {
    $r = Invoke-RestMethod -Method Post -Uri "$Adres/agents/aktualizacja" -Body @{ akcja = "start" }
    Write-Host "Nowe zadania wstrzymane (limit przed aktualizacja: $($r.limit_przed_aktualizacja))."
} catch {
    Write-Host "Siedziba nie odpowiada ($Adres) - przebudowuje od razu."
    $dziala = $false
}

if ($dziala -and $CzekajMin -gt 0) {
    $koniec = (Get-Date).AddMinutes($CzekajMin)
    while ((Get-Date) -lt $koniec) {
        $p = Invoke-RestMethod -Uri "$Adres/agents/praca.json"
        $n = @($p.dzialajace).Count
        if ($n -eq 0) { Write-Host "Wszystkie zadania skonczyly prace."; break }
        $lista = ($p.dzialajace | ForEach-Object { "#$($_.id) $($_.rodzaj) $($_.postep)" }) -join ", "
        Write-Host ("{0:HH:mm:ss}  dziala {1}: {2}" -f (Get-Date), $n, $lista)
        Start-Sleep -Seconds 30
    }
    if ((Get-Date) -ge $koniec) { Write-Host "Minelo $CzekajMin min - przebudowuje mimo dzialajacych zadan (wznowia sie od punktu kontrolnego)." }
}

docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Budowa nieudana. Stary kontener nadal dziala - cofnij wstrzymanie: .\aktualizuj.ps1 -Anuluj"
    exit 1
}
Write-Host "Gotowe. Po starcie Siedziba sama przywroci limit zadan w tle."
