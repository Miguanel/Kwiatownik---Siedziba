# ============================================================
#  Siedziba na laptopie (Ubuntu Server) - obsluga z Windows.  Uruchamiaj w folderze KwiatownikSiedziba.
#    .\laptop.ps1 -Konfiguruj -Adres michal@192.168.1.50   raz: klucz SSH + alias "siedziba" w ~/.ssh/config
#    .\laptop.ps1                    tunel SSH + panel w przegladarce (http://localhost:8000, SearXNG :8081)
#    .\laptop.ps1 -Aktualizuj        nowy kod: git push tutaj, potem pull + bezpieczna przebudowa na laptopie
#    .\laptop.ps1 -Logi              logi Siedziby na zywo (Ctrl+C konczy)
#    .\laptop.ps1 -Stan              kontenery, dysk, pamiec, GPU, bateria
#    .\laptop.ps1 -Ssh               konsola na laptopie
#    .\laptop.ps1 -Zamknij           zamknij tunel
#    .\laptop.ps1 -KopiujObrazy      obrazy Dockera (Ollama, SearXNG, python) z tego komputera na laptopa przez LAN
#    .\laptop.ps1 -KopiujModele      modele Ollamy (wolumen ollama_models) z tego komputera na laptopa przez LAN
#  Panel nie ma logowania - jest tylko na 127.0.0.1 laptopa, a tunel SSH przenosi go na ten komputer.
# ============================================================
param([switch]$Konfiguruj, [string]$Adres, [switch]$Aktualizuj, [int]$CzekajMin = 20, [switch]$Logi,
      [switch]$Stan, [switch]$Ssh, [switch]$Zamknij, [switch]$KopiujObrazy, [switch]$KopiujModele,
      [string]$Serwer = "siedziba")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Kat = "~/kwiatownik/KwiatownikSiedziba"
$Porty = @(8000, 8081)

function Tunel {
    Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" |
        Where-Object { $_.CommandLine -match "8000:127\.0\.0\.1:8000" }
}

if ($Konfiguruj) {
    if ($Adres -notmatch "^([^@\s]+)@([^@\s]+)$") { throw "Podaj -Adres uzytkownik@ip_laptopa, np. -Adres michal@192.168.1.50" }
    $u = $Matches[1]; $h = $Matches[2]
    $dir = Join-Path $HOME ".ssh"; New-Item -ItemType Directory -Force $dir | Out-Null
    $klucz = Join-Path $dir "id_ed25519"
    if (-not (Test-Path $klucz)) {
        Write-Host "Nowy klucz SSH - nacisnij Enter przy pytaniach o haslo (klucz bez hasla)."
        ssh-keygen -t ed25519 -f $klucz -C "siedziba-$env:COMPUTERNAME"
    }
    Write-Host "Kopiuje klucz na laptopa - podaj RAZ haslo uzytkownika $u na laptopie:"
    Get-Content "$klucz.pub" | ssh $Adres "umask 077; mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
    if ($LASTEXITCODE -ne 0) { throw "Nie udalo sie skopiowac klucza (czy laptop ma openssh-server i dobre IP?)" }
    $cfg = Join-Path $dir "config"
    if (-not ((Test-Path $cfg) -and (Select-String -Path $cfg -Pattern "^\s*Host\s+$Serwer\s*$" -Quiet))) {
        $wpis = "`r`nHost $Serwer`r`n    HostName $h`r`n    User $u`r`n    IdentityFile ~/.ssh/id_ed25519`r`n" +
                "    ServerAliveInterval 30`r`n    ServerAliveCountMax 4`r`n"
        Add-Content -Path $cfg -Value $wpis -Encoding ascii
        Write-Host "Dopisano alias '$Serwer' do $cfg"
    } else { Write-Host "Alias '$Serwer' juz jest w $cfg (zmiana IP: popraw HostName tam)." }
    ssh -o BatchMode=yes $Serwer "echo 'Logowanie kluczem dziala:' `$(whoami)@`$(hostname)"
    exit $LASTEXITCODE
}

# cmd /c: potok bajtow docker -> ssh (PowerShell 5 psuje dane binarne w potoku miedzy programami)
if ($KopiujObrazy) {
    $obrazy = @("ollama/ollama:latest", "searxng/searxng:latest", "python:3.11-slim") |
              Where-Object { docker image ls -q $_ }
    if (-not $obrazy) { throw "Brak obrazow w Dockerze na tym komputerze (docker images)" }
    Write-Host "Wysylam na laptopa: $($obrazy -join ', ') - kilka GB, potrwa..."
    cmd /c "docker save $($obrazy -join ' ') | ssh $Serwer docker load"
    if ($LASTEXITCODE -ne 0) { throw "Kopiowanie obrazow nieudane" }
    ssh $Serwer "docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}'"
    exit 0
}
if ($KopiujModele) {
    Write-Host "Modele Ollamy z tego komputera:"
    docker compose run --rm --no-deps -T --entrypoint sh ollama -c 'du -sh /root/.ollama/models; find /root/.ollama/models/manifests -type f | sed s#.*/manifests/##'
    Write-Host "Wysylam caly wolumen modeli na laptopa (moze byc kilkanascie GB)..."
    cmd /c ("docker compose run --rm --no-deps -T --entrypoint tar ollama -c -C /root/.ollama . | " +
            "ssh $Serwer ""cd $Kat && docker compose run --rm --no-deps -T --entrypoint tar ollama -x -C /root/.ollama""")
    if ($LASTEXITCODE -ne 0) { throw "Kopiowanie modeli nieudane" }
    ssh $Serwer "cd $Kat && docker compose run --rm --no-deps -T --entrypoint sh ollama -c 'du -sh /root/.ollama/models; find /root/.ollama/models/manifests -type f | sed s#.*/manifests/##'"
    exit 0
}

if ($Zamknij) { Tunel | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Write-Host "Tunel zamkniety."; exit 0 }
if ($Ssh)  { ssh $Serwer; exit $LASTEXITCODE }
if ($Logi) { ssh -t $Serwer "cd $Kat && docker compose logs -f --tail 200 siedziba"; exit 0 }
if ($Stan) {
    ssh $Serwer ("cd $Kat && docker compose ps --format 'table {{.Name}}\t{{.Status}}'; echo; uptime; " +
                 "df -h / | tail -1; free -h | head -2; " +
                 "command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; " +
                 "cat /sys/class/power_supply/BAT*/capacity 2>/dev/null | sed 's/^/bateria %: /'")
    exit 0
}

if ($Aktualizuj) {
    git fetch -q
    $kod = @(git status --porcelain -- . ":(exclude)data")
    if ($kod.Count -gt 0) {
        Write-Host "Niezacommitowane zmiany kodu (laptop ich NIE dostanie):" -ForegroundColor Yellow
        $kod | Select-Object -First 15 | ForEach-Object { Write-Host "   $_" }
        if ((Read-Host "Kontynuowac mimo to? (t/N)") -ne "t") { exit 1 }
    }
    $n = [int](git rev-list --count "@{u}..HEAD")
    if ($n -gt 0) {
        Write-Host "Wypycham $n commit(ow) na GitHuba..."
        git push; if ($LASTEXITCODE -ne 0) { throw "git push nieudany" }
    }
    ssh -t $Serwer "bash $Kat/serwer/aktualizuj.sh $CzekajMin"
    exit $LASTEXITCODE
}

# ---- domyslnie: tunel + panel
if (-not (Tunel)) {
    foreach ($p in $Porty) {
        $zajety = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($zajety) {
            $kto = (Get-Process -Id $zajety.OwningProcess -ErrorAction SilentlyContinue).ProcessName
            throw "Port $p jest zajety przez '$kto' (lokalna Siedziba w Dockerze? docker compose down w tym folderze)"
        }
    }
    $fw = $Porty | ForEach-Object { "-L"; "${_}:127.0.0.1:${_}" }
    Start-Process ssh -ArgumentList (@("-N", "-o", "ExitOnForwardFailure=yes") + $fw + @($Serwer)) -WindowStyle Hidden
    $ok = $false
    for ($i = 0; $i -lt 20 -and -not $ok; $i++) {
        Start-Sleep -Milliseconds 500
        $ok = [bool](Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)
    }
    if (-not $ok) { throw "Tunel nie wstal - sprawdz: ssh $Serwer" }
    Write-Host "Tunel otwarty (zamkniecie: .\laptop.ps1 -Zamknij)."
}
Start-Process "http://localhost:8000"
