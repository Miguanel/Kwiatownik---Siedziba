#!/usr/bin/env bash
# ============================================================
#  Przygotowanie laptopa z Ubuntu Server pod Siedzibe Kwiatownika. Dwie czesci:
#
#  A) SYSTEM - raz, z konta z sudo (administrator laptopa), dla uzytkownika Siedziby:
#        sudo bash /home/WorkerSiedziba/instaluj_siedziba.sh system WorkerSiedziba
#     Docker Engine + compose, uzytkownik w grupie docker, limit logow, klapa/usypianie wylaczone,
#     strefa Europe/Warsaw, zapora ufw (tylko SSH; panel, Ollama, SearXNG sa na 127.0.0.1).
#     Opcja --gpu na koncu: sterownik NVIDIA + toolkit - tylko karta z CUDA compute >= 5.0 (Maxwell+).
#     Quadro K3000M (Kepler, 3.0, 2 GB) Ollama nie obsluguje - wtedy bez --gpu (Ollama na CPU).
#
#  B) UZYTKOWNIK - bez sudo (przenies_na_laptopa.ps1 uruchamia to sam):
#        bash instaluj_siedziba.sh
#     Repozytoria obok siebie w ~/kwiatownik (taki uklad zaklada docker-compose.yml: ../Kwiatownik2 itd.).
#     Gdy uzytkownik ma sudo, a Dockera jeszcze nie ma, najpierw sam zrobi czesc A.
# ============================================================
set -euo pipefail
krok() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

czesc_systemowa() {  # czesc_systemowa <uzytkownik> <gpu 0|1>  - uruchamiana jako root
    local U="$1" GPU="$2"
    id "$U" >/dev/null 2>&1 || { echo "Nie ma uzytkownika '$U'"; exit 1; }
    krok "Pakiety systemowe"
    apt-get update -y
    apt-get install -y ca-certificates curl git rsync ufw pciutils python3
    krok "Docker Engine"
    command -v docker >/dev/null 2>&1 || curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
    usermod -aG docker "$U"
    if [ ! -f /etc/docker/daemon.json ]; then
        echo '{ "log-driver": "json-file", "log-opts": { "max-size": "20m", "max-file": "3" } }' > /etc/docker/daemon.json
        systemctl restart docker
    fi
    krok "Laptop jako serwer: klapa, usypianie, czas"
    mkdir -p /etc/systemd/logind.conf.d
    printf '[Login]\nHandleLidSwitch=ignore\nHandleLidSwitchExternalPower=ignore\nHandleLidSwitchDocked=ignore\n' \
        > /etc/systemd/logind.conf.d/siedziba.conf
    systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target >/dev/null
    systemctl restart systemd-logind || true
    timedatectl set-timezone Europe/Warsaw || true
    krok "Zapora: tylko SSH"
    ufw allow OpenSSH >/dev/null
    ufw --force enable >/dev/null
    ufw status | head -5
    if [ "$GPU" = 1 ] && lspci | grep -qi nvidia; then
        krok "Karta NVIDIA: sterownik + nvidia-container-toolkit"
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            apt-get install -y ubuntu-drivers-common && ubuntu-drivers install
            echo "!! Zainstalowano sterownik NVIDIA - potrzebny restart: sudo reboot"
        fi
        if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                > /etc/apt/sources.list.d/nvidia-container-toolkit.list
            apt-get update -y && apt-get install -y nvidia-container-toolkit
            nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
        fi
    else
        echo "Ollama bedzie liczyc na CPU."
    fi
    krok "Czesc systemowa gotowa"
    echo "Uzytkownik $U jest w grupie docker (dziala od jego nastepnego logowania)."
    echo "Dalej na Windows: .\\przenies_na_laptopa.ps1"
}

# ---- A) tryb systemowy
if [ "${1:-}" = "system" ]; then
    [ "$(id -u)" = 0 ] || { echo "Czesc systemowa wymaga sudo:  sudo bash $0 system <uzytkownik>"; exit 1; }
    GPU=0; [ "${3:-}" = "--gpu" ] && GPU=1
    czesc_systemowa "${2:-${SUDO_USER:-}}" "$GPU"
    exit 0
fi

# ---- B) tryb uzytkownika
if ! docker info >/dev/null 2>&1; then
    if id -nG | grep -qwE 'sudo|admin'; then
        echo "Docker niedostepny - robie czesc systemowa (zapyta o haslo sudo)."
        GPU=0; [ "${1:-}" = "--gpu" ] && GPU=1
        sudo bash "$0" system "$USER" $([ "$GPU" = 1 ] && echo --gpu)
        echo "Zaloguj sie ponownie (grupa docker), potem uruchom ten skrypt jeszcze raz."
        exit 2
    fi
    if id -nG | grep -qw docker; then
        echo "Jestes w grupie docker, ale ta sesja jeszcze tego nie widzi - zaloguj sie ponownie i powtorz."
    else
        echo "Uzytkownik $USER nie ma sudo, a Docker jest niedostepny."
        echo "Administrator laptopa musi raz uruchomic (na swoim koncie):"
        echo "    sudo bash $HOME/instaluj_siedziba.sh system $USER"
    fi
    exit 3
fi

KAT="${KWIATOWNIK_DIR:-$HOME/kwiatownik}"
GH="https://github.com/Miguanel"
krok "Repozytoria w $KAT"
mkdir -p "$KAT"; cd "$KAT"
git config --global credential.helper store      # token podany raz zostaje w ~/.git-credentials
klonuj() {  # klonuj <repo na GitHubie> <folder> [opcje git clone]
    if [ -d "$2/.git" ]; then echo "$2: juz jest"; return; fi
    echo "Klonuje $1 -> $2  (prywatne repo: Username = login GitHub, Password = token GitHub)"
    git clone "${@:3}" "$GH/$1.git" "$2"
}
klonuj "Kwiatownik---Siedziba" KwiatownikSiedziba
klonuj "KwiatownikNazwa" Kwiatownik2
klonuj "Kwiatownik" Kwiatownik --depth 1          # stary Kwiatownik: tylko archiwum przepisow (do odczytu)
chmod +x KwiatownikSiedziba/serwer/*.sh 2>/dev/null || true
krok "Gotowe: $(ls -d "$KAT"/*/ | xargs -n1 basename | tr '\n' ' ')"
