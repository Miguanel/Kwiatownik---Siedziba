#!/usr/bin/env bash
# ============================================================
#  Jednorazowe przygotowanie laptopa z Ubuntu Server pod Siedzibe Kwiatownika.
#  Uruchom NA LAPTOPIE jako zwykly uzytkownik (skrypt sam pyta o sudo):
#      bash instaluj.sh                 # Ollama na CPU (domyslnie)
#      bash instaluj.sh --gpu           # takze sterownik NVIDIA + toolkit - tylko karta z CUDA compute >= 5.0
#                                         (Maxwell i nowsze). Quadro K3000M (Kepler, 3.0, 2 GB) Ollama nie obsluguje.
#  (przenies_na_laptopa.ps1 z Windows kopiuje ten plik na laptopa i uruchamia go sam)
#
#  1. Docker Engine + compose (oficjalny skrypt get.docker.com), uzytkownik w grupie docker, start z systemem
#  2. Laptop jako serwer: zamkniecie klapy nic nie robi, bez usypiania, strefa czasu Europe/Warsaw
#  3. Zapora ufw: tylko SSH z zewnatrz (panel, Ollama, SearXNG sa na 127.0.0.1 - przez tunel SSH)
#  4. Z --gpu: sterownik NVIDIA + nvidia-container-toolkit (Ollama na GPU)
#  5. Repozytoria obok siebie w ~/kwiatownik (taki uklad zaklada docker-compose.yml: ../Kwiatownik2 itd.)
# ============================================================
set -euo pipefail
KAT="${KWIATOWNIK_DIR:-$HOME/kwiatownik}"
GH="https://github.com/Miguanel"
GPU=0; [ "${1:-}" = "--gpu" ] && GPU=1
krok() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

krok "Pakiety systemowe"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git rsync ufw pciutils python3

krok "Docker Engine"
if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sudo sh
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
# logi kontenerow nie zapelnia dysku
if [ ! -f /etc/docker/daemon.json ]; then
    echo '{ "log-driver": "json-file", "log-opts": { "max-size": "20m", "max-file": "3" } }' \
        | sudo tee /etc/docker/daemon.json >/dev/null
    sudo systemctl restart docker
fi

krok "Laptop jako serwer: klapa, usypianie, czas"
sudo mkdir -p /etc/systemd/logind.conf.d
printf '[Login]\nHandleLidSwitch=ignore\nHandleLidSwitchExternalPower=ignore\nHandleLidSwitchDocked=ignore\n' \
    | sudo tee /etc/systemd/logind.conf.d/siedziba.conf >/dev/null
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target >/dev/null
sudo systemctl restart systemd-logind || true
sudo timedatectl set-timezone Europe/Warsaw || true

krok "Zapora: tylko SSH"
sudo ufw allow OpenSSH >/dev/null
sudo ufw --force enable >/dev/null
sudo ufw status | head -5

RESTART=0
if [ "$GPU" = 1 ] && lspci | grep -qi 'nvidia'; then
    krok "Karta NVIDIA: sterownik + nvidia-container-toolkit"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        sudo apt-get install -y ubuntu-drivers-common
        sudo ubuntu-drivers install
        RESTART=1
    fi
    if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        sudo apt-get update -y && sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    fi
else
    echo "Ollama bedzie liczyc na CPU (sterownik NVIDIA tylko z --gpu i karta compute >= 5.0)."
fi

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

krok "Gotowe"
echo "Uklad: $(ls -d "$KAT"/*/ | xargs -n1 basename | tr '\n' ' ')"
[ "$RESTART" = 1 ] && echo "!! Zainstalowano sterownik NVIDIA - potrzebny restart: sudo reboot"
exit 0
