# Siedziba na laptopie z Ubuntu Server

Siedziba (Docker) dziala na laptopie, a kod pisze sie dalej na Windows. Kod idzie przez GitHuba, a dane
(bazy, cache, sieci, modele) zostaja na laptopie. Panel nie ma logowania, wiec jest tylko na 127.0.0.1 laptopa
i dochodzi na Windows tunelem SSH.

    Windows (PyCharm)  --git push-->  GitHub  --git pull-->  laptop: ~/kwiatownik/KwiatownikSiedziba (docker compose)
    Windows: .\laptop.ps1  ==tunel SSH==>  laptop 127.0.0.1:8000 (panel), :8081 (SearXNG)
    laptop: wdrozeniowiec  --commit+push danych-->  GitHub Kwiatownik2  -->  Render buduje strone

## Raz: przeniesienie
1. Na laptopie: Ubuntu Server z `openssh-server` (zaznaczony przy instalacji); IP: `ip -4 a`.
   Najlepiej stale IP (rezerwacja DHCP w routerze).
2. Na Windows, w folderze KwiatownikSiedziba:

       .\laptop.ps1 -Konfiguruj -Adres michal@192.168.1.50     # klucz SSH + alias "siedziba"
       git add -A; git commit -m "Siedziba na serwer"; git push   # laptop klonuje kod z GitHuba
       .\przenies_na_laptopa.ps1

   Skrypt: instaluje Dockera na laptopie (`serwer/instaluj.sh`: klapa i usypianie wylaczone, zapora tylko SSH,
   sterownik i toolkit NVIDIA, jesli jest karta), klonuje repozytoria do `~/kwiatownik`, wstrzymuje Siedzibe tutaj,
   kopiuje `.env`, `data\`, `models\` oraz dane czekajace na commit w Kwiatowniku2, buduje i uruchamia Siedzibe
   na laptopie, a tutaj robi `docker compose down`.
   Uzytkownik Siedziby bez sudo (np. WorkerSiedziba): administrator laptopa robi raz czesc systemowa
   `sudo bash /home/WorkerSiedziba/instaluj_siedziba.sh system WorkerSiedziba`, potem znow `.\przenies_na_laptopa.ps1`. Klon prywatnego repo pyta o login GitHub i token (nie haslo).
3. Laptop HP EliteBook 8770w ma Quadro K3000M (Kepler, CUDA compute 3.0, 2 GB): Ollama wymaga compute >= 5.0,
   wiec Ollama liczy na CPU i sterownika NVIDIA nie instalujemy. (Inna karta, Maxwell lub nowsza:
   `bash instaluj.sh --gpu`, potem `sudo reboot` i `.\laptop.ps1 -Aktualizuj` - wlaczy GPU.)

## Na co dzien (Windows)
| Polecenie | Co robi |
|---|---|
| `.\laptop.ps1` | tunel + panel w przegladarce (http://localhost:8000) |
| `.\laptop.ps1 -Aktualizuj` | `git push`, potem na laptopie pull + bezpieczna przebudowa (`serwer/aktualizuj.sh`) |
| `.\laptop.ps1 -Logi` / `-Stan` / `-Ssh` / `-Zamknij` | logi, stan (dysk, RAM, GPU, bateria), konsola, zamkniecie tunelu |

Kwiatownik2 na Windows: przed zmianami strony zawsze `git pull`, bo commity danych robi teraz laptop.
Wdrozeniowiec przed pushem sam dociaga z GitHuba zmiany strony (rebase); przy konflikcie w tym samym pliku
rosliny cofa rebase i zglasza to w historii agenta.

## Ustawienia serwera (`.env` na laptopie, dopisuje je aktualizuj.sh)
- `DEPLOY_GIT_AUTOCRLF=input` - kopia Kwiatownika2 na Linuxie ma konce linii LF
- `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml` - tylko gdy dziala `nvidia-smi` (Ollama na GPU)
- Bez GPU Ollama liczy na CPU: Bielik 11B bedzie bardzo wolny (MERGE_TIMEOUT_S i tak przelaczy na chmure).
  Lepiej: `MERGE_MODELS=ollama:SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0`, `OLLAMA_MODELS=qwen3:4b`.

## Kopia zapasowa danych z laptopa
    ssh siedziba "cd ~/kwiatownik/KwiatownikSiedziba && tar -czf ~/siedziba_kopia.tgz .env data"
    scp siedziba:siedziba_kopia.tgz .
(nie `ssh ... > plik` - PowerShell 5 psuje dane binarne w przekierowaniu)
