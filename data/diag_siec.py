"""Diagnostyka sieci kontenera Siedziby (bez zmian w niczym, tylko odczyt).
Uruchom:  docker compose exec siedziba python data/diag_siec.py
Wynik wklej w rozmowie."""
import socket
import time

import httpx

HOSTS = ["kwiatownik-backend.onrender.com", "api.groq.com", "www.wikidata.org", "pl.wikipedia.org",
         "api-free.deepl.com", "generativelanguage.googleapis.com", "pfaf.org", "stronazdrowia.pl"]

try:
    print("resolv.conf:", " | ".join(l.strip() for l in open("/etc/resolv.conf") if l.strip() and not l.startswith("#")))
except OSError as exc:
    print("resolv.conf:", exc)
for host in HOSTS:
    t0 = time.time()
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        addrs = sorted({(i[0].name[-1], i[4][0]) for i in infos})
    except OSError as exc:
        print(f"{host}: DNS BLAD {exc}")
        continue
    tcp = []
    for fam, ip in addrs[:4]:
        s = socket.socket(socket.AF_INET6 if fam == "6" else socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        try:
            s.connect((ip, 443))
            tcp.append(f"{ip} ok")
        except OSError as exc:
            tcp.append(f"{ip} {type(exc).__name__}")
        finally:
            s.close()
    try:
        r = httpx.get(f"https://{host}/", timeout=10)
        http = f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        http = f"{type(exc).__name__}: {str(exc)[:60]}"
    print(f"{host}: {'; '.join(tcp)} | {http} | {time.time() - t0:.1f} s")
