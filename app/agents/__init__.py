"""Agenci Siedziby (/agents).

- Audytor wiedzy (audit.py) - skanuje baze wiedzy Kwiatownika (pliki roslin i przepisy), liczy metryki kodem,
  prosi eksperta LLM o ocene i zapisuje raport + zalecenia (AgentTask "proposed").
- Zleceniodawca (dispatch.py) - zamienia zalecenia audytu na zadania Siedziby (kolejka), z limitami i bez powtorek.
- Planista ciaglosci (planner.py) - sprawdza, czy Siedziba nad czyms pracuje; proponuje kolejne zadanie,
  a w trybie automatycznym sam zleca JEDNO, gdy kolejka stoi.
Historia: AgentRun (przebiegi z raportami) i AgentTask (zadania z wynikami).
"""
