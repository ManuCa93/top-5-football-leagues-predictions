# Schedine top 5 campionati

Ogni settimana, con un solo comando: aggiorna risultati e quote di Serie A, Premier League, La Liga,
Bundesliga e Ligue 1, chiude le schedine della settimana prima, ne propone di nuove dentro un budget
fisso e apre un report con probabilità, quote eque, costo atteso e storia delle squadre.

## Uso

```bash
pip install -r requirements.txt
python run.py --bankroll 100   # solo la prima volta: il bankroll di partenza
python run.py                  # poi ogni settimana, meglio il venerdì pomeriggio
```

Le quote delle prossime partite arrivano da football-data.co.uk di solito il venerdì (weekend) e il
martedì (turno infrasettimanale): se lanci prima, il report lo segnala e chiude solo le schedine passate.

Opzioni: `--budget-pct 5` (percentuale del bankroll da giocare a settimana, default 5%),
`--as-of 2026-10-16` (prova come se fosse quel venerdì, senza toccare il registro), `--no-open`.

Il registro è `data/ledger.csv`. Puoi modificarne due colonne a mano:
`giocata` (0 se una schedina non l'hai giocata) e `quota_presa` (la quota vera che ti ha dato Snai).

## Come vengono scelte le schedine

- **Probabilità**: quelle implicite nelle quote medie dei bookmaker, senza margine (metodo "power",
  il meglio calibrato nei test). Goal/No Goal dai gol attesi impliciti in 1X2 e Over/Under.
- **Quote Snai**: stimate. Su 185 partite con quote Snai vere (copiate a mano nel vecchio codice) l'1X2 Snai
  coincide con la media di mercato, la doppia chance è circa il 99% di quella ricavata dall'1X2 e Goal/No Goal
  si ricostruisce con un errore tipico del 2-3%. Prima di giocare, controlla la quota vera.
- **Schedine**: tre tipi (doppia, tripla, quota alta). Per ognuna si sceglie la combinazione che perde meno
  in media (prodotto di probabilità × quota), con vincoli di probabilità per gamba e di quota totale,
  senza ripetere partite.
- **Storia delle squadre** (classifica, forma, gol, tiri, xG, Elo, riposo, precedenti, modello dei gol):
  è nel report come contesto, ma non decide. Nei test su 9.000 partite aggiungerla alle quote ha peggiorato
  le previsioni.

## Cosa aspettarsi (numeri onesti)

Backtest walk-forward: ogni venerdì il sistema vede solo il passato e gioca il weekend con quote pre-partita.

| | Periodo | ROI | Note |
|---|---|---|---|
| Sistema nuovo | 2020/21–2024/25 (185 settimane) | **−3.6%** | atteso −5.3%; intervallo 90%: −12.8% / +6.6% |
| Sistema nuovo | 2025/26–2026/27 (45 settimane, holdout) | **−28%** | atteso −7%; "quota alta" vinta 1 volta su 44 |
| Vecchio sistema (marzo 2026) | 2025/26 | **−17%** | accuracy dichiarata 66%, reale 43% |

Con le quote Snai **ogni schedina ha un costo atteso negativo**: circa −3% le doppie sui favoriti,
−6% le triple, −10% la quota alta. Nessun modello costruito su dati pubblici ha battuto le quote.
Il vantaggio è dalla tua parte solo quando Snai offre una quota più alta della "quota equa" mostrata nel
report, ad esempio con le quote maggiorate.

Il vecchio sistema mostrava un'accuracy del 66-68% per via del **leakage**: la feature degli scontri diretti
usava anche il risultato della partita stessa e quelle future. Inoltre il momentum era sempre 0
(`reversed()` su un generatore) e Kelly veniva scavalcato dalla puntata minima.

## Struttura

```
run.py                     comando settimanale
predictor/
  data.py                  download e cache dei CSV di football-data.co.uk (2019/20 → oggi)
  odds.py                  probabilità senza margine, quote Snai stimate, esiti
  slips.py                 costruzione delle schedine
  ledger.py                registro e bankroll
  context.py, features.py  storia delle squadre (solo dati precedenti alla partita)
  model.py                 modello dei gol (Poisson / Dixon-Coles)
  report.py                report HTML
  backtest.py              backtest del sistema nuovo (python -m predictor.backtest [--holdout])
  walkforward.py           valutazione walk-forward dei modelli
  legacy.py                il sistema di marzo 2026, portato fedelmente per la simulazione
  backtest_legacy.py       simulazione del vecchio sistema
tests/                     test anti-leakage e del registro
tools/extract_snai_odds.py estrae dalla storia git le quote Snai inserite a mano
archive/                   il vecchio codice (V25-V27), i log e la documentazione originale
```

Le cartelle `data/` e `reports/` sono locali e non vanno su git.

> Giocare comporta il rischio di perdere denaro. Gioca solo cifre che puoi permetterti di perdere.
