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

## Grid search delle feature (ottobre 2026)

Domanda: aggiungendo alle quote la storia delle squadre si ottengono probabilità migliori del mercato,
abbastanza da battere il margine Snai? Dati: 38.000 partite 2005/06–2026/27 di football-data.co.uk più il calendario
europeo di [openfootball](https://github.com/openfootball/champions-league). Modello: regressione logistica che parte
dai logit delle quote e impara solo le correzioni, riaddestrata ogni mese sul solo passato (`python -m predictor.grid`).

Famiglie provate, ognuna con la sua griglia di parametri (circa 290 configurazioni per l'1X2, 180 per il GG/NG):

- peso delle stagioni passate: la scorsa e quella prima valgono m1 e m2 partite (m1 ∈ 0/5/10/19/38, m2 ∈ 0/3/8),
  in alternativa decadimento esponenziale; su punti, differenza reti, tiri, tiri in porta, angoli, e sul rendimento
  rispetto a quanto si aspettava il mercato
- rendimento casa/trasferta separato, forma nelle ultime 3/5/8/12 partite e negli ultimi 14/21/30 giorni, Elo
- riposo e partite negli ultimi 14 giorni contando campionato ed Europa, Europa nei 4 giorni prima/dopo
- peso degli anni vecchi nel training (mezza vita 2/4/8 anni) e prima stagione di training
- infortuni e squalificati pesati per i minuti giocati: il codice c'è (`predictor/sofascore.py`, `sofa_features.py`) ma
  Sofascore blocca i download automatici dopo poche centinaia di richieste, quindi non è entrato nella valutazione

Risultati (log loss: più basso è meglio; quote di mercato DEV = 0.9711):

| Prova | Δ log loss vs quote | Singole 1X2 a quota ≤ 2.5 con p × quota > 1 |
|---|---|---|
| Scelta sul 2020–24, misurata sul 2020–24 | −0.0012 ± 0.0005 | ROI +5.4% ± 2.1%, CLV −1.4% |
| Scelta sul 2020–24, verificata sul 2014–19 | +0.0008 ± 0.0005 | ROI −2.2% ± 1.6% |
| Scelta sul 2014–19, verificata sul 2020–24 | −0.0001 ± 0.0005 | ROI +4.3% ± 2.3%, CLV −2.4% |
| **Holdout 2025/26–2026/27** (scelta su 2014–24, piano scritto prima) | **−0.0002 ± 0.0009** | **ROI −3.9% ± 5.7%, CLV −3.1%** |
| Riferimento: tutti i favoriti a quota ≤ 2.5 | | ROI −2.5% (2020–24), −1.9% (holdout) |

- Il miglioramento visto sul periodo usato per scegliere sparisce sugli altri: con centinaia di configurazioni
  qualcuna sembra buona per caso. Ogni periodo sceglie famiglie diverse.
- La famiglia più stabile sono i pesi per stagione (punti o differenza reti, la stagione scorsa che
  vale circa 10 partite), ma vale al massimo 2-5 decimillesimi di log loss: troppo poco contro un margine del 5-8%.
- Il CLV (quota presa contro quota di chiusura) è negativo in ogni prova: le giocate "di valore" del modello
  sono quote che poi il mercato non ha confermato.
- xG e statistiche avanzate da [Understat](https://understat.com) (dal 2014/15: xG, xG senza rigori, PPDA, passaggi
  profondi, punti attesi; `predictor/understat.py`): stesse verifiche incrociate 2016–19 ↔ 2020–24. La differenza xG
  pesata per stagione vale +0.0004 / +0.0000 di log loss, il pressing (PPDA) +0.0005 / −0.0004, gli xPts rispetto al
  mercato +0.0002 / −0.0001; CLV fra −1.9% e −3.2% in ogni caso. Gli xG sono già nelle quote, anche per il GG/NG.
- GG/NG: con il margine Snai (~8%) in 5 stagioni il modello trova 0–13 giocate con valore sulla carta, tutte con CLV negativo.
- Forma recente, riposo, Europa e casa/trasferta sono già nel prezzo, come trovato anche nella ricerca su NBA e NFL.

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
  research.py              feature candidate e modello "quote + correzioni" per la grid search
  grid.py, betgrid.py      grid search delle feature e delle regole di gioco
  europe.py                calendario europeo (openfootball) per riposo e carico
  understat.py             xG, PPDA, passaggi profondi, punti attesi (Understat, dal 2014/15)
  sofascore.py, sofa_features.py  assenti e calendari da Sofascore (bloccato dopo poche richieste)
  legacy.py                il sistema di marzo 2026, portato fedelmente per la simulazione
  backtest_legacy.py       simulazione del vecchio sistema
tests/                     test anti-leakage e del registro
tools/extract_snai_odds.py estrae dalla storia git le quote Snai inserite a mano
archive/                   il vecchio codice (V25-V27), i log e la documentazione originale
```

Le cartelle `data/` e `reports/` sono locali e non vanno su git.

> Giocare comporta il rischio di perdere denaro. Gioca solo cifre che puoi permetterti di perdere.
