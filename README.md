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

Opzioni: `--due-doppie` (2 doppie con quota totale 1.6–2.6 invece di doppia + tripla + quota alta; per i turni
infrasettimanali lancia il comando anche il martedì), `--budget-pct 5` (percentuale del bankroll da giocare a settimana, default 5%),
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

## Due doppie a giornata, 2010/11 – oggi

`python -m predictor.doubles` simula 2 doppie da 5 € per ogni giornata (weekend e turni infrasettimanali) dei 5
campionati, con le sole quote pre-partita e le quote Snai stimate; il report con le curve è in
`reports/simulazione-due-doppie.html`.

| Strategia | Doppie | Vinte | ROI | ROI 90% | CLV | Stagioni in attivo | In attivo dopo 5 stagioni |
|---|---|---|---|---|---|---|---|
| Meno costosa, quota 1.6–2.6 | 1533 | 54% | −3.4% | −7.2% / +0.5% | −3.7% | 8/16 | 27% |
| Meno costosa, quota 2.5–4 | 1627 | 37% | −3.4% | −8.4% / +1.4% | −4.5% | 7/16 | 24% |
| Favoriti più probabili (1/2) | 1668 | 52% | −5.4% | −9.3% / −1.5% | −3.8% | 6/16 | 16% |
| Doppie chance più probabili | 1668 | 59% | −6.2% | −9.9% / −2.4% | −9.4% | 4/16 | 8% |
| GG/NG più probabili | 1668 | 40% | −8.9% | −13.4% / −4.5% | −16.3% | 5/16 | 1% |
| Casuale (quote 1.3–2.0) | 1668 | 32% | −13.3% | −18.5% / −8.1% | −12.1% | 2/16 | 0% |
| Modello (pesi stagioni) | 1525 | 42% | +1.4% | −3.9% / +6.4% | −4.3% | 9/16 | 66% |

Il modello è positivo solo sulle stagioni usate per scegliere la sua feature (2014–24: +3.1%); su quelle non usate
fa −2.0% (2010–13) e −3.2% (2025–26), e il CLV è −4.3%: non è un vantaggio reale. Le strategie oneste perdono
fra il 3% e il 13% a seconda di quanto margine pagano (doppie chance e GG/NG hanno i margini Snai più alti).

## Quote maggiorate: l'unico valore positivo con Snai

```bash
python run.py --maggiorata Inter 1 2.60 --bankroll 100   # mercati: 1 X 2 1X X2 12 GG NG O U
```

Confronta la quota Snai con la quota equa senza margine. La quota equa viene da Pinnacle o Betfair Exchange, i
riferimenti più precisi; se mancano, dalla media di mercato. Se la quota Snai è sopra la quota equa, il comando
suggerisce 1/4 di Kelly come puntata.

## Caccia alle inefficienze in 20 campionati (ottobre 2026)

`python -m predictor.scan`: 105.000 partite 2012–oggi di 20 campionati (top 5, serie cadette inglesi, italiana,
spagnola, tedesca e francese, Scozia, Olanda, Belgio, Portogallo, Turchia, Grecia), mercati 1X2 e Over/Under 2.5,
alla quota media (proxy Snai). Le regole sono scelte sul 2012–19 e verificate sul 2020–oggi.

| Regola | Giocate test | ROI test | CLV test | Stagioni in attivo |
|---|---|---|---|---|
| A. Quota media sopra la quota equa Pinnacle | 134 | −2.2% ± 6.6% | −1.9% | 3/6 |
| B. Combinazioni campionato/esito/fascia di quota in attivo nel 2012–19 (+8% / +11% sul train) | 2837 | **−7.0% ± 2.4%** | −5.6% | 0/7 |
| C. Gradient boosting su quote, struttura del mercato, Elo, forma, gol, riposo | 702 | −0.7% ± 2.4% | −2.1% | 2/6 |
| Riferimento: esito 1X2 a caso | 46040 | −7.3% | −7.1% | 0/7 |

- B è l'esempio tipico di regola trovata per caso: in attivo 8 stagioni su 8 nel periodo in cui è stata scelta,
  0 su 7 dopo.
- Il gradient boosting ha una log loss peggiore di Pinnacle sia sul 2016–19 (0.5973 contro 0.5968) sia dopo
  (0.6209 contro 0.6207).
- La quota media supera la quota equa Pinnacle in pochissime partite, e quando lo fa il mercato poi si muove contro
  (CLV negativo).

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
  doubles.py               simulazione "2 doppie a giornata" dal 2010/11
  scan.py                  inefficienze in 20 campionati, 1X2 e Over/Under (regole + gradient boosting)
  boost.py                 conviene la quota maggiorata? (run.py --maggiorata)
  sofascore.py, sofa_features.py  assenti e calendari da Sofascore (bloccato dopo poche richieste)
  legacy.py                il sistema di marzo 2026, portato fedelmente per la simulazione
  backtest_legacy.py       simulazione del vecchio sistema
tests/                     test anti-leakage e del registro
tools/extract_snai_odds.py estrae dalla storia git le quote Snai inserite a mano
archive/                   il vecchio codice (V25-V27), i log e la documentazione originale
```

Le cartelle `data/` e `reports/` sono locali e non vanno su git.

> Giocare comporta il rischio di perdere denaro. Gioca solo cifre che puoi permetterti di perdere.
