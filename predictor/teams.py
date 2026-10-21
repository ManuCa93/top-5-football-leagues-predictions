"""Nomi squadra alternativi (italiani, football-data.org, quote Snai) -> nomi football-data.co.uk."""

ALIASES = {
    # Bundesliga
    "Amburgo": "Hamburg", "Hamburger SV": "Hamburg", "Bayern": "Bayern Munich",
    "Colonia": "FC Koln", "Koln": "FC Koln", "Eintracht Francoforte": "Ein Frankfurt",
    "Eintracht Frankfurt": "Ein Frankfurt", "Frankfurt": "Ein Frankfurt", "Friburgo": "Freiburg",
    "Gladbach": "M'gladbach", "Monchengladbach": "M'gladbach", "Leipzig": "RB Leipzig",
    "Lipsia": "RB Leipzig", "St. Pauli": "St Pauli", "Stoccarda": "Stuttgart",
    "Union Berlino": "Union Berlin",
    # Ligue 1
    "Lione": "Lyon", "Marsiglia": "Marseille", "Nizza": "Nice", "PSG": "Paris SG", "Psg": "Paris SG",
    "Paris Fc": "Paris FC", "Strasburgo": "Strasbourg", "Tolosa": "Toulouse",
    # La Liga
    "Athletic Bilbao": "Ath Bilbao", "Athletic Club": "Ath Bilbao", "Atletico": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid", "Barcellona": "Barcelona", "Maiorca": "Mallorca",
    "Espanyol": "Espanol", "Rayo": "Vallecano", "Real Betis": "Betis", "Real Oviedo": "Oviedo",
    "Real Sociedad": "Sociedad", "Siviglia": "Sevilla",
    # Premier League
    "Manchester United": "Man United", "Nottingham": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
}


def canonical(name):
    return ALIASES.get(name, name)
