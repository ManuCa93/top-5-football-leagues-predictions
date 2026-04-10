import flet as ft
import threading
import sys
import json
import traceback
import re

# Importiamo il tuo script (assicurati che il file si chiami script_old.py)
import script_old

# =====================================================
# ⚙️ CONFIGURAZIONE GENERALE (FACILE DA MODIFICARE)
# =====================================================

# 🎨 Palette Colori (Interfaccia)
class UIColors:
    BG_PAGE = "#0A0B0C"          # Sfondo della pagina (Nero profondo)
    BG_CARD = "#141619"          # Sfondo delle schede (Grigio scurissimo)
    ACCENT = "#00E5FF"           # Colore principale (Teal / Verde Acqua Neon)
    ACCENT_DIM = "#00838F"       # Accento scuro (per le ombreggiature)
    TEXT_TITLE = "#FFFFFF"       # Colore testi in evidenza (Bianco puro)
    TEXT_BODY = "#B0BEC5"        # Colore testi secondari (Grigio azzurrino)
    BORDER = "#263238"           # Colore bordi dei pannelli

# 💻 Palette Colori (Terminale Integrato)
class TermColors:
    BG = "#000000"               # Sfondo nero puro per il terminale
    TEXT = "#E0E0E0"             # Testo di log standard
    OK = "#00E676"               # Verde per successi [OK], [START]
    WARN = "#FF9100"             # Arancione per avvisi [WARN]
    ERR = "#FF1744"              # Rosso brillante per errori [ERROR], [CRITICAL]
    INFO = "#2979FF"             # Blu per informazioni generali [INFO]

# 🪟 Impostazioni Finestra App
class AppConfig:
    TITLE = "EUROPEAN FOOTBALL PREDICTOR AI | Dashboard v26"
    WIDTH = 1300
    HEIGHT = 900
    MIN_WIDTH = 1000
    MIN_HEIGHT = 700
    FONT_TECH = "https://github.com/google/fonts/raw/main/ofl/orbitron/Orbitron%5Bwght%5D.ttf"

# ⚽ Impostazioni Dati di Default
DEFAULT_MATCHDAYS = {'SA': 23, 'PL': 24, 'PD': 22, 'BL1': 20, 'FL1': 20}

# =====================================================
# LOGICA REINDIRIZZAMENTO & COLORAZIONE TERMINALE
# =====================================================
class ModernTerminalOutput:
    """Classe avanzata per reindirizzare e colorare l'output del terminale"""
    def __init__(self, page, output_list):
        self.page = page
        self.output_list = output_list

    def write(self, text):
        clean_text = text.strip()
        if clean_text:
            color = TermColors.TEXT
            weight = ft.FontWeight.NORMAL
            
            # Assegnazione colore tramite Regex
            if re.search(r'\[OK\]|\[START\]|\[DONE\]', clean_text):
                color = TermColors.OK
                weight = ft.FontWeight.BOLD
            elif re.search(r'\[WARN\]', clean_text):
                color = TermColors.WARN
            elif re.search(r'\[ERROR\]|\[CRITICAL\]|Traceback', clean_text):
                color = TermColors.ERR
                weight = ft.FontWeight.BOLD
            elif re.search(r'\[INFO\]|\[API\]|\[WAIT\]', clean_text):
                color = TermColors.INFO
            elif clean_text.startswith('$$') or clean_text.startswith('=='):
                color = UIColors.ACCENT
                weight = ft.FontWeight.BOLD

            new_line = ft.Text(
                clean_text, 
                size=12, 
                font_family="Consolas",
                color=color,
                weight=weight,
                selectable=True
            )
            
            self.output_list.controls.append(new_line)
            self.page.update()

    def flush(self):
        pass

# =====================================================
# INTERFACCIA GRAFICA PRINCIPALE
# =====================================================
def main(page: ft.Page):
    # Configurazione Finestra usando le costanti
    page.title = AppConfig.TITLE
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = UIColors.BG_PAGE
    page.padding = 30
    
    page.window.width = AppConfig.WIDTH
    page.window.height = AppConfig.HEIGHT
    page.window.min_width = AppConfig.MIN_WIDTH
    page.window.min_height = AppConfig.MIN_HEIGHT
    page.window.center()

    page.fonts = {"TechFont": AppConfig.FONT_TECH}

    # Carica dati iniziali
    default_odds = script_old.get_odds_mapping()

    # --- HEADER SECTION ---
    header = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.ANALYTICS_OUTLINED, color=UIColors.ACCENT, size=40),
            ft.Column([
                ft.Text("AI FOOTBALL PREDICTOR", size=32, weight=ft.FontWeight.BOLD, color=UIColors.TEXT_TITLE, font_family="TechFont"),
                ft.Text("European Leagues Analysis & Advanced Betting Portfolio Generator", size=14, color=UIColors.TEXT_BODY),
            ], spacing=0)
        ]),
        margin=ft.margin.only(bottom=20)
    )

    # --- PANNELLO SINISTRO (Controlli) ---
    
    # Scheda Giornate
    matchday_inputs = {}
    inputs_grid = ft.GridView(
        expand=1, runs_count=3, max_extent=150,
        child_aspect_ratio=2.0, spacing=15, run_spacing=15,
    )
    
    for league, day in DEFAULT_MATCHDAYS.items():
        tf = ft.TextField(
            label=f"{league}",
            value=str(day),
            width=100, height=50,
            text_align=ft.TextAlign.CENTER,
            keyboard_type=ft.KeyboardType.NUMBER,
            border_color=UIColors.BORDER,
            focused_border_color=UIColors.ACCENT,
            label_style=ft.TextStyle(color=UIColors.TEXT_BODY),
            text_style=ft.TextStyle(color=UIColors.TEXT_TITLE, weight=ft.FontWeight.BOLD),
            cursor_color=UIColors.ACCENT,
            content_padding=5
        )
        matchday_inputs[league] = tf
        inputs_grid.controls.append(tf)

    card_leagues = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.LEADERBOARD_OUTLINED, color=UIColors.ACCENT, size=20),
                ft.Text("ULTIME GIORNATE GIOCATE", size=16, weight=ft.FontWeight.BOLD, color=UIColors.ACCENT),
            ], spacing=10),
            ft.Divider(color=UIColors.BORDER, height=1),
            ft.Container(content=inputs_grid, padding=ft.padding.only(top=10))
        ]),
        bgcolor=UIColors.BG_CARD, border_radius=12, padding=20, border=ft.border.all(1, UIColors.BORDER),
    )

    # =================================================
    # 2. Scheda Editor Quote (CON ZOOM DINAMICO)
    # =================================================
    
    # Variabile per tracciare la grandezza del testo
    editor_font_size = 12

    # Funzioni per lo zoom del testo
    def zoom_in(e):
        nonlocal editor_font_size
        if editor_font_size < 30: # Limite massimo
            editor_font_size += 1
            odds_editor.text_style.size = editor_font_size
            odds_editor.update()

    def zoom_out(e):
        nonlocal editor_font_size
        if editor_font_size > 8: # Limite minimo
            editor_font_size -= 1
            odds_editor.text_style.size = editor_font_size
            odds_editor.update()

    odds_editor = ft.TextField(
        multiline=True, 
        # Rimosso max_lines così si espande dinamicamente con la finestra
        value=json.dumps(default_odds, indent=4),
        text_style=ft.TextStyle(font_family="Consolas", size=editor_font_size, color="#AFAFAF"),
        border_color=ft.Colors.TRANSPARENT, focused_border_color=ft.Colors.TRANSPARENT,
        cursor_color=UIColors.ACCENT, expand=True, content_padding=0,
    )

    card_odds = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.ATTACH_MONEY_ROUNDED, color=UIColors.ACCENT, size=20),
                    ft.Text("MAPPATURA QUOTE (JSON)", size=16, weight=ft.FontWeight.BOLD, color=UIColors.ACCENT),
                ]),
                # Aggiunti i bottoni di Zoom invece del semplice testo
                ft.Row([
                    ft.IconButton(icon=ft.Icons.REMOVE_CIRCLE_OUTLINE, icon_color=UIColors.TEXT_BODY, tooltip="Riduci Testo", on_click=zoom_out, icon_size=18),
                    ft.IconButton(icon=ft.Icons.ADD_CIRCLE_OUTLINE, icon_color=UIColors.TEXT_BODY, tooltip="Ingrandisci Testo", on_click=zoom_in, icon_size=18),
                ], spacing=0)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Divider(color=UIColors.BORDER, height=1),
            ft.Container(
                content=odds_editor, bgcolor="#0C0E11", padding=15, border_radius=8, 
                expand=True, margin=ft.margin.only(top=10), border=ft.border.all(1, "#1A1D21")
            )
        ]),
        bgcolor=UIColors.BG_CARD, border_radius=12, padding=20, border=ft.border.all(1, UIColors.BORDER), expand=True
    )
    
    # --- PANNELLO DESTRO (Console) ---
    output_console_list = ft.ListView(expand=True, spacing=3, auto_scroll=True, padding=10)
    
    card_terminal = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.TERMINAL_ROUNDED, color=UIColors.ACCENT, size=20),
                    ft.Text("AI CORE ENGINE OUTPUT", size=16, weight=ft.FontWeight.BOLD, color=UIColors.ACCENT),
                ]),
                ft.Row([
                    ft.Container(width=10, height=10, bgcolor=ft.Colors.GREEN_400, border_radius=5),
                    ft.Text("LIVE STATUS", size=11, color=ft.Colors.GREEN_400, weight=ft.FontWeight.BOLD)
                ], spacing=5)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Divider(color=UIColors.BORDER, height=1),
            ft.Container(
                content=output_console_list, bgcolor=TermColors.BG, border_radius=8, expand=True,
                margin=ft.margin.only(top=10), border=ft.border.all(1, "#111111"),
                shadow=ft.BoxShadow(spread_radius=1, blur_radius=10, color="#000000")
            )
        ]),
        bgcolor=UIColors.BG_CARD, border_radius=12, padding=20, border=ft.border.all(1, UIColors.BORDER), expand=True
    )

    sys.stdout = ModernTerminalOutput(page, output_console_list)

    # --- LOGICA DI ESECUZIONE ---
    loading_ring = ft.ProgressRing(visible=False, width=24, height=24, color=UIColors.ACCENT, stroke_width=3)
    status_text = ft.Text("Pronto.", size=12, color=UIColors.TEXT_BODY)

    def run_script_thread(e):
        updated_matchdays = {}
        try:
            for league, text_field in matchday_inputs.items():
                updated_matchdays[league] = int(text_field.value)
        except ValueError:
            print("[ERROR] Le giornate devono essere numeri interi!")
            return

        try:
            updated_odds = json.loads(odds_editor.value)
        except Exception as ex:
            print(f"[ERROR] Formato JSON delle quote non valido: {ex}")
            return

        # UI pre-esecuzione
        run_btn.disabled = True
        run_btn.content.controls[0].name = ft.Icons.HOURGLASS_EMPTY
        run_btn.shadow = None
        loading_ring.visible = True
        status_text.value = "AI Engine in esecuzione..."
        status_text.color = UIColors.ACCENT
        output_console_list.controls.clear()
        print(f"[{'START':^10}] Inizializzazione Core V26 Optimized...")
        page.update()

        def background_task():
            try:
                script_old.esegui_previsioni(updated_matchdays, updated_odds)
                status_text.value = "Analisi completata con successo."
                status_text.color = ft.Colors.GREEN_400
            except Exception as ex:
                print(f"\n[{'FATAL':^10}] Eccezione non gestita: {ex}")
                traceback.print_exc()
                status_text.value = "Errore durante l'esecuzione."
                status_text.color = TermColors.ERR
            finally:
                # Ripristina UI post-esecuzione
                run_btn.disabled = False
                run_btn.content.controls[0].name = ft.Icons.ROCKET_LAUNCH_ROUNDED # Corretto icona razzo
                run_btn.shadow = ft.BoxShadow(blur_radius=20, color=UIColors.ACCENT_DIM, offset=ft.Offset(0,0), spread_radius=-2)
                loading_ring.visible = False
                page.update()

        threading.Thread(target=background_task, daemon=True).start()

    # --- BOTTONE D'AVVIO ---
    run_btn = ft.Container(
        content=ft.Row([
            ft.Text("   ", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),        
            ft.Icon(ft.Icons.ROCKET_LAUNCH_ROUNDED, color=UIColors.ACCENT, size=22),
            ft.Text("   AVVIA ENGINE AI   ", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE)        
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=12),
        
        alignment=ft.Alignment(0, 0),
        height=60, border_radius=8,
        bgcolor=UIColors.BG_PAGE,
        border=ft.border.all(2, UIColors.ACCENT),
        shadow=ft.BoxShadow(
            blur_radius=20, color=UIColors.ACCENT_DIM,
            offset=ft.Offset(0, 0), spread_radius=-2
        ),
        on_click=run_script_thread,
        animate_scale=ft.Animation(200, ft.AnimationCurve.DECELERATE)
    )
    
    # --- BARRA INFERIORE ---
    action_bar = ft.Container(
        content=ft.Row([
            ft.Row([run_btn, loading_ring], spacing=20),
            status_text
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        margin=ft.margin.only(top=20)
    )

    # --- LAYOUT FINALE ---
    page.add(
        header,
        ft.Row([
            ft.Column([card_leagues, card_odds], expand=4, spacing=20),
            ft.Column([card_terminal], expand=6)
        ], expand=True),
        action_bar
    )

if __name__ == "__main__":
    ft.app(target=main)