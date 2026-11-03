import math
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LA PÁGINA
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Mapa de Rutas Mínimas", page_icon="🗺️", layout="wide")

# Paleta del mapa
BG = "#17123a"       # tinta índigo (fondo del gráfico)
PANEL = "#241c5a"    # paneles
TEXT = "#ece8ff"     # texto claro
CORAL = "#ff6b6b"
SUN = "#ffd93d"
MINT = "#5ef2b0"
SKY = "#4dd2ff"
GRID = "#3a307f"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stButton button {
    font-family: 'Fredoka', 'Segoe UI', sans-serif;
}
.stApp {
    background: radial-gradient(circle at 15% 0%, #2d2272 0%, #17123a 55%) fixed;
    color: __TEXT__;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #120d2e;
    border-right: 3px dashed __GRID__;
}
h1, h2, h3, h4, label, p, span, li, div { color: __TEXT__; }

/* Hero */
.hero {
    background: linear-gradient(120deg, #3b2a9c 0%, #6a35c9 55%, #ff6b6b 130%);
    border-radius: 28px;
    padding: 26px 32px 18px 32px;
    border: 3px solid #ffffff22;
    box-shadow: 0 10px 0 #0c0824;
    margin-bottom: 22px;
}
.hero h1 {
    font-size: 2.7rem; font-weight: 700; margin: 0 0 4px 0; letter-spacing: .5px;
    text-shadow: 3px 3px 0 #0c0824;
}
.hero p { font-size: 1.1rem; margin: 0 0 14px 0; opacity: .95; max-width: 720px; }
.track {
    position: relative; height: 46px;
    border-bottom: 4px dotted #ffffff88;
    overflow: hidden;
}
.runner {
    position: absolute; bottom: -2px; left: 0; font-size: 2.1rem;
    animation: run 7s linear infinite;
}
@keyframes run { from { left: -4%; } to { left: 100%; } }
@media (prefers-reduced-motion: reduce) { .runner { animation: none; left: 50%; } }

/* Tarjetas de métricas */
.cards { display: flex; gap: 14px; flex-wrap: wrap; margin: 6px 0 18px 0; }
.card {
    flex: 1 1 150px; background: __PANEL__; border-radius: 20px; padding: 14px 18px;
    border: 3px solid var(--c); box-shadow: 0 6px 0 var(--c);
}
.card .v { font-size: 1.9rem; font-weight: 700; line-height: 1.1; color: var(--c); }
.card .k { font-size: .95rem; opacity: .85; }

/* Caja de resultado */
.result {
    background: __PANEL__; border-radius: 20px; padding: 16px 18px;
    border-left: 8px solid var(--c); margin-bottom: 12px;
}
.result b { color: var(--c); }
.pathline { font-size: 1.25rem; font-weight: 600; word-spacing: 4px; }

/* Pestañas y botones */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: __PANEL__; border-radius: 16px 16px 0 0; padding: 10px 18px; font-weight: 600;
}
.stTabs [aria-selected="true"] { background: __SUN__; }
.stTabs [aria-selected="true"] p { color: #17123a; }
.stButton button[kind="primary"] {
    background: __SUN__; color: #17123a; border: none; border-radius: 16px;
    font-weight: 700; box-shadow: 0 5px 0 #b8941a;
}
.stButton button[kind="primary"]:hover { transform: translateY(2px); box-shadow: 0 3px 0 #b8941a; }
.stButton button:focus-visible { outline: 3px solid __SKY__; }
</style>
"""
for token, value in {"__TEXT__": TEXT, "__PANEL__": PANEL, "__GRID__": GRID,
                     "__SUN__": SUN, "__SKY__": SKY}.items():
    CSS = CSS.replace(token, value)
st.markdown(CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# DATOS DEL EJERCICIO
# ──────────────────────────────────────────────────────────────
node_coords = {
    'A': (-5, 4), 'B': (7, -6), 'C': (6, 7), 'D': (-7, -5), 'E': (5, -4),
    'F': (2, 9), 'G': (-4, 6), 'H': (1, -2), 'I': (-6, 8), 'J': (3, -7),
}

raw_edges = [
    ('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'),
    ('B', 'D'), ('B', 'C'), ('B', 'I'), ('B', 'J'),
    ('C', 'A'),
    ('D', 'F'),
    ('E', 'A'), ('E', 'H'), ('E', 'J'),
    ('F', 'D'), ('F', 'G'), ('F', 'H'),
    ('G', 'I'), ('G', 'J'),
    ('H', 'E'), ('H', 'I'),
    ('I', 'B'), ('I', 'F'),
]


def calc_weight(p1, p2):
    """Distancia euclidiana redondeada a 1 decimal."""
    return round(math.hypot(p2[0] - p1[0], p2[1] - p1[1]), 1)


G = nx.DiGraph()
for node, pos in node_coords.items():
    G.add_node(node, pos=pos)
for u, v in raw_edges:
    G.add_edge(u, v, weight=calc_weight(node_coords[u], node_coords[v]))

nodes_sorted = sorted(G.nodes())


def best_route(src, dst):
    """Devuelve (existe, camino, peso). Si src == dst busca el circuito más corto."""
    if src != dst:
        try:
            path = nx.dijkstra_path(G, src, dst, weight='weight')
            dist = nx.dijkstra_path_length(G, src, dst, weight='weight')
            return True, path, round(dist, 2)
        except nx.NetworkXNoPath:
            return False, [], 0
    cycles = []
    for nbr in G.successors(src):
        try:
            p = nx.dijkstra_path(G, nbr, src, weight='weight')
            w = nx.dijkstra_path_length(G, nbr, src, weight='weight') + G[src][nbr]['weight']
            cycles.append((w, [src] + p))
        except nx.NetworkXNoPath:
            pass
    if cycles:
        cycles.sort()
        return True, cycles[0][1], round(cycles[0][0], 2)
    return False, [], 0


# Fuente de emoji disponible (matplotlib solo dibuja emojis a color con estas fuentes)
_available = {f.name for f in font_manager.fontManager.ttflist}
EMOJI_FONT = next((f for f in ("Segoe UI Emoji", "Noto Color Emoji") if f in _available), None)

# ──────────────────────────────────────────────────────────────
# PANEL LATERAL
# ──────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Ruta mínima o circuito")
source_node = st.sidebar.selectbox("Nodo de salida", nodes_sorted, index=0)
target_node = st.sidebar.selectbox("Nodo de llegada", nodes_sorted, index=9)
st.sidebar.caption("Si eliges el mismo nodo, se busca el circuito más corto de regreso.")

st.sidebar.header("🎬 Animación")
AVATARS = {
    "🚗 Carrito": "🚗", "🛵 Motocicleta": "🛵", "🏃 Corredor": "🏃",
    "👾 Alien": "👾", "🐥 Pollito": "🐥", "🚀 Cohete": "🚀",
}
avatar_label = st.sidebar.selectbox("Elige tu personaje", list(AVATARS.keys()))
avatar_emoji = AVATARS[avatar_label]
num_laps = st.sidebar.number_input("Número de vueltas", min_value=1, max_value=10, value=1)
anim_speed = st.sidebar.slider("Pausa entre cuadros (menos = más rápido)", 0.01, 0.15, 0.03, step=0.01)
run_animation = st.sidebar.button("▶️ ¡Animar recorrido!", type="primary", use_container_width=True)

st.sidebar.header("🎨 Diseño")
node_color = st.sidebar.color_picker("Color de los nodos", "#8b6cff")
path_color = st.sidebar.color_picker("Color de la ruta", SUN)
show_weights = st.sidebar.checkbox("Mostrar pesos en las aristas", value=True)
show_trail = st.sidebar.checkbox("Mostrar estela mientras avanza", value=True)

has_path, shortest_path, total_weight = best_route(source_node, target_node)
path_edges = list(zip(shortest_path[:-1], shortest_path[1:])) if has_path else []

# ──────────────────────────────────────────────────────────────
# ENCABEZADO
# ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
        <h1>🗺️ Mapa de rutas mínimas</h1>
        <p>Elige dónde empezar y dónde llegar. Dijkstra encuentra el camino más corto
        y tu {avatar_emoji} lo recorre por ti.</p>
        <div class="track"><span class="runner">{avatar_emoji}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


def cards_html(items):
    out = '<div class="cards">'
    for value, label, color in items:
        out += f'<div class="card" style="--c:{color}"><div class="v">{value}</div><div class="k">{label}</div></div>'
    return out + "</div>"


# ──────────────────────────────────────────────────────────────
# DIBUJO DEL GRAFO
# ──────────────────────────────────────────────────────────────
def draw_graph(current_pos=None, seg=None):
    """
    seg=None  -> vista estática (toda la ruta resaltada).
    seg=int   -> animación: solo las primeras `seg` aristas quedan resaltadas.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    pos = nx.get_node_attributes(G, 'pos')

    if seg is None:
        done_edges, pending_edges = path_edges, []
        visited = set(shortest_path)
    else:
        done_edges = path_edges[:seg] if show_trail else []
        pending_edges = path_edges[seg:] if show_trail else []
        visited = set(shortest_path[:seg + 1]) if show_trail else {shortest_path[0]}

    highlighted = set(path_edges)
    other_edges = [e for e in G.edges() if e not in highlighted]

    style = "arc3,rad=0.08"
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color="#6f63c9", alpha=0.75,
                           arrows=True, arrowsize=16, node_size=720,
                           connectionstyle=style, ax=ax)
    if pending_edges:
        nx.draw_networkx_edges(G, pos, edgelist=pending_edges, edge_color=path_color, alpha=0.35,
                               style="dashed", width=2.5, arrows=True, arrowsize=18,
                               node_size=720, connectionstyle=style, ax=ax)
    if done_edges:
        nx.draw_networkx_edges(G, pos, edgelist=done_edges, edge_color=path_color, width=4.5,
                               arrows=True, arrowsize=24, node_size=720,
                               connectionstyle=style, ax=ax)

    colors = [path_color if n in visited else node_color for n in G.nodes()]
    edge_cols = [TEXT if n in (source_node, target_node) else BG for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=720,
                           edgecolors=edge_cols, linewidths=3, ax=ax)
    label_colors = {n: ("#17123a" if n in visited else "white") for n in G.nodes()}
    for n, (x, y) in pos.items():
        ax.text(x, y, n, color=label_colors[n], fontsize=12, fontweight="bold",
                ha="center", va="center", zorder=6)

    if show_weights:
        labels = nx.get_edge_attributes(G, 'weight')
        kwargs = dict(edge_labels=labels, font_size=8, label_pos=0.4, font_color=TEXT,
                      bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.85), ax=ax)
        try:
            nx.draw_networkx_edge_labels(G, pos, connectionstyle=style, **kwargs)
        except TypeError:  # versiones antiguas de networkx
            nx.draw_networkx_edge_labels(G, pos, **kwargs)

    if has_path:
        sx, sy = node_coords[source_node]
        ax.text(sx, sy - 0.95, "SALIDA", color=MINT, fontsize=8, fontweight="bold", ha="center")
        if target_node != source_node:
            tx, ty = node_coords[target_node]
            ax.text(tx, ty - 0.95, "META", color=CORAL, fontsize=8, fontweight="bold", ha="center")

    if current_pos:
        x, y = current_pos[0], current_pos[1] + 0.9
        if EMOJI_FONT:
            ax.text(x, y, avatar_emoji, fontsize=24, ha="center", va="center", zorder=10,
                    fontname=EMOJI_FONT,
                    bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=path_color, lw=2.5))
        else:  # sin fuente emoji: estrella como personaje
            ax.scatter([x], [y], s=900, marker="*", c=path_color, edgecolors="white",
                       linewidths=2, zorder=10)

    ax.set_title("Red dirigida con pesos euclidianos", color=TEXT, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", color=GRID, alpha=0.8)
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 11)
    return fig


# ──────────────────────────────────────────────────────────────
# PESTAÑAS
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Mapa", "🎯 Reto", "🧮 Matrices", "📊 Métricas"])

# ---------- TAB 1: MAPA ----------
with tab1:
    if has_path:
        laps_total = round(total_weight * int(num_laps), 2)
        st.markdown(
            cards_html([
                (total_weight, "distancia por vuelta", SUN),
                (len(shortest_path), "nodos en el recorrido", MINT),
                (len(path_edges), "aristas recorridas", SKY),
                (laps_total, f"distancia total ({int(num_laps)} vuelta{'s' if num_laps > 1 else ''})", CORAL),
            ]),
            unsafe_allow_html=True,
        )

    col_left, col_right = st.columns([3, 1.2])

    with col_right:
        if has_path:
            tipo = "Circuito cerrado" if source_node == target_node else "Ruta mínima"
            st.markdown(
                f"""<div class="result" style="--c:{MINT}">
                <b>¡{tipo} encontrado!</b><br>
                <span class="pathline">{' ➔ '.join(shortest_path)}</span></div>""",
                unsafe_allow_html=True,
            )
            steps_df = pd.DataFrame(
                [{"Paso": i + 1, "Tramo": f"{u} ➔ {v}", "Peso": G[u][v]['weight']}
                 for i, (u, v) in enumerate(path_edges)]
            )
            st.markdown("**Recorrido paso a paso**")
            st.dataframe(steps_df, hide_index=True)
        else:
            st.markdown(
                f"""<div class="result" style="--c:{CORAL}">
                <b>Sin camino 😿</b><br>No existe ruta ni circuito desde
                <b>{source_node}</b> hacia <b>{target_node}</b>.
                Prueba con otro destino.</div>""",
                unsafe_allow_html=True,
            )
        status_ph = st.empty()
        st.markdown("**Leyenda**")
        st.markdown(
            f"<span style='color:{path_color}'>●</span> Ruta elegida &nbsp; "
            f"<span style='color:#6f63c9'>●</span> Otras conexiones<br>"
            f"<span style='color:{TEXT}'>◯</span> Anillo claro: salida y meta",
            unsafe_allow_html=True,
        )

    with col_left:
        plot_ph = st.empty()
        progress_ph = st.empty()

        if run_animation and has_path:
            steps_per_edge = 12
            frames = []
            weights = [G[u][v]['weight'] for u, v in path_edges]
            for i, (u, v) in enumerate(path_edges):
                p1, p2 = node_coords[u], node_coords[v]
                for t in np.linspace(0, 1, steps_per_edge, endpoint=False):
                    frames.append({
                        "pt": (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])),
                        "seg": i,
                        "dist": sum(weights[:i]) + t * weights[i],
                        "at": u,
                    })
            frames.append({"pt": node_coords[shortest_path[-1]], "seg": len(path_edges),
                           "dist": sum(weights), "at": shortest_path[-1]})

            total_frames = len(frames) * int(num_laps)
            done = 0
            bar = progress_ph.progress(0.0)
            for lap in range(int(num_laps)):
                for fr in frames:
                    fig = draw_graph(current_pos=fr["pt"], seg=fr["seg"])
                    plot_ph.pyplot(fig)
                    plt.close(fig)
                    done += 1
                    bar.progress(done / total_frames)
                    status_ph.markdown(
                        f"""<div class="result" style="--c:{SKY}">
                        🏁 Vuelta <b>{lap + 1}</b> de {int(num_laps)}<br>
                        📍 Cerca del nodo <b>{fr['at']}</b><br>
                        📏 Recorrido: <b>{fr['dist']:.1f}</b> / {total_weight}</div>""",
                        unsafe_allow_html=True,
                    )
                    time.sleep(anim_speed)
                st.toast(f"Vuelta {lap + 1} de {int(num_laps)} completada", icon="🏆")
                time.sleep(0.3)
            st.balloons()
            status_ph.markdown(
                f"""<div class="result" style="--c:{SUN}">
                🏆 <b>¡Llegaste!</b><br>Distancia total: <b>{round(total_weight * int(num_laps), 2)}</b></div>""",
                unsafe_allow_html=True,
            )
        else:
            fig = draw_graph(current_pos=node_coords[source_node] if has_path else None)
            plot_ph.pyplot(fig)
            plt.close(fig)

# ---------- TAB 2: RETO ----------
def new_question():
    pairs = [(a, b) for a in nodes_sorted for b in nodes_sorted
             if a != b and nx.has_path(G, a, b)]
    st.session_state.q = random.choice(pairs)
    st.session_state.last = None


def check_answer():
    a, b = st.session_state.q
    real = round(nx.dijkstra_path_length(G, a, b, weight='weight'), 2)
    guess = st.session_state.guess
    diff = abs(guess - real)
    st.session_state.rounds += 1
    if diff <= 0.15:
        pts, msg = 10, "¡Exacto!"
    elif diff <= 1:
        pts, msg = 5, "¡Casi! Te faltó poquito."
    else:
        pts, msg = 0, "Esta vez no, pero ya sabes la respuesta."
    st.session_state.score += pts
    st.session_state.last = (pts, msg, real, nx.dijkstra_path(G, a, b, weight='weight'))


with tab2:
    for key, default in (("score", 0), ("rounds", 0), ("last", None)):
        st.session_state.setdefault(key, default)
    if "q" not in st.session_state:
        new_question()

    a, b = st.session_state.q
    st.subheader("¿Cuál es la distancia mínima?")
    st.markdown(f"De **{a}** a **{b}**. Mira el mapa, haz tu estimación y compárala con Dijkstra.")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.number_input("Tu respuesta", min_value=0.0, step=0.1, format="%.1f", key="guess")
        b1, b2 = st.columns(2)
        b1.button("✅ Comprobar", type="primary", on_click=check_answer,
                  disabled=st.session_state.last is not None, use_container_width=True)
        b2.button("🔄 Nueva pregunta", on_click=new_question, use_container_width=True)
        last = st.session_state.last
        if last:
            pts, msg, real, path = last
            color = MINT if pts == 10 else SUN if pts == 5 else CORAL
            st.markdown(
                f"""<div class="result" style="--c:{color}"><b>{msg}</b> (+{pts} pts)<br>
                Distancia real: <b>{real}</b><br>Camino: {' ➔ '.join(path)}</div>""",
                unsafe_allow_html=True,
            )
    with c2:
        score = st.session_state.score
        rank = "Novato 🐣" if score < 30 else "Navegante 🧭" if score < 70 else "Maestro de rutas 👑"
        st.markdown(
            cards_html([(score, "puntos", SUN), (st.session_state.rounds, "preguntas", SKY)]),
            unsafe_allow_html=True,
        )
        st.markdown(f"**Rango:** {rank}")

# ---------- TAB 3: MATRICES ----------
with tab3:
    st.subheader("Matrices de la red")
    matrix_type = st.radio(
        "Tipo de matriz",
        ["Binaria (0 / 1)", "Ponderada (pesos)", "Distancias mínimas (todos contra todos)"],
        horizontal=True,
    )

    if matrix_type == "Distancias mínimas (todos contra todos)":
        fw = dict(nx.floyd_warshall(G, weight='weight'))
        df = pd.DataFrame({v: {u: fw[u][v] for u in nodes_sorted} for v in nodes_sorted})
        df = df.replace(math.inf, np.nan).round(1)
        st.caption("Cada celda es la mejor distancia posible de la fila a la columna. “—” significa que no hay camino.")
        st.dataframe(df.style.background_gradient(cmap="YlOrRd", vmin=0).format("{:.1f}", na_rep="—"))
    else:
        df = pd.DataFrame(0.0, index=nodes_sorted, columns=nodes_sorted)
        for u, v, data in G.edges(data=True):
            df.loc[u, v] = 1.0 if matrix_type.startswith("Binaria") else data['weight']
        if matrix_type.startswith("Binaria"):
            st.dataframe(df.astype(int).style.background_gradient(cmap="Purples", vmin=0, vmax=1))
        else:
            st.dataframe(df.style.background_gradient(cmap="Purples", vmin=0).format("{:.1f}"))

# ---------- TAB 4: MÉTRICAS ----------
with tab4:
    st.subheader("Análisis estructural")
    weights_all = [d['weight'] for _, _, d in G.edges(data=True)]
    n_scc = nx.number_strongly_connected_components(G)
    longest = max(G.edges(data=True), key=lambda e: e[2]['weight'])
    shortest = min(G.edges(data=True), key=lambda e: e[2]['weight'])

    st.markdown(
        cards_html([
            (G.number_of_nodes(), "nodos", SKY),
            (G.number_of_edges(), "aristas", MINT),
            (round(nx.density(G), 2), "densidad", SUN),
            (round(sum(weights_all) / len(weights_all), 1), "peso promedio", CORAL),
        ]),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- **Arista más larga:** {longest[0]} ➔ {longest[1]} ({longest[2]['weight']})\n"
        f"- **Arista más corta:** {shortest[0]} ➔ {shortest[1]} ({shortest[2]['weight']})\n"
        f"- **Fuertemente conexa:** {'Sí, desde cualquier nodo se llega a cualquier otro.' if n_scc == 1 else f'No, tiene {n_scc} componentes.'}"
    )

    in_deg, out_deg = dict(G.in_degree()), dict(G.out_degree())
    df_stats = pd.DataFrame({
        "X": [node_coords[n][0] for n in nodes_sorted],
        "Y": [node_coords[n][1] for n in nodes_sorted],
        "Grado de salida": [out_deg[n] for n in nodes_sorted],
        "Grado de entrada": [in_deg[n] for n in nodes_sorted],
    }, index=nodes_sorted)
    st.dataframe(df_stats)
    st.markdown("**Grados por nodo**")
    st.bar_chart(df_stats[["Grado de salida", "Grado de entrada"]], color=[CORAL, SKY])