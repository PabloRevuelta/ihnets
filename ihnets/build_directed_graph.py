from igraph import Graph

# build_directed_graph.py
from igraph import Graph

from igraph import Graph

def build_directed_graph(main_graph):
    """
    Construye una versión dirigida del grafo original según reglas entre redes.

    Reglas:
    - Energy network → Roads/Railway network: arista unidireccional (solo de energía hacia transporte)
    - Roads/Railway → Energy network: no se añade arista inversa
    - Entre nodos de la misma red o entre Roads ↔ Railway: aristas bidireccionales
    """

    g_dir = Graph(directed=True)
    g_dir.add_vertices(len(main_graph.vs))

    # Copiar atributos de nodos
    for i, v in enumerate(main_graph.vs):
        g_dir.vs[i].update_attributes(v.attributes())

    edges = []
    for e in main_graph.es:
        u, v = e.tuple
        net_u = main_graph.vs[u]["network"]
        net_v = main_graph.vs[v]["network"]

        # Caso 1: energía ↔ transporte → siempre energía → transporte
        if (
            {"Energy network", "Roads network"} <= {net_u, net_v}
            or {"Energy network", "Railway network"} <= {net_u, net_v}
        ):
            if net_u == "Energy network":
                edges.append((u, v))
            else:
                edges.append((v, u))

        # Caso 2: misma red o transporte ↔ transporte → bidireccional
        elif net_u == net_v or (
            net_u in {"Roads network", "Railway network"} and
            net_v in {"Roads network", "Railway network"}
        ):
            edges.extend([(u, v), (v, u)])

    g_dir.add_edges(edges)
    return g_dir




def invert_graph(g_dir):
    """
    Crea una versión invertida del grafo dirigido:
    todas las aristas cambian de sentido, se conservan nodos y atributos.

    Devuelve:
        g_inv : igraph.Graph
    """
    if not g_dir.is_directed():
        raise ValueError("El grafo original debe ser dirigido.")

    g_inv = Graph(directed=True)
    g_inv.add_vertices(len(g_dir.vs))

    # Copiar atributos de nodos
    for i, v in enumerate(g_dir.vs):
        g_inv.vs[i].update_attributes(v.attributes())

    # Invertir aristas
    reversed_edges = [(v, u) for (u, v) in g_dir.get_edgelist()]
    g_inv.add_edges(reversed_edges)

    # Copiar atributos de aristas
    for e_inv, e_orig in zip(g_inv.es, g_dir.es):
        e_inv.update_attributes(e_orig.attributes())

    return g_inv
