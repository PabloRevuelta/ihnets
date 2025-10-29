from igraph import Graph

# build_directed_graph.py
from igraph import Graph

def build_directed_graph(main_graph):
    """
    Construye una versión dirigida del grafo original según reglas entre redes.

    Parámetros
    ----------
    main_graph : igraph.Graph
        Grafo original no dirigido, con nodos que incluyen al menos el atributo 'network'.

    Reglas de dirección
    -------------------
    - Energy network → Roads/Railway network: arista unidireccional (solo de energía hacia transporte)
    - Roads/Railway → Energy network: no se añade arista inversa
    - Entre nodos de la misma red o entre Roads ↔ Railway: aristas bidireccionales

    Devuelve
    --------
    g_dir : igraph.Graph
        Grafo dirigido que conserva todos los atributos de los nodos y replica las aristas
        conforme a las reglas de direccionalidad definidas.
    """
    g_dir = Graph(directed=True)
    g_dir.add_vertices(len(main_graph.vs))

    for i, v in enumerate(main_graph.vs):
        g_dir.vs[i].update_attributes(v.attributes())

    edges = []
    for e in main_graph.es:
        u, v = e.tuple
        net_u = main_graph.vs[u]["network"]
        net_v = main_graph.vs[v]["network"]

        if net_u == "Energy network" and net_v in {"Roads network", "Railway network"}:
            edges.append((u, v))
        elif net_v == "Energy network" and net_u in {"Roads network", "Railway network"}:
            continue
        else:
            edges.extend([(u, v), (v, u)])

    g_dir.add_edges(edges)

    return g_dir
