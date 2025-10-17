import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import os

def plots_networks(networks_dic,networks_intercon_dic,combined_graph,gdf_cut):

    for network, dic in networks_dic.items():
        plot_gdf(network, dic['lines gdf'], dic['nodes gdf'], gdf_cut)
        network_nodes_types = [key for key in dic['nodes file paths'].keys()]
        node_colors = ["#D1495B" if v["type"] in network_nodes_types else "#E9C46A" for v in
                       networks_dic[network]["igraph"].vs]
        plot_ig_graph(network, networks_dic[network]['igraph'], gdf_cut, node_colors, "#6BA292")
        layout = networks_dic[network]['igraph'].layout("fr")  # "fr" = Fruchterman-Reingold, similar a nx.spring_layout
        plot_ig_asbtract(network, layout, networks_dic[network]['igraph'], node_colors, '#6BA292')

    for networks, dic in networks_intercon_dic.items():
        network_1 = networks.split(" - ")[0].strip()
        node_colors = ['#D1495B' if v['network'] == network_1 else '#3B7EA1' for v in dic['igraph'].vs]
        plot_ig_graph(networks, dic['igraph'], gdf_cut, node_colors, '#6BA292')

    nodes_color_map = {
        "Roads network": "#D1495B",  # terracota
        "Railway network": "#3B7EA1",  # azul
        "Energy network": "#7CA982"  # verde
    }
    # Colores de aristas (más claros)
    edge_color_map = {
        "Roads network": "#E2959B",
        "Railway network": "#85B1D1",
        "Energy network": "#A0C29D"
    }
    # Asignar colores a cada nodo
    node_colors = [nodes_color_map.get(v["network"]) for v in combined_graph.vs]
    edges_colors = [edge_color_map.get(e["network"], '#F4D35E') for e in combined_graph.es]

    plot_ig_graph("combinado", combined_graph, gdf_cut, node_colors, edges_colors)

    # --- 1️⃣ Separar nodos por red
    nodes_A = [v.index for v in combined_graph.vs if v["network"] == 'Roads network']
    nodes_B = [v.index for v in combined_graph.vs if v["network"] == 'Railway network']
    nodes_C = [v.index for v in combined_graph.vs if v["network"] == 'Energy network']
    sub_A = combined_graph.subgraph(nodes_A)
    sub_B = combined_graph.subgraph(nodes_B)
    sub_C = combined_graph.subgraph(nodes_C)
    # --- 2️⃣ Layouts internos (FR)
    layout_A = np.array(sub_A.layout("fr"))
    layout_B = np.array(sub_B.layout("fr"))
    layout_C = np.array(sub_C.layout("fr"))
    # Normalizar tamaños
    if layout_A.shape[0] > 0:
        layout_A = layout_A / np.max(np.abs(layout_A)) * 5
    if layout_B.shape[0] > 0:
        layout_B = layout_B / np.max(np.abs(layout_B)) * 5
    if layout_C.shape[0] > 0:
        layout_C = layout_C / np.max(np.abs(layout_C)) * 5
    # --- 3️⃣ Trasladar bloques para separar redes
    layout_A[:, 0] -= 10  # A hacia la izquierda
    layout_B[:, 0] += 0  # B centrado
    layout_C[:, 0] += 10  # C hacia la derecha
    # --- 4️⃣ Combinar layouts en el layout final
    layout = np.zeros((combined_graph.vcount(), 2))
    for i, node in enumerate(nodes_A):
        layout[node] = layout_A[i]
    for i, node in enumerate(nodes_B):
        layout[node] = layout_B[i]
    for i, node in enumerate(nodes_C):
        layout[node] = layout_C[i]
    layout = layout.tolist()
    # --- 5️⃣ Llamar a la función de plotting abstracto
    plot_ig_asbtract("Redes completas", layout, combined_graph, node_colors, edges_colors)

    #plot_ig_users_flux(transport_graph, gdf_cut, 'black', 'blue')
    #plot_ig_users(networks_dic['Energy network']["igraph"], gdf_cut, 'black', 'blue')

def plots_resil_vun_analysis(scenarios_dic):

    plt.figure(figsize=(8, 5))
    for node, dic in scenarios_dic.items():
        lista = [sum(x) for x in zip(*dic.values())]
        x = [i * 0.1 for i in range(len(lista))]
        plt.plot(x, lista, marker='.', markersize=1, label=node)  # gráfica de línea con puntos
        # plt.text(x[-1] + 0.1, lista[-1], node, va="center")
    plt.title("Network performance curves with 1 failed asset at t=1h with 100% drop")
    plt.xlabel("Network performance (nº users)")
    plt.ylabel("t (h)")
    plt.grid(True)
    file_name ="performance curves all scens.png"
    plt.savefig(file_name, dpi=1200, bbox_inches="tight")
    plt.close()
    os.startfile(file_name)

    failed_asset = list(scenarios_dic.keys())[1]

    plt.figure(figsize=(8, 5))
    lista = [sum(x) for x in zip(*scenarios_dic[failed_asset].values())]
    x = [i * 0.1 for i in range(len(lista))]
    plt.plot(x, lista, marker='.', markersize=1)  # gráfica de línea con puntos
    plt.title("Network performance curve with 1 failed asset at t=1h with 100% drop. Failed asset: "+failed_asset)
    plt.xlabel("Network performance (nº users)")
    plt.ylabel("t (h)")
    plt.grid(True)
    file_name ="performance curve failed "+failed_asset+".png"
    plt.savefig(file_name, dpi=1200, bbox_inches="tight")
    plt.close()
    os.startfile(file_name)

    plt.figure(figsize=(8, 5))
    for node, lista in scenarios_dic[failed_asset].items():
        # Solo plotear si hay algún cambio en la lista
        if max(lista) - lista[0] > 1 or lista[0] - min(lista) > 1:
            x = [i * 0.1 for i in range(len(lista))]
            plt.plot(x, lista, marker='.', markersize=1, label=node)
    plt.title(failed_asset)
    plt.title("Node performance curve with 1 failed asset at t=1h with 100% drop. Failed asset: " + failed_asset)
    plt.xlabel("Node performance (nº users)")
    plt.ylabel("t (h)")
    plt.grid(True)
    plt.legend()
    file_name ="nodes performance failed "+failed_asset+".png"
    plt.savefig(file_name, dpi=1200, bbox_inches="tight")
    plt.close()
    os.startfile(file_name)

def plot_gdf(network,gdf_lines,gdf_nodes,gdf_cut):
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_cut.plot(ax=ax, color="lightgrey", edgecolor="black", figsize=(8, 8))
    gdf_lines.plot(ax=ax, color="#6BA292", edgecolor="black", linewidth=0.8)
    gdf_nodes.plot(ax=ax, color="#D1495B", markersize=6,edgecolor="black",linewidth=0.3)

    plt.title("Mapa de "+network)
    file_name="mapa "+network+".png"
    plt.savefig(file_name, dpi=600, bbox_inches="tight")
    plt.close()
    os.startfile(file_name)

def plot_ig_graph(network, g_ig, gdf_cut, node_colors, edge_colors):
    # Extraer coordenadas reales de los nodos
    x_coords = [v["geometry"].x for v in g_ig.vs]
    y_coords = [v["geometry"].y for v in g_ig.vs]
    coords_layout = list(zip(x_coords, y_coords))

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("white")

    # 1️⃣ Capa base (área de corte)
    if gdf_cut is not None:
        gdf_cut.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.8, zorder=0)

      # Dibujar grafo
    ig.plot(
        g_ig,
        target=ax,
        layout=coords_layout,
        vertex_size=30,
        vertex_color=node_colors,
        vertex_frame_width=0.5,
        vertex_frame_color="black",
        edge_color=edge_colors,
        edge_width=1.5,
        edge_curved=False,
        vertex_label=None,
        margin=30
    )

    # Ajustar límites al área del corte
    if gdf_cut is not None:
        xmin, ymin, xmax, ymax = gdf_cut.total_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    plt.title("Grafo geométrico de "+network)

    # Guardar archivo
    file_name = f"grafo_geom_{network}.png"
    plt.tight_layout()
    plt.savefig(file_name, dpi=600, bbox_inches="tight")
    plt.close()
    os.startfile(file_name)

def plot_ig_asbtract(network,layout, g_ig, node_colors,edge_colors):

    # 2️⃣ Dibujar el grafo
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("white")

       # Dibujar grafo
    ig.plot(
        g_ig,
        target=ax,
        layout=layout,
        vertex_size=30,
        vertex_color=node_colors,
        vertex_frame_width=0.5,
        vertex_frame_color="black",
        edge_color=edge_colors,
        edge_width=1.5,
        edge_curved=False,
        vertex_label=None,
        margin=30
    )

    plt.title("Grafo abstracto de "+network)

    # Guardar y abrir
    file_name ="grafo abs "+network+".png"
    plt.savefig(file_name, dpi=600, bbox_inches="tight")
    plt.close()
    os.startfile(file_name)

def plot_ig_users_flux(g_ig, gdf_cut, node_colors,edge_colors):
    """
    Dibuja un grafo no dirigido mostrando flechas simuladas desde cada nodo
    para indicar el flujo que sale por cada arista.
    Cada arista tiene dos atributos: flujo desde cada nodo.
    """

    # Extraer coordenadas
    x_coords = [v['geometry'].x for v in g_ig.vs]
    y_coords = [v['geometry'].y for v in g_ig.vs]
    coords_layout = list(zip(x_coords, y_coords))
    '''layout = g_ig.layout("kk")
    coords_layout = [(x, y) for x, y in layout.coords]
    x_coords = [c[0] for c in coords_layout]
    y_coords = [c[1] for c in coords_layout]'''

    # Creamos listas para flechas y etiquetas
    arrows = []

    for e in g_ig.es:
        u_idx, v_idx = e.source, e.target
        u_name = str(g_ig.vs[u_idx]["name"])
        v_name = str(g_ig.vs[v_idx]["name"])

        # Flujo desde cada nodo
        flow_u = e[u_name]
        flow_v = e[v_name]

        # Simulamos flecha desde u a v
        arrows.append(((coords_layout[u_idx], coords_layout[v_idx]), flow_u))
        # Simulamos flecha desde v a u
        arrows.append(((coords_layout[v_idx], coords_layout[u_idx]), flow_v))

    # Plot con matplotlib
    fig, ax = plt.subplots(figsize=(12,10))
    gdf_cut.plot(ax=ax, facecolor="none", edgecolor="black")
    ax.set_aspect('equal')
    ax.set_title("Flujos de nodos")

    # Dibujar nodos
    ax.scatter(x_coords, y_coords, s=5, c='red', zorder=3)
    for i, v in enumerate(g_ig.vs):
        ax.text(
            x_coords[i],
            y_coords[i],
             f"{v['name']}\n({v['users']})",  # Texto: nombre (usuarios)
            fontsize=1,
            color=node_colors,        # Contrasta mejor con el nodo rojo
            ha="center",          # Centrado horizontal
            va="center",          # Centrado vertical
            zorder=4,
        )


    # Dibujar flechas y etiquetas
    for (start, end), flow in arrows:
        # Dibujar la flecha
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="->", color=edge_colors, lw=0.8),
            zorder=2
        )

        # Vector de la arista
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = (dx**2 + dy**2)**0.5

        # Posición del texto: cerca del inicio, a un 15% de la arista
        t = 0.15
        x_text = start[0] + t * dx
        y_text = start[1] + t * dy

        # Desplazamiento perpendicular opcional para evitar solapamientos
        if length > 0:
            perp_offset = 0.02 * length
            x_text += -perp_offset * dy / length
            y_text += perp_offset * dx / length

        # Dibujar la etiqueta
        ax.text(x_text, y_text, f"{flow:.2f}",
                fontsize=1, color='black', zorder=4,
                ha='center', va='center')

    xmin, ymin, xmax, ymax = gdf_cut.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.title("Grafo de usuarios y flujos de transporte")

    file_name ="grafo_ig_transport_users_flux.png"
    plt.savefig(file_name, dpi=600, bbox_inches="tight")
    plt.close()
    os.startfile(file_name)

def plot_ig_users(g_ig, gdf_cut, node_colors, edge_colors):
    """
    Dibuja un grafo no dirigido mostrando solo nodos y aristas.
    Cada nodo tiene su nombre y número de usuarios.
    """
    # Extraer coordenadas
    x_coords = [v['geometry'].x for v in g_ig.vs]
    y_coords = [v['geometry'].y for v in g_ig.vs]
    coords_layout = list(zip(x_coords, y_coords))

    # Crear figura
    fig, ax = plt.subplots(figsize=(12,10))
    ax.set_facecolor("white")

    # Dibujar capa base si existe
    if gdf_cut is not None:
        gdf_cut.plot(ax=ax, facecolor="none", edgecolor="black")

    # Dibujar aristas
    for e in g_ig.es:
        u_idx, v_idx = e.source, e.target
        x = [coords_layout[u_idx][0], coords_layout[v_idx][0]]
        y = [coords_layout[u_idx][1], coords_layout[v_idx][1]]
        ax.plot(x, y, color=edge_colors, lw=0.8, zorder=1)

    # Dibujar nodos
    ax.scatter(x_coords, y_coords, s=20, c=node_colors, edgecolor="black", linewidth=0.5, zorder=2)

    # Etiquetas de nodos: nombre y número de usuarios
    for i, v in enumerate(g_ig.vs):
        ax.text(
            x_coords[i],
            y_coords[i],
            f"{v['name']}\n({v['users']})",
            fontsize=6,
            color="black",
            ha="center",
            va="center",
            zorder=3
        )

    # Ajustar límites al área del corte
    if gdf_cut is not None:
        xmin, ymin, xmax, ymax = gdf_cut.total_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    plt.title("Grafo de usuarios de energía")

    # Guardar y mostrar
    file_name ="grafo_ig_energy_users.png"
    plt.tight_layout()
    plt.savefig(file_name, dpi=600, bbox_inches="tight")
    plt.close()
    os.startfile(file_name)