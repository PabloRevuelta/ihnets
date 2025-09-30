import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import os

def plots_networks(networks_dic,networks_intercon_dic,combined_graph,gdf_cut):
    for network, dic in networks_dic.items():

        plot_gdf(network,networks_dic[network]['lines gdf'],networks_dic[network]['nodes gdf'],gdf_cut)

        node_colors = ['red' if v['type'] == 'station' else 'yellow' for v in networks_dic[network]['igraph'].vs]
        plot_ig_graph(network, networks_dic[network]['igraph'], gdf_cut, node_colors,'blue')

        layout = networks_dic[network]['igraph'].layout("fr")  # "fr" = Fruchterman-Reingold, similar a nx.spring_layout
        plot_ig_asbtract(network, layout, networks_dic[network]['igraph'], node_colors, 'blue')

    for networks, g_ig in networks_intercon_dic.items():
        network_1 = networks.split("+")[0].strip()
        node_colors = [
            'red' if v['F_CODE'] in networks_dic[network_1]['nodes gdf']["F_CODE"].unique().tolist() else 'yellow' for v
            in g_ig.vs]
        plot_ig_graph("interconexiones", g_ig, gdf_cut, node_colors, 'blue')

    node_colors = ['red' if v['network'] == 'Roads network' else 'yellow' for v in combined_graph.vs]
    plot_ig_graph("combinado", combined_graph, gdf_cut, node_colors,'blue')

    nodes_A = [v.index for v in combined_graph.vs if v["network"] == 'Roads network']
    nodes_B = [v.index for v in combined_graph.vs if v["network"] == 'Railway network']
    sub_A = combined_graph.subgraph(nodes_A)
    sub_B = combined_graph.subgraph(nodes_B)
    # --- 2️⃣ Layouts internos (FR)
    layout_A = np.array(sub_A.layout("fr"))
    layout_B = np.array(sub_B.layout("fr"))
    # Normalizar tamaños
    if layout_A.shape[0] > 0:
        layout_A = layout_A / np.max(np.abs(layout_A)) * 5
    if layout_B.shape[0] > 0:
        layout_B = layout_B / np.max(np.abs(layout_B)) * 5
    # --- 3️⃣ Trasladar bloques
    layout_A[:, 0] -= 10   # A hacia la izquierda
    layout_B[:, 0] += 10   # B hacia la derecha
    # --- 4️⃣ Combinar layouts
    layout = np.zeros((combined_graph.vcount(), 2))
    for i, node in enumerate(nodes_A):
        layout[node] = layout_A[i]
    for i, node in enumerate(nodes_B):
        layout[node] = layout_B[i]
    layout=layout.tolist()
    plot_ig_asbtract("combinado",layout,combined_graph,node_colors,'blue')

    plot_ig_users_flux(combined_graph, gdf_cut, 'black', 'blue')

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

    '''for node, dic in scenarios_dic.items():
        plt.figure(figsize=(8, 5))
        for nodes, lista in scenarios_dic[node].items():
            # Solo plotear si hay algún cambio en la lista
            if max(lista) - lista[0] > 1 or lista[0] - min(lista) > 1:
                x = [i * 0.1 for i in range(len(lista))]
                plt.plot(x, lista, marker='.', markersize=1, label=nodes)
        plt.title(node)
        plt.title("Node performance curve with 1 failed asset at t=1h with 100% drop. Failed asset: " + failed_asset)
        plt.xlabel("Node performance (nº users)")
        plt.ylabel("t (h)")
        plt.grid(True)
        plt.legend()
        plt.show()'''

def plot_gdf(network,gdf_lines,gdf_nodes,gdf_cut):
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_cut.plot(ax=ax, color="lightgrey", edgecolor="black", figsize=(8, 8))
    gdf_lines.plot(ax=ax, color="green", edgecolor="black", linewidth=0.5)
    gdf_nodes.plot(ax=ax, color="red", markersize=0.5)

    plt.title("Mapa de "+network)
    file_name="mapa "+network+".png"
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()
    os.startfile(file_name)

def plot_ig_graph(network, g_ig, gdf_cut, node_colors,edge_colors):
    # Extraer coordenadas de los nodos
    x_coords = [v['geometry'].x for v in g_ig.vs]
    y_coords = [v['geometry'].y for v in g_ig.vs]
    coords_layout = list(zip(x_coords, y_coords))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Dibujar capa de cortes si existe
    if gdf_cut is not None:
        gdf_cut.plot(ax=ax, facecolor="none", edgecolor="black")

    ig.plot(
        g_ig,
        target=ax,
        layout=coords_layout,  # <-- usamos las coordenadas reales
        vertex_size=20,
        vertex_color=node_colors,
        edge_color=edge_colors,
        vertex_frame_width=3,
        edge_width=1,
        vertex_label=None,
        margin=20
    )

    # Ajustar límites
    if gdf_cut is not None:
        xmin, ymin, xmax, ymax = gdf_cut.total_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    plt.title("Grafo geométrico de "+network)

    # Guardar y abrir
    file_name ="grafo geom "+network+".png"
    plt.savefig(file_name, dpi=1200, bbox_inches="tight")
    plt.close()
    os.startfile(file_name)

def plot_ig_asbtract(network,layout, g_ig, node_colors,edge_colors):

    # 2️⃣ Dibujar el grafo
    fig, ax = plt.subplots(figsize=(10, 8))

    g_ig = g_ig.simplify(combine_edges=None)

    ig.plot(
        g_ig,
        target=ax,  # dibujar sobre matplotlib
        layout=layout,  # layout abstracto
        vertex_size=1,
        edge_width=0.5,  # tamaño de los nodos
        vertex_color=node_colors,  # color de nodos
        edge_color=edge_colors,  # color de aristas
        vertex_label=None,  # sin etiquetas
        bbox=(1000, 800),  # tamaño de la imagen en pixeles
        margin=20
    )
    plt.title("Grafo abstracto de "+network)

    # Guardar y abrir
    file_name ="grafo abs "+network+".png"
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
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
    edge_labels = []
    edge_widths = []

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

    plt.title("Grafo de usuarios y flujos")

    file_name ="grafo_ig_users_flux.png"
    plt.savefig(file_name, dpi=1200, bbox_inches="tight")
    plt.close()
    os.startfile(file_name)