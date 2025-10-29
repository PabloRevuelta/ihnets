import osmnx as ox
import json

import networks_creation
import networks_intercon_users_flux
import resil_vuln_analysis
import build_directed_graph
import plots

import matplotlib.pyplot as plt
from igraph import plot


def main():

    roads_file_path = r"C:\Users\santamariace\PycharmProjects\ihnets\data\RoadL\RoadL.shp"
    cities_file_path = r"C:\Users\santamariace\PycharmProjects\ihnets\data\BuiltupP\BuiltupP.shp"

    rail_file_path = r"C:\Users\santamariace\PycharmProjects\ihnets\data\RailrdL\RailrdL.shp"
    stops_file_path = r"C:\Users\santamariace\PycharmProjects\ihnets\data\RailrdC\RailrdC.shp"

    energy_lines_file_path = r"C:\Users\santamariace\PycharmProjects\ihnets\data\osm_power_lines_cantabria\osm_power_lines_cantabria.shp"
    generation_points_file_path = r"C:\Users\santamariace\PycharmProjects\ihnets\data\combined_osm_power_cantabria_global_power\combined_osm_power_cantabria_global_power.shp"
    substations_file_path = r"C:\Users\santamariace\PycharmProjects\ihnets\data\osm_power_subest_cantabria\osm_power_subest_cantabria.shp"

    networks_dic = {
        'Roads network': {'lines file paths': {'roads': roads_file_path},
                          'nodes file paths': {'cities': cities_file_path},
                          'buffer distance': 0.01, 'buffer option': 'to lines'},
        'Railway network': {'lines file paths': {'train tracks': rail_file_path},
                            'nodes file paths': {'stations': stops_file_path},
                            'buffer distance': None, 'buffer option': None},
        'Energy network': {'lines file paths': {'lines': energy_lines_file_path},
                           'nodes file paths': {'power sources': generation_points_file_path,
                                                'substations': substations_file_path},
                           'buffer distance': 0.005, 'buffer option': 'to nodes'}
    }

    interconnections_dic = {
        'Roads network - Railway network': {'connected elements': ['all', 'all'], 'method': 'all in buffer distance',
                                            'buffer distance': 0.015},
        'Railway network - Energy network': {'connected elements': ['all', 'substations'], 'method': 'closest'},
        'Roads network - Energy network': {'connected elements': ['tunnel', 'substations'], 'method': 'closest'},
    }

    extra_file_paths = {
        'tunnels file path': r"C:\Users\santamariace\PycharmProjects\ihnets\data\osm_tunnels_cantabria\osm_tunnels_cantabria.shp",
        'cities file path': r"C:\Users\santamariace\PycharmProjects\ihnets\data\BuiltupP\BuiltupP.shp"
        }

    ################
    # En la versión final, habrá ver como meter sin simular usuarios y flujos en nodos y aristas de las redes y las interconexiones
    ################

    gdf_cut = ox.geocode_to_gdf("Cantabria, Spain")

    networks_creation.networks_creation(networks_dic,gdf_cut,extra_file_paths)

    main_graph=networks_intercon_users_flux.networks_interconnection_users_flux(networks_dic,interconnections_dic,extra_file_paths)

    print("Atributos de nodos:", main_graph.vs.attribute_names())
    print("Atributos de aristas:", main_graph.es.attribute_names())

    print('Original Network created')

    #Comienzo con la creación de mi nueva red dirigida

    directed_graph = build_directed_graph.build_directed_graph(main_graph)

    # Comparar número de nodos y aristas
    print("Nodos (original vs dirigido):", len(main_graph.vs), len(directed_graph.vs))
    print("Aristas (original vs dirigido):", len(main_graph.es), len(directed_graph.es))

    # Comparar nombres de atributos de nodos
    attrs_main = set(main_graph.vs.attribute_names())
    attrs_dir = set(directed_graph.vs.attribute_names())
    print("Atributos de nodos (original):", attrs_main)
    print("Atributos de nodos (dirigido):", attrs_dir)
    print("Coinciden los atributos:", attrs_main == attrs_dir)

    # Ver atributos de los primeros 5 nodos
    print("\nPrimeros 5 nodos del grafo original:")
    for v in main_graph.vs[:5]:
        print(v.index, v.attributes())

    print("\nPrimeros 5 nodos del grafo dirigido:")
    for v in directed_graph.vs[:5]:
        print(v.index, v.attributes())

    one_way_edges = [
        (e.source, e.target)
        for e in directed_graph.es
        if not directed_graph.are_adjacent(e.target, e.source)
    ]
    g_oneway = directed_graph.subgraph_edges(one_way_edges, delete_vertices=False)

    # Layout basado en coordenadas geográficas
    layout = [(v["geometry"].x, v["geometry"].y) for v in g_oneway.vs]

    # Colores por tipo de red
    color_map = []
    for v in g_oneway.vs:
        tipo = v["network"]
        if tipo == "Energy network":
            color_map.append("gold")
        elif tipo == "Roads network":
            color_map.append("lightcoral")
        elif tipo == "Railway network":
            color_map.append("skyblue")
        else:
            color_map.append("gray")

    # Plot del grafo unidireccional
    fig, ax = plt.subplots(figsize=(12, 9))
    plot(
        g_oneway,
        target=ax,
        layout=layout,
        vertex_color=color_map,
        vertex_size=8,
        vertex_label=[str(v.index) for v in g_oneway.vs],
        vertex_label_size=12,
        edge_color="black",
        edge_arrow_size=0.6,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Aristas unidireccionales (energía → transporte)")
    plt.show()

    return

    #plots.plots_networks(networks_dic, interconnections_dic, main_graph, gdf_cut)

    fail_drop = 1.0  # Total
    t_0 = 1.0
    params_dic = {'tFa': 1.0, 'Rc0': 1.0, 'tRc': 4.0}
    dt = 0.1  # (h)
    n = 1

    main_graph.vs["energy"] = [1] * main_graph.vcount()
    main_graph.es["energy"] = [1] * main_graph.ecount()

    scenarios_dic=resil_vuln_analysis.resil_vun_analysis(main_graph,t_0,n,params_dic, dt, fail_drop)

    with open("datos.json", "w", encoding="utf-8") as f:
        json.dump(scenarios_dic, f, ensure_ascii=False, indent=4)

    print('Analysis finished')

    #plots.plots_resil_vun_analysis(scenarios_dic)


########################################################################################################################


if __name__ == "__main__":
    main()