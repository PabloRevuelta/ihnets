import osmnx as ox
import json

import networks_creation
import networks_intercon_users_flux
import resil_vuln_analysis
import plots


def main():

    roads_file_path = "C:\\Users\\revueltaap\\UNICAN\\EMCAN 2024 A2 ADAPTA - Documentos\\02_Tareas\\Proyecto redes\\DATABASES\\euro-global-map-shp\\euro-global-map-shp\\euro-global-map-shp\\RoadL.shp"
    cities_file_path = "C:\\Users\\revueltaap\\UNICAN\\EMCAN 2024 A2 ADAPTA - Documentos\\02_Tareas\\Proyecto redes\\DATABASES\\euro-global-map-shp\\euro-global-map-shp\\euro-global-map-shp\\BuiltupP.shp"

    rail_file_path = "C:\\Users\\revueltaap\\UNICAN\\EMCAN 2024 A2 ADAPTA - Documentos\\02_Tareas\\Proyecto redes\\DATABASES\\euro-regional-map-shp\\FullEurope\\data\\RailrdL.shp"
    stops_file_path = "C:\\Users\\revueltaap\\UNICAN\\EMCAN 2024 A2 ADAPTA - Documentos\\02_Tareas\\Proyecto redes\\DATABASES\\euro-regional-map-shp\\FullEurope\\data\\RailrdC.shp"

    energy_lines_file_path = "C:\\Users\\revueltaap\\UNICAN\\EMCAN 2024 A2 ADAPTA - Documentos\\02_Tareas\\Proyecto redes\\DATABASES\\OSM\\osm_power_lines_cantabria.shp"
    generation_points_file_path = "C:\\Users\\revueltaap\\UNICAN\\EMCAN 2024 A2 ADAPTA - Documentos\\02_Tareas\\Proyecto redes\\DATABASES\\Combined\\combined_osm_power_cantabria_global_power.shp"
    substations_file_path = "C:\\Users\\revueltaap\\UNICAN\\EMCAN 2024 A2 ADAPTA - Documentos\\02_Tareas\\Proyecto redes\\DATABASES\\OSM\\osm_power_subest_cantabria.shp"

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

    extra_file_paths={'tunnels file path':"C:\\Users\\revueltaap\\UNICAN\\EMCAN 2024 A2 ADAPTA - Documentos\\02_Tareas\\Proyecto redes\\DATABASES\\OSM\\osm_tunnels_cantabria.shp",
                      'cities file path':"C:\\Users\\revueltaap\\UNICAN\\EMCAN 2024 A2 ADAPTA - Documentos\\02_Tareas\\Proyecto redes\\DATABASES\\euro-global-map-shp\\euro-global-map-shp\\euro-global-map-shp\\BuiltupP.shp"
    }

    ################
    # En la versión final, habrá ver como meter sin simular usuarios y flujos en nodos y aristas de las redes y las interconexiones
    ################

    gdf_cut = ox.geocode_to_gdf("Cantabria, Spain")

    networks_creation.networks_creation(networks_dic,gdf_cut,extra_file_paths)

    main_graph=networks_intercon_users_flux.networks_interconnection_users_flux(networks_dic,interconnections_dic,extra_file_paths)

    print('Networks created')

    plots.plots_networks(networks_dic, interconnections_dic, main_graph, gdf_cut)

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

    plots.plots_resil_vun_analysis(scenarios_dic)


########################################################################################################################


if __name__ == "__main__":
    main()