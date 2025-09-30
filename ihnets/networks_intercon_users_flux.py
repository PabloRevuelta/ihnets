import igraph as ig
import random
import geopandas as gpd

def networks_interconnection_users_flux(networks_dic, interconnections_dic, extra_file_paths):
    networks_intercon(networks_dic, interconnections_dic)

    transport_igraphs_list = [networks_dic['Roads network']["igraph"], networks_dic['Railway network']["igraph"],
                              interconnections_dic['Roads network - Railway network']["igraph"]]
    transport_graph = ig.union(transport_igraphs_list, byname=True)
    assign_users_and_flux_transport(transport_graph)

    assign_users_and_flux_energy(networks_dic['Energy network']["igraph"], networks_dic['Energy network']["nodes gdf"],
                                 extra_file_paths['cities file path'])
    assign_users_and_flux_energy_intercons(interconnections_dic, networks_dic)

    all_graphs_list = [interconnections_dic['Roads network - Energy network']["igraph"],
                       interconnections_dic['Railway network - Energy network']["igraph"], transport_graph,
                       networks_dic['Energy network']["igraph"]]
    main_graph = ig.union(all_graphs_list, byname=True)

    return main_graph

def intercon_igraph(interconnection, g_combined, intercon_dic):
    network_1, network_2 = interconnection.split(" - ")

    if intercon_dic['connected elements'][0] == 'all':
        target_nodes_1 = [v for v in g_combined.vs if v['network'] == network_1]
    else:
        target_nodes_1 = [v for v in g_combined.vs if
                          v['network'] == network_1 and v['type'] == intercon_dic['connected elements'][0]]

    if intercon_dic['connected elements'][1] == 'all':
        target_nodes_2 = [v for v in g_combined.vs if v['network'] == network_2]
    else:
        target_nodes_2 = [v for v in g_combined.vs if
                          v['network'] == network_2 and v['type'] == intercon_dic['connected elements'][1]]

        # 2️⃣ Iterar sobre pares y conectar si cumplen distancia
    if intercon_dic['method'] == 'all in buffer distance':
        for v1 in target_nodes_1:
            geom1 = v1['geometry']
            for v2 in target_nodes_2:
                geom2 = v2['geometry']
                if geom1.distance(geom2) <= intercon_dic['buffer distance']:
                    # Añadir arista entre los nodos
                    g_combined.add_edge(v1.index, v2.index, network=interconnection, type='interconnection',
                                        name=f"{interconnection} edge {v1.index}-{v2.index}")

    elif intercon_dic['method'] == 'closest':
        # Iterar sobre todos los nodos de net1
        for v1 in target_nodes_1:
            geom1 = v1["geometry"]

            # Buscar el nodo más cercano en net2
            nearest_node = min(
                target_nodes_2,
                key=lambda v2: geom1.distance(v2["geometry"])
            )

            # Añadir arista al grafo con tipo y geometry
            g_combined.add_edge(v1.index, nearest_node.index, network=interconnection, type="interconnection",
                                name=f"{interconnection} edge {v1.index}-{nearest_node.index}")

    return g_combined

def networks_intercon(networks_dic, interconnections_dic):
    allowed_types = ['tunnel']
    for dic in networks_dic.values():
        for node_keys in dic['nodes file paths'].keys():
            allowed_types.append(node_keys)

    for interconnection, intercon_dic in interconnections_dic.items():

        network_1, network_2 = interconnection.split(" - ")

        # Crear un grafo vacío
        g_combined = ig.Graph(directed=False)

        # Añadir los nodos de g1
        for v in networks_dic[network_1]['igraph'].vs:
            if v["type"] in allowed_types:
                g_combined.add_vertex(**v.attributes())
        # Añadir los nodos de g2
        for v in networks_dic[network_2]['igraph'].vs:
            if v["type"] in allowed_types:
                g_combined.add_vertex(**v.attributes())

        intercon_dic['igraph'] = intercon_igraph(interconnection, g_combined, intercon_dic)

def assign_users_and_flux_transport(g_ig):
    # -------------------------------
    # Usuarios por nodo
    # -------------------------------
    g_ig.vs["users"] = [random.randint(50, 200) for _ in g_ig.vs]

    # -------------------------------
    # Inicializar flujos de salida por aristas
    # -------------------------------

    shares_by_node = {v["name"]: {} for v in g_ig.vs}

    # Asignar porcentajes de salida a las aristas (sum ≤ 1)
    for v in g_ig.vs:
        neighbors = g_ig.neighbors(v, mode="ALL")
        if not neighbors:
            continue

        weights = [random.random() for _ in neighbors]
        total = sum(weights)
        factor = random.uniform(0.5, 0.9)  # suma de salidas ≤1
        shares = [round((w / total) * factor, 2) for w in weights]

        for n, s in zip(neighbors, shares):
            shares_by_node[v["name"]][g_ig.vs[n]["name"]] = s

    # Asignar atributos a las aristas
    for e in g_ig.es:
        u_name, v_name = g_ig.vs[e.source]["name"], g_ig.vs[e.target]["name"]
        e[u_name] = shares_by_node[u_name].get(v_name)
        e[v_name] = shares_by_node[v_name].get(u_name)

    # -------------------------------
    # Calcular flujo de llegada
    # -------------------------------

    for v in g_ig.vs:
        name = v["name"]
        incoming_flows = []
        total_incoming = 0

        # Recoger todos los flujos entrantes
        for e in g_ig.es:
            u_name, w_name = g_ig.vs[e.source]["name"], g_ig.vs[e.target]["name"]
            if w_name == name:  # flujo desde u hacia v
                flow_value = e[u_name] * g_ig.vs[e.source]["users"]
                incoming_flows.append((e, "source", flow_value))
                total_incoming += flow_value
            elif u_name == name:  # flujo desde w hacia v
                flow_value = e[w_name] * g_ig.vs[e.target]["users"]
                incoming_flows.append((e, "target", flow_value))
                total_incoming += flow_value

        # Escalar flujos si exceden users
        if total_incoming > v["users"]:
            factor = v["users"] / total_incoming
            for e, attr, _ in incoming_flows:
                u_name, v_name = g_ig.vs[e.source]["name"], g_ig.vs[e.target]["name"]
                if attr == "source":
                    e[u_name] *= factor
                else:
                    e[v_name] *= factor

def assign_users_and_flux_energy(g_ig_energy, gdf_energy_nodes, cities_file_path):
    import warnings
    from collections import Counter

    warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

    gdf_cities = gpd.read_file(cities_file_path)
    gdf_cities = gdf_cities.to_crs(gdf_energy_nodes.crs)

    # --- 0. Filtrar subestaciones ---
    gdf_substations_sel = gdf_energy_nodes[gdf_energy_nodes["type"] == "substations"]

    # --- 1. Subestación -> ciudad más cercana ---
    sub_to_city = {}
    for i, sub_geom in gdf_substations_sel.geometry.items():
        dists = gdf_cities.geometry.distance(sub_geom)
        idx_nearest = dists.idxmin()
        sub_to_city[i] = idx_nearest

    # --- 2. Contar cuántas subestaciones por ciudad ---
    city_counts = Counter(sub_to_city.values())

    # --- 3. Calcular PPL_per_sub en un diccionario auxiliar ---
    city_to_ppl = {}
    for idx, row in gdf_cities.iterrows():
        n_subs = city_counts.get(idx, 0)
        if n_subs > 0:
            city_to_ppl[idx] = row["PPL"] / n_subs
        else:
            city_to_ppl[idx] = None

    # --- 4. Construir diccionario final solo con esas subestaciones ---
    sub_to_ppl_dict = {}
    for sub_idx, city_idx in sub_to_city.items():
        sub_geom = gdf_substations_sel.geometry.loc[sub_idx]
        ppl_value = city_to_ppl[city_idx]
        sub_to_ppl_dict[sub_geom] = ppl_value

    # --- Añadir atributo "users" a todos los nodos ---
    users_vals = []
    for v in g_ig_energy.vs:
        geom = v["geometry"]
        if geom in sub_to_ppl_dict:
            users_vals.append(sub_to_ppl_dict[geom])
        else:
            users_vals.append(0)

    g_ig_energy.vs["users"] = [round(u) for u in users_vals]
    g_ig_energy.vs["energy"] = 1

def assign_users_and_flux_energy_intercons(interconnections_dic, networks_dic):
    for e in interconnections_dic['Railway network - Energy network']['igraph'].es:
        u = interconnections_dic['Railway network - Energy network']['igraph'].vs[e.source]
        v = interconnections_dic['Railway network - Energy network']['igraph'].vs[e.target]

        geom_u = u["geometry"]
        geoms_ig_energy_set = set(networks_dic['Energy network']['igraph'].vs["geometry"])
        if geom_u in geoms_ig_energy_set:
            e[u['name']] = 1
        else:
            e[v['name']] = 1

    for e in interconnections_dic['Roads network - Energy network']['igraph'].es:
        u = interconnections_dic['Roads network - Energy network']['igraph'].vs[e.source]
        v = interconnections_dic['Roads network - Energy network']['igraph'].vs[e.target]

        geom_u = u["geometry"]
        geoms_ig_energy_set = set(networks_dic['Energy network']['igraph'].vs["geometry"])
        if geom_u in geoms_ig_energy_set:
            e[u['name']] = 1
        else:
            e[v['name']] = 1