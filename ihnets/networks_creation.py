import itertools
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString, Point
import warnings
import igraph as ig
warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

def networks_creation(networks_dic,gdf_cut,extra_file_paths):
    for network, dic in networks_dic.items():
        networks_dic[network]['lines gdf'] = combine_gdfs(gdf_cut, dic['lines file paths'])
        networks_dic[network]['nodes gdf'] = combine_gdfs(gdf_cut, dic['nodes file paths'])

    add_tunnels(networks_dic['Roads network']['lines gdf'], extra_file_paths['tunnels file path'])

    for network, dic in networks_dic.items():
        networks_dic[network]['igraph'] = nx_to_igraph(network, gdf_to_nx(
            networks_dic[network]['lines gdf'],
            networks_dic[network]['nodes gdf'],
            networks_dic[network]['buffer distance'], networks_dic[network]['buffer option'],gdf_cut))

        dic['igraph'].vs["network"] = [network] * dic['igraph'].vcount()
        dic['igraph'].es["network"] = [network] * dic['igraph'].ecount()

def combine_gdfs(gdf_cut,paths_dic):
    """
    Une varios shapefiles en un solo GeoDataFrame:
    - reproyectados al CRS del ref_gdf
    - recortados (clip) al ref_gdf
    - con columna extra 'type' = clave del diccionario
    - solo mantiene 'type' + geometry
    """
    gdfs = []

    for key, filepath in paths_dic.items():
        # 1. Leer shapefile
        gdf = gpd.read_file(filepath)

        # 2. Reproyectar al CRS de referencia
        gdf = gdf.to_crs(gdf_cut.crs)

        # 3. Clip con el gdf de referencia
        gdf = gpd.clip(gdf, gdf_cut)

        # 4. Añadir columna con la clave
        gdf["type"] = key

        # 5. Mantener solo la columna 'source' + geometry
        gdf = gdf[["type", "geometry"]]

        gdfs.append(gdf)

    # 6. Concatenar todos en un único GeoDataFrame
    result = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdf_cut.crs)

    return result

def add_tunnels(lines_gdf,tunnels_path_file):
    import warnings
    warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

    # 1️⃣ Leer túneles y re-proyectar
    tunnels_gdf = gpd.read_file(tunnels_path_file)
    tunnels_gdf = tunnels_gdf.to_crs(lines_gdf.crs)

    # 2️⃣ Preparar carreteras y asegurar columna 'tunnel'
    gdf_roads = lines_gdf.copy()
    if 'tunnel' not in gdf_roads.columns:
        gdf_roads['tunnel'] = 'no'

    # 3️⃣ Crear buffer para los túneles
    buffer_distance = 0.001  # ajustar según CRS
    gdf_tunnels_buffer = tunnels_gdf.copy()
    gdf_tunnels_buffer['geometry'] = gdf_tunnels_buffer.geometry.buffer(buffer_distance)

    # 4️⃣ Spatial join
    intersects = gpd.sjoin(
        gdf_roads,
        gdf_tunnels_buffer,
        how="left",
        predicate="intersects"
    )

    # 5️⃣ Índices de carreteras que intersectan túneles
    roads_with_tunnel = intersects[~intersects['index_right'].isna()].index

    # 6️⃣ Marcar la columna 'tunnel'
    lines_gdf['tunnel'] = 'no'  # inicializar
    lines_gdf.loc[roads_with_tunnel, 'tunnel'] = 'yes'

def combine_lines_gdf(gdf):
    # 1. Asegurarse de que todas las geometrías son LineString
    def explode_multilines(gdff):
        out_geoms = []
        out_attrs = []
        for idx, row in gdff.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString):
                out_geoms.append(geom)
                out_attrs.append(row.drop("geometry"))
            elif isinstance(geom, MultiLineString):
                for part in geom.geoms:
                    out_geoms.append(part)
                    out_attrs.append(row.drop("geometry"))
        exploded = gpd.GeoDataFrame(out_attrs, geometry=out_geoms, crs=gdff.crs)
        return exploded

    gdf = explode_multilines(gdf)

    # 2. Construir grafo topológico
    g_nx = nx.Graph()
    for i, line in enumerate(gdf.geometry):
        coords = list(line.coords) # type: ignore
        for j in range(len(coords) - 1):
            p1, p2 = coords[j], coords[j + 1]
            g_nx.add_edge(p1, p2, line_id=i)

    # 3. Encontrar cadenas
    def find_chains(g_nx):
        visited_edges = set()
        chains = []

        for u, v in g_nx.edges():
            if (u, v) in visited_edges or (v, u) in visited_edges:
                continue

            chain = [u, v]
            visited_edges.add((u, v))
            visited_edges.add((v, u))

            # expandir hacia u
            current, prev = u, v
            while g_nx.degree[current] == 2:
                neighbors = [n for n in g_nx.neighbors(current) if n != prev]
                if not neighbors:
                    break
                nxt = neighbors[0]
                chain.insert(0, nxt)
                visited_edges.add((current, nxt))
                visited_edges.add((nxt, current))
                prev, current = current, nxt

            # expandir hacia v
            current, prev = v, u
            while g_nx.degree[current] == 2:
                neighbors = [n for n in g_nx.neighbors(current) if n != prev]
                if not neighbors:
                    break
                nxt = neighbors[0]
                chain.append(nxt)
                visited_edges.add((current, nxt))
                visited_edges.add((nxt, current))
                prev, current = current, nxt

            chains.append(chain)

        return chains

    chains = find_chains(g_nx)

    # 4️⃣ Fusionar geometrías y atributos
    merged_records = []
    for chain in chains:
        merged_line = LineString(chain)

        # índices de líneas originales en la cadena
        line_indices = set()
        for j in range(len(chain) - 1):
            edge_data = g_nx.get_edge_data(chain[j], chain[j + 1])
            line_indices.add(edge_data['line_id'])
        sub_gdf = gdf.iloc[list(line_indices)]

        new_record = {}

        # Caso especial: tunnel
        if "tunnel" in gdf.columns:
            if (sub_gdf["tunnel"] == "yes").any(): # type: ignore
                new_record["tunnel"] = "yes"
            else:
                new_record["tunnel"] = "no"

        # Otros atributos: conservar si son constantes
        for col in gdf.columns:
            if col in ("geometry", "tunnel"):
                continue
            values = sub_gdf[col].dropna().unique()
            if len(values) == 1:
                new_record[col] = values[0]
            else:
                new_record[col] = None  # conflicto → se pierde

        new_record["geometry"] = merged_line
        merged_records.append(new_record)

    merged_gdf = gpd.GeoDataFrame(merged_records, crs=gdf.crs)
    return merged_gdf

def gdf_to_nx_buffer_to_lines(lines_gdf,nodes_gdf,buffer_distance):

    # Grafo
    g_nx = nx.Graph()

    # Añadir nodos de estaciones
    for idx, est in nodes_gdf.iterrows():
        g_nx.add_node(idx, **est.to_dict())

    # Diccionario para localizar nodos por coordenadas (tuplas)
    coord_to_node = {tuple(est.geometry.coords[0]): idx for idx, est in nodes_gdf.iterrows()}

    cross_id = 0

    # Diccionario para registrar la mejor vía asignada a cada estación
    station_to_best_via = {}

    # --- Primera pasada: calcular mejor vía para cada estación ---
    for i, via in lines_gdf.iterrows():
        via_buffer = via.geometry.buffer(buffer_distance)

        # Estaciones dentro del buffer
        estaciones_cercanas = nodes_gdf[nodes_gdf.geometry.intersects(via_buffer)].copy()

        if not estaciones_cercanas.empty:
            estaciones_cercanas['distance_to_via'] = estaciones_cercanas.geometry.apply(
                lambda p: via.geometry.distance(p))

            for idx, est in estaciones_cercanas.iterrows():
                dist = est['distance_to_via']
                if (idx not in station_to_best_via) or (dist < station_to_best_via[idx]["dist"]):
                    station_to_best_via[idx] = {"via": i, "dist": dist}

        # --- Crear nodos inicial/final de la vía ---
        line_start = tuple(via.geometry.coords[0])
        if line_start not in coord_to_node:
            start_node = f"start_{cross_id}"
            g_nx.add_node(start_node, geometry=Point(line_start), type="intersection")
            coord_to_node[line_start] = start_node
            cross_id += 1

        line_end = tuple(via.geometry.coords[-1])
        if line_end not in coord_to_node:
            end_node = f"end_{cross_id}"
            g_nx.add_node(end_node, geometry=Point(line_end), type="intersection")
            coord_to_node[line_end] = end_node
            cross_id += 1

    # --- Segunda pasada: asignar vías definitivas y crear aristas ---

    for i, via in lines_gdf.iterrows():
        # Estaciones que quedaron asignadas a esta vía
        estaciones_via = [s for s, best in station_to_best_via.items() if best["via"] == i]

        if len(estaciones_via) > 0:
            subset = nodes_gdf.loc[estaciones_via].copy()
            subset["pos"] = subset.geometry.apply(lambda p: via.geometry.project(p))
            estaciones_ordenadas = subset.sort_values("pos").index.tolist()

            # Conectar estaciones consecutivas
            for j in range(len(estaciones_ordenadas) - 1):
                n1 = estaciones_ordenadas[j]
                n2 = estaciones_ordenadas[j + 1]
                g_nx.add_edge(n1, n2, **via.to_dict())

            # Conectar extremos con estaciones
            line_start = tuple(via.geometry.coords[0])
            line_end = tuple(via.geometry.coords[-1])
            g_nx.add_edge(coord_to_node[line_start], estaciones_ordenadas[0], **via.to_dict())
            g_nx.add_edge(estaciones_ordenadas[-1], coord_to_node[line_end], **via.to_dict())
        else:
            # Sin estaciones asignadas: conectar extremos
            line_start = tuple(via.geometry.coords[0])
            line_end = tuple(via.geometry.coords[-1])
            g_nx.add_edge(coord_to_node[line_start], coord_to_node[line_end], **via.to_dict())

    return g_nx

def gdf_to_nx_no_buffer(lines_gdf, nodes_gdf):

    g_nx = nx.Graph()

    # Añadir nodos de estaciones
    for idx, est in nodes_gdf.iterrows():
        g_nx.add_node(idx, **est.to_dict())

    # Diccionario para localizar nodos por coordenadas (tuplas)
    coord_to_node = {tuple(est.geometry.coords[0]): idx for idx, est in nodes_gdf.iterrows()}

    # Contador para nodos nuevos
    cross_id = 0

    # Procesar cada vía
    for i, via in lines_gdf.iterrows():
        estaciones_conectadas = nodes_gdf[nodes_gdf.intersects(via.geometry)].copy()

        # --- Nodo inicial ---
        line_start = tuple(via.geometry.coords[0])
        if line_start in coord_to_node:
            start_node = coord_to_node[line_start]
        else:
            start_node = f"start_{cross_id}"
            g_nx.add_node(start_node, geometry=Point(line_start), type="intersection")
            coord_to_node[line_start] = start_node
            cross_id += 1

        # --- Nodo final ---
        line_end = tuple(via.geometry.coords[-1])
        if line_end in coord_to_node:
            end_node = coord_to_node[line_end]
        else:
            end_node = f"end_{cross_id}"
            g_nx.add_node(end_node, geometry=Point(line_end), type="intersection")
            coord_to_node[line_end] = end_node
            cross_id += 1

        if len(estaciones_conectadas) >= 1:
            # Ordenar estaciones según la proyección sobre la línea
            estaciones_conectadas["pos"] = estaciones_conectadas.geometry.apply(
                lambda p: via.geometry.project(p)
            )
            estaciones_ordenadas = estaciones_conectadas.sort_values("pos").index.tolist()

            # Conectar estaciones consecutivas
            for j in range(len(estaciones_ordenadas) - 1):
                n1 = estaciones_ordenadas[j]
                n2 = estaciones_ordenadas[j + 1]
                g_nx.add_edge(n1, n2, **via.to_dict())

            # Conectar primera estación con nodo inicial
            g_nx.add_edge(start_node, estaciones_ordenadas[0], **via.to_dict())
            # Conectar última estación con nodo final
            g_nx.add_edge(estaciones_ordenadas[-1], end_node, **via.to_dict())
        else:
            # Si no hay estaciones, conectar directamente inicio y fin
            g_nx.add_edge(start_node, end_node, **via.to_dict())

    return g_nx

def gdf_to_nx_buffer_to_nodes(lines_gdf,nodes_gdf,buffer_distance,gdf_cut):

    g_nx= nx.Graph()
    coord_to_node = {}
    cross_id = 0

    boundary = gdf_cut.union_all()
    boundary_line = boundary.boundary  # línea del borde

    def merge_extreme_to_station(g_nx, ext_node, station_node):
        """Conecta vecinos del nodo extremo a la estación y elimina el nodo extremo."""
        neighbors = [n for n in g_nx.neighbors(ext_node) if n != station_node]
        for neighbor in neighbors:
            attr = g_nx.get_edge_data(ext_node, neighbor)
            g_nx.add_edge(station_node, neighbor, **attr)
        g_nx.remove_node(ext_node)

    for i, via in lines_gdf.iterrows():
        line_start = Point(via.geometry.coords[0])
        line_end = Point(via.geometry.coords[-1])

        # Nodo inicio
        if tuple(line_start.coords[0]) in coord_to_node:
            start_node = coord_to_node[tuple(line_start.coords[0])]
        else:
            start_node = f"start_{cross_id}"
            g_nx.add_node(start_node, geometry=line_start, type="intersection")
            coord_to_node[tuple(line_start.coords[0])] = start_node
            cross_id += 1

        # Nodo fin
        if tuple(line_end.coords[0]) in coord_to_node:
            end_node = coord_to_node[tuple(line_end.coords[0])]
        else:
            end_node = f"end_{cross_id}"
            g_nx.add_node(end_node, geometry=line_end, type="intersection")
            coord_to_node[tuple(line_end.coords[0])] = end_node
            cross_id += 1

        # Añadir arista que representa la vía
        g_nx.add_edge(start_node, end_node, **via.to_dict())

    g_nx.remove_edges_from(nx.selfloop_edges(g_nx))

    # 1️⃣ Añadir todas las estaciones al grafo
    for idx, est in nodes_gdf.iterrows():
        g_nx.add_node(idx, **est.to_dict())

    # 2️⃣ Iterar sobre nodos extremos válidos
    for n, data in list(g_nx.nodes(data=True)):
        if data.get("type") != "intersection":
            continue
        if g_nx.degree(n) != 1:
            continue

        # Buscar estaciones cercanas
        candidate_stations = [
            idx for idx, est in nodes_gdf.iterrows()
            if est.geometry.distance(data["geometry"]) <= buffer_distance
        ]

        if candidate_stations:
            # Seleccionar la estación más cercana
            nearest_station = min(
                candidate_stations,
                key=lambda idx: nodes_gdf.loc[idx].geometry.distance(data["geometry"])
            )

            # Conectar nodo extremo con estación
            g_nx.add_edge(nearest_station, n, type="train tracks")

            # Hacer merge: la estación reemplaza al nodo extremo
            merge_extreme_to_station(g_nx, n, nearest_station)

    # Obtener nodos extremos (degree == 1)
    end_nodes = [n for n, d in g_nx.degree() if d == 1]

    # Revisar todos los pares de nodos extremos
    for n1, n2 in itertools.combinations(end_nodes, 2):
        if n1 not in g_nx or n2 not in g_nx:
            continue

        geom1 = g_nx.nodes[n1]["geometry"]
        geom2 = g_nx.nodes[n2]["geometry"]

        # Solo procesar si están dentro de la distancia
        if geom1.distance(geom2) <= buffer_distance:
            type1 = g_nx.nodes[n1].get("type")
            type2 = g_nx.nodes[n2].get("type")

            types=['substations','power sources']

            if type1 in types and type2 in types:
                # Caso especial: ambos son estaciones, solo añadir arista entre ellos
                g_nx.add_edge(n1, n2, type="train tracks")
            else:
                geom = g_nx.nodes[n1]["geometry"]
                if geom.distance(boundary_line)<1e-6:
                    neighbors_n2 = list(g_nx.neighbors(n2))
                    for v in neighbors_n2:
                        if n1 == v:
                            continue
                        g_nx.add_edge(n1, v, type="train tracks")
                    g_nx.remove_node(n2)
                else:
                    # Caso normal: merge/fusión de nodos extremos
                    neighbors_n1 = list(g_nx.neighbors(n1))
                    neighbors_n2 = list(g_nx.neighbors(n2))

                    # Conectar todos los vecinos entre sí (evitando los nodos extremos)
                    for u in neighbors_n1:
                        for v in neighbors_n2:
                            if u == v:
                                continue
                            g_nx.add_edge(u, v, type="train tracks")

                    # Eliminar los dos nodos extremos
                    g_nx.remove_node(n1)
                    g_nx.remove_node(n2)



    # Suponemos que gdf_cut es un polígono (o multipolygon) que define los límites

    isolated_nodes = [n for n, d in g_nx.degree() if d == 0]

    for n in isolated_nodes:
        geom_n = g_nx.nodes[n]["geometry"]

        # Buscar el nodo más cercano que tenga grado > 0
        candidate_nodes = [m for m, d in g_nx.degree() if d > 0]
        if not candidate_nodes:
            continue

        nearest_node = min(
            candidate_nodes,
            key=lambda m: geom_n.distance(g_nx.nodes[m]["geometry"])
        )
        geom_nearest = g_nx.nodes[nearest_node]["geometry"]

        # Línea entre nodo aislado y más cercano
        line = LineString([geom_n, geom_nearest])

        if boundary.contains(line):
            # Caso 1: la línea está dentro -> conectar directamente
            g_nx.add_edge(n, nearest_node, type="train tracks")
        else:
            # Caso 2: la línea se sale -> buscar intersección con el límite
            inter = line.intersection(boundary.boundary)

            if not inter.is_empty:
                # Si hay varias intersecciones, cogemos la más cercana al nodo aislado
                if inter.geom_type == "MultiPoint":
                    inter_point = min(inter.geoms, key=lambda p: geom_n.distance(p)) # type: ignore
                else:
                    inter_point = inter

                # Crear un nuevo nodo en el punto de salida
                new_node = f"boundary_{n}"
                g_nx.add_node(new_node, geometry=inter_point, type="boundary")

                # Conectar aislado hasta el límite
                g_nx.add_edge(n, new_node, type="train tracks")

    # Obtener lista de componentes conexas
    components = list(nx.connected_components(g_nx))

    checked = set()  # componentes que ya no se pueden conectar

    while len(components) > 1:
        progress = False  # para saber si en esta iteración conectamos algo

        for smallest_comp in components:
            comp_key = tuple(sorted(map(str, smallest_comp)))
            if comp_key in checked:
                continue

            # Todas las otras componentes
            other_comps = [c for c in components if c != smallest_comp]

            # Inicializar variables para la arista más corta
            min_dist = float("inf")
            closest_pair = None

            # Buscar el par de nodos (uno en smallest_comp, otro en otra componente) más cercano
            for n1 in smallest_comp:
                geom1 = g_nx.nodes[n1]["geometry"]
                for comp in other_comps:
                    for n2 in comp:
                        geom2 = g_nx.nodes[n2]["geometry"]
                        dist = geom1.distance(geom2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = (n1, n2)

            # Conectar si está por debajo del umbral
            if closest_pair and min_dist <= 10*buffer_distance:
                n1, n2 = closest_pair
                g_nx.add_edge(
                    n1, n2, type="train tracks")
                progress = True
                break  # volvemos a recalcular componentes tras una conexión
            else:
                # Marcar esta componente como no conectable en este paso
                checked.add(comp_key)

        if not progress:
            # No hemos podido conectar ninguna componente más
            break

        # Recalcular componentes
        components = list(nx.connected_components(g_nx))
        checked.clear()  # reiniciar porque las componentes han cambiado

    # 3️⃣ Recorrer nodos y comprobar si su geometry toca el borde
    for n, data in g_nx.nodes(data=True):
        geom = data.get("geometry")  # debe ser shapely.Point
        if geom.distance(boundary_line)<1e-6:
            #print(n)
            g_nx.nodes[n]["type"] = "boundary"



    return g_nx

def gdf_to_nx(lines_gdf,nodes_gdf,buffer_distance,buffer_option,gdf_cut):

    lines_gdf=combine_lines_gdf(lines_gdf)

    if buffer_distance:
        if buffer_option=='to lines':
            g_nx=gdf_to_nx_buffer_to_lines(lines_gdf,nodes_gdf,buffer_distance)
        elif buffer_option=='to nodes':
            g_nx=gdf_to_nx_buffer_to_nodes(lines_gdf,nodes_gdf,buffer_distance,gdf_cut)
        else:
            g_nx=None
    else:
        g_nx = gdf_to_nx_no_buffer(lines_gdf, nodes_gdf)

    if buffer_option != 'to nodes':
        # --- Eliminar componentes conexas de solo 2 o 3 nodos ---
        for comp in list(nx.connected_components(g_nx)):
            if len(comp) == 2 or len(comp)==3:
                g_nx.remove_nodes_from(comp)


    '''# --- Colapsar nodos de intersección ---
    nodes_to_collapse = [n for n, d in g_nx.degree() if d > 2]

    for n in nodes_to_collapse:
        neighbors = list(g_nx.neighbors(n))

        # Conectar todos los vecinos entre sí con MultiGraph (cada conexión será una arista adicional)
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                g_nx.add_edge(neighbors[i], neighbors[j], via_id=f"collapsed_from_{n}")

        # Eliminar nodo central
        g_nx.remove_node(n)'''

    return g_nx

def nx_to_igraph(network, g_nx):
    # 1️⃣ Lista de nodos y mapeo a índices
    nodes = list(g_nx.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # 2️⃣ Preparar lista de aristas y atributos
    edges = []
    edges_attrs = []

    # Nodos nuevos por túnel
    new_nodes = []
    new_nodes_attrs = {}
    cross_id = 0

    for u, v, attr in g_nx.edges(data=True):
        # Si la arista tiene tunnel="yes", crear nodo intermedio
        if attr.get("tunnel") == "yes":
            #Usar promedio de coordenadas de nodos si existen
            coords_u = g_nx.nodes[u].get("geometry")
            coords_v = g_nx.nodes[v].get("geometry")
            midpoint = Point((coords_u.x + coords_v.x)/2,(coords_u.y + coords_v.y)/2)
            new_node_name = f"tunnel_{cross_id}"
            cross_id += 1

            new_nodes.append(new_node_name)
            new_nodes_attrs[new_node_name] = {"geometry": midpoint, "type": "tunnel"}

            # Partir arista en dos
            edges.append((node_to_idx[u], len(nodes) + len(new_nodes) - 1))
            edges_attrs.append(attr.copy())
            edges.append((len(nodes) + len(new_nodes) - 1, node_to_idx[v]))
            edges_attrs.append(attr.copy())
        else:
            # Arista normal
            edges.append((node_to_idx[u], node_to_idx[v]))
            edges_attrs.append(attr.copy())

    # 3️⃣ Crear grafo vacío
    g_ig = ig.Graph(directed=False)
    g_ig.add_vertices(len(nodes) + len(new_nodes)) # type: ignore

    for n, idx in node_to_idx.items():
        g_ig.vs[idx]["name"] = network +' node '+str(n)
        for k, v in g_nx.nodes[n].items():
                g_ig.vs[idx][k] = v

    # 5️⃣ Añadir nodos nuevos (túneles)
    for i, n in enumerate(new_nodes):
        idx = len(nodes) + i
        g_ig.vs[idx]["name"] = network +' node '+str(n)
        for k, v in new_nodes_attrs[n].items():
            g_ig.vs[idx][k] = v

    # 6️⃣ Añadir aristas
    if edges:
        g_ig.add_edges(edges) # type: ignore

    # 7️⃣ Añadir atributos de aristas
    for e_idx, attr in enumerate(edges_attrs):
        for k, v in attr.items():
            g_ig.es[e_idx][k] = v
        # Añadir un nombre a la arista
        src, tgt = g_ig.es[e_idx].tuple  # obtener los nodos de la arista
        g_ig.es[e_idx]["name"] = f"{network} edge {src}-{tgt}"

    return g_ig