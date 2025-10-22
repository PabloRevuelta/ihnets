import numpy as np
import igraph as ig
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm


def build_cache(g):
    # Guardamos cosas que no cambian entre simulaciones
    ener_nodes = [v.index for v in g.vs if v["network"] == "Energy network"]
    ener_edges = [e.index for e in g.es if
                  g.vs[e.tuple[0]]["network"] == "Energy network" and
                  g.vs[e.tuple[1]]["network"] == "Energy network"]

    boundary_nodes = [v['name'] for v in g.vs if 'type' in v.attributes() and v['type'] == "boundary"]
    power_sources = [v['name'] for v in g.vs if 'type' in v.attributes() and v['type'] == "power sources"]
    generator_nodes = set(boundary_nodes + power_sources)

    v_name_to_idx = {v["name"]: v.index for v in g.vs if "name" in v.attributes()}
    e_name_to_idx = {e["name"]: e.index for e in g.es if "name" in e.attributes()}

    return {
        "ener_nodes": ener_nodes,
        "ener_edges": ener_edges,
        "generator_nodes": generator_nodes,
        "v_name_to_idx": v_name_to_idx,
        "e_name_to_idx": e_name_to_idx,
    }




def run_scenario(args):
    """Función externa para ejecutar un escenario en paralelo."""
    kind, idx, combined_graph, t_0, params_dic, dt, fail_drop = args
    g_copy = combined_graph.copy()

    if kind == "node":
        v = g_copy.vs[idx]
        return v["name"], simulate_scenario([v], t_0, g_copy, params_dic, dt, fail_drop)
    else:
        e = g_copy.es[idx]
        return e["name"], simulate_scenario([e], t_0, g_copy, params_dic, dt, fail_drop)



def resil_vun_analysis(combined_graph, t_0, n, params_dic, dt, fail_drop):
    scenarios_dic = {}

    if n == 1:
        cache = build_cache(combined_graph)

        nodes = list(range(combined_graph.vcount()))
        edges = list(range(combined_graph.ecount()))
        total_jobs = len(nodes) + len(edges)

        # preparamos argumentos para todos los escenarios
        tasks = [("node", i, combined_graph, t_0, params_dic, dt, fail_drop) for i in nodes] + \
                [("edge", i, combined_graph, t_0, params_dic, dt, fail_drop) for i in edges]


        # Barra de progreso global
        with ProcessPoolExecutor(max_workers=os.cpu_count()-2) as executor:
            futures = [executor.submit(run_scenario, args) for args in tasks]
            for f in tqdm(as_completed(futures), total=total_jobs, desc="Simulaciones"):
                name, result = f.result()
                scenarios_dic[name] = result



    return scenarios_dic

def simulate_scenario(a_f_list,t_0,g_ig,params_dic, dt, fail_drop):
    scenario_dic={} #se guardan aqui (en cada tiempo) los usuarios activos de cada nodo.
    total_users=0
    for v in g_ig.vs:
        scenario_dic[v['name']]=[v['users']]
        total_users+=v['users']
    #print(total_users) #numero de usuarios iniciales
    state_flag='initial' #estado del sistema, actualmente inicial
    t=dt #en horas

    g_ig_energy=g_ig.subgraph([v.index for v in g_ig.vs if v["network"] == 'Energy network'])
    boundary_nodes = [v['name'] for v in g_ig_energy.vs if v["type"] == "boundary"]
    power_source_nodes = [v['name'] for v in g_ig_energy.vs if v["type"] == "power sources"]
    generator_nodes = list(set(boundary_nodes + power_source_nodes))
    v_ener_name_to_idx = {v["name"]: v.index for v in g_ig_energy.vs if "name" in v.attributes()}
    e_ener_name_to_idx = {e["name"]: e.index for e in g_ig_energy.es if "name" in e.attributes()}

    # No copiamos el grafo completo; solo marcamos lo que ha fallado:
    for element in a_f_list:
        element["energy"] = 0
        # refleja el fallo en el subgrafo de energía
        if element['network'] == 'Energy network':
            if isinstance(element, ig.Vertex):
                idx = v_ener_name_to_idx.get(element["name"])
                if idx is not None:
                    g_ig_energy.vs[idx]["energy"] = 0
            elif isinstance(element, ig.Edge):
                idx = e_ener_name_to_idx.get(element["name"])
                if idx is not None:
                    g_ig_energy.es[idx]["energy"] = 0

    # Calculamos las componentes del sistema de energía al inicio
    active_nodes = [v.index for v in g_ig_energy.vs if v["energy"] == 1]
    active_edges = [e.index for e in g_ig_energy.es if e["energy"] == 1]

    g_active = g_ig_energy.subgraph_edges(active_edges, delete_vertices=False).induced_subgraph(active_nodes)

    comps_energy = g_active.components(mode="weak")
    membership = comps_energy.membership
    comp_sets = [set(c) for c in comps_energy]

    all_names = np.array(g_active.vs["name"])
    v_name_to_idx = {v["name"]: v.index for v in g_active.vs if "name" in v.attributes()}

    while state_flag!='finished':
        actual_users=0
        t_ref = np.maximum(dt, round(t - params_dic['tFa'],1))
        index = int(round(t_ref / dt)) - 1

        # 1. Calcular fallos (sin copiar grafo)
        change_failed_v = False
        change_failed_e = False
        for element in a_f_list:
            if element['network'] == 'Energy network':
                if isinstance(element, ig.Vertex):
                    capacity_profile=asset_failure_profile(t,1,t_0,1,params_dic)
                    energy_state=element["energy"]
                    if capacity_profile < 1 and energy_state==1:
                        state_flag = "failure"
                        change_failed_v=True
                        element["energy"] = 0
                        g_ig_energy.vs[v_ener_name_to_idx[element["name"]]]["energy"]=0
                    elif capacity_profile== 1 and energy_state==0:
                        element["energy"] = 1
                        g_ig_energy.vs[v_ener_name_to_idx[element["name"]]]["energy"]=1
                        change_failed_v=True
                elif isinstance(element, ig.Edge):
                    flux = asset_failure_profile(t, 1, t_0, 1, params_dic)
                    energy_state=element["energy"]
                    if flux < 1 and energy_state==1:
                        state_flag = "failure"
                        change_failed_e=True
                        element["energy"] = 0
                        g_ig_energy.es[e_ener_name_to_idx[element["name"]]]["energy"]=0
                    elif flux== 1 and energy_state==0:
                        element["energy"] = 1
                        g_ig_energy.es[e_ener_name_to_idx[element["name"]]]["energy"]=1
                        change_failed_e=True



        # 3. Solo si hubo fallos → recomputar componentes energía
        if change_failed_v or change_failed_e:
            # Si hubo cambios, recalculamos componentes de energía
            active_nodes = [v.index for v in g_ig_energy.vs if v["energy"] == 1]
            active_edges = [e.index for e in g_ig_energy.es if e["energy"] == 1]

            g_active = g_ig_energy.subgraph_edges(active_edges, delete_vertices=False).induced_subgraph(active_nodes)

            comps_energy = g_active.components(mode="weak")
            membership = comps_energy.membership
            comp_sets = [set(c) for c in comps_energy]

            all_names = np.array(g_active.vs["name"])
            v_name_to_idx = {v["name"]: v.index for v in g_active.vs if "name" in v.attributes()}

        for v in g_ig.vs:
            if v['network'] == 'Energy network':
                if v['energy'] == 1 and v["name"] in v_name_to_idx:
                    node_idx = v_name_to_idx[v["name"]]
                    comp_id = membership[node_idx]
                    reachable_nodes = comp_sets[comp_id]
                    reachable_names = all_names[list(reachable_nodes)]
                    node_users = v['users'] if set(generator_nodes) & set(reachable_names) else 0
                else:
                    node_users = 0
                scenario_dic[v['name']].append(node_users)
                actual_users += node_users

            else:
                neighbors_list = g_ig.vs[g_ig.neighbors(v)]
                node_users = v['users']
                power = 1
                for u in neighbors_list:
                    e = g_ig.es[g_ig.get_eid(u.index, v.index)]
                    if u['network'] == 'Energy network':
                        if e in a_f_list:
                            flux_e = asset_failure_profile(t, e[u['name']], t_0, 1, params_dic)
                            if flux_e < 1:
                                state_flag = 'failure'
                        else:
                            flux_e = e[u['name']]
                        power *= u['energy'] * flux_e
                    else:
                        if e in a_f_list:
                            flux_e = asset_failure_profile(t, e[u['name']], t_0, fail_drop, params_dic)
                            if flux_e < e[u['name']]:
                                state_flag = 'failure'
                        else:
                            flux_e = e[u['name']]
                        reference_value = scenario_dic[u['name']][index]
                        node_users = node_users - (u['users'] * e[u['name']] - reference_value * flux_e)
                if power == 0:
                    node_users = 0
                elif power == 1:
                    if v in a_f_list:
                        capacity_profile = asset_failure_profile(t, v['users'], t_0, fail_drop, params_dic)
                        if capacity_profile < v['users']:
                            state_flag = 'failure'
                        if node_users > capacity_profile:
                            node_users = capacity_profile
                scenario_dic[v['name']].append(node_users)
                actual_users += node_users

        if state_flag == 'failure' and abs(actual_users - total_users) < 0.1:
            state_flag='finished'
        #print(actual_users,t,state_flag)
        t=round(t+dt,1)
    return scenario_dic

def asset_failure_profile(t,initial_value,t_f,fail_drop,params_dic):
    if t<t_f+params_dic['tFa']:
        return initial_value
    elif t_f+params_dic['tFa']<=t<t_f+params_dic['tFa']+params_dic['Rc0']+params_dic['tRc']:
        return initial_value*(1-fail_drop)
    else:
        return initial_value