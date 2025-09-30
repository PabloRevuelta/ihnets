import numpy as np
import igraph as ig

def resil_vun_analysis(combined_graph,t_0,n,params_dic, dt, fail_drop):

    scenarios_dic = {}

    if n==1:

        '''v_f=combined_graph.vs.find(name= 'Energy network 34999')
        print('Failed asset ' + str(0) + '/' + str(combined_graph.vcount()) + ': node ' + v_f["name"])
        scenarios_dic[v_f["name"]] = simulate_scenario([v_f], t_0, combined_graph, params_dic, dt, fail_drop)
        # plot_failure_scen(v_f["name"],scenarios_dic)'''

        print('Nodos: ' + str(combined_graph.vcount()))
        i = 1
        for v_f in combined_graph.vs:
            #if v_f['network']=='Energy network':
                #v_f=combined_graph.vs.find(name= 'Energy network node end_182')
            print('Failed asset ' + str(i) + '/' + str(combined_graph.vcount()) + ': node ' + v_f["name"])
            scenarios_dic[v_f["name"]] = simulate_scenario([v_f], t_0, combined_graph, params_dic, dt, fail_drop)
            # plot_failure_scen(v_f["name"],scenarios_dic)
            i += 1

        print('Aristas: ' + str(combined_graph.ecount()))
        i = 1
        for e_f in combined_graph.es:
            print('Failed asset ' + str(i) + '/' + str(combined_graph.ecount()) + ': edge ' + e_f["name"])
            scenarios_dic[e_f["name"]] = simulate_scenario([e_f], t_0, combined_graph, params_dic, dt, fail_drop)
            # plot_failure_scen(e_f.index,scenarios_dic)
            i += 1

    return scenarios_dic

def simulate_scenario(a_f_list,t_0,g_ig,params_dic, dt, fail_drop):
    scenario_dic={}
    total_users=0
    for v in g_ig.vs:
        scenario_dic[v['name']]=[v['users']]
        total_users+=v['users']
    #print(total_users)
    state_flag='initial'
    t=dt

    g_ig_energy=g_ig.subgraph([v.index for v in g_ig.vs if v["network"] == 'Energy network'])
    boundary_nodes = [v['name'] for v in g_ig_energy.vs if v["type"] == "boundary"]
    power_source_nodes = [v['name'] for v in g_ig_energy.vs if v["type"] == "power sources"]
    generator_nodes = list(set(boundary_nodes + power_source_nodes))

    while state_flag!='finished':
        actual_users=0
        t_ref = np.maximum(dt, round(t - params_dic['tFa'],1))
        index = int(round(t_ref / dt)) - 1

        g_ig_failed_elements_energy=g_ig_energy.copy()

        e_to_delete=[]
        v_to_delete=[]
        for element in a_f_list:
            if element['network']=='Energy network':
                if isinstance(element, ig.Vertex):
                    capacity_profile=asset_failure_profile(t,1,t_0,1,params_dic)
                    element['energy']=capacity_profile
                    if capacity_profile<1:
                        state_flag='failure'
                        v_energy=g_ig_failed_elements_energy.vs.find(name= element["name"])
                        #print(element["name"])
                        #print(v_energy["name"])
                        v_to_delete.append(v_energy.index)
                elif isinstance(element, ig.Edge):
                    flux_e=asset_failure_profile(t,1,t_0,1,params_dic)
                    if flux_e<1:
                        state_flag='failure'
                        e_energy=g_ig_failed_elements_energy.es.find(name= element["name"])
                        e_to_delete.append(e_energy.index)
        g_ig_failed_elements_energy.delete_edges(e_to_delete)
        g_ig_failed_elements_energy.delete_vertices(v_to_delete)

        '''e_to_delete=[]
        for e in g_ig_failed_elements_energy.es:
            e_main=g_ig.es.find(name= e["name"])
            if e_main in a_f_list:
                flux_e=asset_failure_profile(t,1,t_0,1,params_dic)
                if flux_e<1:
                    state_flag='failure'
                    e_to_delete.append(e.index)
        g_ig_failed_elements_energy.delete_edges(e_to_delete)
        v_to_delete=[]
        for v in g_ig_failed_elements_energy.vs:
            v_main=g_ig.vs.find(name= v["name"])
            if v_main in a_f_list:
                capacity_profile=asset_failure_profile(t,1,t_0,1,params_dic)
                v_main['energy']=capacity_profile
                if capacity_profile<1:
                    state_flag='failure'
                    #print(v_main['name'],v_main['users'])

                    #neighbors_idx = g_ig_failed_elements_energy.neighbors(v)
                    #neighbors_names = [g_ig_failed_elements_energy.vs[i]["name"] for i in neighbors_idx]
                    #print(neighbors_names)

                    v_to_delete.append(v.index)'''

        all_names = np.array(g_ig_failed_elements_energy.vs["name"])

        comps_energy = g_ig_failed_elements_energy.components(mode="weak")

        for v in g_ig.vs:
            if v['network']=='Energy network':

                if v['energy']==1:
                    reachable_nodes = comps_energy[comps_energy.membership[g_ig_failed_elements_energy.vs.find(name= v["name"]).index]]
                    reachable_names = all_names[reachable_nodes]
                    #if v['name']=="Energy network node 143":
                        #print([g_ig_failed_elements_energy.vs[i]["name"] for i in reachable_nodes])
                        #print(generator_nodes)
                    if set(generator_nodes) & set(reachable_names):
                      node_users=v['users']
                    else:
                       #print(v['name'])
                       node_users=0
                else:
                    node_users=0
                scenario_dic[v['name']].append(node_users)
                #if node_users<v['users']:
                    #print(v['name'],v['users'],node_users)
                actual_users+=node_users

            else:
                neighbors_list = g_ig.vs[g_ig.neighbors(v)]
                node_users = v['users']
                power=1
                for u in neighbors_list:
                    e=g_ig.es[g_ig.get_eid(u.index, v.index)]
                    if u['network']=='Energy network':
                        if e in a_f_list:
                            flux_e=asset_failure_profile(t,e[u['name']],t_0,1,params_dic)
                            if flux_e<1:
                                state_flag='failure'
                        else:
                            flux_e=e[u['name']]
                        power*=u['energy']*flux_e
                        #if power==0:
                            #print(u['energy'],flux_e)

                    else:
                        if e in a_f_list:
                            flux_e=asset_failure_profile(t,e[u['name']],t_0,fail_drop,params_dic)
                            #print(e[u['name']])
                            #print(flux_e)
                            if flux_e<e[u['name']]:
                                state_flag='failure'
                        else:
                            flux_e=e[u['name']]
                        reference_value=scenario_dic[u['name']][index]
                        node_users=node_users-(u['users']*e[u['name']] - reference_value*flux_e)
                if power==0:
                    node_users=0
                elif power==1:
                    if v in a_f_list:
                        capacity_profile=asset_failure_profile(t,v['users'],t_0,fail_drop,params_dic)
                        if capacity_profile<v['users']:
                            state_flag='failure'
                        if node_users>capacity_profile:
                            node_users=capacity_profile
                #if node_users<v['users']:
                    #print(power)
                    #print(v['name'])
                    #print(v['users'])
                    #print(node_users)
                scenario_dic[v['name']].append(node_users)
                actual_users+=node_users

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