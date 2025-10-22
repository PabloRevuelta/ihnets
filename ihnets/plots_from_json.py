import ijson
import matplotlib.pyplot as plt
import os

def plots_resil_vun_analysis_from_json(json_path):
    plt.figure(figsize=(8, 5))

    # Leer el JSON de forma incremental
    with open(json_path, "r", encoding="utf-8") as archivo:
        # Itera sobre los pares clave-valor del JSON raíz
        parser = ijson.kvitems(archivo, "")
        for node, dic in parser:
            # 'dic' es el subdiccionario asociado a cada nodo
            # Calcular la suma elemento a elemento de las listas internas
            lista = [sum(x) for x in zip(*dic.values())]
            x = [i * 0.1 for i in range(len(lista))]
            plt.plot(x, lista, marker='.', markersize=1, label=node)

    plt.title("Network performance curves with 1 failed asset at t=1h with 100% drop")
    plt.ylabel("Network performance (nº users)")
    plt.xlabel("t (h)")
    plt.grid(True)
    file_name = "performance_curves_all_scens.png"
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()
    os.startfile(file_name)

def plots_resil_vun_analysis_from_json_ind_asset(json_path,asset_name):
    with open(json_path, "r", encoding="utf-8") as archivo:
        # Itera sobre los pares clave-valor del primer nivel (cada failed_asset)
        parser = ijson.kvitems(archivo, "")
        for failed_asset, dic in parser:
            if failed_asset==asset_name:
                plt.figure(figsize=(8, 5))
                lista = [sum(x) for x in zip(*dic.values())]
                x = [i * 0.1 for i in range(len(lista))]
                plt.plot(x, lista, marker='.', markersize=1)  # gráfica de línea con puntos
                plt.title("Network performance curve with 1 failed asset at t=1h with 100% drop. Failed asset: "+failed_asset)
                plt.ylabel("Network performance (nº users)")
                plt.xlabel("t (h)")
                plt.grid(True)
                file_name ="performance curve failed "+failed_asset+".png"
                plt.savefig(file_name, dpi=1200, bbox_inches="tight")
                plt.close()
                os.startfile(file_name)

                plt.figure(figsize=(8, 5))
                for node, lista in dic.items():
                    # Solo plotear si hay algún cambio en la lista
                    if max(lista) - lista[0] > 1 or lista[0] - min(lista) > 1:
                        x = [i * 0.1 for i in range(len(lista))]
                        plt.plot(x, lista, marker='.', markersize=1, label=node)
                plt.title(failed_asset)
                plt.title("Node performance curve with 1 failed asset at t=1h with 100% drop. Failed asset: " + failed_asset)
                plt.ylabel("Node performance (nº users)")
                plt.xlabel("t (h)")
                plt.grid(True)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # (x, y) en coordenadas de la figura
                file_name ="nodes performance failed "+failed_asset+".png"
                plt.savefig(file_name, dpi=1200, bbox_inches="tight")
                plt.close()
                os.startfile(file_name)

def plots_resil_vun_analysis_from_json_ind_asset_all_screen(json_path):
    with open(json_path, "r", encoding="utf-8") as archivo:
        # Itera sobre los pares clave-valor del primer nivel (cada failed_asset)
        parser = ijson.kvitems(archivo, "")
        for failed_asset, dic in parser:
            plt.figure(figsize=(8, 5))
            for node, lista in dic.items():
                # Solo plotear si hay algún cambio en la lista
                if max(lista) - lista[0] > 1 or lista[0] - min(lista) > 1:
                    x = [i * 0.1 for i in range(len(lista))]
                    plt.plot(x, lista, marker='.', markersize=1, label=node)
            plt.title(failed_asset)
            plt.title("Node performance curve with 1 failed asset at t=1h with 100% drop. Failed asset: " + failed_asset)
            plt.ylabel("Node performance (nº users)")
            plt.xlabel("t (h)")
            plt.grid(True)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # (x, y) en coordenadas de la figura
            plt.show()

if __name__ == "__main__":
    json_path="C:\\Users\\revueltaap\\Desktop\\datos.json"
    plots_resil_vun_analysis_from_json(json_path)
    #Uso este caso para ver las curvas de caída de todos los fallos simulados y escoger cual me gusta
    #para plotear y presentar
    plots_resil_vun_analysis_from_json_ind_asset_all_screen(json_path)
    #Una vez escogida la que me gusta, la pongo aquí y obtengo las gráficas de esa
    asset_name="Energy network node 48"
    plots_resil_vun_analysis_from_json_ind_asset(json_path,asset_name)
