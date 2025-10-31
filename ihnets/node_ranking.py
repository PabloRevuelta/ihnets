import numpy as np
import build_directed_graph

def pagerank_creation_and_analysis(G, damping_factor=0.85,top_n=10,debug=False):
    """
    Calcula el PageRank de un grafo y devuelve estadísticas descriptivas.

    Parámetros
    ----------
    G : igraph.Graph
        Grafo dirigido.
    damping_factor : float
        Factor de amortiguamiento (por defecto 0.85).
    top_n : int
        Número de nodos principales a mostrar.

    Devuelve
    --------
    dict con:
        - top_values
        - mean, median, std, min, max
        - gini (desigualdad)
        - skewness (asimetría)
        - distribution (histograma normalizado)
    """

    pr = np.array(G.pagerank(damping=damping_factor))
    G.vs["pagerank_value"] = pr

    # Estadísticos básicos
    mean_val = np.mean(pr)
    median_val = np.median(pr)
    std_val = np.std(pr)
    min_val = np.min(pr)
    max_val = np.max(pr)

    # Medida de desigualdad (Gini)
    pr_sorted = np.sort(pr)
    n = len(pr_sorted)
    gini = (2 * np.sum((np.arange(1, n + 1) * pr_sorted))) / (n * np.sum(pr_sorted)) - (n + 1) / n

    # Asimetría (skewness)
    skewness = np.mean(((pr - mean_val) / std_val) ** 3)

    # Distribución de frecuencias
    hist, bin_edges = np.histogram(pr, bins=20, density=True)
    distribution = {"bins": bin_edges.tolist(), "density": hist.tolist()}

    # Top-N nodos
    top_values = sorted(zip(G.vs["name"], pr), key=lambda x: x[1], reverse=True)[:top_n]

    # Resultados consolidados
    result = {
        "top_values": top_values,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "gini": gini,
        "skewness": skewness,
        "distribution": distribution
    }

    # Prints básicos (puedes quitarlos si devuelves a main)
    if (debug):
        print("\n--- Estadísticas de PageRank ---")
        print(f"Mean: {mean_val:.6f}, Median: {median_val:.6f}, Std: {std_val:.6f}")
        print(f"Min: {min_val:.6f}, Max: {max_val:.6f}")
        print(f"Gini: {gini:.4f}, Skewness: {skewness:.4f}")

        print(f"\nTop {top_n} nodos por PageRank:")
        for n, v in top_values:
            print(f"{n}: {v:.6f}")

    return result