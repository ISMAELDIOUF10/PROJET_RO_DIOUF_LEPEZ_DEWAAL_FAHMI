import sys
import os
import time
import random
import csv
import contextlib
import copy




EPSILON = 1e-9


DOSSIER_PROBLEMES = "problemes_de_transport" 




class ProblemeDeTransport:
    """
    Classe regroupant toutes les données et les algorithmes de résolution.
    """
    def __init__(self):
        self.n = 0  # Nombre de fournisseurs
        self.m = 0  # Nombre de clients
        self.couts = []       # Matrice des coûts (A)
        self.provisions = []  # Vecteur P
        self.commandes = []   # Vecteur C
        
        
        self.proposition = []
        
        
        self.bases = set()

        
        self.potentiels_u = [] 
        self.potentiels_v = []
        self.marginaux = []

    def lire_fichier(self, chemin_fichier):
        """Lit un fichier .txt formaté selon les consignes du projet."""
        try:
            with open(chemin_fichier, 'r', encoding='utf-8') as f:
                content = f.read().split() # Lit tout comme une suite de mots/nombres

            if not content: raise ValueError("Fichier vide.")
            
            iterator = iter(content)
            try:
                
                self.n = int(next(iterator))
                self.m = int(next(iterator))

                
                self.couts = [[0.0] * self.m for _ in range(self.n)]
                self.provisions = [0.0] * self.n
                self.commandes = [0.0] * self.m
                self.proposition = [[0.0] * self.m for _ in range(self.n)]
                self.bases = set()

                
                for i in range(self.n):
                    for j in range(self.m):
                        self.couts[i][j] = float(next(iterator))
                    self.provisions[i] = float(next(iterator))

                
                for j in range(self.m):
                    self.commandes[j] = float(next(iterator))

                

            except StopIteration:
                print("Erreur : Le fichier est incomplet.")
            except ValueError:
                print("Erreur : Le fichier contient des caractères non numériques.")

        except FileNotFoundError:
            print(f"Erreur : Impossible d'ouvrir '{chemin_fichier}'.")

    
    def _ajouter_base(self, i, j, val):
        self.proposition[i][j] = float(val)
        self.bases.add((i, j))

    def _retirer_base(self, i, j):
        self.proposition[i][j] = 0.0
        if (i, j) in self.bases:
            self.bases.remove((i, j))

    def _is_basic(self, i, j):
        return (i, j) in self.bases

    
    def _print_separator(self, col_widths):
        print("    +" + "+".join(["-" * (w + 2) for w in col_widths]) + "+")

    def _afficher_matrice_dynamique(self, data_accessor, title, show_borders=False):
        """
        Affiche une matrice avec alignement automatique des colonnes.
        data_accessor: fonction lambda(i, j) retournant la valeur à afficher.
        """
        print(f"\n=== {title} ===")
        
        
        col_widths = []
        for j in range(self.m):
            header_len = len(f"C{j+1}")
            max_w = header_len
            for i in range(self.n):
                val_str = str(data_accessor(i, j))
                max_w = max(max_w, len(val_str))
            
            if show_borders:
                max_w = max(max_w, len(f"{self.commandes[j]:g}"))
            col_widths.append(max_w)

        supply_width = 0
        if show_borders:
            supply_width = max(len("Prov."), max(len(f"{p:g}") for p in self.provisions))

        
        header_str = "    |"
        for j, w in enumerate(col_widths):
            header_str += f" {f'C{j+1}':^{w}} |"
        if show_borders:
            header_str += f" {'Prov.':^{supply_width}} |"
        
        print("-" * len(header_str))
        print(header_str)
        self._print_separator(col_widths + ([supply_width] if show_borders else []))

        
        for i in range(self.n):
            row_str = f" P{i+1:<2}|"
            for j in range(self.m):
                val_str = str(data_accessor(i, j))
                row_str += f" {val_str:^{col_widths[j]}} |"
            
            if show_borders:
                row_str += f" {self.provisions[i]:^{supply_width}g} |"
            print(row_str)
        
        self._print_separator(col_widths + ([supply_width] if show_borders else []))

        
        if show_borders:
            dem_str = " Com|"
            for j in range(self.m):
                dem_str += f" {self.commandes[j]:^{col_widths[j]}g} |"
            print(dem_str)
            print("-" * len(header_str))

    def affichage_global(self):
        """Affiche toutes les informations pertinentes."""
        
        self._afficher_matrice_dynamique(
            lambda i, j: f"{self.couts[i][j]:g}", 
            "MATRICE DES COÛTS", 
            show_borders=True
        )

        
        def get_prop(i, j):
            val = self.proposition[i][j]
            
            if (i, j) in self.bases:
                return f"{val:g}" if val > EPSILON else "eps"
            return "."
        
        total = self.calcul_cout_total()
        self._afficher_matrice_dynamique(get_prop, f"PROPOSITION DE TRANSPORT (Coût: {total:g})")

        
        if self.potentiels_u and self.potentiels_u[0] is not None:
            print("\n--- Potentiels ---")
            u_str = ", ".join([f"U{i+1}={u:g}" for i, u in enumerate(self.potentiels_u) if u is not None])
            v_str = ", ".join([f"V{j+1}={v:g}" for j, v in enumerate(self.potentiels_v) if v is not None])
            print(f"Lignes : {u_str}")
            print(f"Colonnes : {v_str}")

        
        if self.marginaux and self.marginaux[0][0] is not None:
            def get_marg(i, j):
                return f"{self.marginaux[i][j]:g}" if self.marginaux[i][j] is not None else "."
            self._afficher_matrice_dynamique(get_marg, "TABLE DES COÛTS MARGINAUX")

    def calcul_cout_total(self):
        total = 0
        for (i, j) in self.bases:
            if self.proposition[i][j] > EPSILON:
                total += self.proposition[i][j] * self.couts[i][j]
        return total

    

    def nord_ouest(self, verbose=True):
        if verbose: print("\n[Algo] Exécution Nord-Ouest...")
        self.proposition = [[0.0]*self.m for _ in range(self.n)]
        self.bases = set()
        prov = list(self.provisions)
        cmd = list(self.commandes)
        
        i, j = 0, 0
        while i < self.n and j < self.m:
            q = min(prov[i], cmd[j])
            self._ajouter_base(i, j, q)
            prov[i] -= q
            cmd[j] -= q
            
            if prov[i] == 0 and i < self.n - 1:
                i += 1
            elif cmd[j] == 0 and j < self.m - 1:
                j += 1
            else: # Cas fin ou diagonale
                i += 1; j += 1
        
        
        self.rendre_connexe(verbose=False)

    def balas_hammer(self, verbose=True):
        if verbose: print("\n[Algo] Exécution Balas-Hammer...")
        self.proposition = [[0.0]*self.m for _ in range(self.n)]
        self.bases = set()
        prov = list(self.provisions)
        cmd = list(self.commandes)
        
        rows_done = [False] * self.n
        cols_done = [False] * self.m
        
        # Boucle principale
        for _ in range(self.n * self.m * 2): # Sécurité
            if all(rows_done) or all(cols_done): break
            
            penalties = []
            # Pénalités Lignes
            for i in range(self.n):
                if not rows_done[i]:
                    costs = sorted([(self.couts[i][j], j) for j in range(self.m) if not cols_done[j]], key=lambda x: x[0])
                    if len(costs) >= 2: pen = costs[1][0] - costs[0][0]
                    elif len(costs) == 1: pen = costs[0][0]
                    else: pen = 0
                    if costs: penalties.append((pen, 'row', i, costs[0][1])) 

            # Pénalités Colonnes
            for j in range(self.m):
                if not cols_done[j]:
                    costs = sorted([(self.couts[i][j], i) for i in range(self.n) if not rows_done[i]], key=lambda x: x[0])
                    if len(costs) >= 2: pen = costs[1][0] - costs[0][0]
                    elif len(costs) == 1: pen = costs[0][0]
                    else: pen = 0
                    if costs: penalties.append((pen, 'col', costs[0][1], j))

            if not penalties: break
            
            # Trier par pénalité décroissante
            penalties.sort(key=lambda x: x[0], reverse=True)
            best = penalties[0]
            r, c = best[2], best[3]
            
            q = min(prov[r], cmd[c])
            self._ajouter_base(r, c, q)
            prov[r] -= q
            cmd[c] -= q
            
            if prov[r] == 0: rows_done[r] = True
            else: cols_done[c] = True
            
        self.rendre_connexe(verbose=False)

    #  MÉTHODE DU MARCHE-PIED 

    def _build_adj(self):
        """Construit le graphe d'adjacence des bases."""
        adj = {k: [] for k in range(self.n + self.m)}
        for (i, j) in self.bases:
            u, v = i, self.n + j
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def est_connexe(self):
        """Vérifie si le graphe des bases est connexe (BFS)."""
        adj = self._build_adj()
        start = 0
        visited = set([start])
        queue = [start]
        while queue:
            curr = queue.pop(0)
            for neigh in adj[curr]:
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append(neigh)
        return len(visited) == (self.n + self.m), visited

    def rendre_connexe(self, verbose=True):
        """
        Si le graphe n'est pas connexe (solution dégénérée), 
        ajoute des variables de base artificielles (epsilon).
        """
        is_conn, visited = self.est_connexe()
        while not is_conn:
            
            unvisited = set(range(self.n + self.m)) - visited
            best_edge = None
            min_c = float('inf')
            
            
            for u in visited:
                if u < self.n: # Ligne
                    for v in unvisited:
                        if v >= self.n:
                            cost = self.couts[u][v-self.n]
                            if cost < min_c: min_c = cost; best_edge = (u, v-self.n)
                else: # Colonne
                    for v in unvisited:
                        if v < self.n:
                            cost = self.couts[v][u-self.n]
                            if cost < min_c: min_c = cost; best_edge = (v, u-self.n)
            
            if best_edge:
                self._ajouter_base(best_edge[0], best_edge[1], EPSILON)
            else:
                # Fallback sécurité
                u = list(visited)[0]
                v = list(unvisited)[0]
                r, c = (u, v-self.n) if u < self.n else (v, u-self.n)
                self._ajouter_base(r, c, EPSILON)
            
            is_conn, visited = self.est_connexe()

    def calcul_potentiels(self):
        """Calcule Ui et Vj tel que Ui + Vj = Cij pour les bases."""
        self.potentiels_u = [None]*self.n
        self.potentiels_v = [None]*self.m
        self.potentiels_u[0] = 0 # Base arbitraire
        
        adj = self._build_adj()
        queue = [0]
        
        while queue:
            curr = queue.pop(0)
            is_row = curr < self.n
            idx = curr if is_row else curr - self.n
            val = self.potentiels_u[idx] if is_row else self.potentiels_v[idx]
            
            if val is None: continue 
            
            for neigh in adj[curr]:
                is_neigh_row = neigh < self.n
                n_idx = neigh if is_neigh_row else neigh - self.n
                
                # U_i + V_j = Cost_ij
                if is_row and not is_neigh_row: # Ligne vers Colonne
                    if self.potentiels_v[n_idx] is None:
                        self.potentiels_v[n_idx] = self.couts[idx][n_idx] - val
                        queue.append(neigh)
                elif not is_row and is_neigh_row: # Colonne vers Ligne
                    if self.potentiels_u[n_idx] is None:
                        self.potentiels_u[n_idx] = self.couts[n_idx][idx] - val
                        queue.append(neigh)

    def calcul_marginaux(self):
        """Calcule Delta_ij = Cij - Ui - Vj pour les cases hors-base."""
        self.marginaux = [[None]*self.m for _ in range(self.n)]
        min_delta = 0
        entry = None
        
        for i in range(self.n):
            for j in range(self.m):
                if not self._is_basic(i, j):
                    if self.potentiels_u[i] is not None and self.potentiels_v[j] is not None:
                        delta = self.couts[i][j] - self.potentiels_u[i] - self.potentiels_v[j]
                        self.marginaux[i][j] = delta
                        # On cherche le plus négatif (Blan -> plus petit indice si égalité)
                        if delta < min_delta - EPSILON:
                            min_delta = delta
                            entry = (i, j)
                    else:
                        self.marginaux[i][j] = 0
        return min_delta, entry

    def get_cycle(self, start_pos):
        """Trouve le cycle unique créé en ajoutant l'arête start_pos."""
        start_u = start_pos[0]
        target = self.n + start_pos[1]
        adj = self._build_adj()
        
        queue = [(start_u, [start_u])]
        visited = {start_u}
        
        while queue:
            curr, path = queue.pop(0)
            if curr == target:
                # Reconstruction du cycle
                full_path = path 
                coords = [start_pos]
                for k in range(len(full_path)-1):
                    n1, n2 = full_path[k], full_path[k+1]
                    if n1 < self.n: coords.append((n1, n2-self.n))
                    else: coords.append((n2, n1-self.n))
                return coords
            
            for neigh in adj[curr]:
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append((neigh, path + [neigh]))
        return None

    def maximiser_cycle(self, entry, verbose=True):
        """Fait pivoter les quantités sur le cycle."""
        cycle = self.get_cycle(entry)
        if not cycle: return
        
        if verbose: print(f" -> Cycle détecté (longueur {len(cycle)})")
        
        
        minus_vals = []
        for k in range(1, len(cycle), 2):
            r, c = cycle[k]
            minus_vals.append(self.proposition[r][c])
            
        theta = min(minus_vals)
        if verbose: print(f" -> Theta (quantité déplacée) = {theta:g}")
        
        # Mise à jour
        self._ajouter_base(entry[0], entry[1], 0.0) # Entre dans la base
        
        for k, (r, c) in enumerate(cycle):
            if k % 2 == 0: # +
                self.proposition[r][c] += theta
            else: # -
                self.proposition[r][c] -= theta
        
        # Sortie de la variable (la première qui tombe à 0)
        removed = False
        for k in range(1, len(cycle), 2):
            r, c = cycle[k]
            if not removed and self.proposition[r][c] <= EPSILON:
                self._retirer_base(r, c)
                removed = True
                if verbose: print(f" -> Sortie de la base : ({r}, {c})")
            elif self.proposition[r][c] <= EPSILON:
                self.proposition[r][c] = EPSILON # Reste base mais nulle (dégénéré)

    def resoudre_marche_pied(self, verbose=True):
        """Boucle principale d'optimisation."""
        iter_max = 1000
        it = 0
        while it < iter_max:
            it += 1
            if verbose: print(f"\n--- Itération {it} ---")
            
            # 1. Assurer connexité
            self.rendre_connexe(verbose)
            
            # 2. Potentiels
            self.calcul_potentiels()
            
            # 3. Marginaux
            min_delta, entry = self.calcul_marginaux()
            
            if verbose: self.affichage_global()
            
            # Critère d'arrêt
            if min_delta >= -EPSILON:
                if verbose: print("\n>>> OPTIMUM ATTEINT <<<")
                return
            
            if verbose: print(f"\n[Amélioration] Candidat entrée : {entry} (Gain: {min_delta:g})")
            self.maximiser_cycle(entry, verbose)



def generer_probleme_aleatoire(n, m):
    """Génère un problème aléatoire équilibré."""
    pb = ProblemeDeTransport()
    pb.n, pb.m = n, m
    pb.couts = [[random.randint(1, 100) for _ in range(m)] for _ in range(n)]
    
    # Génération équilibrée
    temp = [[random.randint(1, 100) for _ in range(m)] for _ in range(n)]
    pb.provisions = [sum(temp[i]) for i in range(n)]
    pb.commandes = [sum(temp[i][j] for i in range(n)) for j in range(m)]
    
    # Init structures
    pb.proposition = [[0.0]*m for _ in range(n)]
    pb.bases = set()
    return pb

def generer_trace_fichier(pb_source, algo_type, nom_fichier_sortie):
    """Sauvegarde l'exécution dans un fichier .txt."""
    pb = copy.deepcopy(pb_source)
    
    with open(nom_fichier_sortie, 'w', encoding='utf-8') as f:
        with contextlib.redirect_stdout(f):
            print(f"TRACE - {nom_fichier_sortie}")
            print("="*60)
            pb.affichage_global()
            
            if algo_type == '1': pb.nord_ouest(verbose=True)
            else: pb.balas_hammer(verbose=True)
            
            print("\n--- Début Marche-Pied ---")
            pb.resoudre_marche_pied(verbose=True)

def etude_complexite():
    """Génère un CSV et des graphiques de temps d'exécution."""
    print("\n--- LANCEMENT BENCHMARK COMPLEXITÉ ---")
    tailles = [10, 20, 40, 80]
    nb_runs = 10
    results = []
    
    for n in tailles:
        print(f"Test taille N={n}...")
        for _ in range(nb_runs):
            pb = generer_probleme_aleatoire(n, n)
            
            # Test Nord-Ouest
            pb_no = copy.deepcopy(pb)
            t0 = time.perf_counter()
            pb_no.nord_ouest(verbose=False)
            t_init_no = time.perf_counter() - t0
            t0 = time.perf_counter()
            pb_no.resoudre_marche_pied(verbose=False)
            t_opt_no = time.perf_counter() - t0
            
            # Test Balas-Hammer
            pb_bh = copy.deepcopy(pb)
            t0 = time.perf_counter()
            pb_bh.balas_hammer(verbose=False)
            t_init_bh = time.perf_counter() - t0
            t0 = time.perf_counter()
            pb_bh.resoudre_marche_pied(verbose=False)
            t_opt_bh = time.perf_counter() - t0
            
            results.append({
                "N": n,
                "Total_NO": t_init_no + t_opt_no,
                "Total_BH": t_init_bh + t_opt_bh
            })
            
    # Export CSV
    with open("resultats_complexite.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("Données sauvegardées dans 'resultats_complexite.csv'.")
    
    # Graphique
    try:
        import matplotlib.pyplot as plt
        avg_no = {n: 0 for n in tailles}
        avg_bh = {n: 0 for n in tailles}
        counts = {n: 0 for n in tailles}
        
        for r in results:
            n = r["N"]
            avg_no[n] += r["Total_NO"]
            avg_bh[n] += r["Total_BH"]
            counts[n] += 1
            
        x = sorted(tailles)
        y_no = [avg_no[n]/counts[n] for n in x]
        y_bh = [avg_bh[n]/counts[n] for n in x]
        
        plt.figure(figsize=(10,6))
        plt.plot(x, y_no, label="Nord-Ouest + MP", marker='o')
        plt.plot(x, y_bh, label="Balas-Hammer + MP", marker='s')
        plt.xlabel("Taille N")
        plt.ylabel("Temps (s)")
        plt.title("Comparaison Complexité")
        plt.legend()
        plt.grid(True)
        plt.savefig("graphique_complexite.png")
        print("Graphique sauvegardé : 'graphique_complexite.png'")
        
    except ImportError:
        print("Matplotlib non installé. Pas de graphique.")


def main():
    while True:
        print("\n" + "="*50)
        print("   PROJET RECHERCHE OPÉRATIONNELLE")
        print("="*50)
        print("1. Charger un fichier spécifique (Affichage Console)")
        print("2. Générer TOUTES les traces (.txt)")
        print("3. Étude de complexité (CSV + Graph)")
        print("4. Test aléatoire rapide")
        print("Q. Quitter")
        
        choix = input(">>> Choix : ").strip().lower()
        
        if choix == 'q': break
        
        elif choix == '1':
            f_name = input("Nom du fichier (ex: proposition1.txt) : ")
            # Cherche dans le dossier configuré ou à la racine
            path = os.path.join(DOSSIER_PROBLEMES, f_name)
            if not os.path.exists(path):
                # Essai à la racine
                path = f_name
                if not os.path.exists(path):
                    print("Fichier introuvable.")
                    continue
            
            pb = ProblemeDeTransport()
            pb.lire_fichier(path)
            if pb.n == 0: continue
            
            pb.affichage_global()
            algo = input("Algo (1:Nord-Ouest, 2:Balas-Hammer) : ")
            
            t0 = time.time()
            if algo == '1': pb.nord_ouest(verbose=True)
            else: pb.balas_hammer(verbose=True)
            print(f"Temps init: {time.time()-t0:.4f}s")
            
            input("Appuyez sur Entrée pour lancer l'optimisation...")
            t0 = time.time()
            pb.resoudre_marche_pied(verbose=True)
            print(f"Temps opti: {time.time()-t0:.4f}s")
            
        elif choix == '2':
            if not os.path.exists("traces"): os.makedirs("traces")
            # Liste les fichiers dans le dossier ou à la racine
            targets = []
            if os.path.exists(DOSSIER_PROBLEMES):
                targets = [os.path.join(DOSSIER_PROBLEMES, f) for f in os.listdir(DOSSIER_PROBLEMES) if f.endswith(".txt")]
            
            # Ajoute ceux de la racine si le dossier est vide ou inexistant
            root_targets = [f for f in os.listdir(".") if f.startswith("proposition") and f.endswith(".txt")]
            targets.extend(root_targets)
            
            # Nettoyage doublons et tri
            targets = sorted(list(set(targets)))
            
            print(f"Génération pour {len(targets)} fichiers...")
            
            for path in targets:
                fname = os.path.basename(path)
                pb = ProblemeDeTransport()
                pb.lire_fichier(path)
                
                out_no = os.path.join("traces", f"trace_{fname.replace('.txt','')}_NO.txt")
                generer_trace_fichier(pb, '1', out_no)
                print(f" -> {out_no}")
                
                out_bh = os.path.join("traces", f"trace_{fname.replace('.txt','')}_BH.txt")
                generer_trace_fichier(pb, '2', out_bh)
                print(f" -> {out_bh}")
            print("Terminé.")
            
        elif choix == '3':
            etude_complexite()
            
        elif choix == '4':
            try:
                n = int(input("Taille N : "))
                pb = generer_probleme_aleatoire(n, n)
                pb.affichage_global()
                pb.balas_hammer(verbose=True)
                pb.resoudre_marche_pied(verbose=True)
            except ValueError: print("Entier requis.")

if __name__ == "__main__":
    main()
