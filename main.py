import sys
import os
if __name__ == "__main__":
    while True:
        print("\n" + "="*60)
        print("   PROJET RECHERCHE OPÉRATIONNELLE - MENU PRINCIPAL")
        print("="*60)
        
        # 1. Choisir le numéro du problème
        choix_pb = input(">>> Choisir le numéro du problème (1-12) ou 'q' pour quitter : ")
        if choix_pb.lower() == 'q':
            print("Fin du programme.")
            break
        
        filename = f"proposition{choix_pb}.txt"
        if not os.path.exists(filename):
            print(f"(!) Erreur : Le fichier '{filename}' n'existe pas dans ce dossier.")
            continue
            
        # 2. Lecture et Affichage initial
        print(f"\n--- Lecture de {filename} ---")
        tp = TransportProblem(filename)
        if tp.n == 0: continue # Erreur chargement

        tp.display_matrix(tp.costs, title="MATRICE DES COÛTS", show_supply_demand=True)
        
        # 3. Choix de l'algorithme
        print("\nAlgorithmes disponibles :")
        print("  1. Nord-Ouest")
        print("  2. Balas-Hammer")
        choix_algo = input(">>> Votre choix (1 ou 2) : ")
        
        algo_name = "Inconnu"
        if choix_algo == '1':
            tp.solve_north_west()
            algo_name = "Nord-Ouest"
        elif choix_algo == '2':
            tp.solve_balas_hammer()
            algo_name = "Balas-Hammer"
        else:
            print("(!) Choix invalide. Utilisation de Balas-Hammer par défaut.")
            tp.solve_balas_hammer()
            algo_name = "Balas-Hammer"

        # Affichage après algo initial
        print(f"\n--- Proposition Initiale ({algo_name}) ---")
        tp.display_matrix(tp.transport, title="PROPOSITION DE TRANSPORT", show_supply_demand=False)
        print(f"Coût Initial : {tp.get_total_cost()}")

        # 4. Méthode du Marche-pied (Boucle d'optimisation)
        iteration = 1
        print(f"\n--- Début de la méthode du Marche-Pied ---")
        
        while True:
            print(f"\n[ITÉRATION {iteration}]")
            
            # Test Dégénérescence (Connexité)
            is_connected = tp.check_connectivity_bfs()
            status = "Connexe" if is_connected else "Non connexe (Dégénéré)"
            print(f"* Test dégénérescence : {status}")
            
            if not is_connected:
                print("(!) Le graphe n'est pas connexe. Impossible de calculer les potentiels correctement.")
                # Dans une version avancée, on ajouterait des epsilons ici.
                # Pour l'instant, on sort pour éviter les crashs sur potentiels None.
                break 

            # Calcul Potentiels & Marginaux
            tp.calculate_potentials()
            tp.display_potentials()
            
            tp.calculate_marginals()
            tp.display_matrix(tp.marginals, title="COÛTS MARGINAUX")
            
            # Test Optimalité (Recherche du coût marginal le plus négatif)
            min_marginal = 0
            entry_cell = None
            
            for i in range(tp.n):
                for j in range(tp.m):
                    if tp.transport[i][j] == 0:
                        val = tp.marginals[i][j]
                        if val is not None and val < min_marginal:
                            min_marginal = val
                            entry_cell = (i, j)
            
            if entry_cell:
                # NON OPTIMAL
                print(f"\n>>> Solution NON optimale.")
                print(f"* Arête à ajouter : {entry_cell} (Gain marginal: {min_marginal})")
                
                # Maximisation sur cycle
                cycle = tp.get_cycle_bfs(entry_cell[0], entry_cell[1])
                if cycle:
                    print(f"* Cycle détecté : {cycle}")
                    theta = tp.update_transport_on_cycle(cycle)
                    
                    if theta == 0:
                        print("(!) Theta = 0. C'est une itération dégénérée. On continue.")
                        # Risque de cyclage infini ici si pas géré, mais ok pour ce stade.
                else:
                    print("(!) Erreur critique : Pas de cycle trouvé. Arrêt.")
                    break
                
                # Affichage intermédiaire
                print(f"-> Nouveau coût : {tp.get_total_cost()}")
                iteration += 1
            else:
                # OPTIMAL
                print("\n>>> Tous les coûts marginaux sont positifs ou nuls.")
                break
        
        # Fin de la boucle tant que
        print("\n" + "*"*40)
        print(f" RÉSULTAT FINAL ({algo_name})")
        print("*"*40)
        tp.display_matrix(tp.transport, title="PROPOSITION OPTIMALE")
        print(f"Coût minimal trouvé : {tp.get_total_cost()}")
        
        input("\nAppuyez sur Entrée pour continuer vers un autre problème...")
