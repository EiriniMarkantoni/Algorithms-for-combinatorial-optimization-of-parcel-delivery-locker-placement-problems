"""
===============================================================================
ΜΟΝΤΕΛΟ MILP ΓΙΑ ΕΤΕΡΟΓΕΝΕΣ ΠΡΟΒΛΗΜΑ ΧΩΡΟΘΕΤΗΣΗΣ ΘΥΡΙΔΩΝ ΠΑΡΑΔΟΣΗΣ ΜΕ GUROBI
===============================================================================

Αυτός ο κώδικας υλοποιεί ένα Mixed Integer Linear Programming (MILP) μοντέλο
για το πρόβλημα τοποθέτησης ετερογενών εγκαταστάσεων (parcel lockers) με:
- Τρεις τύπους εγκαταστάσεων (small/type 0, medium/type 1, large/type 2)
- Πέντε τρόπους μεταφοράς με διαφορετικά όρια προσβασιμότητας
- Περιορισμούς χωρητικότητας, ζήτησης και προϋπολογισμού
- Ενσωματωμένο κόστος μετακίνησης και εκπομπών CO₂

ΒΑΣΙΚΑ ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Ανάγνωση προβλημάτων από αρχεία .txt
- Επίλυση με Gurobi optimizer
- Μαζική επεξεργασία δεδομένων πολλών προβλημάτων
- Αποθήκευση λεπτομερών αποτελεσμάτων και στατιστικών

ΧΡΗΣΗ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python gurobi_model.py <directory> [--problems problem1.txt problem2.txt ...]

ΣΥΓΓΡΑΦΕΑΣ: Ειρήνη Μαρκαντώνη
ΗΜΕΡΟΜΗΝΙΑ: Ιανουάριος 2026
ΣΚΟΠΟΣ: Διπλωματική Εργασία 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import gurobipy as gp
import time
from gurobipy import *
from gurobipy import GRB
import numpy as np
import argparse
import os

# ===========================================================================
# ΟΡΙΣΜΟΣ ΚΑΘΟΛΙΚΩΝ ΣΤΑΘΕΡΩΝ
# ===========================================================================
TIME_LIMIT = 3600 # Χρονικό όριο εκτέλεσης του μοντέλου (σε δευτερόλεπτα) - 1 ώρα
E = 3 # Αριθμός τύπων εγκαταστάσεων (small=0, medium=1, large=2)

# ============================================================================
# ΣΥΝΑΡΤΗΣΗ ΑΝΑΓΝΩΣΗΣ ΔΕΔΟΜΕΝΩΝ ΑΠΟ ΑΡΧΕΙΟ
# ============================================================================
def read_data(filename):
    """
    Διαβάζει τα δεδομένα ενός instance από αρχείο κειμένου
    Args:
        filename (str): Διαδρομή προς το αρχείο instance
    
    Returns:
        tuple: (P, N, CL, clients, points, fp_sp_distances, cl_sp_distances, 
                capacities, facility_types, transport_mode, Cmax, demand, cost_types)
    """

    # Άνοιγμα και ανάγνωση αρχείου
    f = open(filename, "rt")
    lines = f.readlines()
    # Καθαρισμός γραμμών: αφαίρεση newlines και κενών γραμμών
    lines = [f.replace('\n', '') for f in lines if f != '\n']

    # -------------------------------------------------------------------------
    # ΑΝΑΓΝΩΣΗ ΒΑΣΙΚΩΝ ΠΑΡΑΜΕΤΡΩΝ
    # -------------------------------------------------------------------------
    # Πρώτη γραμμή: <total_points> <clients> <facilities> <facilities_to_open>
    line = lines[0].split(' ')
    P = int(line[3])  # Αριθμός εγκαταστάσεων που πρέπει να ανοίξουν 
    N = int(line[2])  # Αριθμός πιθανών τοποθεσιών για εγκαταστάσεις
    CL = int(line[1]) # Αριθμός πελατών

    # -------------------------------------------------------------------------
    # ΑΝΑΓΝΩΣΗ ΛΙΣΤΑΣ ΠΕΛΑΤΩΝ
    # -------------------------------------------------------------------------
    # Ξεκινάμε από γραμμή 2 (index 1: header, index 2: πρώτος πελάτης)
    line_idx = 2
    clients = np.zeros(CL) # Array για αποθήκευση δεικτών πελατών
    for i in range(CL):
        clients[i] = int(lines[line_idx])  # Γραμμικός δείκτης πελάτη στο grid
        line_idx+=1

    # -------------------------------------------------------------------------
    # ΑΝΑΓΝΩΣΗ ΛΙΣΤΑΣ ΥΠΟΨΗΦΙΩΝ ΕΓΚΑΤΑΣΤΑΣΕΩΝ
    # -------------------------------------------------------------------------
    line_idx+=1   # Παράλειψη header γραμμής "N candidate facilities:"
    line = lines[line_idx]
    points = np.zeros(N)  # Array για αποθήκευση δεικτών εγκαταστάσεων
    for i in range(N):
        points[i] = int(lines[line_idx])  # Γραμμικός δείκτης εγκατάστασης στο grid
        line_idx+=1

    # -------------------------------------------------------------------------
    # ΑΝΑΓΝΩΣΗ ΤΥΠΩΝ ΕΓΚΑΤΑΣΤΑΣΕΩΝ
    # -------------------------------------------------------------------------
    # Κάθε θέση έχει μέγιστο επιτρεπτό τύπο βάσει διαθέσιμου όγκου
    # Τιμές: 0 (small), 1 (medium), 2 (large)
    line_idx += 1  # Παράλειψη header "N facilities types:"
    facility_types = np.zeros(N, dtype=int)  
    for i in range(N):
        line = lines[line_idx].strip()  # Διαβάζουμε τη γραμμή και αφαιρούμε τα περιττά κενά
        facility_types[i] = int(line.split('type ')[-1])  # Εξαγωγή αριθμού μετά το "type " (π.χ. "type 2" -> 2)
        line_idx += 1

    # -------------------------------------------------------------------------
    # ΑΝΑΓΝΩΣΗ ΑΠΟΣΤΑΣΕΩΝ ΜΕΤΑΞΥ ΕΓΚΑΤΑΣΤΑΣΕΩΝ
    # -------------------------------------------------------------------------
    # Οι αποστάσεις είναι συνθετικές και δίνονται σε km
    line_idx+=1  # Παράλειψη header "Ν x N facility distances:"
    line = lines[line_idx]
    # Πίνακας για αποστάσεις facility-to-facility (N x N)
    fp_sp_distances = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            if(i==j):  # Παράλειψη διαγωνίου (απόσταση από θέση σε εαυτή = 0)
                continue
            line = lines[line_idx].split(' ')
            fp_sp_distances[i][j] = float(line[2])
            line_idx+=1
    

    # -------------------------------------------------------------------------
    # ΑΝΑΓΝΩΣΗ ΑΠΟΣΤΑΣΕΩΝ ΠΕΛΑΤΩΝ-ΕΓΚΑΤΑΣΤΑΣΕΩΝ
    # -------------------------------------------------------------------------
    # Συνθετικές αποστάσεις που προσομοιώνουν πραγματικές διαδρομές με διαφορετικά μέσα μεταφοράς (mode-dependent accessibility)
    line_idx+=1  # Παράλειψη header "CL x N client-facility distances:"
    line = lines[line_idx]
    # Πίνακας για αποστάσεις client-to-facility (CL x N)
    cl_sp_distances = np.zeros(shape=(CL,N))
    for i in range(CL):
            for j in range(N):
                line = lines[line_idx].split(' ')
                cl_sp_distances[i][j] = float(line[2])
                line_idx+=1
    
    # -------------------------------------------------------------------------
    # ΑΝΑΓΝΩΣΗ ΚΟΣΤΩΝ ΘΥΡΙΔΩΝ
    # -------------------------------------------------------------------------
    # Κόστη περιλαμβάνουν: αρχικό κόστος + κόστος υπολογιστή + παρούσα αξία λειτουργικών κοστών (15 έτη)
    line_idx += 1 # Παράλειψη header "Cost of lockers:"
    cost_types = np.zeros(E)  # Array για κόστη [Small, Medium, Large]
    for c in range(E):
        line = lines[line_idx].strip()  # Διαβάζουμε τη γραμμή και αφαιρούμε τα περιττά κενά
        cost_types[c] = float(line.split(":")[1].strip()) # Εξαγωγή αριθμού μετά το ":" (π.χ. "Small: 123.45" -> 123.45)
        line_idx += 1

    
    # -------------------------------------------------------------------------
    # ΑΝΑΓΝΩΣΗ ΔΙΑΘΕΣΙΜΟΥ ΚΕΦΑΛΑΙΟΥ (ΠΡΟΫΠΟΛΟΓΙΣΜΟΣ)
    # -------------------------------------------------------------------------
    line_idx += 1  # Παράλειψη header "Total available capital:"
    Cmax = float(lines[line_idx])  # Διαθέσιμο κεφάλαιο (€)
    line_idx += 1

    # -------------------------------------------------------------------------
    # ΑΝΑΓΝΩΣΗ ΖΗΤΗΣΗΣ ΠΕΛΑΤΩΝ
    # -------------------------------------------------------------------------
    # Ζήτηση σε αριθμό δεμάτων (ακέραιος)
    line_idx += 1  # Παράλειψη header "Demand of customers:"
    demand = np.zeros(CL)  # Array για ζήτηση κάθε πελάτη
    for d in range(CL):
        line = lines[line_idx].strip()  # Διαβάζουμε τη γραμμή και αφαιρούμε τα περιττά κενά
        # Εξαγωγή αριθμού μετά το "Parcels: " 
        # (π.χ. "Area: 1, Demand: 2.5, Parcels: 3" -> 3)
        demand[d] = int(line.split('Parcels: ')[-1]) 
        line_idx += 1
   
    # -------------------------------------------------------------------------
    # ΑΝΑΓΝΩΣΗ ΧΩΡΗΤΙΚΟΤΗΤΩΝ ΕΓΚΑΤΑΣΤΑΣΕΩΝ
    # -------------------------------------------------------------------------
    # Μέγιστη χωρητικότητα (σε δέματα) για κάθε τύπο [Small, Medium, Large]
    line_idx += 1
    capacities = np.zeros(E)
    for p in range(E):
        capacities[p] = int(lines[line_idx])
        line_idx += 1

    # -------------------------------------------------------------------------
    # ΑΝΑΓΝΩΣΗ ΜΕΣΩΝ ΜΕΤΑΦΟΡΑΣ ΠΕΛΑΤΩΝ
    # -------------------------------------------------------------------------
    # Κωδικοί: 0=πόδια, 1=ποδήλατο, 2=ΜΜΜ, 3=όχημα εσωτερικής καύσης, 4=ηλεκτρικό όχημα
    line_idx += 1  # Παράλειψη header "Transportation of customers:"
    transport_mode = np.zeros(CL)   # Array για μέσο μεταφοράς κάθε πελάτη
    for k in range(CL):
        transport_mode[k] = int(lines[line_idx])
        line_idx += 1  

    # Επιστροφή όλων των δεδομένων ως tuple
    return P, N, CL, clients, points, fp_sp_distances, cl_sp_distances, capacities, facility_types, transport_mode, Cmax, demand, cost_types



# ======================================================================
# ΣΥΝΑΡΤΗΣΗ ΔΗΜΙΟΥΡΓΙΑΣ ΚΑΙ ΕΠΙΛΥΣΗΣ ΜΟΝΤΕΛΟΥ
# ======================================================================
def run_model(filename):
    """
    Δημιουργεί και επιλύει το MILP μοντέλο για ένα πρόβλημα τοποθέτησης εγκαταστάσεων
    
    Το μοντέλο περιλαμβάνει:
    ΜΕΤΑΒΛΗΤΕΣ ΑΠΟΦΑΣΗΣ:
    - x[i,j]: Binary - 1 αν η εγκατάσταση στη θέση i ανοίγει με τύπο j, 0 αλλιώς
    - y[k,i]: Binary - 1 αν ο πελάτης k εξυπηρετείται από την εγκατάσταση i, 0 αλλιώς
    
    ΑΝΤΙΚΕΙΜΕΝΙΚΗ ΣΥΝΑΡΤΗΣΗ:
    Minimize: Συνολική απόσταση που διανύουν οι πελάτες
    
    ΠΕΡΙΟΡΙΣΜΟΙ:
    1. Κάθε θέση έχει το πολύ έναν τύπο εγκατάστασης
    2. Ακριβώς p εγκαταστάσεις ανοίγουν
    3. Οι εγκαταστάσεις ανοίγουν μόνο με επιτρεπτούς τύπους
    4. Κάθε πελάτης εξυπηρετείται από ακριβώς μία εγκατάσταση
    5. Εξυπηρέτηση μόνο από ανοιχτές εγκαταστάσεις
    6. Προσβασιμότητα βάσει μέσου μεταφοράς (mode-dependent)
    7. Χωρητικότητα εγκαταστάσεων
    8. Προϋπολογισμός (εγκατάσταση + μετακίνηση + CO₂)
    
    Args:
        filename (str): Διαδρομή προς το αρχείο instance
    
    Returns:
        tuple: (status, termination_condition, objective_value, solving_time, nodes_explored)
    """

    # -------------------------------------------------------------------------
    # ΑΝΑΓΝΩΣΗ ΔΕΔΟΜΕΝΩΝ
    # -------------------------------------------------------------------------
    P, N, CL, clients, points,fp_sp_distances, cl_sp_distances, capacities, facility_types,transport_mode, Cmax, demand, cost_types  = read_data(filename)

    # -------------------------------------------------------------------------
    # ΔΗΜΙΟΥΡΓΙΑ ΜΟΝΤΕΛΟΥ GUROBI
    # -------------------------------------------------------------------------
    model = gp.Model("FacilityLocation")

    # -------------------------------------------------------------------------
    # ΜΕΤΑΒΛΗΤΕΣ ΑΠΟΦΑΣΗΣ
    # -------------------------------------------------------------------------
    # Μεταβλητή x[i,j] : δυαδική - Αν θα ανοίξει εγκατάσταση τύπου j στη θέση i
    # i ∈ {0, ..., N-1}: Υποψήφιες θέσεις εγκαταστάσεων
    # j ∈ {0, 1, 2}: Τύποι εγκαταστάσεων (small=0, medium=1, large=2)
    x = model.addVars(N, E, vtype=GRB.BINARY, name="x")  

    # Μεταβλητή y[k,i] : δυαδική - Αν ο πελάτης k εξυπηρετείται από την εγκατάσταση στη θέση i
    # k ∈ {0, ..., CL-1}: Πελάτες
    # i ∈ {0, ..., N-1}: Υποψήφιες θέσεις εγκαταστάσεων
    y = model.addVars(CL, N, vtype=GRB.BINARY, name="y")


    # -------------------------------------------------------------------------
    # ΟΡΙΣΜΟΣ ΜΕΣΩΝ ΜΕΤΑΦΟΡΑΣ ΚΑΙ ΕΚΠΟΜΠΩΝ CO₂
    # -------------------------------------------------------------------------

    # Λεξικό για τους τύπους μέσων μεταφοράς
    Transport_Modes = {
         0: "Walking",           # Πόδια
         1: "Bicycle",           # Ποδήλατο
         2: "Public Transport",  # Μέσα μαζικής μεταφοράς
         3: "Exhaust Cars",      # Οχήματα με κινητήρα εσωτερικής καύσης
         4: "BEVs"               # Ηλεκτρικά οχήματα μπαταρίας
                      }

    # -------------------------------------------------------------------------
    # ΕΚΠΟΜΠΕΣ CO₂ ΚΑΙ ΚΟΣΤΟΣ ΜΕΤΑΚΙΝΗΣΗΣ
    # -------------------------------------------------------------------------

    # Εκπομπές CO2 σε γραμμάρια ανά χιλιόμετρο για κάθε μέσο μεταφοράς  (g CO₂/km)
    emission_values = {
        0: 0,     # Walking - Περπάτημα (0 g/km)
        1: 0,     # Bicycle - Ποδήλατο (0 g/km)
        2: 42.5,  # Public Transport - Μέσα Μαζικής Μεταφοράς (42.5 g/km)
        3: 162,   # Exhaust Cars - Οχήματα με κινητήρα εσωτερικής καύσης (162 g/km)
        4: 0      # BEVs - Ηλεκτρικά οχήματα μπαταρίας (0 g/km)
                     }
    
    # Κόστος μετακίνησης ανά km ανά μέσο μεταφοράς (€ / km)
    transport_cost_per_km = {
        0: 0.0,    # Walking - Πόδια
        1: 0.0,    # Bicycle - Ποδήλατο
        2: 0.05,   # Public Transport - Μέσα Μαζικής Μεταφοράς
        3: 0.2,    # Exhaust Cars - Οχήματα με κινητήρα εσωτερικής καύσης
        4: 0.1     # BEVs - Ηλεκτρικά οχήματα μπαταρίας
    }

    # Συντελεστής μετατροπής εκπομπών CO₂ σε οικονομικό κόστος (€/g CO₂)
    # Επιλέγεται συντηρητικά υψηλότερη τιμή (1€/kg) από τις συμβατικές τιμές αγοράς (~0.085€/kg EU ETS) για την 
    # ενσωμάτωση από την αρχή πιο βιώσιμων λύσεων
    co2_cost_factor = 0.001  # 1€ ανά kg CO₂ = 0.001€/g CO₂

    # Υπολογισμός συνδυασμένου κόστους ανά km για κάθε μέσο μεταφοράς
    # Συνδυασμένο κόστος ανά km: οικονομικό κόστος μετακίνησης + περιβαλλοντικό κόστος
    # Αυτό επιτρέπει την ενσωμάτωση του περιβαλλοντικού αντίκτυπου στο μοντέλο
    combined_cost_per_km = {}
    for mode in emission_values:
        combined_cost_per_km[mode] = transport_cost_per_km[mode] + emission_values[mode] * co2_cost_factor


    # Ποσοστά των πελατών που χρησιμοποιούν το κάθε μέσο μεταφοράς 
    # Σημείωση: Το άθροισμα των ποσοστών πρέπει να είναι ακριβώς 1.0
    customers_percentages = {
        0: 0.40875,  # 40.875% χρησιμοποιούν Πόδια
        1: 0.07125,  # 7.125% χρησιμοποιούν Ποδήλατο
        2: 0.07875,  # 7.875% χρησιμοποιούν Μέσα Μαζικής Μεταφοράς
        3: 0.42625,  # 42.625% χρησιμοποιούν Οχήματα με κινητήρα εσωτερικής καύσης
        4: 0.015    # 1.5% χρησιμοποιούν Ηλεκτρικά οχήματα μπαταρίας
}
 
    # ------------------------------------------------------------------------
    # ΟΡΙΣΜΟΣ ΜΕΓΙΣΤΗΣ ΑΠΟΣΤΑΣΗΣ ΠΡΟΣΒΑΣΙΜΟΤΗΤΑΣ ΑΝΑ ΜΕΣΟ ΜΕΤΑΦΟΡΑΣ
    # ------------------------------------------------------------------------
    # Eπιστρέφει τη μέγιστη απόσταση που είναι διατεθειμένος να διανύσει ένας πελάτης με συγκεκριμένο μέσο μεταφοράς (km)
    def get_max_distance(mode):
        if mode == 0:   # Μέσο μεταφοράς πόδια
           return 0.5  
        elif mode == 1: # Μέσο μεταφοράς ποδήλατο
           return 1.0  
        elif mode == 2: # Μέσο μεταφοράς Μέσα Μαζικής Μεταφοράς
           return 2.0
        elif mode == 3: # Μέσο μεταφοράς όχημα με κινητήρα εσωτερικής καύσης
           return 3.5
        else:           # Μέσο μεταφοράς ηλεκτρικό όχημα μπαταρίας
           return 6.0   
    
    # Υπολογισμός μέγιστης αποδεκτής απόστασης για κάθε πελάτη
    # Δημιουργία λίστας με τη μέγιστη απόσταση κάθε πελάτη βάσει του μέσου του
    max_distance = [get_max_distance(transport_mode[k]) for k in range(CL)]
    

    # -------------------------------------------------------------------------
    # ΠΕΡΙΟΡΙΣΜΟΙ ΜΟΝΤΕΛΟΥ
    # -------------------------------------------------------------------------

    # ΠΕΡΙΟΡΙΣΜΟΣ 1: Σε κάθε θέση μπορεί να ανοίξει το πολύ 1 τύπος εγκατάστασης
    # Σκοπός: Αποφυγή πολλαπλών εγκαταστάσεων στην ίδια θέση
    # Μαθηματικά: Σ(j=0 έως E-1) x[i,j] ≤ 1, ∀i
    for i in range(N):
        model.addConstr(gp.quicksum(x[i, j] for j in range(E)) <= 1, name=f"single_type_{i}") # Άθροισμα όλων των τύπων εγκατάστασης στη θέση i
    

    # ΠΕΡΙΟΡΙΣΜΟΣ 2: Ακριβώς P εγκαταστάσεις πρέπει να ανοίξουν (p-median)
    # Σκοπός: Καθορισμός μεγέθους δικτύου
    # Μαθηματικά: Σ(i=0 έως N-1) Σ(j=0 έως E-1) x[i,j] = P
    model.addConstr(gp.quicksum(x[i, j] for i in range(N) for j in range(E)) == P, name="total_facilities")


    # ΠΕΡΙΟΡΙΣΜΟΣ 3: Οι εγκαταστάσεις μπορούν να ανοίξουν μόνο αν το επιτρέπει ο τύπος της θέσης
    # Σκοπός: Σεβασμός περιορισμών διαθέσιμου όγκου ανά θέση
    # Λογική: Αν facility_types[i] = 1 (medium), επιτρέπονται μόνο small(0) και medium(1)
    #         Αν facility_types[i] = 0 (small), επιτρέπεται μόνο small(0)
    # Μαθηματικά: x[i,j] = 0, ∀i, ∀j > facility_types[i]
    for i in range(N):
        for j in range(E):
            if facility_types[i] < j:
                # Τύπος j ΔΕΝ επιτρέπεται στη θέση i
                model.addConstr(x[i, j] == 0, name=f"type_forbid_{i}_{j}")


    # ΠΕΡΙΟΡΙΣΜΟΣ 4: Κάθε πελάτης εξυπηρετείται από ακριβώς μία εγκατάσταση
    # Σκοπός: Πλήρης κάλυψη ζήτησης - κανένας πελάτης δεν μένει ανεξυπηρέτητος
    # Μαθηματικά: Σ(i=0 έως N-1) y[k,i] = 1, ∀k
    for k in range(CL):
        model.addConstr(gp.quicksum(y[k, i] for i in range(N)) == 1, name=f"single_assignment_{k}")



    # ΠΕΡΙΟΡΙΣΜΟΣ 5: Κάθε πελάτης εξυπηρετείται μόνο από ανοιχτές εγκαταστάσεις
    # Σκοπός: Αποτροπή ανάθεσης σε κλειστές εγκαταστάσεις
    # Μαθηματικά: y[k,i] ≤ Σ(j=0 έως E-1) x[i,j], ∀k, ∀i
    # Εξήγηση: Αν καμία εγκατάσταση δεν ανοίγει στο i (όλα τα x[i,j]=0), τότε y[k,i] πρέπει να είναι 0
    for k in range(CL):
        for i in range(N):
            model.addConstr(y[k, i] <= gp.quicksum(x[i, j] for j in range(E)), name=f"service_open_{k}_{i}")


    # ΠΕΡΙΟΡΙΣΜΟΣ 6: Απόσταση πελάτη ≤ μέγιστη επιτρεπτή (mode-dependent accessibility)
    # Σκοπός: Εξασφάλιση προσβασιμότητας βάσει μέσου μεταφοράς
    # Μαθηματικά: y[k,i] = 0, ∀k, ∀i όπου cl_sp_distances[k,i] > max_distance[k]
    # Εξήγηση: Αν η απόσταση υπερβαίνει τη μέγιστη που ο πελάτης μπορεί να διανύσει με το μέσο του, δεν επιτρέπεται η ανάθεση
    for k in range(CL):
        for i in range(N):
            if cl_sp_distances[k,i] > max_distance[k]:
                model.addConstr(y[k,i] == 0, name=f"distance_forbid_{k}_{i}")


    # ΠΕΡΙΟΡΙΣΜΟΣ 7: Η ζήτηση που εξυπηρετείται δεν υπερβαίνει τη χωρητικότητα
    # Σκοπός: Σεβασμός φυσικών περιορισμών χωρητικότητας θυρίδων
    # Μαθηματικά: Σ(k∈CL) demand[k]·y[k,i] ≤ Σ(j∈E) capacities[j]·x[i,j], ∀i
    # Εξήγηση: Το άθροισμα ζήτησης που ανατίθεται στην εγκατάσταση i δεν μπορεί να υπερβαίνει τη χωρητικότητα του τύπου που έχει εγκατασταθεί
    for i in range(N):
        model.addConstr(
            gp.quicksum(demand[k] * y[k, i] for k in range(CL)) <= gp.quicksum(capacities[j] * x[i, j] for j in range(E)),
            name=f"capacity_constraint_{i}"
        )

    # -------------------------------------------------------------------------
    # ΥΠΟΛΟΓΙΣΜΟΣ ΣΥΝΟΛΙΚΟΥ ΚΟΣΤΟΥΣ ΜΕΤΑΚΙΝΗΣΗΣ
    # -------------------------------------------------------------------------
    # Κόστος Μετακίνησης = Σ(απόσταση × συνδυασμένο_κόστος_ανά_km)
    # Το συνδυασμένο κόστος περιλαμβάνει:
    # 1. Άμεσο οικονομικό κόστος μετακίνησης (καύσιμα, εισιτήρια)
    # 2. Περιβαλλοντικό κόστος (νομισματοποιημένες εκπομπές CO₂)
    total_transport_cost = gp.quicksum(
        y[k, i] * cl_sp_distances[k][i] * combined_cost_per_km[transport_mode[k]]
        for k in range(CL) for i in range(N)
    )
    
    # ΠΕΡΙΟΡΙΣΜΟΣ 8: Συνολικό κόστος ≤ Διαθέσιμο κεφάλαιο (προϋπολογισμός)
    # Σκοπός: Οικονομική εφικτότητα του δικτύου
    # Συνολικό Κόστος = Κόστος εγκατάστασης + Κόστος μετακίνησης (με CO₂)
    # Μαθηματικά: Σ(x[i,j]·cost_types[j]) + total_transport_cost ≤ Cmax
    model.addConstr(
        gp.quicksum(x[i, j] * cost_types[j] for i in range(N) for j in range(E)) + total_transport_cost <= Cmax,
        name="budget_constraint"
   )

    # -------------------------------------------------------------------------
    # ΑΝΤΙΚΕΙΜΕΝΙΚΗ ΣΥΝΑΡΤΗΣΗ
    # -------------------------------------------------------------------------
    # Minimize: Συνολική απόσταση που διανύουν οι πελάτες
    # Σκοπός: Ελαχιστοποίηση της απόστασης που διανύουν πελατών για την παραλαβή δεμάτων
    # Μαθηματικά: min Σ(k∈CL) Σ(i∈N) y[k,i]·cl_sp_distances[k,i]
    obj = gp.quicksum(y[k, i] * cl_sp_distances[k][i] for k in range(CL) for i in range(N))
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # -------------------------------------------------------------------------
    # ΡΥΘΜΙΣΕΙΣ SOLVER
    # -------------------------------------------------------------------------
    model.setParam(GRB.Param.TimeLimit, TIME_LIMIT)  # Μέγιστος χρόνος: 3600s - 1 ώρα
    model.setParam("OutputFlag", 1)                  # Εμφάνιση προόδου Gurobi

    # -------------------------------------------------------------------------
    # ΑΠΟΘΗΚΕΥΣΗ ΣΤΑΤΙΣΤΙΚΩΝ ΜΟΝΤΕΛΟΥ
    # -------------------------------------------------------------------------
    with open("model_stats.txt", "w") as f:
        f.write(f"Αριθμός μεταβλητών: {model.NumVars}\n")
        f.write(f"Αριθμός περιορισμών: {model.NumConstrs}\n")

    # -------------------------------------------------------------------------
    # ΕΠΙΛΥΣΗ ΜΟΝΤΕΛΟΥ
    # -------------------------------------------------------------------------
    start_time = time.time()                 # Καταγραφή χρόνου έναρξης
    model.optimize()                         # Εκκίνηση Gurobi optimizer
    model.write("model_debug.lp")            # Αποθήκευση μοντέλου σε LP format για debugging
    solving_time = time.time() - start_time  # Υπολογισμός χρόνου επίλυσης
    
    # Ανάκτηση αριθμού εξερευνημένων κόμβων του Branch and Bound δέντρου
    nodes_explored = model.NodeCount
    
    # -------------------------------------------------------------------------
    # ΕΠΕΞΕΡΓΑΣΙΑ ΑΠΟΤΕΛΕΣΜΑΤΩΝ
    # -------------------------------------------------------------------------
    # ΠΕΡΙΠΤΩΣΗ 1: Βρέθηκε βέλτιστη λύση
    if model.status == GRB.OPTIMAL:
        print("Model solved to optimality")
        print(f"Objective value = {model.objVal:.2f}\n\n")
        print(f"Nodes explored = {nodes_explored}\n\n")
        
    
        # ---------------------------------------------------------------------
        # ΕΚΤΥΠΩΣΗ ΑΝΟΙΧΤΩΝ ΕΓΚΑΤΑΣΤΑΣΕΩΝ
        # ---------------------------------------------------------------------
        print("Εγκαταστάσεις που θα ανοίξουν (x[i, j]):")
        for i in range(N):            # Για κάθε θέση
            for j in range(E):        # Για κάθε τύπο εγκατάστασης
                if x[i, j].x > 0.5:   # Αν x[i,j] = 1 (με tolerance για floating point)
                    print(f"x[{i}, {j}] = {x[i, j].x}")
    
        # ---------------------------------------------------------------------
        # ΕΚΤΥΠΩΣΗ ΑΝΑΘΕΣΕΩΝ ΠΕΛΑΤΩΝ
        # ---------------------------------------------------------------------
        print("\nΑνάθεση πελατών σε εγκαταστάσεις (y[k, i]):")
        for k in range(CL):           # Για κάθε πελάτη
            for i in range(N):        # Για κάθε θέση
                if y[k, i].x > 0.5:   # Μόνο οι ενεργές μεταβλητές y[k, i]
                    print(f"y[{k}, {i}] = {y[k, i].x}")
        
        # Επιστροφή επιτυχούς αποτελέσματος
        return model.status, "Optimal", model.objVal,solving_time, nodes_explored

    # ΠΕΡΙΠΤΩΣΗ 2: Το πρόβλημα είναι ανεφικτό (infeasible)
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible\n")
        print(f"Nodes explored = {nodes_explored}\n\n")
        return model.status, "Infeasible", -1, solving_time, nodes_explored
    
    # ΠΕΡΙΠΤΩΣΗ 3: Άλλη κατάσταση (Time limit, διακοπή από χρήστη, κλπ.)
    else:
        print(f"Solver status: {model.status}\n")
        print(f"Nodes explored = {nodes_explored}\n\n")
        return model.status, "Unknown", -1, solving_time, nodes_explored


# ============================================================================
# ΣΥΝΑΡΤΗΣΗ ΧΕΙΡΙΣΜΟΥ ΟΡΙΣΜΑΤΩΝ ΓΡΑΜΜΗΣ ΕΝΤΟΛΩΝ
# ============================================================================
def parse_arguments():
    """
    Διαχειρίζεται τα ορίσματα εισόδου από τη γραμμή εντολών
    
    Αναμενόμενα ορίσματα:
    1. dir: Φάκελος που περιέχει τα προβλήματα (υποχρεωτικό)
    2. --problems: Λίστα συγκεκριμένων αρχείων προς επίλυση (προαιρετικό)
    
    Παραδείγματα χρήσης:
    - Επίλυση όλων: python gurobi_model.py instances/10_100_50_5/
    - Επίλυση συγκεκριμένων: python gurobi_model.py instances/10_100_50_5/ --problems 0.txt 1.txt
    
    Returns:
        Namespace object με τα parsed ορίσματα
    """

    parser = argparse.ArgumentParser(description="Facility Location Problem Solver with Gurobi")
    # Υποχρεωτικό όρισμα: φάκελος με προβλήματα
    parser.add_argument("dir", type=str, help="Directory containing problem files")
    # Προαιρετικό όρισμα: συγκεκριμένα προβλήματα
    parser.add_argument("--problems", nargs="*", type=str, help="Specific problems to solve") # nargs="*" : Δέχεται 0 ή περισσότερα ορίσματα
    return parser.parse_args()


# ===========================================================================
# ΚΛΑΣΗ ΑΠΟΘΗΚΕΥΣΗΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ
# ===========================================================================
class Problem_Results:
    """
    Κλάση για αποθήκευση αποτελεσμάτων επίλυσης ενός προβλήματος
    
    Attributes:
        name (str): Όνομα προβλήματος
        objective (float): Αντικειμενική τιμή (-1 αν ανεφικτό)
        status (int): Gurobi status code
        termination_condition (str): Περιγραφή κατάστασης ("Optimal", "Infeasible", κλπ.)
        solving_time (float): Χρόνος επίλυσης σε δευτερόλεπτα
        nodes_explored (int): Αριθμός εξερευνημένων κόμβων B&B δέντρου
    """

    def __init__(self, name="", objective=-1, status=None, termination_condition=None, solving_time=0, nodes_explored=0):
        self.name = name
        self.objective = objective
        self.status = status
        self.termination_condition = termination_condition
        self.solving_time = solving_time
        self.nodes_explored = nodes_explored


# ============================================================================
# ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ - ΣΗΜΕΙΟ ΕΝΑΡΞΗΣ
# ============================================================================
if __name__ == "__main__":
    print("\n\n===================== Gurobi Facility Location Solver =====================\n\n")

    # -------------------------------------------------------------------------
    # ΒΗΜΑ 1: ΧΕΙΡΙΣΜΟΣ ΟΡΙΣΜΑΤΩΝ
    # -------------------------------------------------------------------------
    args = parse_arguments()
    dir = args.dir  # Φάκελος με τα προβλήματα

    # Λίστα για αποθήκευση αποτελεσμάτων κάθε προβλήματος
    problem_results = list()

    # -------------------------------------------------------------------------
    # ΒΗΜΑ 2: ΕΠΙΛΥΣΗ ΠΡΟΒΛΗΜΑΤΩΝ
    # -------------------------------------------------------------------------
    # ΠΕΡΙΠΤΩΣΗ Α: Επίλυση συγκεκριμένων προβλημάτων
    if args.problems is not None:
        problems = args.problems

        for problem in problems:
            try:
                print(f"****************Solving problem {problem}****************\n\n")
                # Επίλυση instance
                st, tc, obj, solving_time, nodes_explored = run_model(dir+problem)
                # Αποθήκευση αποτελεσμάτων
                problem_results.append(Problem_Results(problem, obj, st, tc, solving_time, nodes_explored))
            except Exception as e:
                # Διαχείριση σφαλμάτων 
                print(f"{str(e)}\n\n")

    # ΠΕΡΙΠΤΩΣΗ Β: Επίλυση όλων των προβλημάτων στον φάκελο
    else:
        for name in sorted(os.listdir(dir)):
            print(f"****************Solving problem {name}****************\n\n")
            # Επίλυση instance
            st, tc, obj, solving_time, nodes_explored = run_model(dir+name)
            # Αποθήκευση αποτελεσμάτων
            problem_results.append(Problem_Results(name, obj, st, tc, solving_time, nodes_explored))

    # -------------------------------------------------------------------------
    # ΒΗΜΑ 3: ΚΑΤΗΓΟΡΙΟΠΟΙΗΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ
    # -------------------------------------------------------------------------
    # Δημιουργία λιστών για κάθε κατηγορία αποτελέσματος
    infeasible = list()           # Ανεφικτά προβλήματα
    feasible = list()             # Προβλήματα με βέλτιστη λύση
    aborted = list()              # Προβλήματα που διακόπηκαν (time limit, error, κλπ.)
    objectives = list()           # Αντικειμενικές τιμές εφικτών προβλημάτων
    solving_times = list()        # Χρόνοι επίλυσης εφικτών προβλημάτων
    nodes_explored_list = list()  # Κόμβοι B&B εφικτών προβλημάτων
   
    for pr in problem_results:
        if pr.status == 2:          # GRB.OPTIMAL - Βρέθηκε βέλτιστη λύση
           feasible.append(pr.name)
           objectives.append(pr.objective)
           solving_times.append(pr.solving_time)
           nodes_explored_list.append(pr.nodes_explored)
        elif pr.status == 3:        # GRB.INFEASIBLE - Πρόβλημα ανεφικτό
           infeasible.append(pr.name)
        else:                       # Οποιοδήποτε άλλο status (TIME_LIMIT, INTERRUPTED, κλπ.)
           aborted.append(pr.name)

    # -------------------------------------------------------------------------
    # ΒΗΜΑ 4: ΥΠΟΛΟΓΙΣΜΟΣ ΣΤΑΤΙΣΤΙΚΩΝ
    # -------------------------------------------------------------------------
    # Μέση αντικειμενική τιμή για εφικτά προβλήματα
    mean_obj = np.mean(objectives) 
    # Μέσος χρόνος επίλυσης για εφικτά προβλήματα
    sum_mean_time = sum(solving_times) / len(solving_times) if solving_times else 0

    # -------------------------------------------------------------------------
    # ΒΗΜΑ 5: ΕΚΤΥΠΩΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΣΤΟ TERMINAL
    # -------------------------------------------------------------------------
    print("\n\n=============================== RESULTS ==================================\n\n")
    print(f"Average objective value for all feasible problems: {round(mean_obj, 3)}.")
    print(f"Average run time for all feasible problems: {round(sum_mean_time,3)} s.\n")
    print(f"Time of every feasible problem: {[round(t,3) for t in solving_times]} s. \n")
    print(f"Nodes explored for every feasible problem: {nodes_explored_list}\n")
    print(f"\nInfeasible problems: {infeasible}")
    print(f"Feasible problems: {feasible}")
    print(f"Aborted problems: {aborted}")
    print()
    print("==========================================================================\n\n")

    # -------------------------------------------------------------------------
    # ΒΗΜΑ 6: ΑΠΟΘΗΚΕΥΣΗ ΛΕΠΤΟΜΕΡΩΝ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΣΕ ΑΡΧΕΙΟ
    # -------------------------------------------------------------------------
    with open("18_324_129_25.txt", "w") as file:
        # ΕΝΟΤΗΤΑ 1: Λεπτομερή αποτελέσματα ανά πρόβλημα
        file.write("=" * 80 + "\n")
        file.write("DETAILED RESULTS FOR EACH PROBLEM\n")
        file.write("=" * 80 + "\n\n")
     
        for pr in problem_results:
            file.write(f"Problem: {pr.name}\n")
            file.write(f"  Status: {pr.termination_condition}\n")
            if pr.status == 2:  # Αν είναι βέλτιστο, γράψε όλα τα στατιστικά
                 file.write(f"  Objective value: {round(pr.objective, 3)}\n")
                 file.write(f"  Solving time: {round(pr.solving_time, 3)} s\n")
                 file.write(f"  Nodes explored: {pr.nodes_explored}\n")
            file.write("\n")
     
        # ΕΝΟΤΗΤΑ 2: Συνοπτικά στατιστικά
        file.write("=" * 80 + "\n")
        file.write("SUMMARY STATISTICS\n")
        file.write("=" * 80 + "\n\n")
     
        file.write(f"Total problems solved: {len(objectives)} out of {len(problem_results)}\n\n")
     
        if objectives:  # Αν υπάρχουν εφικτά προβλήματα
             file.write(f"Average objective value for all feasible problems: {round(mean_obj, 3)}\n")
             file.write(f"Average run time for all feasible problems: {round(sum_mean_time, 3)} s\n")
         
             file.write(f"Time of every feasible problem: {[round(t, 3) for t in solving_times]} s\n")
             file.write(f"Nodes explored for every feasible problem: {nodes_explored_list}\n\n")
     
        file.write(f"Feasible problems: {feasible}\n")
        file.write(f"Infeasible problems: {infeasible}\n")
        file.write(f"Aborted problems: {aborted}\n\n")


    print("Τα αποτελέσματα αποθηκεύτηκαν στο αρχείο '18_324_129_25.txt'.\n")

