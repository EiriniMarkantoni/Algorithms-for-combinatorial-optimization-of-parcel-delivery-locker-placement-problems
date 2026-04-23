"""
=========================================================================================
ΒΑΣΙΚΟΣ ΑΛΓΟΡΙΘΜΟΣ BRANCH AND BOUND ΓΙΑ ΕΤΕΡΟΓΕΝΕΣ ΠΡΟΒΛΗΜΑ ΧΩΡΟΘΕΤΗΣΗΣ ΘΥΡΙΔΩΝ ΠΑΡΑΔΟΣΗΣ
=========================================================================================

ΠΕΡΙΓΡΑΦΗ ΑΛΓΟΡΙΘΜΟΥ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Αυτός ο κώδικας υλοποιεί την βασική (baseline) έκδοση του αλγορίθμου 
Branch and Bound για την επίλυση μικτών ακέραιων γραμμικών προβλημάτων (MILP)
Ο αλγόριθμος χρησιμοποιεί:

- Στρατηγική εξερεύνησης: Depth-First Search (DFS) με χρήση στοίβας (stack)
- Κανόνας επιλογής μεταβλητής: First Fractional Variable (απλός κανόνας)
- Κλάδεμα: Bounding με χρήση άνω και κάτω φραγμάτων (pruning by bounds)
- Βελτιστοποίηση: Warm start με χρήση simplex bases από γονικούς κόμβους

Η υλοποίηση αποτελεί τη βάση σύγκρισης (baseline) για την αξιολόγηση 
της αποδοτικότητας της βελτιωμένης εκδόσης του αλγορίθμου που αναπτύχθηκε
στα πλαίσια της διπλωματικής εργασίας

ΧΡΗΣΗ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ο αλγόριθμος δέχεται ως είσοδο ένα μοντέλο Gurobi και επιστρέφει τη βέλτιστη 
ακέραια λύση μαζί με στατιστικά εκτέλεσης (χρόνος, αριθμός κόμβων)

ΣΥΓΓΡΑΦΕΑΣ: Ειρήνη Μαρκαντώνη
ΗΜΕΡΟΜΗΝΙΑ: Ιανουάριος 2026
ΣΚΟΠΟΣ: Διπλωματική Εργασία 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from collections import deque
import model_for_bnb as md


# ===========================================================================
# ΚΑΘΟΛΙΚΕΣ ΣΤΑΘΕΡΕΣ ΚΑΙ ΜΕΤΑΒΛΗΤΕΣ
# ===========================================================================
TIME_LIMIT = 3600     # Χρονικό όριο εκτέλεσης του αλγορίθμου (σε δευτερόλεπτα) - 1 ώρα

isMax = None          # Καθορισμός της βελτιστοποίησης (μεγιστοποίηση ή ελαχιστοποίηση)
DEBUG_MODE = False    # Ενεργοποίηση/απενεργοποίηση λειτουργίας αποσφαλμάτωσης (debug mode)
nodes = 0             # Μετρητής συνολικού αριθμού κόμβων που εξερευνήθηκαν στο δέντρο αναζήτησης
lower_bound = -np.inf # Κάτω φράγμα (lower bound) του προβλήματος - αρχικοποιείται στο -∞
upper_bound = np.inf  # Άνω φράγμα (upper bound) του προβλήματος - αρχικοποιείται στο +∞


# ===========================================================================
# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# ===========================================================================
def is_nearly_integer(value, tolerance=1e-6):
    """
    Ελέγχει αν μια τιμή είναι πολύ κοντά σε ακέραιο αριθμό
    
    Παράμετροι:
        value: η τιμή προς έλεγχο
        tolerance: η ανοχή για την απόκλιση από ακέραιο (προεπιλογή 1e-6)
    
    Επιστρέφει:
        True αν η τιμή είναι εντός της ανοχής από το πλησιέστερο ακέραιο
    """

    return abs(value - round(value)) <= tolerance


# ===========================================================================
# ΔΟΜΗ ΔΕΔΟΜΕΝΩΝ ΚΟΜΒΟΥ (CLASS NODE)
# ===========================================================================
class Node:
    """
    Κλάση που αναπαριστά έναν κόμβο στο δέντρο Branch and Bound
    
    Κάθε κόμβος αποθηκεύει:
    - Τα όρια (bounds) των μεταβλητών για αυτόν τον κόμβο
    - Το βάθος του στο δέντρο
    - Τις βάσεις (basis) για warm start του simplex
    - Τη μεταβλητή διακλάδωσης (branching variable)
    - Μια ετικέτα αναγνώρισης (πχ "Left", "Right", "root")
    """

    def __init__(self, ub, lb, depth, vbasis, cbasis, branching_var, label=""):
        self.ub = ub                        # Άνω όρια (upper bounds) των μεταβλητών
        self.lb = lb                        # Κάτω όρια (lower bounds) των μεταβλητών
        self.depth = depth                  # Βάθος κόμβου στο δέντρο
        self.vbasis = vbasis                # Variable basis για warm start
        self.cbasis = cbasis                # Constraint basis για warm start
        self.branching_var = branching_var  # Η μεταβλητή στην οποία έγινε η διακλάδωση
        self.label = label                  # Ετικέτα κόμβου (πχ "Left", "Right")


# ==========================================================================
# ΣΥΝΑΡΤΗΣΗ ΕΚΤΥΠΩΣΗΣ DEBUG ΠΛΗΡΟΦΟΡΙΩΝ
# ==========================================================================
def debug_print(node: Node = None, x_obj=None, sol_status=None):
    """
    Εκτυπώνει πληροφορίες αποσφαλμάτωσης κατά την εκτέλεση του αλγορίθμου
    
    Παράμετροι:
        node: ο τρέχων κόμβος (προαιρετικό)
        x_obj: η τιμή της αντικειμενικής συνάρτησης (προαιρετικό)
        sol_status: η κατάσταση της λύσης (προαιρετικό)
    """

    print("\n\n-----------------  DEBUG OUTPUT  -----------------\n\n")
    print(f"UB:{upper_bound}")
    print(f"LB:{lower_bound}")
    if node is not None:
        print(f"Branching Var: {node.branching_var}")
    if node is not None:
        print(f"Child: {node.label}")
    if node is not None:
        print(f"Depth: {node.depth}")
    if x_obj is not None:
        print(f"Simplex Objective: {x_obj}")
    if sol_status is not None:
        print(f"Solution status: {sol_status}")

    print("\n\n--------------------------------------------------\n\n")


# =========================================================================================================================
#                                             ΣΥΝΑΡΤΗΣΗ BRANCH AND BOUND
# ==========================================================================================================================
def branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth,start, vbasis=[], cbasis=[], depth=0):
    """
    Υλοποίηση του αλγορίθμου Branch and Bound για επίλυση ακέραιων προβλημάτων
    
    Ο αλγόριθμος λειτουργεί ως εξής:
    1. Επιλύει το χαλαρωμένο πρόβλημα (LP relaxation)
    2. Αν η λύση είναι ακέραια, την αποθηκεύει
    3. Αν όχι, δημιουργεί δύο υποπροβλήματα (κόμβους-παιδιά) με διακλάδωση σε μια κλασματική μεταβλητή
    4. Επαναλαμβάνει τη διαδικασία χρησιμοποιώντας στοίβα (stack) για την εξερεύνηση
    5. Κόβει κλάδους (pruning) όταν τα φράγματα δείχνουν ότι δεν υπάρχει καλύτερη λύση
    
    Παράμετροι:
        model: το μοντέλο Gurobi προς επίλυση
        ub: τα αρχικά άνω όρια των μεταβλητών
        lb: τα αρχικά κάτω όρια των μεταβλητών
        integer_var: boolean λίστα που δείχνει ποιες μεταβλητές πρέπει να είναι ακέραιες
        best_bound_per_depth: πίνακας με το καλύτερο φράγμα σε κάθε βάθος
        nodes_per_depth: πίνακας με τον αριθμό κόμβων ανά βάθος
        start: χρόνος έναρξης για έλεγχο χρονικού ορίου
        vbasis: αρχική variable basis για warm start (προαιρετικό)
        cbasis: αρχική constraint basis για warm start (προαιρετικό)
        depth: αρχικό βάθος (προεπιλογή 0)
    
    Επιστρέφει:
        solutions: λίστα με όλες τις λύσεις που βρέθηκαν (κάθε λύση έχει τη μορφή [τιμές μεταβλητών, τιμή αντικειμενικής, βάθος])
        best_sol_idx: δείκτης της καλύτερης λύσης
        solutions_found: συνολικός αριθμός λύσεων που βρέθηκαν
    """

    global nodes, lower_bound, upper_bound

    # Δημιουργία στοίβας (stack) για την αποθήκευση των κόμβων προς εξερεύνηση
    # Χρησιμοποιείται deque για αποδοτικές προσθήκες/αφαιρέσεις από το τέλος
    stack = deque()


    solutions = list()    # Αρχικοποίηση λίστας για την αποθήκευση των λύσεων που βρέθηκαν
    solutions_found = 0   # Μετρητής λύσεων
    best_sol_idx = 0      # Δείκτης της καλύτερης λύσης

    # Αρχικοποίηση της καλύτερης τιμής αντικειμενικής συνάρτησης
    # Ανάλογα με το αν είναι πρόβλημα μεγιστοποίησης ή ελαχιστοποίησης
    if isMax:
        best_sol_obj = -np.inf  # Αρχικοποίηση στο -∞ για μεγιστοποίηση
    else:
        best_sol_obj = np.inf   # Αρχικοποιήση στο +∞ για ελαχιστοποίηση

    # Δημιουργία του ριζικού κόμβου (root node) του δέντρου
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")
    nodes_per_depth[0] -= 1 

    
    # ===========================================================================
    # ΕΠΙΛΥΣΗ ΡΙΖΙΚΟΥ ΚΟΜΒΟΥ (ROOT NODE)
    # ===========================================================================
    if DEBUG_MODE:
        debug_print()

    # Επίλυση του χαλαρωμένου προβλήματος (χωρίς περιορισμούς ακεραιότητας)
    model.optimize()

    # Έλεγχος αν το μοντέλο επιλύθηκε βέλτιστα
    # Αν όχι, το πρόβλημα είναι μη εφικτό (infeasible)
    if model.status != GRB.OPTIMAL:
        if isMax:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], -np.inf, depth
        else:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], np.inf, depth

    # Ανάκτηση των τιμών των μεταβλητών από τη λύση
    x_candidate = model.getAttr('X', model.getVars()) 

    # Ανάκτηση της τιμής της αντικειμενικής συνάρτησης
    x_obj = model.ObjVal
    best_bound_per_depth[0] = x_obj # Ενημέρωση του καλύτερου φράγματος στο βάθος 0

    # Έλεγχος αν όλες οι μεταβλητές που πρέπει να είναι ακέραιες έχουν πράγματι ακέραιες τιμές
    # Αν βρεθεί κλασματική μεταβλητή, επιλέγεται για διακλάδωση
    vars_have_integer_vals = True
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var and not is_nearly_integer(x_candidate[idx]):
            vars_have_integer_vals = False
            selected_var_idx = idx  # Επιλογή πρώτης κλασματικής μεταβλητής
            break

    # Αν όλες οι μεταβλητές είναι ακέραιες, βρέθηκε εφικτή λύση
    if vars_have_integer_vals:
        # Αποθήκευση της λύσης στη ρίζα - αυτό σημαίνει ότι η χαλάρωση ήδη έδωσε ακέραια λύση
        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1

        if DEBUG_MODE:
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        # Επιστροφή αμέσως αφού βρέθηκε βέλτιστη λύση στη ρίζα
        return solutions, best_sol_idx, solutions_found

    # Αν η λύση στη ρίζα είναι κλασματική, ενημέρωση των φραγμάτων
    else:
        if isMax:
            upper_bound = x_obj  # Για μεγιστοποίηση: η LP relaxation δίνει άνω φράγμα
        else:
            lower_bound = x_obj  # Για ελαχιστοποίηση: η LP relaxation δίνει κάτω φράγμα

    if DEBUG_MODE:
        debug_print(node=root_node, x_obj=x_obj, sol_status="Fractional")

    # Ανάκτηση των βάσεων (bases) για warm start στους επόμενους κόμβους
    # Η χρήση warm start επιταχύνει την επίλυση των υποπροβλημάτων
    vbasis = model.getAttr("VBasis", model.getVars())
    cbasis = model.getAttr("CBasis", model.getConstrs())

    # Δημιουργία αντιγράφων των ορίων για τους κόμβους-παιδιά
    left_lb = np.copy(lb)
    left_ub = np.copy(ub)
    right_lb = np.copy(lb)
    right_ub = np.copy(ub)

    # Διακλάδωση στην επιλεγμένη κλασματική μεταβλητή:
    # - Αριστερός κόμβος: x_i ≤ ⌊x_i⌋ (στρογγυλοποίηση προς τα κάτω - προς το μικτότερο ακέραιο)
    # - Δεξιός κόμβος: x_i ≥ ⌈x_i⌉ (στρογγυλοποίηση προς τα πάνω - προς το μεγαλύτερο ακέραιο)
    left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
    right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

    # Δημιουργία των κόμβων-παιδιών με τα νέα όρια
    left_child = Node(left_ub, left_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
    right_child = Node(right_ub, right_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

    # Προσθήκη των κόμβων-παιδιών στη στοίβα
    # Σημείωση: Ο δεξιός προστίθεται πρώτος ώστε ο αριστερός να εξερευνηθεί πρώτος (LIFO)
    stack.append(right_child)
    stack.append(left_child)

    
    # ===========================================================================
    # ΕΠΙΛΥΣΗ ΥΠΟΠΡΟΒΛΗΜΑΤΩΝ (SUBPROBLEMS)
    # ===========================================================================
    # Η κύρια επαναληπτική διαδικασία: συνεχίζει όσο υπάρχουν κόμβοι στη στοίβα.
    # Ο βρόχος μπορεί να διακοπεί εσωτερικά αν:
    #   - ξεπεραστεί το χρονικό όριο
    #   - βρεθεί βέλτιστη λύση (σύγκλιση φραγμάτων)

    while (len(stack) != 0):
        print("\n********************************  NEW NODE BEING EXPLORED  ******************************** ")

        # ------ ΕΛΕΓΧΟΣ ΧΡΟΝΙΚΟΥ ΟΡΙΟΥ ------
        elapsed_time = time.time() - start
        if elapsed_time >= TIME_LIMIT:
            print(f"\n Time limit of {TIME_LIMIT/60:.0f} minutes reached.")
            print("Returning best solution found so far...\n")
            
             # Αν δεν βρέθηκε καμία λύση εντός του χρονικού ορίου
            if len(solutions) == 0:
                print("No feasible solution found within time limit.")
                return [], 0, 0
            # Επιστροφή της καλύτερης λύσης που βρέθηκε μέχρι στιγμής
            return solutions, best_sol_idx, solutions_found  

        # Αύξηση του μετρητή των εξερευνημένων κόμβων
        nodes += 1

        # Λήψη του επόμενου κόμβου από την κορυφή της στοίβας (LIFO - Last In First Out)
        current_node = stack[-1]

        # Αφαίρεση του κόμβου από τη στοίβα
        stack.pop()

        # Ενημέρωση του αριθμού κόμβων που απομένουν σε αυτό το βάθος
        nodes_per_depth[current_node.depth] -= 1

        # ----- WARM START ------
        # Χρήση των βάσεων που κληρονόμησε από τον γονικό κόμβο για να ξεκινήσει ο simplex από ένα καλό σημείο εκκίνησης
        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
            model.setAttr("VBasis", model.getVars(), current_node.vbasis)
            model.setAttr("CBasis", model.getConstrs(), current_node.cbasis)

        # ----- ΕΝΗΜΕΡΩΣΗ ΜΟΝΤΕΛΟΥ ΜΕ ΤΑ ΟΡΙΑ ΤΟΥ ΤΡΕΧΟΝΤΟΣ ΚΟΜΒΟΥ -----
        # Αλλαγή των ορίων των μεταβλητών σύμφωνα με τις διακλαδώσεις που έγιναν
        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)
        model.update()

        if DEBUG_MODE:
            debug_print()

        # ----- ΕΠΙΛΥΣΗ ΥΠΟΠΡΟΒΛΗΜΑΤΟΣ -----
        model.optimize()

        # ----- ΕΛΕΓΧΟΣ ΕΦΙΚΤΟΤΗΤΑΣ -----
        # Αν το υποπρόβλημα δεν επιλύθηκε βέλτιστα, είναι μη εφικτό
        infeasible = False
        if model.status != GRB.OPTIMAL:
            if isMax:
                infeasible = True
                x_obj = -np.inf
            else:
                infeasible = True
                x_obj = np.inf
            # Αφαιρείται ολόκληρο το υποδέντρο από αυτόν τον κόμβο και κάτω επειδή το υποπρόβλημα είναι μη εφικτό
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

        else:
            # Ανάκτηση των τιμών των μεταβλητών από τη λύση
            x_candidate = model.getAttr('X', model.getVars()) 

            # Ανάκτηση της τιμής της αντικειμενικής συνάρτησης
            x_obj = model.ObjVal

            # ----- ΕΝΗΜΕΡΩΣΗ ΚΑΛΥΤΕΡΟΥ ΦΡΑΓΜΑΤΟΣ ΑΝΑ ΒΑΘΟΣ -----
            # Διατήρηση του καλύτερου φράγματος που έχει βρεθεί σε αυτό το βάθος
            if isMax == True and x_obj > best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj
            elif isMax == False and x_obj < best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj

            # ----- ΕΝΗΜΕΡΩΣΗ ΚΑΘΟΛΙΚΩΝ ΦΡΑΓΜΑΤΩΝ -----
            # Όταν ολοκληρωθούν όλοι οι κόμβοι ενός βάθους, ενημερώνεται το καθολικό φράγμα
            if nodes_per_depth[current_node.depth] == 0:
                if isMax == True:
                    upper_bound = best_bound_per_depth[current_node.depth]
                else:
                    lower_bound = best_bound_per_depth[current_node.depth]

        # ----- ΧΕΙΡΙΣΜΟΣ ΜΗ ΕΦΙΚΤΟΥ ΚΟΜΒΟΥ -----
        # Αν ο κόμβος είναι μη εφικτός, δεν δημιουργούνται παιδιά και συνεχίζεται η αναζήτηση
        if infeasible:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

        # ----- ΕΛΕΓΧΟΣ ΑΚΕΡΑΙΟΤΗΤΑΣ -----
        # Έλεγχος αν όλες οι ακέραιες μεταβλητές έχουν πράγματι ακέραιες τιμές
        vars_have_integer_vals = True
        for idx, is_int_var in enumerate(integer_var):
            if is_int_var and not is_nearly_integer(x_candidate[idx]):
                vars_have_integer_vals = False
                selected_var_idx = idx  # Επιλογή πρώτης κλασματικής μεταβλητής
                break

        # ----- ΧΕΙΡΙΣΜΟΣ ΑΚΕΡΑΙΑΣ ΛΥΣΗΣ -----
        if vars_have_integer_vals: 
            # Βρέθηκε εφικτή ακέραια λύση
            if isMax:
                # ----- ΜΕΓΙΣΤΟΠΟΙΗΣΗ -----
                if lower_bound < x_obj: 
                    # Βρέθηκε καλύτερη λύση - ενημέρωση κάτω φράγματος
                    lower_bound = x_obj

                    if abs(lower_bound - upper_bound) < 1e-6: 
                        # ----- ΕΠΙΤΕΥΧΘΗΚΕ ΒΕΛΤΙΣΤΟΤΗΤΑ -----
                        # Το κάτω και άνω φράγμα συναντήθηκαν - βέλτιστη λύση
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) < 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1

                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found

                    # ----- ΟΧΙ ΒΕΛΤΙΣΤΗ ΛΥΣΗ ΑΚΟΜΑ -----
                    # Αποθήκευση της λύσης, αύξηση μετρητή λύσεων και ενημέρωση καλύτερης λύσης
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # Δεν γίνεται περαιτέρω διακλάδωση από αυτόν τον κόμβο
                    # Δεν δημιουργούνται παιδιά και αφαιρείται το υποδένδρο
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 * (i - current_node.depth)

                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue

            else:
                # ----- ΕΛΑΧΙΣΤΟΠΟΙΗΣΗ -----
                if upper_bound > x_obj: 
                    # Βρέθηκε καλύτερη λύση - ενημέρωση άνω φράγματος
                    upper_bound = x_obj 

                    if abs(lower_bound - upper_bound) < 1e-6: 
                        # ----- ΕΠΙΤΕΥΧΘΗΚΕ ΒΕΛΤΙΣΤΟΤΗΤΑ -----
                        # Το κάτω και άνω φράγμα συναντήθηκαν - βέλτιστη λύση
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1
                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found

                    # ----- ΟΧΙ ΒΕΛΤΙΣΤΗ ΛΥΣΗ ΑΚΟΜΑ -----
                    # Αποθήκευση της λύσης, αύξηση μετρητή λύσεων και ενημέρωση καλύτερης λύσης
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # Δεν γίνεται περαιτέρω διακλάδωση από αυτόν τον κόμβο
                    # Δεν δημιουργούνται παιδιά και αφαιρείται το υποδένδρο
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 * (i - current_node.depth)

                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue

            # ----- ΙΣΟΔΥΝΑΜΗ ΛΥΣΗ -----
            # Αν η ακέραια λύση δεν βελτιώνει το τρέχον φράγμα, απορρίπτεται και δεν γίνεται επέκταση των παιδιών
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj,
                            sol_status="Integer (Rejected -- Doesn't improve incumbent)")
            continue

        # ----- ΚΛΑΔΕΜΑ ΜΕ ΒΑΣΗ ΦΡΑΓΜΑΤΑ (BOUNDING) -----
        # Αν η λύση του υποπροβλήματος δεν μπορεί να βελτιώσει το τρέχον φράγμα, κλαδεύεται
        if isMax:
            # Για μεγιστοποίηση: κλάδεμα αν x_obj ≤ lower_bound
            if x_obj < lower_bound or abs(x_obj - lower_bound) < 1e-6: 
                # Αφαίρεση των κόμβων-απογόνων (δεν θα δημιουργηθούν)
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 * (i - current_node.depth)
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue
        else:
            # Για ελαχιστοποίηση: κλάδεμα αν x_obj ≥ upper_bound
            if x_obj > upper_bound or abs(x_obj - upper_bound) < 1e-6: 
                # Αφαίρεση των κόμβων-απογόνων (δεν θα δημιουργηθούν)
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 * (i - current_node.depth)
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue

        if DEBUG_MODE:
            debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional")


        # ----- ΔΙΑΚΛΑΔΩΣΗ (BRANCHING) -----
        # Η λύση είναι κλασματική και δεν κλαδεύτηκε - δημιουργία κόμβων-παιδιών
        
        # Ανάκτηση βάσεων για warm start των παιδιών
        vbasis = model.getAttr("VBasis", model.getVars())
        cbasis = model.getAttr("CBasis", model.getConstrs())

        # Δημιουργία αντιγράφων των ορίων για τους κόμβους-παιδιά
        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)

        # Εφαρμογή διακλάδωσης στην επιλεγμένη κλασματική μεταβλητή
        left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

        # Δημιουργία των κόμβων-παιδιών
        left_child = Node(left_ub, left_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx,
                          "Left")
        right_child = Node(right_ub, right_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx,
                           "Right")

        # Προσθήκη των κόμβων-παιδιών στη στοίβα για μελλοντική εξερεύνηση
        stack.append(right_child)
        stack.append(left_child)

    # Επιστροφή όλων των λύσεων που βρέθηκαν 
    return solutions, best_sol_idx, solutions_found


# =============================================================================================================================
#                                                 ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ (MAIN)
# =============================================================================================================================
if __name__ == "__main__": 
    
    # Αρχικοποίηση λιστών για την αποθήκευση των αποτελεσμάτων όλων των προβλημάτων
    objective_values = []    # Τιμές αντικειμενικής συνάρτησης
    elapsed_times = []       # Χρόνοι εκτέλεσης

    # Άνοιγμα αρχείου για την εγγραφή των αποτελεσμάτων
    with open("10_100_30_9.txt", "w") as f_out:
        # Βρόχος για την επίλυση όλων των προβλημάτων (0.txt έως 9.txt)
        for i in range(10):
            problem_file = f"problems/10_100_30_9/{i}.txt"
            f_out.write(f"========= Problem {i} =========\n")
            print(f"Solving {problem_file} ...")

            # ----- ΦΟΡΤΩΣΗ ΜΟΝΤΕΛΟΥ -----
            # Φόρτωση των δεδομένων του προβλήματος και δημιουργία του μοντέλου
            model, ub, lb, integer_var, num_vars, N , E, CL, P, cl_sp_distances, transport_mode, emission_values, cost_types, facility_types, Cmax, capacities, demand, get_max_distance ,combined_cost_per_km = md.run_model(problem_file)

            # ----- ΕΠΑΝΑΦΟΡΑ ΚΑΘΟΛΙΚΩΝ ΜΕΤΑΒΛΗΤΩΝ -----
            # Σημαντικό: Μηδενισμός των καθολικών μεταβλητών για κάθε νέο πρόβλημα
            nodes = 0
            lower_bound = -np.inf
            upper_bound = np.inf

            # ----- ΑΡΧΙΚΟΠΟΙΗΣΗ ΔΟΜΩΝ BRANCH-AND-BOUND -----
            # Δημιουργία πινάκων για την παρακολούθηση των φραγμάτων και κόμβων ανά βάθος
            if isMax:
                best_bound_per_depth = np.array([-np.inf for _ in range(num_vars)])
            else:
                best_bound_per_depth = np.array([np.inf for _ in range(num_vars)])

            # Υπολογισμός του θεωρητικού αριθμού κόμβων σε κάθε βάθος
            # Σε κάθε βάθος διπλασιάζονται οι κόμβοι (πλήρες δυαδικό δέντρο)
            nodes_per_depth = np.zeros(num_vars + 1, dtype=float)
            nodes_per_depth[0] = 1
            for j in range(1, num_vars + 1):
                nodes_per_depth[j] = nodes_per_depth[j-1] * 2

            # ----- ΕΚΤΕΛΕΣΗ BRANCH-AND-BOUND -----
            start = time.time()      # Καταγραφή χρόνου έναρξης
            solutions, best_sol_idx, solutions_found = branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth, start)
            end = time.time()        # Καταγραφή χρόνου λήξης
            elapsed_time = end - start

            # ----- ΕΠΕΞΕΡΓΑΣΙΑ ΑΠΟΤΕΛΕΣΜΑΤΩΝ -----
            if len(solutions) == 0:
                print("No feasible solution found!")
            else:
                # Εύρεση της πραγματικά βέλτιστης λύσης από όλες τις λύσεις που βρέθηκαν 
                if isMax:
                    # Για μεγιστοποίηση: η λύση με τη μέγιστη τιμή
                    best_sol_idx = np.argmax([sol[1] for sol in solutions])
                    objective_values.append(solutions[best_sol_idx][1])
                else:
                    # Για ελαχιστοποίηση: η λύση με την ελάχιστη τιμή
                    best_sol_idx = np.argmin([sol[1] for sol in solutions])
                    objective_values.append(solutions[best_sol_idx][1])

                elapsed_times.append(elapsed_time)


            # ----- ΕΓΓΡΑΦΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΣΤΟ ΑΡΧΕΙΟ -----
            if len(solutions) > 0:
                f_out.write(f"Objective Value: {round(solutions[best_sol_idx][1], 3)}\n")
                f_out.write(f"Tree depth: {solutions[best_sol_idx][2]}\n")
            else:
                f_out.write("No feasible solution found.\n")

            f_out.write(f"Time Elapsed: {round(elapsed_time,3)} seconds\n")
            f_out.write(f"Total nodes: {nodes}\n")
            f_out.write("\n")

        
        # ----- ΥΠΟΛΟΓΙΣΜΟΣ ΜΕΣΩΝ ΟΡΩΝ -----
        # Υπολογισμός του μέσου όρου της τιμής αντικειμενικής συνάρτησης
        if len(objective_values) > 0:
            avg_obj = round(sum(objective_values)/len(objective_values), 3)
        else:
            avg_obj = None

        # Υπολογισμός του μέσου όρου του χρόνου εκτέλεσης
        avg_time = round(sum(elapsed_times)/len(elapsed_times), 3)

        # Εγγραφή των μέσων όρων στο αρχείο
        f_out.write("========= Averages =========\n")
        if avg_obj is not None:
            f_out.write(f"Average Objective Value: {avg_obj}\n")
        f_out.write(f"Average Time Elapsed: {avg_time} seconds\n")
