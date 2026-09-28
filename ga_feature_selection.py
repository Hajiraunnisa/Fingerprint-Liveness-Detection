
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ==========================
# LOAD CNN FEATURES
# ==========================
X = np.load("train_features.npy")
y = np.load("train_labels.npy")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

POP_SIZE = 30
GENERATIONS = 20
MUTATION_RATE = 0.05
GENES = X.shape[1]
ELITE_SIZE = 2

def fitness(chromosome):
    idx = np.where(chromosome == 1)[0]
    if len(idx) == 0:
        return 0.0

    Xtr = X_train[:, idx]
    Xva = X_val[:, idx]

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=10,
        random_state=42
    )

    clf.fit(Xtr, y_train)
    pred = clf.predict(Xva)

    acc = accuracy_score(y_val, pred)
    f1  = f1_score(y_val, pred)

    return 0.7 * acc + 0.3 * f1

def initialize_population():
    pop = np.random.randint(0,2,(POP_SIZE,GENES))
    for i in range(POP_SIZE):
        if pop[i].sum()==0:
            pop[i, np.random.randint(GENES)] = 1
    return pop

def tournament_selection(pop,scores,k=3):
    ids = random.sample(range(len(pop)),k)
    best = ids[0]
    for i in ids[1:]:
        if scores[i] > scores[best]:
            best = i
    return pop[best].copy()

def crossover(p1,p2):
    pt = random.randint(1,GENES-2)
    c1 = np.concatenate((p1[:pt],p2[pt:]))
    c2 = np.concatenate((p2[:pt],p1[pt:]))
    return c1,c2

def mutate(child):
    for i in range(GENES):
        if random.random() < MUTATION_RATE:
            child[i]=1-child[i]
    if child.sum()==0:
        child[random.randint(0,GENES-1)] = 1
    return child

population = initialize_population()

best_score = -1
best_chromosome = None
history=[]

for gen in range(GENERATIONS):

    scores = np.array([fitness(ind) for ind in population])

    elite_idx = np.argsort(scores)[-ELITE_SIZE:]
    elites = population[elite_idx].copy()

    if scores.max() > best_score:
        best_score = scores.max()
        best_chromosome = population[np.argmax(scores)].copy()

    history.append(best_score)

    print(f"Generation {gen+1}/{GENERATIONS}")
    print(f"Best Fitness : {best_score:.4f}")
    print(f"Average Fitness : {scores.mean():.4f}")
    print(f"Selected Features : {best_chromosome.sum()}")
    print("-"*40)

    new_pop = list(elites)

    while len(new_pop) < POP_SIZE:
        p1 = tournament_selection(population,scores)
        p2 = tournament_selection(population,scores)

        c1,c2 = crossover(p1,p2)

        c1 = mutate(c1)
        c2 = mutate(c2)

        new_pop.append(c1)
        if len(new_pop) < POP_SIZE:
            new_pop.append(c2)

    population = np.array(new_pop)

# =====================================================
# SAVE BEST FEATURE INDICES
# =====================================================

selected_idx = np.where(best_chromosome == 1)[0]

print("\n========== FINAL RESULTS ==========")
print("Best Fitness :", best_score)
print("Selected Features :", len(selected_idx))
print("Selected Feature Indices:")
print(selected_idx)

# =====================================================
# CREATE REDUCED DATASETS
# =====================================================

# Load complete datasets
X_train_full = np.load("train_features.npy")
y_train_full = np.load("train_labels.npy")

X_val_full = np.load("test_features.npy")      # Keep this if your file is named test_features.npy
y_val_full = np.load("test_labels.npy")        # Keep this if your file is named test_labels.npy

# If you saved them as val_features.npy / val_labels.npy,
# replace the above two lines with:
#
# X_val_full = np.load("val_features.npy")
# y_val_full = np.load("val_labels.npy")

# Select only GA-selected features
X_train_selected = X_train_full[:, selected_idx]
X_val_selected = X_val_full[:, selected_idx]

# =====================================================
# SAVE EVERYTHING
# =====================================================

np.save("best_chromosome.npy", best_chromosome)
np.save("best_features.npy", selected_idx)

np.save("train_selected_features.npy", X_train_selected)
np.save("val_selected_features.npy", X_val_selected)

np.save("train_selected_labels.npy", y_train_full)
np.save("val_selected_labels.npy", y_val_full)

# Save selected feature indices to a text file
with open("selected_feature_indices.txt", "w") as f:
    f.write("Selected Feature Indices\n")
    f.write("=========================\n\n")
    f.write(",".join(map(str, selected_idx)))

# =====================================================
# FITNESS HISTORY GRAPH
# =====================================================

plt.figure(figsize=(8,5))
plt.plot(range(1, GENERATIONS + 1), history, marker='o')
plt.title("GA Fitness History")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.tight_layout()
plt.savefig("fitness_history.png", dpi=300)
plt.show()

print("\n==============================")
print("FILES SAVED SUCCESSFULLY")
print("==============================")
print("✓ best_chromosome.npy")
print("✓ best_features.npy")
print("✓ train_selected_features.npy")
print("✓ val_selected_features.npy")
print("✓ train_selected_labels.npy")
print("✓ val_selected_labels.npy")
print("✓ selected_feature_indices.txt")
print("✓ fitness_history.png")