import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
import time
import warnings
from src.features import META_COLS

warnings.filterwarnings("ignore")


class FeatureSelectionArena:
    # Compare GA and BPSO
    def __init__(self, pop_size=30, n_iterations=20):
        # Population Size
        self.pop_size = pop_size
        # Iteration Times
        self.n_iterations = n_iterations

    def _fitness(self, solution_mask, X_train, y_train, X_test, y_test):
        if np.sum(solution_mask) == 0:
            return 0.0
        # Extract the selected feature subset column
        X_tr_sub = X_train[:, solution_mask == 1]
        X_te_sub = X_test[:, solution_mask == 1]
        # Rapid assessment using lightweight forests
        clf = RandomForestClassifier(
            n_estimators=30, max_depth=10, random_state=42, n_jobs=-1
        )
        clf.fit(X_tr_sub, y_train)
        preds = clf.predict(X_te_sub)
        return f1_score(y_test, preds, average="macro")

    def run_ga(self, X_tr, y_tr, X_te, y_te, n_features):
        # Genetic Algorithm
        mutation_rate = 0.1
        population = np.random.randint(0, 2, size=(self.pop_size, n_features))
        best_mask, best_score = None, -1
        for gen in range(self.n_iterations):
            scores = np.array(
                [self._fitness(ind, X_tr, y_tr, X_te, y_te) for ind in population]
            )
            if np.max(scores) > best_score:
                best_score = np.max(scores)
                best_mask = population[np.argmax(scores)].copy()
            new_pop = []
            for _ in range(self.pop_size):
                tournament = np.random.choice(self.pop_size, size=3, replace=False)
                winner = tournament[np.argmax(scores[tournament])]
                new_pop.append(population[winner])
            new_pop = np.array(new_pop)
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    cx_point = np.random.randint(1, n_features - 1)
                    new_pop[i, cx_point:], new_pop[i + 1, cx_point:] = (
                        new_pop[i + 1, cx_point:].copy(),
                        new_pop[i, cx_point:].copy(),
                    )
            mutation_mask = np.random.rand(self.pop_size, n_features) < mutation_rate
            new_pop ^= mutation_mask
            population = new_pop
        return best_mask, best_score

    def run_bpso(self, X_tr, y_tr, X_te, y_te, n_features):
        # PSO
        # Inertia weights and learning factors
        w, c1, c2 = 0.8, 1.5, 1.5
        positions = np.random.randint(0, 2, size=(self.pop_size, n_features))
        velocities = np.random.uniform(-1, 1, size=(self.pop_size, n_features))
        pbest_positions = positions.copy()
        pbest_scores = np.zeros(self.pop_size)
        gbest_position, gbest_score = None, -1
        for it in range(self.n_iterations):
            scores = np.array(
                [self._fitness(pos, X_tr, y_tr, X_te, y_te) for pos in positions]
            )
            for i in range(self.pop_size):
                if scores[i] > pbest_scores[i]:
                    pbest_scores[i] = scores[i]
                    pbest_positions[i] = positions[i].copy()

                if pbest_scores[i] > gbest_score:
                    gbest_score = pbest_scores[i]
                    gbest_position = pbest_positions[i].copy()
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                # PSO core formular
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (pbest_positions[i] - positions[i])
                    + c2 * r2 * (gbest_position - positions[i])
                )
                sigmoid_v = 1 / (1 + np.exp(-velocities[i]))
                positions[i] = (np.random.rand(n_features) < sigmoid_v).astype(int)
        return gbest_position, gbest_score

    def fight(self, csv_path: str):
        df = pd.read_csv(csv_path)
        feature_names = np.array([c for c in df.columns if c not in META_COLS])

        X = df[feature_names].values
        y = df["label"].values
        groups = df["subject_id"].values
        n_features = len(feature_names)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        print(f"All features dimension: {n_features}")
        print(
            f"Iteration parameters: population {self.pop_size} | iteration {self.n_iterations} \n"
        )

        # GA
        t0 = time.time()
        ga_mask, ga_score = self.run_ga(X_tr, y_tr, X_te, y_te, n_features)
        ga_time = time.time() - t0
        # BPSO
        t0 = time.time()
        pso_mask, pso_score = self.run_bpso(X_tr, y_tr, X_te, y_te, n_features)
        pso_time = time.time() - t0

        ga_features = feature_names[ga_mask == 1].tolist()
        pso_features = feature_names[pso_mask == 1].tolist()

        print("\n================ Results ================")
        print(f"GA Algorithm")
        print(f"   Cost: {ga_time:.1f} second | Final F1-Score: {ga_score:.4f}")
        print(f"   Choose {len(ga_features)} features: {ga_features}\n")

        print(f"BPSO Algorithm")
        print(f"   Cost: {pso_time:.1f} second | Final F1-Score: {pso_score:.4f}")
        print(f"   Choose {len(pso_features)} features: {pso_features}\n")

        if pso_score > ga_score:
            print("Suggest to choose BPSO")
        elif ga_score > pso_score:
            print("Suggest to choose GA")
        else:
            if len(pso_features) < len(ga_features):
                print("BPSO better. Although same score, but BPSO use less features")
            else:
                print("GA better. Although same score, but GA use less features")


if __name__ == "__main__":
    ENHANCED_CSV = "data/dataset/enhanced.csv"
    arena = FeatureSelectionArena(pop_size=20, n_iterations=15)
    arena.fight(ENHANCED_CSV)
