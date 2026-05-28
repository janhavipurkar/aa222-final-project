# Notes:
# Builds p(x): probability a real NEA exists near x = [a, e, i]
# Used in the combined objective f(x)*p(x).

# Gaussian RBF surrogate (Ch. 14.3.3) -- one basis function per NEA,
# centered at its orbital coordinates with a minimum bandwidth enforced.

# from existence_model import load_existence_model
# p = load_existence_model()
# p([1.2, 0.15, 5.0])  -> float in [0, 1]

import csv
import numpy as np

# minimum bandwidth per orbital element -- controls how wide each Gaussian is
# set to roughly one std of spread across NEA orbital families
MIN_SIGMA = np.array([0.15, 0.05, 4.0])  # sigma_a (au), sigma_e, sigma_i (deg)


def load_centers(filename="dataset_cleaned.csv"):
    # load [a, e, i] for NEAs only (neo=Y), with per-asteroid bandwidths
    # per-asteroid sigma from JPL orbit uncertainties, clamped to MIN_SIGMA
    # orbital validity: a in [0.5, 4.2] AU (excludes ~13 misclassified outliers)
    centers, sigmas = [], []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_sigma = all(k in reader.fieldnames for k in ["sigma_a", "sigma_e", "sigma_i"])
        for r in reader:
            if r.get("neo") != "Y":
                continue
            try:
                a = float(r["a"])
                e = float(r["e"])
                i = float(r["i"])
            except (ValueError, KeyError):
                continue
            if not (0.5 <= a <= 4.2) or not (0.0 <= e < 1.0):
                continue
            centers.append([a, e, i])
            if has_sigma:
                try:
                    sa = float(r["sigma_a"]) if r["sigma_a"] not in ("", "None") else MIN_SIGMA[0]
                    se = float(r["sigma_e"]) if r["sigma_e"] not in ("", "None") else MIN_SIGMA[1]
                    si = float(r["sigma_i"]) if r["sigma_i"] not in ("", "None") else MIN_SIGMA[2]
                    sigmas.append([sa, se, si])
                except (ValueError, KeyError):
                    sigmas.append(MIN_SIGMA.tolist())
            else:
                sigmas.append(MIN_SIGMA.tolist())

    C = np.array(centers)
    S = np.maximum(np.array(sigmas), MIN_SIGMA)  # clamp: raw sigmas are too small to use directly
    return C, S


def p_unnormalized(x, C, S):
    # sum of Gaussian RBFs: psi(r) = exp(-r^2/2), Ch. 14.3.3
    # r_i = scaled distance from x to asteroid i: ||(x - c_i) / sigma_i||
    diff = (x - C) / S
    r2   = np.sum(diff**2, axis=1)
    return np.sum(np.exp(-0.5 * r2))


def save_model(C, S, p_max,
               centers_file="existence_centers.npy",
               sigmas_file="existence_sigmas.npy",
               pmax_file="existence_pmax.npy"):
    np.save(centers_file, C)
    np.save(sigmas_file, S)
    np.save(pmax_file, np.array([p_max]))
    print("saved: existence_centers.npy, existence_sigmas.npy, existence_pmax.npy")


def load_existence_model(centers_file="existence_centers.npy",
                          sigmas_file="existence_sigmas.npy",
                          pmax_file="existence_pmax.npy"):
    # returns p as a callable: p(x) -> float in [0, 1], x = [a, e, i]
    C     = np.load(centers_file)
    S     = np.load(sigmas_file)
    p_max = float(np.load(pmax_file)[0])

    def p(x):
        return float(p_unnormalized(np.array(x, dtype=float), C, S) / p_max)

    return p


if __name__ == "__main__":
    np.random.seed(0)

    print("loading NEA centers from dataset_cleaned.csv...")
    C, S = load_centers("dataset_cleaned.csv")
    print(f"  {len(C)} NEAs loaded")
    print(f"  a: [{C[:,0].min():.3f}, {C[:,0].max():.3f}] au")
    print(f"  e: [{C[:,1].min():.3f}, {C[:,1].max():.3f}]")
    print(f"  i: [{C[:,2].min():.3f}, {C[:,2].max():.3f}] deg")

    # estimate p_max by dense sampling within realistic NEA orbital bounds
    lo      = np.array([0.5,  0.0,   0.0])
    hi      = np.array([4.2,  1.0, 180.0])
    samples = np.random.uniform(lo, hi, size=(5000, 3))
    p_vals  = np.array([p_unnormalized(s, C, S) for s in samples])
    p_max   = max(float(p_vals.max()), 1e-9)
    print(f"  p_max = {p_max:.4f}")

    save_model(C, S, p_max)

    p = load_existence_model()

    print("\nsanity check -- real NEAs (expect p close to 1):")
    for row in C[:5]:
        print(f"  [a={row[0]:.3f}, e={row[1]:.3f}, i={row[2]:.2f}]  ->  p = {p(row):.4f}")

    print("\nsanity check -- random points in NEA space (expect lower p):")
    for row in np.random.uniform(lo, hi, size=(5, 3)):
        print(f"  [a={row[0]:.3f}, e={row[1]:.3f}, i={row[2]:.2f}]  ->  p = {p(row):.4f}")