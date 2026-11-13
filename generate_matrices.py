import sys
import os
import random

def save_matrix(path, N):
    with open(path, "w") as f:
        f.write(f"{N}\n")
        for _ in range(N):
            row = " ".join(f"{random.uniform(0, 10.0):.17g}" for _ in range(N))
            f.write(row + "\n")

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    random.seed(42)
    save_matrix(os.path.join(outdir, "A.txt"), N)
    save_matrix(os.path.join(outdir, "B.txt"), N)

if __name__ == "__main__":
    main()