import pandas as pd
import numpy as np


# Loading dataset
df = pd.read_csv("/boredom_trap.csv")

print("=" * 65)
print("CGS616 Assignment 4 : Causal Inference Analysis")
print("Estimating P(Y = Doomscrolling | do(Content Score))")
print("=" * 65)


print("\nDataset Loaded Successfully")
print("Rows :", len(df))
print("Columns :", len(df.columns))

# Labels
content_labels = {
    0: "Low-Effort Text / Meme",
    1: "High-Effort Deepfake Video"
}

boredom_labels = {
    0: "Busy / Engaged",
    1: "Highly Bored"
}

# Population weights of confounder Z

total = len(df)

p_z0 = len(df[df["Boredom_Z"] == 0]) / total
p_z1 = len(df[df["Boredom_Z"] == 1]) / total

print("\nPopulation Distribution of Boredom (Z)")
print(f"P(Z=0) = {p_z0:.3f} --> {boredom_labels[0]}")
print(f"P(Z=1) = {p_z1:.3f} --> {boredom_labels[1]}")

print("\n" + "-" * 65)


# Function for causal estimate

def backdoor_adjustment(x):

    # Naive observational probability
    naive = df[df["Content_Score_X"] == x]["Active_Scrolling_Y"].mean()

    # Conditional probabilities
    z0_data = df[(df["Content_Score_X"] == x) & (df["Boredom_Z"] == 0)]
    z1_data = df[(df["Content_Score_X"] == x) & (df["Boredom_Z"] == 1)]

    p_y_x_z0 = z0_data["Active_Scrolling_Y"].mean()
    p_y_x_z1 = z1_data["Active_Scrolling_Y"].mean()

    # Handle empty groups
    if np.isnan(p_y_x_z0):
        p_y_x_z0 = 0

    if np.isnan(p_y_x_z1):
        p_y_x_z1 = 0

    # Backdoor formula
    causal = (p_y_x_z0 * p_z0) + (p_y_x_z1 * p_z1)

    return naive, causal, p_y_x_z0, p_y_x_z1

# Compute for X = 0 and X = 1

for x in [0, 1]:

    print(f"\nContent Score X = {x}")
    print(f"Type : {content_labels[x]}")

    naive, causal, cond0, cond1 = backdoor_adjustment(x)

    print(f"Naive P(Y=1 | X={x})            = {naive:.3f}")
    print(f"P(Y=1 | X={x}, Z=0)            = {cond0:.3f}")
    print(f"P(Y=1 | X={x}, Z=1)            = {cond1:.3f}")
    print(f"Causal P(Y=1 | do(X={x}))      = {causal:.3f}")
    print(f"Bias Removed                   = {naive - causal:+.3f}")

    print("-" * 65)


# Average Treatment Effect

_, do_x0, _, _ = backdoor_adjustment(0)
_, do_x1, _, _ = backdoor_adjustment(1)

ate = do_x1 - do_x0

print("\nAverage Treatment Effect (ATE)")
print(f"ATE = P(Y=1|do(X=1)) - P(Y=1|do(X=0))")
print(f"ATE = {do_x1:.3f} - {do_x0:.3f} = {ate:.3f}")

if ate > 0:
    print("Interpretation: High-effort content increases doomscrolling.")
elif ate < 0:
    print("Interpretation: High-effort content reduces doomscrolling.")
else:
    print("Interpretation: No causal effect found.")

print("=" * 65)
