import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)


# -------------------------------------------------------
# paths - built relative to this script so it runs from
# anywhere without breaking
# -------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "../data/students.csv")
VISUALS_DIR = os.path.join(BASE_DIR, "../visuals")

os.makedirs(VISUALS_DIR, exist_ok=True)


# -------------------------------------------------------
# load the data
# dataset: kaggle.com/datasets/spscientist/students-performance-in-exams
# download and save as data/students.csv before running
# -------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print(df.shape)
print(df.head())
print(df.dtypes)
print(df.isnull().sum())


# -------------------------------------------------------
# clean up column names
# -------------------------------------------------------

# strip spaces, lowercase, replace spaces and slashes with underscores
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("/", "_")
)

print("\ncleaned columns:", df.columns.tolist())

# average score across the three subjects
df["average_score"] = (
    df["math_score"] + df["reading_score"] + df["writing_score"]
) / 3
df["average_score"] = df["average_score"].round(2)

# 60+ = pass
df["result"] = df["average_score"].apply(lambda x: "Pass" if x >= 60 else "Fail")

print(df[["math_score", "reading_score", "writing_score", "average_score", "result"]].head(10))


# -------------------------------------------------------
# exploratory analysis
# -------------------------------------------------------

def save_fig(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, filename), dpi=150)
    plt.show()


# score distributions
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Score Distributions by Subject", fontsize=15, fontweight="bold")

subjects = ["math_score", "reading_score", "writing_score"]
colors   = ["#4C72B0", "#DD8452", "#55A868"]

for ax, col, color in zip(axes, subjects, colors):
    sns.histplot(df[col], ax=ax, color=color, kde=True, bins=20)
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("Score")

save_fig("score_distributions.png")


# gender vs average score
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="gender", y="average_score", palette="Set2")
plt.title("Average Score by Gender", fontsize=13, fontweight="bold")
plt.xlabel("Gender")
plt.ylabel("Average Score")
save_fig("score_by_gender.png")


# parental education - sorted highest to lowest for readability
edu_order = (
    df.groupby("parental_level_of_education")["average_score"]
    .mean()
    .sort_values(ascending=False)
    .index
)

plt.figure(figsize=(10, 5))
sns.barplot(data=df,
            x="parental_level_of_education",
            y="average_score",
            order=edu_order,
            palette="Blues_d")
plt.title("Average Score by Parental Education Level", fontsize=13, fontweight="bold")
plt.xlabel("Parental Education")
plt.ylabel("Average Score")
plt.xticks(rotation=30, ha="right")
save_fig("score_by_parent_education.png")


# test prep course impact
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="test_preparation_course", y="average_score", palette="pastel")
plt.title("Impact of Test Preparation Course on Score", fontsize=13, fontweight="bold")
plt.xlabel("Test Preparation Course")
plt.ylabel("Average Score")
save_fig("score_by_test_prep.png")


# correlation heatmap
corr = df[["math_score", "reading_score", "writing_score", "average_score"]].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, linewidths=0.5)
plt.title("Correlation Between Scores", fontsize=13, fontweight="bold")
save_fig("correlation_heatmap.png")


# -------------------------------------------------------
# quick summary
# -------------------------------------------------------

prep_avg = df.groupby("test_preparation_course")["average_score"].mean()
diff     = prep_avg["completed"] - prep_avg["none"]
print(f"\ntest prep effect: +{diff:.1f} pts")
print(f"  completed → {prep_avg['completed']:.1f}  |  none → {prep_avg['none']:.1f}")

top_edu   = df.groupby("parental_level_of_education")["average_score"].mean().idxmax()
top_score = df.groupby("parental_level_of_education")["average_score"].mean().max()
print(f"\nhighest scoring parental education group: '{top_edu}' ({top_score:.1f})")

gender_avg = df.groupby("gender")["average_score"].mean()
print("\ngender averages:")
for g, s in gender_avg.items():
    print(f"  {g.title()}: {s:.1f}")

pass_rate = (df["result"] == "Pass").mean() * 100
print(f"\noverall pass rate: {pass_rate:.1f}%")

# note: high pass rate means class imbalance — keep that in mind
# when reading the model accuracy below


# -------------------------------------------------------
# logistic regression - predict pass/fail
# using only demographic features, no score columns
# -------------------------------------------------------

df_ml = df.copy()

# after column cleaning, race/ethnicity is now race_ethnicity
cat_cols = ["gender", "race_ethnicity", "parental_level_of_education",
            "lunch", "test_preparation_course"]
cat_cols = [c for c in cat_cols if c in df_ml.columns]

df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)

# exclude all score-related columns so the model only uses demographics
drop_cols = ["result", "average_score", "math_score", "reading_score", "writing_score"]
features  = [c for c in df_ml.columns if c not in drop_cols]

X = df_ml[features]
y = (df_ml["result"] == "Pass").astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\nmodel accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"])
plt.title("Confusion Matrix", fontsize=13, fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
save_fig("confusion_matrix.png")