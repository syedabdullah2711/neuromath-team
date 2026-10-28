
from pathlib import Path

from flask import Flask, request, jsonify
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

app = Flask(__name__)

# File paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"

QUESTIONS_FILE = DATA_DIR / "questions.csv"
RISK_FILE = DATA_DIR / "risk_training.csv"
ATTEMPTS_FILE = DATA_DIR / "attempts.csv"

# Load datasets
df = pd.read_csv(QUESTIONS_FILE)
risk_df = pd.read_csv(RISK_FILE)

# Train the risk prediction model ONCE
FEATURES = ["accuracy", "category_score"]

X_train = risk_df[FEATURES]
y_train = risk_df["risk"]

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

RISK_MAP = {
    0: "Low Risk",
    1: "At Risk",
    2: "High Risk"
}


def load_attempts():
    """Load saved attempts or create an empty dataset."""
    if ATTEMPTS_FILE.exists():
        return pd.read_csv(ATTEMPTS_FILE)

    return pd.DataFrame(
        columns=["student", "category", "accuracy"]
    )


def save_attempt(student, category, accuracy):
    """Save one assessment attempt."""
    attempts = load_attempts()

    new_attempt = pd.DataFrame([{
        "student": student,
        "category": category,
        "accuracy": accuracy
    }])

    attempts = pd.concat(
        [attempts, new_attempt],
        ignore_index=True
    )

    attempts.to_csv(ATTEMPTS_FILE, index=False)


def calculate_risk(accuracy, category_score):
    """Predict risk using the trained model."""
    prediction = model.predict([[
        accuracy,
        category_score
    ]])[0]

    return RISK_MAP.get(
        int(prediction),
        "Unknown"
    )


# Home route
@app.route("/")
def home():
    return jsonify({
        "message": "Dyscalculia Engine Running"
    })


# Retrieve questions
@app.route("/questions/<category>/<age_group>/<int:count>")
def get_questions(category, age_group, count):

    if count < 1:
        return jsonify({"error": "Count must be positive"}), 400

    filtered = df[
        (df["category"] == category) &
        (df["age_group"] == age_group)
    ]

    if filtered.empty:
        return jsonify({
            "error": "No questions found for this category and age group"
        }), 404

    sample = filtered.sample(
        n=min(count, len(filtered)),
        random_state=None
    )

    student_view = sample.drop(
        columns=["answer"]
    )

    return jsonify(
        student_view.to_dict(orient="records")
    )


# Submit answers from a client
@app.route("/submit", methods=["POST"])
def submit():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object"}), 400

    answers = data.get("answers")

    if not isinstance(answers, list) or not answers:
        return jsonify({
            "error": "Answers must be a non-empty list"
        }), 400

    score = 0
    total = 0
    category_correct = {}
    category_total = {}

    for item in answers:
        if not isinstance(item, dict):
            continue

        # Questions are identified by their question text
        question = item.get("question")

        # Also accept question_id if it contains question text
        if question is None:
            question = item.get("question_id")

        selected = item.get("selected")

        if question is None or selected is None:
            continue

        matching = df[df["question"] == str(question)]

        if matching.empty:
            continue

        row = matching.iloc[0]
        category = str(row["category"])
        correct_answer = str(row["answer"]).strip()

        total += 1
        category_total[category] = (
            category_total.get(category, 0) + 1
        )

        if str(selected).strip() == correct_answer:
            score += 1
            category_correct[category] = (
                category_correct.get(category, 0) + 1
            )

    if total == 0:
        return jsonify({
            "error": "No valid answers matched the question dataset"
        }), 400

    accuracy = (score / total) * 100

    category_accuracies = [
        (category_correct.get(category, 0) /
         category_total[category]) * 100
        for category in category_total
    ]

    category_score = (
        sum(category_accuracies) / len(category_accuracies)
    )

    risk = calculate_risk(accuracy, category_score)

    student = str(data.get("student", "demo_student"))
    category_name = str(data.get("category", "mixed"))

    save_attempt(student, category_name, accuracy)

    return jsonify({
        "score": score,
        "total": total,
        "accuracy": round(accuracy, 2),
        "category_score": round(category_score, 2),
        "risk": risk
    })


# Simple demonstration assessment
@app.route("/submit_demo")
def submit_demo():

    answers = [
        {"question": "2+2=?", "selected": "4"},
        {"question": "5+3=?", "selected": "7"}
    ]

    score = 0

    for item in answers:
        correct = df.loc[
            df["question"] == item["question"],
            "answer"
        ]

        if not correct.empty:
            if str(item["selected"]) == str(correct.iloc[0]):
                score += 1

    total = len(answers)
    accuracy = (score / total) * 100

    save_attempt("demo_student", "arithmetic", accuracy)

    return jsonify({
        "score": score,
        "total": total,
        "accuracy": accuracy
    })


# View progress across demonstration attempts
@app.route("/progress_demo")
def progress_demo():

    attempts = load_attempts()

    if not {"student", "accuracy"}.issubset(attempts.columns):
        return jsonify({
            "error": "Attempts dataset has an unexpected format"
        }), 500

    student_attempts = attempts[
        attempts["student"] == "demo_student"
    ]

    scores = student_attempts["accuracy"].tolist()

    if len(scores) < 2:
        status = "Need More Attempts"
    elif scores[-1] > scores[-2]:
        status = "Improving"
    elif scores[-1] == scores[-2]:
        status = "Stable"
    else:
        status = "Needs Help"

    return jsonify({
        "attempts": scores,
        "status": status
    })


# Demonstration using predefined answers
@app.route("/submit_real_demo")
def submit_real_demo():

    answers = [
        {"question": "2+2=?", "selected": "4"},
        {"question": "5+3=?", "selected": "8"},
        {"question": "7-2=?", "selected": "4"}
    ]

    score = 0

    for item in answers:
        correct = df.loc[
            df["question"] == item["question"],
            "answer"
        ]

        if not correct.empty:
            if str(item["selected"]) == str(correct.iloc[0]):
                score += 1

    total = len(answers)
    accuracy = (score / total) * 100

    save_attempt("demo_student", "mixed", accuracy)

    risk = calculate_risk(accuracy, accuracy)

    return jsonify({
        "score": score,
        "total": total,
        "accuracy": round(accuracy, 2),
        "risk": risk
    })


# Calculate performance by category
@app.route("/category_demo")
def category_demo():

    answers = [
        {"question": "2+2=?", "selected": "4"},
        {"question": "5+3=?", "selected": "7"},
        {"question": "2 4 6 __", "selected": "8"}
    ]

    category_scores = {}

    for item in answers:
        matching = df[df["question"] == item["question"]]

        if matching.empty:
            continue

        row = matching.iloc[0]
        category = str(row["category"])

        if category not in category_scores:
            category_scores[category] = {
                "correct": 0,
                "total": 0
            }

        category_scores[category]["total"] += 1

        if str(item["selected"]) == str(row["answer"]):
            category_scores[category]["correct"] += 1

    performance = {}

    for category, result in category_scores.items():
        performance[category] = round(
            result["correct"] / result["total"] * 100,
            2
        )

    return jsonify(performance)


# Demonstration of risk prediction
@app.route("/predict_demo")
def predict_demo():

    accuracy = 65
    category_score = 60

    risk = calculate_risk(accuracy, category_score)

    return jsonify({
        "accuracy": accuracy,
        "category_score": category_score,
        "risk": risk
    })


if __name__ == "__main__":
    app.run(debug=True)