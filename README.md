# NeuroMath — Dyscalculia Risk Classification Engine

NeuroMath is a Python and Flask-based educational prototype that evaluates mathematical question performance and demonstrates machine-learning-based risk classification.

## Features

- **Question API:** Retrieves questions by category and age group.
- **Answer Evaluation:** Checks submitted answers against a question dataset.
- **Performance Metrics:** Calculates assessment scores and accuracy.
- **Machine Learning:** Demonstrates risk classification using a Gradient Boosting Classifier.
- **Category Analysis:** Summarizes performance across question categories.
- **Progress Tracking:** Records demo assessment attempts and compares scores.
- **REST API:** Exposes backend functionality through Flask endpoints.

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Gradient Boosting Classifier

## Project Structure

```text
neuromath-team/
├── app.py
├── datasets/
│   ├── questions.csv
│   ├── risk_training.csv
│   └── attempts.csv
├── README.md
└── requirements.txt
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/syedabdullah2711/neuromath-team.git
cd neuromath-team
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the Flask server

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

## API Routes

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Check whether the engine is running |
| GET | `/questions/<category>/<age_group>/<count>` | Retrieve questions |
| POST | `/submit` | Evaluate submitted answers and predict risk |
| GET | `/submit_demo` | Run a demo assessment |
| GET | `/progress_demo` | View demo assessment history |
| GET | `/submit_real_demo` | Run another predefined assessment demo |
| GET | `/category_demo` | Calculate category performance |
| GET | `/predict_demo` | Demonstrate risk prediction |

## Example: Submit Answers

Send a POST request to `/submit` with a JSON body:

```json
{
  "student": "demo_student",
  "category": "arithmetic",
  "answers": [
    {
      "question": "2+2=?",
      "selected": "4"
    },
    {
      "question": "5+3=?",
      "selected": "8"
    }
  ]
}
```

## Limitations

This project is an educational prototype, not a validated dyscalculia screening or diagnostic tool. The risk-classification model uses a very small training dataset, so its predictions are illustrative and have not been clinically validated.

## Future Improvements

- Expand and validate the training dataset.
- Add automated tests and further input validation.
- Develop a user-friendly frontend.
- Improve assessment history and progress visualizations.
- Evaluate model performance using appropriate validation methods.

## Project Status

Prototype — backend API and demonstration endpoints implemented.

## Author

Developed as an educational Python and machine-learning project.
