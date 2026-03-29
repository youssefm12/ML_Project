# Retail Customer Behavioral Analysis

## Business Context
This project focuses on analyzing customer behavior for an e-commerce gift company. The primary goals are:
- **Personalized Marketing**: Segmenting customers to tailor marketing efforts.
- **Churn Reduction**: Identifying customers at risk of leaving to improve retention.
- **Revenue Optimization**: Analyzing transaction patterns to maximize sales.

The dataset includes 52 features related to transactions, demographics, and customer interactions.

## Project Structure
```text
projet_ml_retail/
├── data/                    # Database storage
│   ├── raw/                # Original raw data (ignored by git)
│   ├── processed/          # Cleaned and prepared data
│   └── train_test/         # Split datasets for modeling
├── notebooks/              # Jupyter notebooks for prototyping
├── src/                    # Production-ready Python scripts
│   ├── preprocessing.py    # Data cleaning and feature engineering
│   ├── train_model.py      # Model training (clustering, classif, reg)
│   ├── predict.py          # Inference script
│   └── utils.py            # Shared utility functions
├── models/                 # Saved models (.pkl, .joblib)
├── app/                    # Flask-based web application
├── reports/                # Visualizations and performance reports
├── requirements.txt        # Core project dependencies
├── requirements-dev.txt    # Development and testing tools
├── README.md              # Project documentation
└── .gitignore             # Git exclusion rules
```

## Installation

### Prerequisites
- Python 3.9+
- Git

### Steps
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd projet_ml_retail
   ```

2. **Set up Virtual Environment**:
   ```bash
   # Create venv
   python -m venv .venv

   # Activate (Windows)
   .venv\Scripts\activate

   # Activate (Linux/Mac)
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   # Core dependencies
   pip install -r requirements.txt

   # Development dependencies (optional)
   pip install -r requirements-dev.txt
   ```

## Usage

### 1. Data Preprocessing
Clean and prepare the raw data for modeling.
```bash
python src/preprocessing.py
```

### 2. Model Training
Train the clustering, classification, and regression models.
```bash
python src/train_model.py
```

### 3. Inference
Make predictions on new data.
```bash
python src/predict.py
```

### 4. Exploration & Profiling

You have three ways to explore the data:

**Option 1: Quick View (Recommended)**
Open the generated HTML report for an instant overview of data quality, missing values, and distributions.
- Go to `reports/data_quality_report.html`
- Right-click -> "Open in Default Browser" or "Show Preview"

**Option 2: Interactive Notebook (VS Code)**
Run the deep-dive analysis interactively.
1. Open `notebooks/01_initial_exploration.ipynb` in VS Code.
2. Select the kernel: `.venv (Python 3.9.x)`.
3. Click "Run All" or execute cells individually.

**Option 3: Automated Profiling (Advanced)**
Generate a full profiling report using `ydata-profiling`.
```bash
# Ensure ydata-profiling is installed
pip install -r requirements-dev.txt

# Run generation script (calls generate_profiling_report)
python -c "from src.utils import load_raw_data, generate_profiling_report; df = load_raw_data('data/raw/RetailCustomers.csv'); generate_profiling_report(df)"
```
This will create `reports/profile_report.html` with correlation heatmaps and detailed feature analysis.

### 5. Web Application
Launch the Flask dashboard.
```bash
python app/main.py
```

## Contributing Guidelines
- Follow **PEP 8** style guide.
- Use `black` for formatting and `flake8` for linting.
- Ensure all new features are accompanied by tests in `tests/`.
- Use a feature-branch workflow (e.g., `feature/awesome-feature`).

## License
[Insert License Here]