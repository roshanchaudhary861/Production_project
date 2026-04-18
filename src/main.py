# 🔹 Import modules
from data_preprocessing import load_data, clean_data
from feature_engineering import create_features, create_trend_label, select_features
from train_models import (
    prepare_data,
    train_models,
    save_models,
    cross_validate_models,
    plot_cv_results
)
from evaluate_model import evaluate


def main():
    print("🚀 Starting Habit Trend ML Pipeline...\n")

    # 🔹 Step 1: Load data
    df = load_data("data/dailyActivity_merged.csv")

    # 🔹 Step 2: Clean data
    df = clean_data(df)

    # 🔹 Step 3: Feature engineering
    df = create_features(df)
    df = create_trend_label(df)

    # 🔹 Step 4: Select features
    X, y, feature_names = select_features(df)

    # 🔹 Step 5: Prepare data (split + scale)
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # 🔹 Step 6: Train models
    models = train_models(X_train, y_train)

    # 🔹 Step 7: Cross-validation
    print("\n📊 Running Cross-Validation...")
    cv_results = cross_validate_models(models, X, y)
    plot_cv_results(cv_results)

    # 🔹 Step 8: Evaluate models
    print("\n📈 Evaluating Models...")
    evaluate(models, X_test, y_test, feature_names)

    # 🔹 Step 9: Save models
    save_models(models)

    print("\n✅ Pipeline completed successfully!")


# 🔹 Run program
if __name__ == "__main__":
    main()