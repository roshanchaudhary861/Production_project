from data_preprocessing import load_data, clean_data
from feature_engineering import create_features, create_trend_label, select_features
from train_models import prepare_data, train_models, save_models
from evaluate_model import evaluate


def main():
    # 1. Load dataset
    df = load_data("data/dailyActivity_merged.csv")

    # 2. Clean dataset
    df = clean_data(df)

    # 3. Create new features
    df = create_features(df)

    # 4. Create target variable (VERY IMPORTANT)
    df = create_trend_label(df)

    # 5. Select features and target
    X, y = select_features(df)

    # 6. Split + scale data
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # 7. Train models
    models = train_models(X_train, y_train)

    # 8. Save models
    save_models(models)

    # 9. Evaluate models
    evaluate(models, X_test, y_test, X.columns)


if __name__ == "__main__":
    main()