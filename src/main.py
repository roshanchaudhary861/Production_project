from data_preprocessing import load_data, clean_data
from feature_engineering import create_activity_score, create_trend_label
from train_models import prepare_data, train_models, save_models
from evaluate_model import evaluate


def main():

    df = load_data("data/dailyActivity_merged.csv")
    df = clean_data(df)
    df = create_activity_score(df)
    df = create_trend_label(df)

    X_train, X_test, y_train, y_test = prepare_data(df)

    models = train_models(X_train, y_train)

    evaluate(models, X_test, y_test)

    save_models(models)


if __name__ == "__main__":
    main()