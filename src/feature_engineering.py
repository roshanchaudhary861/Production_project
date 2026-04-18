import pandas as pd


# 🔹 Create new features
def create_features(df):
    # Total activity minutes
    df['total_activity'] = (
        df['VeryActiveMinutes'] +
        df['FairlyActiveMinutes'] +
        df['LightlyActiveMinutes']
    )

    # Activity ratio (avoid divide by zero)
    df['activity_ratio'] = df['VeryActiveMinutes'] / (df['SedentaryMinutes'] + 1)

    return df


# 🔹 Create target variable (Trend) WITHOUT DATA LEAKAGE
def create_trend_label(df):
    # Convert date to datetime (important)
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])

    # Sort by user and date
    df = df.sort_values(by=['Id', 'ActivityDate'])

    # Next day steps
    df['next_steps'] = df.groupby('Id')['TotalSteps'].shift(-1)

    # Step difference (future - current)
    df['step_change'] = df['next_steps'] - df['TotalSteps']

    # Create Trend label
    def label_trend(x):
        if x > 1000:
            return 'Improving'
        elif x < -1000:
            return 'Declining'
        else:
            return 'Stable'

    df['Trend'] = df['step_change'].apply(label_trend)

    # Drop rows where next_steps is NaN (last day per user)
    df = df.dropna()

    return df


# 🔹 Select features and target
def select_features(df):
    features = [
        'TotalSteps',
        'Calories',
        'VeryActiveMinutes',
        'SedentaryMinutes',
        'activity_ratio'
    ]

    X = df[features]
    y = df['Trend']

    return X, y, features