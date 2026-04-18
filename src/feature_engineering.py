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


# 🔥 Create target variable (Trend)
def create_trend_label(df):
    def label(row):
        if row['TotalSteps'] < 5000:
            return "Declining"
        elif row['TotalSteps'] < 10000:
            return "Stable"
        else:
            return "Improving"

    df['Trend'] = df.apply(label, axis=1)

    return df


def select_features(df):
    # Input features
    features = [
        'TotalSteps',
        'Calories',
        'VeryActiveMinutes',
        'SedentaryMinutes', 
        'activity_ratio'
    ]

    X = df[features]

    # Target variable
    y = df['Trend']

    return X, y