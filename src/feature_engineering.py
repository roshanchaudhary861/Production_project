def create_activity_score(df):
    df['ActivityScore'] = (
        df['VeryActiveMinutes'] * 2 +
        df['FairlyActiveMinutes'] * 1.5 +
        df['LightlyActiveMinutes'] * 1
    )
    return df


def create_trend_label(df):
    df = df.sort_values(['Id', 'ActivityDate'])

    df['ActivityTrend'] = df.groupby('Id')['ActivityScore'].diff()

    df['HabitLabel'] = df['ActivityTrend'].apply(
        lambda x: 'Improving' if x > 0 
        else ('Declining' if x < 0 else 'Stable')
    )

    df = df.dropna()
    return df