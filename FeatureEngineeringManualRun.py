import numpy as np

def getFeatures(df):
    logDmg = np.log(df['damageDealt'] + 1)
    logKills = np.log(df['kills'] + 1)
    poking = logDmg / (logKills + 1)
    logHeals = np.log(df['heals'] + 1)  # podobna charakterystyka do kills
    combatEfficiency = logDmg / (logHeals + 1)


    executing = (logKills) / (logDmg + 1)

    takedowns = df['kills'] + df['assists'] + df['DBNOs']
    bonuses = df['heals'] + df['boosts'] + df['weaponsAcquired']
    totalDistance = np.sqrt(df['walkDistance']) + np.log(df['rideDistance'] + df['swimDistance'] + 1)
    collectorScore = np.log(totalDistance * bonuses + 1)
    executorScore = np.sqrt(executing / (totalDistance + 1))
    skirmisherScore = np.log(totalDistance * takedowns + 1)
    warlordScore = np.sqrt(poking * combatEfficiency)
    adventurerScore = collectorScore * df['killStreaks']

    f1 = [executing, takedowns, bonuses, totalDistance, collectorScore, executorScore, skirmisherScore,
          warlordScore, adventurerScore]

    for ind, feat in enumerate(f1):
        df['f1_' + str(ind)] = feat

    groupedDf = df.groupby('groupId')
    teamKills = groupedDf['kills'].sum().rename('TeamKills')
    teamTakedowns = groupedDf['f1_1'].sum().rename('TeamTakedowns')
    teamAssists = groupedDf['assists'].sum().rename('TeamAssists')
    TeamWalkDistanceMean = groupedDf['walkDistance'].mean().rename('TeamWalkDistanceMean')
    TeamMeanRankPoints = groupedDf['rankPoints'].mean().rename('TeamMeanRankPoints')
    Teamwork = (teamAssists / (teamKills + 1)).rename('Teamwork')

    f2 = [teamKills, teamTakedowns, teamAssists, TeamWalkDistanceMean, TeamMeanRankPoints, Teamwork]

    for ind, feat in enumerate(f2):
        df = df.join(feat, on='groupId')

    return df