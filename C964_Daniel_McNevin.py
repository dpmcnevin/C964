import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    from IPython.display import display, clear_output
    from sklearn.calibration import CalibrationDisplay
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import PrecisionRecallDisplay
    from sklearn.metrics import RocCurveDisplay
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return mo, np, pd, px


@app.cell
def _():
    HOME_DIR = "."

    RETROSHEET_DIR = f"{HOME_DIR}/retrosheet"
    CACHE_DIR = f"{HOME_DIR}/cached"
    DAILY_FILES = f"{HOME_DIR}/retrosplits/daybyday"

    START_YEAR = 1980
    END_YEAR = 2025
    return DAILY_FILES, END_YEAR, RETROSHEET_DIR, START_YEAR


@app.cell
def _(mo):
    BATTING_STATS = [
        'ba',
        'slg',
        'obp',
        'ops',
        'run_diff',
    ]

    BATTING_PERIODS = [
        5,
        20,
        100,
        162,
    ]

    ALL_BATTING_STATS = []
    for stat in BATTING_STATS:
        # first base
        ALL_BATTING_STATS.append(f"team_{stat}")
        # then each period in order
        for period in BATTING_PERIODS:
            ALL_BATTING_STATS.append(f"team_{stat}_{period}")

    BATTING_STATS_DROPDOWN = mo.ui.dropdown(options=ALL_BATTING_STATS, value=ALL_BATTING_STATS[0], searchable=True)
    return ALL_BATTING_STATS, BATTING_PERIODS, BATTING_STATS


@app.cell
def _():
    PITCHING_STATS = [
        'P_ERA',
        'P_WHIP'
    ]

    MERGED_PITCHING_STATS = [
        'era',
        'whip'
    ]

    PITCHING_PERIODS = [
        5,
        20,
    ]

    PITCHING_STAT_MAPPER = {
        'P_ERA_5': "starting_pitcher_era_5",
        'P_ERA_20': "starting_pitcher_era_20",
        'P_WHIP_5': "starting_pitcher_whip_5",
        'P_WHIP_20': "starting_pitcher_whip_20",
    }
    return PITCHING_PERIODS, PITCHING_STATS


@app.cell
def _(RETROSHEET_DIR, pd):
    TEAMS_DF = pd.read_csv(f"{RETROSHEET_DIR}/reference/teams.csv.zip")

    TEAMS = TEAMS_DF[TEAMS_DF['LAST'] == 2024].dropna()
    TEAMS.set_index('TEAM', inplace=True)

    TEAMS_LIST = {code: f"{city} {nickname}" 
                  for code, city, nickname in zip(TEAMS.index, TEAMS['CITY'], TEAMS['NICKNAME'])}

    TEAMS_LIST_REVERSED = {v: k for k, v in TEAMS_LIST.items()}
    return TEAMS, TEAMS_LIST_REVERSED


@app.cell
def _(TEAMS):
    def get_team(team_code):
        return TEAMS.loc[TEAMS.index == team_code]

    def get_team_name(team_code):
        return get_team(team_code).apply(lambda tt: f"{tt['CITY']} {tt['NICKNAME']}", axis=1).iloc[0]
    return


@app.cell
def _(RETROSHEET_DIR, pd):
    PLAYERS_DF = pd.read_csv(f"{RETROSHEET_DIR}/reference/biofile.csv.zip")
    PLAYERS_DF.set_index('PLAYERID', inplace=True)

    def get_player_name(player_id):
        row = PLAYERS_DF.loc[player_id][["NICKNAME", "LAST"]]
        return f"{row['NICKNAME']} {row['LAST']}"
    return


@app.cell
def _(END_YEAR, RETROSHEET_DIR, START_YEAR, pd):
    def read_all_seasons(start_year, end_year):
        all_seasons = []

        print("Reading Seasons ", end="")

        for season_year in range(start_year, end_year):
            print(f"{season_year} ", end="")
            _season = pd.read_csv(f"{RETROSHEET_DIR}/seasons/{season_year}/GL{season_year}.TXT.zip", header=None)
            _season['season'] = season_year
            all_seasons.append(_season)

        return pd.concat(all_seasons, axis=0, ignore_index=True)

    ALL_SEASONS_DF = read_all_seasons(START_YEAR, END_YEAR)
    return (ALL_SEASONS_DF,)


@app.cell
def _():
    GAMELOG_COLUMNS = [
        'date', 'game_num', 'day_of_week', 'visiting_team',
        'visiting_team_league', 'visiting_team_game_num', 'home_team',
        'home_team_league', 'home_team_game_num', 'visiting_score',
        'home_score', 'num_outs', 'day_night', 'completion_info',
        'forfeit_info', 'protest_info', 'park_id', 'attendance',
        'time_of_game_minutes', 'visiting_line_score',
        'home_line_score', 'visiting_abs', 'visiting_hits',
        'visiting_doubles', 'visiting_triples', 'visiting_homeruns',
        'visiting_rbi', 'visiting_sac_hits', 'visiting_sac_flies',
        'visiting_hbp', 'visiting_bb', 'visiting_iw', 'visiting_k',
        'visiting_sb', 'visiting_cs', 'visiting_gdp', 'visiting_ci',
        'visiting_lob', 'visiting_pitchers_used',
        'visiting_individual_er', 'visiting_er', 'visiting_wp',
        'visiting_balks', 'visiting_po', 'visiting_assists',
        'visiting_errors', 'visiting_pb', 'visiting_dp',
        'visiting_tp', 'home_abs', 'home_hits', 'home_doubles',
        'home_triples', 'home_homeruns', 'home_rbi',
        'home_sac_hits', 'home_sac_flies', 'home_hbp', 'home_bb',
        'home_iw', 'home_k', 'home_sb', 'home_cs', 'home_gdp',
        'home_ci', 'home_lob', 'home_pitchers_used',
        'home_individual_er', 'home_er', 'home_wp', 'home_balks',
        'home_po', 'home_assists', 'home_errors', 'home_pb',
        'home_dp', 'home_tp', 'ump_home_id', 'ump_home_name',
        'ump_first_id', 'ump_first_name', 'ump_second_id',
        'ump_second_name', 'ump_third_id', 'ump_third_name',
        'ump_lf_id', 'ump_lf_name', 'ump_rf_id', 'ump_rf_name',
        'visiting_manager_id', 'visiting_manager_name',
        'home_manager_id', 'home_manager_name',
        'winning_pitcher_id', 'winning_pitcher_name',
        'losing_pitcher_id', 'losing_pitcher_name',
        'save_pitcher_id', 'save_pitcher_name',
        'game_winning_rbi_id', 'game_winning_rbi_name',
        'visiting_starting_pitcher_id', 'visiting_starting_pitcher_name',
        'home_starting_pitcher_id', 'home_starting_pitcher_name',
        'visiting_1_id', 'visiting_1_name', 'visiting_1_pos',
        'visiting_2_id', 'visiting_2_name', 'visiting_2_pos',
        'visiting_3_id', 'visiting_3_name', 'visiting_3_pos',
        'visiting_4_id', 'visiting_4_name', 'visiting_4_pos',
        'visiting_5_id', 'visiting_5_name', 'visiting_5_pos',
        'visiting_6_id', 'visiting_6_name', 'visiting_6_pos',
        'visiting_7_id', 'visiting_7_name', 'visiting_7_pos',
        'visiting_8_id', 'visiting_8_name', 'visiting_8_pos',
        'visiting_9_id', 'visiting_9_name', 'visiting_9_pos',
        'home_1_id', 'home_1_name', 'home_1_pos',
        'home_2_id', 'home_2_name', 'home_2_pos',
        'home_3_id', 'home_3_name', 'home_3_pos',
        'home_4_id', 'home_4_name', 'home_4_pos',
        'home_5_id', 'home_5_name', 'home_5_pos',
        'home_6_id', 'home_6_name', 'home_6_pos',
        'home_7_id', 'home_7_name', 'home_7_pos',
        'home_8_id', 'home_8_name', 'home_8_pos',
        'home_9_id', 'home_9_name', 'home_9_pos',
        'misc', 'acquisition_info'
    ]
    return (GAMELOG_COLUMNS,)


@app.cell
def _(ALL_SEASONS_DF, GAMELOG_COLUMNS, pd):
    ## Add Columns to dataframe
    ALL_SEASONS_DF.columns = [*GAMELOG_COLUMNS, *['season']]
    ALL_SEASONS_DF['datetime'] = pd.to_datetime(ALL_SEASONS_DF['date'], format='%Y%m%d')
    ALL_SEASONS_DF['game_id'] = (ALL_SEASONS_DF["date"].astype(str)
                                 + "_"
                                 + ALL_SEASONS_DF["game_num"].astype(str)
                                 + "_" + ALL_SEASONS_DF["home_team"]
                                 + "_" + ALL_SEASONS_DF["visiting_team"])
    return


@app.cell
def _(ALL_SEASONS_DF):
    ties = ALL_SEASONS_DF[ALL_SEASONS_DF['home_score'] == ALL_SEASONS_DF['visiting_score']]
    (ties.shape[0] / ALL_SEASONS_DF.shape[0]) * 100
    return


@app.cell
def _(ALL_SEASONS_DF):
    indexes_to_drop = ALL_SEASONS_DF.index[ALL_SEASONS_DF['home_score'] == ALL_SEASONS_DF['visiting_score']]
    ALL_SEASONS_DF.drop(indexes_to_drop, inplace=True)
    return


@app.cell
def _(ALL_SEASONS_DF, np):
    ALL_SEASONS_DF['home_win'] = np.where(ALL_SEASONS_DF['home_score'] > ALL_SEASONS_DF['visiting_score'], 1, 0)
    ALL_SEASONS_DF['visiting_win'] = np.where(ALL_SEASONS_DF['home_score'] > ALL_SEASONS_DF['visiting_score'], 0, 1)
    return


@app.cell
def _(ALL_SEASONS_DF):
    def calculate_batting_stats(dataframe):
        for location in ['home', 'visiting']:
            ## Batting average
            dataframe[f"{location}_ba"] = dataframe[f"{location}_hits"] / dataframe[f"{location}_abs"]

            ## Get the singles
            dataframe[f"{location}_singles"] = (
                dataframe[f"{location}_hits"] 
                - dataframe[f"{location}_doubles"] 
                - dataframe[f"{location}_triples"] 
                - dataframe[f"{location}_homeruns"]
            )

            ## Calculate the slugging percentage
            ##  Total Bases = (1B) + (2×2B) + (3×3B) + (4×HR)
            ##  Slugging = Total Bases / At Bats
            dataframe[f"{location}_slg"] = (
                dataframe[f"{location}_singles"] 
                + 2 * dataframe[f"{location}_doubles"] 
                + 3 * dataframe[f"{location}_triples"] 
                + 4 * dataframe[f"{location}_homeruns"]
            ) / dataframe[f"{location}_abs"]

            ## Calculate On-base Percentage (OBP)
            ##  (Hits + Walks + Hit-by-Pitch) / (At Bats + Walks + Hit-by-Pitch + Sacrifice Flies)
            dataframe[f"{location}_obp"] = (
                dataframe[f"{location}_hits"] 
                + dataframe[f"{location}_bb"] 
                + dataframe[f"{location}_hbp"]
            ) / (
                dataframe[f"{location}_abs"] 
                + dataframe[f"{location}_bb"] 
                + dataframe[f"{location}_hbp"] 
                + dataframe[f"{location}_sac_flies"]
            )

            ## OPS
            dataframe[f"{location}_ops"] = dataframe[f"{location}_obp"] + dataframe[f"{location}_slg"]

        return dataframe

    calculate_batting_stats(ALL_SEASONS_DF)
    return


@app.cell
def _(pd):
    def team_games(team_games_dataframe, team_code):
        team_games_dataframe = team_games_dataframe.copy()

        # Home games (team is home)
        home = team_games_dataframe[team_games_dataframe['home_team'] == team_code].copy()
        home['is_home_game'] = True

        # Rename columns
        home = home.rename(columns=lambda col: f"team_{col.removeprefix('home_')}" if col.startswith('home_') else col)
        home = home.rename(
            columns=lambda col: f"opponent_{col.removeprefix('visiting_')}" if col.startswith('visiting_') else col)

        # Away games (team is visitor)
        away = team_games_dataframe[team_games_dataframe['visiting_team'] == team_code].copy()
        away['is_home_game'] = False

        # Rename columns
        away = away.rename(
            columns=lambda col: f"team_{col.removeprefix('visiting_')}" if col.startswith('visiting_') else col)
        away = away.rename(columns=lambda col: f"opponent_{col.removeprefix('home_')}" if col.startswith('home_') else col)

        # Combine both
        combined = pd.concat([home, away], ignore_index=True).sort_values(by=['date', 'game_num']).copy()
        combined['team_run_diff'] = combined['team_score'] - combined['opponent_score']

        return combined
    return (team_games,)


@app.cell
def _(ALL_SEASONS_DF, TEAMS, team_games):
    def extract_team_dictionary(dataframe):
        all_teams = {}

        print("Loading Teams ", end="")

        for team in TEAMS.index:
            print(f"{team} ", end="")
            all_teams[team] = team_games(dataframe, team)

        return all_teams

    TEAM_DATA = extract_team_dictionary(ALL_SEASONS_DF)
    return (TEAM_DATA,)


@app.cell
def _(BATTING_PERIODS, BATTING_STATS, TEAMS, TEAM_DATA, pd):
    def calculate_team_batting_rolling_averages(team, team_df):

        team_df = team_df.copy()

        print(f"{team} ", end="")
        new_cols = {}

        for stat in BATTING_STATS:
            for period in BATTING_PERIODS:
                col_name = f"team_{stat}_{period}"

                ## Shift 1 so we don't include the current game
                new_cols[col_name] = team_df[f"team_{stat}"].shift(1).rolling(window=period, min_periods=1).mean()

        # Add all new columns at once
        new_cols_df = pd.DataFrame(new_cols, index=team_df.index)
        return pd.concat([team_df, new_cols_df], axis=1).drop_duplicates()

    print("Processing Team: ", end="")
    for team in TEAMS.index:
        TEAM_DATA[team] = calculate_team_batting_rolling_averages(team, TEAM_DATA[team])
    return


@app.cell
def _(TEAM_DATA, pd):
    TEAM_DATA_DF = pd.concat(list(TEAM_DATA.values())).sort_values(by='datetime')
    return (TEAM_DATA_DF,)


@app.cell
def _(ALL_BATTING_STATS, TEAMS_LIST_REVERSED, mo):
    teams_dropdown = mo.ui.dropdown(options=TEAMS_LIST_REVERSED, value=list(TEAMS_LIST_REVERSED.keys())[0], searchable=True)
    batting_stats_dropdown = mo.ui.dropdown(options=ALL_BATTING_STATS, value=ALL_BATTING_STATS[0], searchable=True)
    return batting_stats_dropdown, teams_dropdown


@app.cell
def _(TEAM_DATA, batting_stats_dropdown, mo, px, teams_dropdown):
    _df = TEAM_DATA[teams_dropdown.value].groupby('season')[batting_stats_dropdown.value].mean().reset_index()
    _plot = px.line(_df, x='season', y=batting_stats_dropdown.value, title=f"Mean {batting_stats_dropdown.value} Per Season for {batting_stats_dropdown.value}")

    mo.vstack([
        mo.hstack([
            teams_dropdown,
            batting_stats_dropdown,
        ], justify='start'),
        mo.ui.plotly(_plot)
    ])
    return


@app.cell
def _(TEAMS_LIST_REVERSED, mo):
    teams_multi_dropdown = mo.ui.multiselect(options=TEAMS_LIST_REVERSED, value=list(TEAMS_LIST_REVERSED.keys())[:1], max_selections=5)
    return (teams_multi_dropdown,)


@app.cell
def _(TEAM_DATA_DF, batting_stats_dropdown, mo, px, teams_multi_dropdown):
    _fig = px.line(
        TEAM_DATA_DF[TEAM_DATA_DF['team_team'].isin(teams_multi_dropdown.value)],
        x='datetime',
        y=batting_stats_dropdown.value,
        color='team_team',
        markers=False
    )

    mo.vstack([
        mo.hstack([
            teams_multi_dropdown,
            batting_stats_dropdown,
        ], justify='start'),
        mo.ui.plotly(_fig)
    ])
    return


@app.cell
def _(ALL_SEASONS_DF, BATTING_PERIODS, BATTING_STATS, TEAMS, TEAM_DATA):
    def merge_team_rolled_stats(all_seasons_df, teams_dict):
    
        merged_df = all_seasons_df.copy()
        merged_df.set_index('game_id', inplace=True)
    
        fields_to_merge = [
            'game_id',
            'team_team',
            'is_home_game',
            *[f"team_{stat}_{period}" for stat in BATTING_STATS for period in BATTING_PERIODS]
        ]
    
        for loc in ['home', 'visiting']:
    
            ## Make sure the new fields are present to update
            for field in [f"{loc}_{stat}_{period}" for stat in BATTING_STATS for period in BATTING_PERIODS]:
                merged_df[field] = None
    
            rename_map = {
                f"team_{stat}_{period}": f"{loc}_{stat}_{period}"
                for stat in BATTING_STATS
                for period in BATTING_PERIODS
            }
    
            for _team in TEAMS.index:
                team_df = teams_dict[_team][fields_to_merge].copy()
    
                is_home_game = (loc == 'home')
                team_df = team_df.query('is_home_game == @is_home_game').copy()
                team_df.rename(columns=rename_map, inplace=True)
                team_df.set_index('game_id', inplace=True)
                merged_df.update(team_df)
            
        return merged_df

    MERGED_DF = merge_team_rolled_stats(ALL_SEASONS_DF, TEAM_DATA)
    MERGED_DF.reset_index(inplace=True)
    return


@app.cell
def _(DAILY_FILES, END_YEAR, START_YEAR, pd):
    def get_pitching_dataframe():
        _dfs = []

        print("Getting Pitching Data: ", end="")
        ## Read data from 1980-2024
        for year in range(START_YEAR, END_YEAR):
            print(f"{year} ", end="")
            _season = pd.read_csv(f"{DAILY_FILES}/playing-{year}.csv.zip")
            _season['season'] = year
            _season['game.datetime'] = pd.to_datetime(_season['game.date'], format='%Y-%m-%d')
    
            ## Filter for only pitchers and remove batting and fielding data
            year_pitching_data = _season[
                                     ## Events
                                     (_season['game.source'] == 'evt')
                                     ## Regular Season
                                     & (_season['season.phase'] == 'R')
                                     ## Total batters faced is more than 0
                                     & (_season['P_TBF'] > 0)
                                     ].loc[:,~(_season.columns.str.startswith('B_') | _season.columns.str.startswith('F_'))].copy()
        
            _dfs.append(year_pitching_data)
    
        return pd.concat(_dfs, axis=0, ignore_index=True)

    PITCHING_DF = get_pitching_dataframe().sort_values(by=['person.key', 'game.datetime'])
    return (PITCHING_DF,)


@app.cell
def _(PITCHING_DF):
    ## Calculate ERA, WHIP
    PITCHING_DF['P_IP'] = PITCHING_DF['P_OUT'] / 3.0
    PITCHING_DF['P_ERA'] = (PITCHING_DF['P_ER'] * 9) / PITCHING_DF['P_IP']
    PITCHING_DF['P_WHIP'] = (PITCHING_DF['P_BB'] + PITCHING_DF['P_H']) / PITCHING_DF['P_IP']
    return


@app.cell
def _(PITCHING_DF, PITCHING_PERIODS, PITCHING_STATS):
    for _period in PITCHING_PERIODS:
        for _stat in PITCHING_STATS:
            PITCHING_DF[f"{_stat}_{_period}"] = (
                PITCHING_DF
                .groupby('person.key')[_stat]
                .transform(lambda s: s.shift(1).rolling(window=_period, min_periods=1).mean())
            )
        
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
