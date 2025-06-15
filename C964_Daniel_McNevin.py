import marimo

__generated_with = "0.13.15"
app = marimo.App(
    width="medium",
    layout_file="layouts/C964_Daniel_McNevin.grid.json",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # WGU C964 Task 2

    *Name*: Daniel McNevin

    *Email*: dmcnev2@wgu.edu

    ```
     The information used here was obtained free of
     charge from and is copyrighted by Retrosheet.  Interested
     parties may contact Retrosheet at "www.retrosheet.org".
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Global Setup""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Global Imports

    Install packages and import needed libraries
    """
    )
    return


@app.cell
def _():
    import io

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    import tabulate
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
    return (
        CalibrationDisplay,
        ConfusionMatrixDisplay,
        LogisticRegression,
        PrecisionRecallDisplay,
        RocCurveDisplay,
        StandardScaler,
        classification_report,
        go,
        io,
        make_pipeline,
        mo,
        np,
        pd,
        plt,
        px,
        sns,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ### General Setup

    Set up global variables
    """
    )
    return


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
    for _stat in BATTING_STATS:
        ALL_BATTING_STATS.append(f"team_{_stat}")
        for _period in BATTING_PERIODS:
            ALL_BATTING_STATS.append(f"team_{_stat}_{_period}")

    BATTING_STATS_DROPDOWN = mo.ui.dropdown(options=ALL_BATTING_STATS, value=ALL_BATTING_STATS[0], searchable=True)
    return ALL_BATTING_STATS, BATTING_PERIODS, BATTING_STATS


@app.cell
def _(mo):
    mo.md(r"""#### Pitching Stats""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data Setup""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data Import""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Team Import

    Get a list of all teams that are all active in 2024

    This will set a `TEAMS` global variable that has a DataFrame of all teams
    """
    )
    return


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

    ALL_PITCHING_STATS = []
    for _stat in PITCHING_STATS:
        ALL_PITCHING_STATS.append(f"{_stat}")
        for _period in PITCHING_PERIODS:
            ALL_PITCHING_STATS.append(f"{_stat}_{_period}")
    return (
        ALL_PITCHING_STATS,
        MERGED_PITCHING_STATS,
        PITCHING_PERIODS,
        PITCHING_STATS,
        PITCHING_STAT_MAPPER,
    )


@app.cell
def _(RETROSHEET_DIR, pd):
    TEAMS_DF = pd.read_csv(f"{RETROSHEET_DIR}/reference/teams.csv.zip")

    TEAMS = TEAMS_DF[TEAMS_DF['LAST'] == 2024].dropna()
    TEAMS.set_index('TEAM', inplace=True)

    TEAMS_LIST = {code: f"{city} {nickname}" 
                  for code, city, nickname in zip(TEAMS.index, TEAMS['CITY'], TEAMS['NICKNAME'])}

    TEAMS_LIST_REVERSED = {v: k for k, v in TEAMS_LIST.items()}
    return TEAMS, TEAMS_LIST, TEAMS_LIST_REVERSED


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Team Helper Methods

    Methods to get the team and team name from the `TEAMS` DataFrame
    """
    )
    return


@app.cell
def _(TEAMS):
    def get_team(team_code):
        return TEAMS.loc[TEAMS.index == team_code]

    def get_team_name(team_code):
        return get_team(team_code).apply(lambda tt: f"{tt['CITY']} {tt['NICKNAME']}", axis=1).iloc[0]
    return


@app.cell
def _(mo):
    mo.md(r"""### Player Lookup Methods""")
    return


@app.cell
def _(RETROSHEET_DIR, pd):
    PLAYERS_DF = pd.read_csv(f"{RETROSHEET_DIR}/reference/biofile.csv.zip")
    PLAYERS_DF.set_index('PLAYERID', inplace=True)

    def get_player_name(player_id):
        row = PLAYERS_DF.loc[player_id][["NICKNAME", "LAST"]]
        return f"{row['NICKNAME']} {row['LAST']}"
    return PLAYERS_DF, get_player_name


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Game Log Information

    Read data from each season. Range end is exclusive, so it will read up to, but not including `END_YEAR`
    """
    )
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
def _(mo):
    mo.md(r"""#### Header Columns""")
    return


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
def _(mo):
    mo.md(
        r"""
    #### Add Columns and Fields to DataFrame

    * Add the column headers to the DataFrame
    * Add `datetime` as a parsed version of the games' `date` for easier compairision and graphing
    * Add a `game_id` for a consistent way to reference a game, includes:
        * `date` - The game date
        * `game_num` - Used for doubleheader games
        * `home_team` - The home team
        * `visiting_team` - The visiting team
    """
    )
    return


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
def _(mo):
    mo.md(
        r"""
    #### Drop columns that we don't need

    Out model doesn't currently include any information about position players, so remove it from the DataFrame
    """
    )
    return


@app.cell
def _(ALL_SEASONS_DF):
    ALL_SEASONS_DF.drop([
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
    ], axis=1, inplace=True)
    return


@app.cell
def _(mo):
    mo.md(r"""## Data Exploration and Cleaning""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Remove Ties

    There are exceptional situations where a game can end in a tie. These games can be safely removed if they don't make up a large percentage of the total games
    """
    )
    return


@app.cell
def _(ALL_SEASONS_DF):
    ties = ALL_SEASONS_DF[ALL_SEASONS_DF['home_score'] == ALL_SEASONS_DF['visiting_score']]
    (ties.shape[0] / ALL_SEASONS_DF.shape[0]) * 100
    return


@app.cell
def _(mo):
    mo.md(r"""#### Remove ties""")
    return


@app.cell
def _(ALL_SEASONS_DF):
    indexes_to_drop = ALL_SEASONS_DF.index[ALL_SEASONS_DF['home_score'] == ALL_SEASONS_DF['visiting_score']]
    ALL_SEASONS_DF.drop(indexes_to_drop, inplace=True)
    return


@app.cell
def _(mo):
    mo.md(r"""#### Set the binary value of a win or loss based on the score""")
    return


@app.cell
def _(ALL_SEASONS_DF, np):
    ALL_SEASONS_DF['home_win'] = np.where(ALL_SEASONS_DF['home_score'] > ALL_SEASONS_DF['visiting_score'], 1, 0)
    ALL_SEASONS_DF['visiting_win'] = np.where(ALL_SEASONS_DF['home_score'] > ALL_SEASONS_DF['visiting_score'], 0, 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Batting Stats

    $\text{Batting Average (BA)} = \frac{\text{Hits}}{\text{At Bats}}$

    $\text{Slugging Percentage (SLG)} = \frac{1B + (2 \times 2B) + (3 \times 3B) + (4 \times HR)}{\text{At Bats}}$

    $\text{On-Base Percentage (OBP)} = \frac{H + BB + HBP}{AB + BB + HBP + SF}$

    $\text{On-Base Plus Slugging Percentages (OPS)} = \text{OBP} + \text{SLG}$
    """
    )
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
    print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Per-Team Statistics

    Create a dataframe per-team that includes all the games for that team. This will make functions later like rolling averages much easier.

    We normalize the selected team for each stat with the `team_` prefix (ie. `team_ba`) regardless of whether it is a home or away game. The opponent similarly is normalized with the `opponent_` prefix

    * `is_home_game` is added to the DataFrame
    """
    )
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
def _(mo):
    mo.md(r"""### Extract the team data for each team""")
    return


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
def _(mo):
    mo.md(
        r"""
    ### Calculate the Rolling Average

    Add rolling averages for each selected batting stat over a period of 5, 20, 100, and 162 games

    The rolling averages are all calculate for all values prior to th current column, since using the current game would be determinisitic
    """
    )
    return


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
def _(mo):
    mo.md(
        r"""
    ### Per-Team Batting Data Exploration

    Explore some of the stats that have been calculated
    """
    )
    return


@app.cell
def _(TEAM_DATA, pd):
    TEAM_DATA_DF = pd.concat(list(TEAM_DATA.values())).sort_values(by='datetime')
    return (TEAM_DATA_DF,)


@app.cell
def _(ALL_BATTING_STATS, TEAMS_LIST_REVERSED, mo):
    season_teams_dropdown = mo.ui.dropdown(options=TEAMS_LIST_REVERSED, value=list(TEAMS_LIST_REVERSED.keys())[0], searchable=True)
    season_batting_stats_dropdown = mo.ui.dropdown(options=ALL_BATTING_STATS, value=ALL_BATTING_STATS[-1], searchable=True)
    return season_batting_stats_dropdown, season_teams_dropdown


@app.cell
def _(
    TEAMS_LIST,
    TEAM_DATA,
    mo,
    px,
    season_batting_stats_dropdown,
    season_teams_dropdown,
):
    _df = TEAM_DATA[season_teams_dropdown.value].groupby('season')[season_batting_stats_dropdown.value].mean().reset_index()

    _plot = px.line(
        _df, 
        x='season', 
        y=season_batting_stats_dropdown.value, 
        title=f"{TEAMS_LIST[season_teams_dropdown.value]} :: Mean {season_batting_stats_dropdown.value} Per Season"
    )

    mo.vstack([
        mo.hstack([
            season_teams_dropdown,
            season_batting_stats_dropdown,
        ], justify='start'),
        mo.ui.plotly(_plot)
    ])
    return


@app.cell
def _(ALL_BATTING_STATS, TEAMS_LIST_REVERSED, mo):
    teams_multi_dropdown = mo.ui.multiselect(options=TEAMS_LIST_REVERSED, value=list(TEAMS_LIST_REVERSED.keys())[:1], max_selections=5)
    teams_batting_stats_dropdown = mo.ui.dropdown(options=ALL_BATTING_STATS, value=ALL_BATTING_STATS[-1], searchable=True)
    return teams_batting_stats_dropdown, teams_multi_dropdown


@app.cell
def _(mo):
    mo.md(r"""### Plot the Rolling Average of a Batting Statistic by Team""")
    return


@app.cell
def _(
    TEAM_DATA_DF,
    mo,
    px,
    season_batting_stats_dropdown,
    teams_batting_stats_dropdown,
    teams_multi_dropdown,
):
    _fig = px.line(
        TEAM_DATA_DF[TEAM_DATA_DF['team_team'].isin(teams_multi_dropdown.value)],
        x='datetime',
        y=teams_batting_stats_dropdown.value,
        color='team_team',
        markers=False,
        title=f"Teams {season_batting_stats_dropdown.value}"
    )

    mo.vstack([
        mo.hstack([
            teams_multi_dropdown,
            teams_batting_stats_dropdown,
        ], justify='start'),
        mo.ui.plotly(_fig)
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Add the Aggregated Data Back Into the Game Logs

    Merge the batting rolling averages in to a new DataFrame that has the original game data plus the rolling averages for the home and away team going into that game
    """
    )
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
    return (MERGED_DF,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Pitching Statistics

    Pitching statistics are stored in a separte daily event file

    We will create a DataFrame for all pitchers' events
    """
    )
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
def _(mo):
    mo.md(
        r"""
    ### Calculate Per-Game Pitching Stats


    $\text{Earned Run Average (ERA)} = \left( \frac{\text{Earned Runs}}{\text{Innings Pitched}} \right) \times 9$

    $\text{Walks and Hits per Innings Pitched (WHIP)} = \frac{\text{Walks} + \text{Hits}}{\text{Innings Pitched}}$
    """
    )
    return


@app.cell
def _(PITCHING_DF):
    ## Calculate ERA, WHIP
    PITCHING_DF['P_IP'] = PITCHING_DF['P_OUT'] / 3.0
    PITCHING_DF['P_ERA'] = (PITCHING_DF['P_ER'] * 9) / PITCHING_DF['P_IP']
    PITCHING_DF['P_WHIP'] = (PITCHING_DF['P_BB'] + PITCHING_DF['P_H']) / PITCHING_DF['P_IP']
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Calculate Pitching Rolling Averages

    Add rolling averages for pitching stats for the previous 5 and 20 games. Since these are individual stats, they can be shorter timeframes to represent short and longer term performance
    """
    )
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
def _(mo):
    mo.md(r"""### Pitching Data Exploration""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Compare Pitching Statistics for Selected Pitchers""")
    return


@app.cell
def _(PITCHING_DF, PLAYERS_DF):
    _pitchers = PLAYERS_DF.loc[PITCHING_DF['person.key'].unique()].copy()
    _pitchers['NAME'] = _pitchers['NICKNAME'] + " " + _pitchers['LAST'] + " (" + _pitchers.index + ")"

    PITCHERS_LIST = _pitchers['NAME'].to_dict()
    PITCHERS_LIST_REVERSED = {v: k for k, v in PITCHERS_LIST.items()}
    return PITCHERS_LIST, PITCHERS_LIST_REVERSED


@app.cell
def _(ALL_PITCHING_STATS, END_YEAR, PITCHERS_LIST_REVERSED, START_YEAR, mo):
    _default_pitcher_ids = ['imans001', 'kersc001', 'salec001', 'skenp001']
    _default_list = {k: v for k, v in PITCHERS_LIST_REVERSED.items() if v in _default_pitcher_ids}

    pitchers_multiselect = mo.ui.multiselect(options=PITCHERS_LIST_REVERSED, max_selections=5, value=_default_list)
    pitchers_stats_dropdown = mo.ui.dropdown(options=ALL_PITCHING_STATS, value=ALL_PITCHING_STATS[-1], searchable=True)

    pitchers_start_season_dropdown = mo.ui.dropdown(options=range(START_YEAR, END_YEAR), value=END_YEAR - 1)
    pitchers_end_season_dropdown = mo.ui.dropdown(options=range(START_YEAR, END_YEAR), value=END_YEAR - 1)
    return (
        pitchers_end_season_dropdown,
        pitchers_multiselect,
        pitchers_start_season_dropdown,
        pitchers_stats_dropdown,
    )


@app.cell
def _(
    PITCHING_DF,
    get_player_name,
    mo,
    pitchers_end_season_dropdown,
    pitchers_multiselect,
    pitchers_start_season_dropdown,
    pitchers_stats_dropdown,
    px,
):
    _data = PITCHING_DF[PITCHING_DF['person.key'].isin(pitchers_multiselect.value)]
    _data = _data[(_data['season'] >= pitchers_start_season_dropdown.value) & (_data['season'] <= pitchers_end_season_dropdown.value)]

    _data['player_name'] = _data['person.key'].map(get_player_name)

    _fig = px.line(
        _data,
        x='game.datetime',
        y=pitchers_stats_dropdown.value,
        color='player_name',
        markers=False,
        title=f"{pitchers_stats_dropdown.value} ({pitchers_start_season_dropdown.value} - {pitchers_end_season_dropdown.value})"
    )

    mo.vstack([
        mo.hstack([
            pitchers_multiselect,
            pitchers_stats_dropdown,
            pitchers_start_season_dropdown,
            pitchers_end_season_dropdown,
        ], justify='start'),
        mo.ui.plotly(_fig)
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""#### Pitching Statistics for All Pitchers on a Team""")
    return


@app.cell
def _(ALL_PITCHING_STATS, END_YEAR, START_YEAR, TEAMS_LIST_REVERSED, mo):
    team_pitching_teams_dropdown = mo.ui.dropdown(options=TEAMS_LIST_REVERSED, value=list(TEAMS_LIST_REVERSED.keys())[0], searchable=True)
    team_pitching_stats_dropdown = mo.ui.dropdown(options=ALL_PITCHING_STATS, value=ALL_PITCHING_STATS[-1], searchable=True)
    team_pitching_start_season_dropdown = mo.ui.dropdown(options=range(START_YEAR, END_YEAR), value=END_YEAR - 1)
    team_pitching_end_season_dropdown = mo.ui.dropdown(options=range(START_YEAR, END_YEAR), value=END_YEAR - 1)
    return (
        team_pitching_end_season_dropdown,
        team_pitching_start_season_dropdown,
        team_pitching_stats_dropdown,
        team_pitching_teams_dropdown,
    )


@app.cell
def _(
    PITCHING_DF,
    TEAMS_LIST,
    get_player_name,
    mo,
    px,
    team_pitching_end_season_dropdown,
    team_pitching_start_season_dropdown,
    team_pitching_stats_dropdown,
    team_pitching_teams_dropdown,
):
    _data = PITCHING_DF[PITCHING_DF['team.key'] == team_pitching_teams_dropdown.value]
    _data = _data[(_data['season'] >= team_pitching_start_season_dropdown.value) & (_data['season'] <= team_pitching_end_season_dropdown.value)]

    _data['player_name'] = _data['person.key'].map(get_player_name)

    _fig = px.line(
        _data,
        x='game.datetime',
        y=team_pitching_stats_dropdown.value,
        color='player_name',
        markers=False,
        title=f"{TEAMS_LIST[team_pitching_teams_dropdown.value]} :: {team_pitching_stats_dropdown.value} ({team_pitching_start_season_dropdown.value} - {team_pitching_end_season_dropdown.value})"
    )

    mo.vstack([
        mo.hstack([
            team_pitching_teams_dropdown,
            team_pitching_stats_dropdown,
            team_pitching_start_season_dropdown,
            team_pitching_end_season_dropdown,
        ], justify='start'),
        mo.ui.plotly(_fig)
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Add Starting Pitching Data to Games

    We need to find the data for a pitcher as of the game date to add in the pitcher's data that is representative of their stats up until, but not including that game.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""#### Data Preparation""")
    return


@app.cell
def _(MERGED_DF):
    MERGED_DF_WITH_PITCHING = MERGED_DF.reset_index().rename(columns={'index': 'original_game_index'}).copy()
    return (MERGED_DF_WITH_PITCHING,)


@app.cell
def _(PITCHING_DF, PITCHING_STAT_MAPPER):
    # Prepare the pitcher stats lookup table (the "right" side of the merge)
    # We select and rename columns for the merge and sort by time, which is required.
    _stats_to_get = list(PITCHING_STAT_MAPPER.keys())
    pitcher_lookup = PITCHING_DF[['person.key', 'game.datetime'] + _stats_to_get].rename(columns={
        'person.key': 'pitcher_id',
        'game.datetime': 'datetime'
    }).sort_values('datetime')
    return (pitcher_lookup,)


@app.cell
def _(mo):
    mo.md(r"""#### Unpivot the Games DataFrame""")
    return


@app.cell
def _(MERGED_DF_WITH_PITCHING, pd):
    # Reshape g_df from "wide" to "long" format so we can process all pitchers in one call
    home_pitchers = MERGED_DF_WITH_PITCHING[[
        'original_game_index',
        'datetime',
        'home_starting_pitcher_id'
    ]].rename(
        columns={'home_starting_pitcher_id': 'pitcher_id'}
    )
    home_pitchers['role'] = 'home'

    visiting_pitchers = MERGED_DF_WITH_PITCHING[[
        'original_game_index',
        'datetime',
        'visiting_starting_pitcher_id'
    ]].rename(
        columns={'visiting_starting_pitcher_id': 'pitcher_id'}
    )
    visiting_pitchers['role'] = 'visiting'

    # Combine and sort by time, as required by merge_asof
    merged_df_long = pd.concat([home_pitchers, visiting_pitchers]).sort_values('datetime')
    return (merged_df_long,)


@app.cell
def _(mo):
    mo.md(r"""#### Perform the Time-Series Merge""")
    return


@app.cell
def _(merged_df_long, pd, pitcher_lookup):
    merged_long = pd.merge_asof(
        merged_df_long,
        pitcher_lookup,
        on='datetime',
        by='pitcher_id',
        direction='backward'  # This finds the last value <= the key
    )
    return (merged_long,)


@app.cell
def _(mo):
    mo.md(r"""#### Pivot Results Back to Wide Format""")
    return


@app.cell
def _(PITCHING_STAT_MAPPER, merged_long):
    _stats_to_get = list(PITCHING_STAT_MAPPER.keys())

    # Pivot the merged stats back to the original format with home/visiting columns
    pivoted_stats = merged_long.pivot_table(
        index='original_game_index',
        columns='role',
        values=_stats_to_get
    )

    # Flatten the multi-level column index created by the pivot
    pivoted_stats.columns = [f"{role}_{PITCHING_STAT_MAPPER[stat]}" for stat, role in pivoted_stats.columns]
    return (pivoted_stats,)


@app.cell
def _(mo):
    mo.md(
        r"""
    #### Final Join

    Join the new stats columns back to the original games DataFrame
    """
    )
    return


@app.cell
def _(MERGED_DF_WITH_PITCHING, pivoted_stats):
    MERGED_DF_WITH_PITCHING_FINAL = MERGED_DF_WITH_PITCHING.set_index('original_game_index').join(pivoted_stats)
    return (MERGED_DF_WITH_PITCHING_FINAL,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Modeling""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Model Training""")
    return


@app.cell
def _(
    BATTING_PERIODS,
    BATTING_STATS,
    MERGED_DF_WITH_PITCHING_FINAL,
    MERGED_PITCHING_STATS,
    PITCHING_PERIODS,
):
    _training_df = MERGED_DF_WITH_PITCHING_FINAL.copy()

    TRAINING_FIELDS = [
        'home_win',
        *[f"{ha}_{stat}_{period}" for ha in ['home', 'visiting'] for stat in BATTING_STATS for period in BATTING_PERIODS],
        *[f"{ha}_starting_pitcher_{stat}_{period}" for ha in ['home', 'visiting'] for stat in MERGED_PITCHING_STATS for
          period in PITCHING_PERIODS]
    ]

    TRAINER_DF = _training_df[TRAINING_FIELDS].dropna()

    x = TRAINER_DF.drop('home_win', axis=1)
    y = TRAINER_DF['home_win']
    return TRAINER_DF, TRAINING_FIELDS, x, y


@app.cell
def _(mo):
    mo.md(
        r"""
    #### Split up the dataset with training and test set

    * `x_train`	Features for training
    * `x_test` Features for testing
    * `y_train`	Targets for training
    * `y_test` Targets for testing
    """
    )
    return


@app.cell
def _(train_test_split, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_test, x_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(
        r"""
    #### Create a Pipeline

    Using a standard scaler to ensure that all the features are on the same scale and then using the Logistic Regression model

    * `penalty='l2'`: Adds regularization to prevent overfitting.
    * `max_iter=5000`: Gives the solver more time to converge
    """
    )
    return


@app.cell
def _(LogisticRegression, StandardScaler, make_pipeline):
    PIPELINE = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver='saga',
            penalty='l2',
            max_iter=5000,
        )
    )

    LOGMODEL = PIPELINE.named_steps['logisticregression']
    return LOGMODEL, PIPELINE


@app.cell
def _(mo):
    mo.md(r"""#### Fit the training data into the model""")
    return


@app.cell
def _(PIPELINE, x_train, y_train):
    PIPELINE.fit(x_train, y_train)
    return


@app.cell
def _(mo):
    mo.md(r"""#### Get the prediction results for the test set""")
    return


@app.cell
def _(PIPELINE, x_test):
    predictions = PIPELINE.predict(x_test)
    return (predictions,)


@app.cell
def _(mo):
    mo.md(r"""## Model Metrics""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### Classification Report

    Compare the `y_test` set vs. the actual predictions to get a better sense of the model's accuracy
    """
    )
    return


@app.cell
def _(classification_report, mo, pd, predictions, y_test):
    _report_df = pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose()
    mo.md(_report_df.to_markdown())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### Confusion Matrix

    This shows similar data to the classifiaction report in a graphical format
    """
    )
    return


@app.cell
def _(ConfusionMatrixDisplay, PIPELINE, io, mo, plt, x_test, y_test):
    _fig, _ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(PIPELINE, x_test, y_test, ax=_ax)

    _buffer = io.BytesIO()
    _fig.savefig(_buffer, format='png')
    _buffer.seek(0)

    mo.image(_buffer)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### ROC Curve

    We are looking for a higher Area Under the Curve (AUC) score that shows that the model is good at distinguishing wins and losses
    """
    )
    return


@app.cell
def _(PIPELINE, RocCurveDisplay, io, mo, plt, x_test, y_test):
    _fig, _ax = plt.subplots()
    RocCurveDisplay.from_estimator(PIPELINE, x_test, y_test, ax=_ax)

    _buffer = io.BytesIO()
    _fig.savefig(_buffer, format='png')
    _buffer.seek(0)

    mo.image(_buffer)
    return


@app.cell
def _(mo):
    mo.md(r"""#### Precision Recall Graph""")
    return


@app.cell
def _(PIPELINE, PrecisionRecallDisplay, io, mo, plt, x_test, y_test):
    _fig, _ax = plt.subplots()
    PrecisionRecallDisplay.from_estimator(PIPELINE, x_test, y_test, ax=_ax)

    _buffer = io.BytesIO()
    _fig.savefig(_buffer, format='png')
    _buffer.seek(0)

    mo.image(_buffer)
    return


@app.cell
def _(mo):
    mo.md(r"""#### Calibration Display""")
    return


@app.cell
def _(CalibrationDisplay, PIPELINE, io, mo, plt, x_test, y_test):
    _fig, _ax = plt.subplots()
    CalibrationDisplay.from_estimator(PIPELINE, x_test, y_test, ax=_ax)

    _buffer = io.BytesIO()
    _fig.savefig(_buffer, format='png')
    _buffer.seek(0)

    mo.image(_buffer)
    return


@app.cell
def _(mo):
    mo.md(r"""#### Histogram of Predicted Probabilities""")
    return


@app.cell
def _(PIPELINE, plt, x_test):
    y_proba = PIPELINE.predict_proba(x_test)[:, 1]
    plt.hist(y_proba, bins=20)
    plt.xlabel('Predicted Probability for Class 1')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Probabilities')
    return


@app.cell
def _(mo):
    mo.md(r"""#### Which Features Had the Biggest Impact to the Predictions""")
    return


@app.cell
def _(LOGMODEL, pd, plt, x_train):
    _coefficients = pd.Series(LOGMODEL.coef_[0], index=x_train.columns)

    _fig, _ax = plt.subplots(figsize=(10, 8))
    _coefficients.abs().sort_values(ascending=False).plot(kind='barh', ax=_ax, title='Feature Impact')
    _ax.set_xlabel("Absolute Coefficient Value (Impact)")
    _ax.invert_yaxis()
    _fig.tight_layout()

    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""#### How the Features Correlated to a Win or Loss""")
    return


@app.cell
def _(TRAINER_DF, TRAINING_FIELDS, plt):
    _correlations = TRAINER_DF[TRAINING_FIELDS].corr()

    _target = 'home_win'
    _correlations_target = _correlations[_target].drop(_target).sort_values()

    _fig, _ax = plt.subplots(figsize=(10, 8))
    _correlations_target.plot(kind='barh', ax=_ax, title='Correlation with Target: home_win')
    _ax.set_xlabel("Correlation")
    _fig.tight_layout()

    _fig 
    return


@app.cell
def _(mo):
    mo.md(r"""#### Correlation Heatmap""")
    return


@app.cell
def _(TRAINER_DF, sns):
    sns.heatmap(TRAINER_DF.corr(), cmap='coolwarm', annot=True)
    return


@app.cell
def _(mo):
    mo.md(r"""#### How Often a Win or Loss is Predicted""")
    return


@app.cell
def _(MERGED_DF_WITH_PITCHING_FINAL, plt, sns):
    _training_df = MERGED_DF_WITH_PITCHING_FINAL.copy()

    _fig, _ax = plt.subplots()
    sns.countplot(x='home_win', data=_training_df, ax=_ax)
    _ax.set_title('Class Distribution: home_win')
    _ax.set_xlabel('Home Win (1 = win, 0 = loss)')
    _ax.set_ylabel('Count')
    _fig.tight_layout()

    _fig
    return


@app.cell
def _(PITCHING_DF):
    last_season_pitching_data = PITCHING_DF[
        (PITCHING_DF['season'] == 2024)
    ].copy()

    ## Get the pitchers that have thrown more than 10 innings
    def get_pitchers(pitcher_data, players_df, team_code):
        pitchers = pitcher_data[
            pitcher_data['team.key'] == team_code
            ][['person.key', 'P_IP']].groupby('person.key').sum().query('P_IP > 10')

        return players_df[players_df.index.isin(pitchers.index)]
    return (last_season_pitching_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Predictions""")
    return


@app.cell
def _(TEAM_DATA):
    def get_last_team_stats(team_id):
        return TEAM_DATA[team_id].iloc[-1]
    return (get_last_team_stats,)


@app.cell
def _(last_season_pitching_data):
    def get_last_pitching_stats(pitcher_id):
        return last_season_pitching_data[last_season_pitching_data['person.key'] \
                                         == pitcher_id].groupby('person.key').last().iloc[-1]
    return (get_last_pitching_stats,)


@app.cell
def _(mo):
    mo.md(
        r"""
    #### Create a Single Game Series

    Given the team and the pitcher, create a Series with the batting and pitching statistics that we can use to generate predictions
    """
    )
    return


@app.cell
def _(BATTING_PERIODS, BATTING_STATS, PITCHING_STAT_MAPPER, pd):
    def create_game_series(home_team_stats, home_pitcher_stats, away_team_stats, away_pitcher_stats):
        ## Batting
        batting_fields = [f"team_{s}_{batting_period}" for s in BATTING_STATS for batting_period in BATTING_PERIODS]

        home_batting = home_team_stats[batting_fields].rename(
            lambda f: f.replace('team_', 'home_') if f.startswith('team_') else f)
        away_batting = away_team_stats[batting_fields].rename(
            lambda f: f.replace('team_', 'visiting_') if f.startswith('team_') else f)

        batting = pd.concat([home_batting, away_batting], axis=0)

        ## Pitching
        home_pitching = home_pitcher_stats[PITCHING_STAT_MAPPER.keys()].rename(lambda z: f"home_{PITCHING_STAT_MAPPER[z]}")
        away_pitching = away_pitcher_stats[PITCHING_STAT_MAPPER.keys()].rename(lambda z: f"visiting_{PITCHING_STAT_MAPPER[z]}")

        pitching = pd.concat([home_pitching, away_pitching], axis=0)

        return pd.concat([batting, pitching])
    return (create_game_series,)


@app.cell
def _(mo):
    mo.md(
        r"""
    #### Generate a Game Prediction

    Given 2 teams and their starting pitchers, generate a prediction for a win or loss
    """
    )
    return


@app.cell
def _(create_game_series, get_last_pitching_stats, get_last_team_stats, pd):
    def predict_game(model, home_team, home_pitcher_id, visiting_team, visiting_pitcher_id):
        home_team_stats = get_last_team_stats(home_team)
        home_pitcher_stats = get_last_pitching_stats(home_pitcher_id)

        away_team_stats = get_last_team_stats(visiting_team)
        away_pitcher_stats = get_last_pitching_stats(visiting_pitcher_id)

        game = create_game_series(home_team_stats, home_pitcher_stats, away_team_stats, away_pitcher_stats)

        game_df = pd.DataFrame([game])
        probs = model.predict_proba(game_df)

        return probs[0][1], probs[0][0]
    return (predict_game,)


@app.cell
def _(mo):
    mo.md(r"""### Predict the Outcome of a Game""")
    return


@app.cell
def _(TEAMS_LIST_REVERSED, mo):
    prediction_home_team_dropdown = mo.ui.dropdown(options=TEAMS_LIST_REVERSED, value=list(TEAMS_LIST_REVERSED.keys())[0], searchable=True)
    return (prediction_home_team_dropdown,)


@app.cell
def _(TEAMS_LIST_REVERSED, mo, prediction_home_team_dropdown):
    ## All teams except the selected home team
    _teams_list = {k: v for k, v in TEAMS_LIST_REVERSED.items() if v not in [prediction_home_team_dropdown.value]}

    prediction_away_team_dropdown = mo.ui.dropdown(options=_teams_list, value=list(_teams_list.keys())[1], searchable=True)
    return (prediction_away_team_dropdown,)


@app.function
def pitchers_list(team_name, season, pitching_df, pitchers_list_df):
    pitchers = pitching_df[(pitching_df['team.key'] == team_name) & (pitching_df['season'] == season)].dropna()['person.key'].to_list()
    _pitchers_list = {k: pitchers_list_df[k] for k in pitchers if k in pitchers_list_df}
    _pitchers_list_reversed = {v: k for k, v in _pitchers_list.items()}

    return _pitchers_list, _pitchers_list_reversed


@app.cell
def _(
    PITCHERS_LIST,
    PITCHING_DF,
    mo,
    prediction_away_team_dropdown,
    prediction_home_team_dropdown,
):
    _season = 2024

    ## Home
    home_pitchers_list, home_pitchers_list_reversed = pitchers_list(
        prediction_home_team_dropdown.value,
        _season,
        PITCHING_DF,
        PITCHERS_LIST
    )

    prediction_home_pitcher_dropdown = mo.ui.dropdown(options=home_pitchers_list_reversed, value=list(home_pitchers_list_reversed.keys())[1], searchable=True)

    ## Away
    away_pitchers_list, away_pitchers_list_reversed = pitchers_list(
        prediction_away_team_dropdown.value,
        _season,
        PITCHING_DF,
        PITCHERS_LIST
    )

    prediction_away_pitcher_dropdown = mo.ui.dropdown(options=away_pitchers_list_reversed, value=list(away_pitchers_list_reversed.keys())[1], searchable=True)
    return prediction_away_pitcher_dropdown, prediction_home_pitcher_dropdown


@app.cell
def _(
    PIPELINE,
    TEAMS_LIST,
    go,
    mo,
    predict_game,
    prediction_away_pitcher_dropdown,
    prediction_away_team_dropdown,
    prediction_home_pitcher_dropdown,
    prediction_home_team_dropdown,
):
    _home_team_win, _away_team_win = predict_game(
        PIPELINE, 
        prediction_home_team_dropdown.value, 
        prediction_home_pitcher_dropdown.value, 
        prediction_away_team_dropdown.value, 
        prediction_away_pitcher_dropdown.value, 
    )

    _fig = go.Figure([
        go.Scatter(
            x=[0, 100],
            y=[0, 0],
            mode='lines',
            line=dict(width=10, color='lightgrey'),
            showlegend=False
        ),

        ## Home Team
        go.Scatter(
            x=[_away_team_win * 100],
            y=[0],
            mode='markers',
            marker=dict(size=12, color='blue'),
            name=f"{TEAMS_LIST[prediction_away_team_dropdown.value]} {_away_team_win * 100:.2f}%"),

        ## Away Team
        go.Scatter(
            x=[_home_team_win * 100],
            y=[0],
            mode='markers',
            marker=dict(size=12, color='red'),
            name=f"{TEAMS_LIST[prediction_home_team_dropdown.value]} {_home_team_win * 100:.2f}%"),

    ])

    _fig.update_layout(
        xaxis_range=[0, 100],
        xaxis_title='Prediction Confidence (%)',
        yaxis_visible=False,
        title=f'{TEAMS_LIST[prediction_away_team_dropdown.value]} at {TEAMS_LIST[prediction_home_team_dropdown.value]} Win Probability',
        height=300
    )

    mo.vstack([
        mo.hstack([
            mo.vstack([
                mo.md("#### Away"),
                prediction_away_team_dropdown,
                prediction_away_pitcher_dropdown
            ]),
            mo.vstack([
                mo.md("#### Home"),
                prediction_home_team_dropdown,     
                prediction_home_pitcher_dropdown
            ]),
        ], justify="start"),
        mo.ui.plotly(_fig)
    ])

    return


@app.cell
def _(mo):
    mo.md(r"""### Preidct the Outcome of a Game for All Starting Pitchers""")
    return


@app.cell
def _(PIPELINE, PLAYERS_DF, pd, predict_game):
    def create_pitching_prediction_dataframe(home_team, away_team, player_data):
        home_pitchers_list = player_data[home_team]
        away_pitchers_list = player_data[away_team]

        # Run predictions and store results
        results = []

        for h_pitcher in home_pitchers_list:
            row = {}
            for a_pitcher in away_pitchers_list:
                away_pitcher_name = PLAYERS_DF.loc[a_pitcher]['LAST']
                home_prob, away_prob = predict_game(PIPELINE, home_team, h_pitcher, away_team, a_pitcher)
                row[away_pitcher_name] = home_prob * 100
            home_pitcher_name = PLAYERS_DF.loc[h_pitcher]['LAST']
            results.append(pd.Series(row, name=home_pitcher_name))

        pitching_prediction_dataframe = pd.DataFrame(results)
        pitching_prediction_dataframe.index.name = 'Home Pitcher'
        pitching_prediction_dataframe.columns.name = 'Away Pitcher'

        return pitching_prediction_dataframe.dropna()
    return (create_pitching_prediction_dataframe,)


@app.cell
def _(create_pitching_prediction_dataframe, px):
    def create_prediction_heatmap(home_team, away_team, data):

        df = create_pitching_prediction_dataframe(home_team, away_team, data)

        df_reset = df.reset_index().melt(id_vars='Home Pitcher', var_name='Away Pitcher', value_name='Win %')

        figure = px.density_heatmap(
            df_reset,
            x='Away Pitcher',
            y='Home Pitcher',
            z='Win %',
            color_continuous_scale='RdBu_r',
            range_color=[0, 100],
            text_auto='.1f'
        )

        figure.update_layout(
            title=f"{home_team} Win Probability vs {away_team} Pitchers",
            xaxis_title=away_team,
            yaxis_title=home_team,
            coloraxis_colorbar=dict(title="Win %"),
            yaxis=dict(autorange='reversed'),
            height=800
        )

        return figure
    return


@app.cell
def _(TEAMS_LIST_REVERSED, mo):
    ## Home
    all_pitchers_home_team_dropdown = mo.ui.dropdown(options=TEAMS_LIST_REVERSED, value=list(TEAMS_LIST_REVERSED.keys())[0], searchable=True)

    ## Away
    all_pitchers_away_team_dropdown = mo.ui.dropdown(options=TEAMS_LIST_REVERSED, value=list(TEAMS_LIST_REVERSED.keys())[1], searchable=True)
    return all_pitchers_away_team_dropdown, all_pitchers_home_team_dropdown


@app.cell
def _(
    PITCHERS_LIST,
    PITCHING_DF,
    all_pitchers_away_team_dropdown,
    all_pitchers_home_team_dropdown,
):
    _season = 2024

    all_pitchers_home_pitchers_list, all_pitchers_home_pitchers_list_reversed = pitchers_list(
        all_pitchers_home_team_dropdown.value,
        _season,
        PITCHING_DF,
        PITCHERS_LIST
    )

    all_pitchers_away_pitchers_list, all_pitchers_away_pitchers_list_reversed = pitchers_list(
        all_pitchers_away_team_dropdown.value,
        _season,
        PITCHING_DF,
        PITCHERS_LIST
    )
    return all_pitchers_away_pitchers_list, all_pitchers_home_pitchers_list


@app.cell
def _(
    TEAMS_LIST,
    all_pitchers_away_pitchers_list,
    all_pitchers_away_team_dropdown,
    all_pitchers_home_pitchers_list,
    all_pitchers_home_team_dropdown,
    create_pitching_prediction_dataframe,
    mo,
    px,
):
    _team_data = {}
    _team_data[all_pitchers_home_team_dropdown.value] = all_pitchers_home_pitchers_list.keys()
    _team_data[all_pitchers_away_team_dropdown.value] = all_pitchers_away_pitchers_list.keys()

    _home_team = all_pitchers_home_team_dropdown.value
    _away_team = all_pitchers_away_team_dropdown.value

    _df = create_pitching_prediction_dataframe(_home_team, _away_team, _team_data)

    # Plotly
    _df_reset = _df.reset_index().melt(id_vars='Home Pitcher', var_name='Away Pitcher', value_name='Win %')

    _figure = px.density_heatmap(
        _df_reset,
        x='Away Pitcher',
        y='Home Pitcher',
        z='Win %',
        color_continuous_scale='RdBu_r',
        range_color=[0, 100],
        text_auto='.1f'
    )

    _figure.update_layout(
        title=f"{TEAMS_LIST[_away_team]} at {TEAMS_LIST[_home_team]} Win Probability by Pitcher",
        xaxis_title=f"{TEAMS_LIST[_away_team]}",
        yaxis_title=f"{TEAMS_LIST[_home_team]}",
        coloraxis_colorbar=dict(title="Win %"),
        yaxis=dict(autorange='reversed'),
        height=800
    )


    mo.vstack([
        mo.hstack([
            mo.vstack([
                mo.md("#### Away"),
                all_pitchers_away_team_dropdown
            ]),
            mo.vstack([
                mo.md("#### Home"),
                all_pitchers_home_team_dropdown
            ]),        
        ], justify="start"),
        mo.ui.plotly(_figure)
    ])
    return


if __name__ == "__main__":
    app.run()
