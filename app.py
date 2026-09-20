import streamlit as st
import pandas as pd
import numpy as np
import ast

st.set_page_config(layout="wide", page_title="My Stats")

@st.cache_data
def load_data():
    df = pd.read_csv('all_games_data.csv')
    if 'All Counts' in df.columns:
        df['All Counts'] = df['All Counts'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
    if 'Game Date' in df.columns:
        df['Parsed Date'] = pd.to_datetime(df['Game Date'], format='%m/%d/%Y', errors='coerce').dt.date
    return df

try:
    df_all_games = load_data()
except FileNotFoundError:
    st.error("Data file not found. Please run the notebook cell that exports 'all_games_data.csv'.")
    st.stop()

def get_csw(df_source, team, pitcher=None, batter=None):
    if pitcher:
        p_df = df_source[(df_source['Pitcher'] == pitcher) & (df_source['Pitching Team'] == team)]
    elif batter:
        p_df = df_source[(df_source['Batter'] == batter) & (df_source['Hitting Team'] == team)]
    else:
        return 0
    seqs = p_df['Pitch Sequence'].dropna().astype(str).tolist()
    c_and_s = sum(s.upper().count('C') + s.upper().count('S') for s in seqs)
    total_p = sum(len([c for c in s if c.upper() in 'BIVPCKSF']) for s in seqs)
    return c_and_s / total_p if total_p > 0 else 0

def calculate_first_pitch_strikes(df_source, team, pitcher=None, batter=None):
    if pitcher:
        target_df = df_source[(df_source['Pitcher'] == pitcher) & (df_source['Pitching Team'] == team)]
    elif batter:
        target_df = df_source[(df_source['Batter'] == batter) & (df_source['Hitting Team'] == team)]
    else:
        return 0.0
    pa_df = target_df[target_df['Pitch Sequence'].notna() & (target_df['Pitch Sequence'] != 'N/A') & (target_df['Pitch Sequence'] != '')].copy()
    if pa_df.empty: return 0.0
    pa_df['First Pitch'] = pa_df['Pitch Sequence'].str[0].str.upper()
    pa_df['Is First Pitch Strike'] = pa_df['First Pitch'].isin(['S', 'F', 'K'])
    batters_faced = pa_df['Plate Appearance'].count()
    return pa_df['Is First Pitch Strike'].sum() / batters_faced if batters_faced > 0 else 0.0

def get_split_stats(df):
    split_pivot = df.pivot_table(index=['Hitting Team', 'Batter'], columns='Grouped Play Type', aggfunc='size', fill_value=0).reset_index()
    if split_pivot.empty: return pd.DataFrame()

    expected_cols = ['Single', 'Double', 'Triple', 'Home Run', 'Walk', 'Hit By Pitch', 'Strikeout', 'Out', 'Fielders Choice', 'Reached on Error', 'Sacrifice Fly', 'Sacrifice Hit']
    for col in expected_cols:
        if col not in split_pivot.columns: split_pivot[col] = 0

    split_pivot['Hits'] = split_pivot['Single'] + split_pivot['Double'] + split_pivot['Triple'] + split_pivot['Home Run']
    split_pivot['At Bats'] = split_pivot['Hits'] + split_pivot['Strikeout'] + split_pivot['Out'] + split_pivot['Fielders Choice'] + split_pivot['Reached on Error']
    split_pivot['Plate Appearances'] = split_pivot['At Bats'] + split_pivot['Walk'] + split_pivot['Hit By Pitch'] + split_pivot['Sacrifice Fly'] + split_pivot['Sacrifice Hit']
    split_pivot = split_pivot[split_pivot['Plate Appearances'] > 0].copy()
    split_pivot['Batting Average'] = (split_pivot['Hits'] / split_pivot['At Bats']).fillna(0)

    obp_denom = split_pivot['At Bats'] + split_pivot['Walk'] + split_pivot['Hit By Pitch'] + split_pivot['Sacrifice Fly']
    split_pivot['On Base Percentage'] = ((split_pivot['Hits'] + split_pivot['Walk'] + split_pivot['Hit By Pitch']) / obp_denom).fillna(0)
    split_pivot['Slugging Percentage'] = ((split_pivot['Single'] + 2*split_pivot['Double'] + 3*split_pivot['Triple'] + 4*split_pivot['Home Run']) / split_pivot['At Bats']).fillna(0)
    split_pivot['OPS'] = split_pivot['On Base Percentage'] + split_pivot['Slugging Percentage']
    split_pivot['Strikeout Percentage'] = (split_pivot['Strikeout'] / split_pivot['Plate Appearances']).fillna(0)
    split_pivot['Walk Percentage'] = (split_pivot['Walk'] / split_pivot['Plate Appearances']).fillna(0)
    split_pivot['CSW%'] = split_pivot.apply(lambda row: get_csw(df, row['Hitting Team'], batter=row['Batter']), axis=1)
    split_pivot['FPS%'] = split_pivot.apply(lambda row: calculate_first_pitch_strikes(df, row['Hitting Team'], batter=row['Batter']), axis=1)

    total_pitches_df = df[df['Pitch Sequence'].notna()].copy()
    total_pitches_df['Pitches'] = total_pitches_df['Pitch Sequence'].apply(lambda x: len(x) if x != 'N/A' else 0)
    total_pitches_per_player = total_pitches_df.groupby(['Hitting Team', 'Batter'])['Pitches'].sum().reset_index()
    split_pivot = pd.merge(split_pivot, total_pitches_per_player, on=['Hitting Team', 'Batter'], how='left').fillna({'Pitches': 0})
    split_pivot['P/PA'] = np.where(split_pivot['Plate Appearances'] > 0, split_pivot['Pitches'] / split_pivot['Plate Appearances'], 0.0)

    split_pivot.rename(columns={'Hitting Team': 'Team', 'Batter': 'Player', 'Plate Appearances': 'PA', 'At Bats': 'AB', 'Hits': 'H', 'Single': '1B', 'Double': '2B', 'Triple': '3B', 'Home Run': 'HR', 'Walk': 'BB', 'Hit By Pitch': 'HBP', 'Strikeout': 'SO', 'Batting Average': 'AVG', 'On Base Percentage': 'OBP', 'Slugging Percentage': 'SLG', 'Strikeout Percentage': 'K%', 'Walk Percentage': 'BB%'}, inplace=True)

    games_df = df.groupby(['Hitting Team', 'Batter'])['Game File'].nunique().reset_index()
    games_df.rename(columns={'Hitting Team': 'Team', 'Batter': 'Player', 'Game File': 'G'}, inplace=True)
    split_pivot = pd.merge(split_pivot, games_df, on=['Team', 'Player'], how='left')

    rbi_df = df.groupby(['Hitting Team', 'Batter'])['Runs Scored On Play'].sum().reset_index()
    rbi_df.rename(columns={'Hitting Team': 'Team', 'Batter': 'Player', 'Runs Scored On Play': 'RBI'}, inplace=True)
    split_pivot = pd.merge(split_pivot, rbi_df, on=['Team', 'Player'], how='left')
    split_pivot['R'] = 0

    final_cols = ['Player', 'Team', 'G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'HBP', 'AVG', 'OBP', 'SLG', 'OPS', 'K%', 'BB%', 'CSW%', 'FPS%', 'P/PA']
    for col in final_cols:
        if col not in split_pivot.columns: split_pivot[col] = 0
    return split_pivot[final_cols]

def get_pitching_split_stats(df):
    temp_df = df.copy()
    temp_df['Hitting Team'] = temp_df['Pitching Team']
    temp_df['Batter'] = temp_df['Pitcher']

    hitting_stats_against_pitchers = get_split_stats(temp_df)

    if not hitting_stats_against_pitchers.empty:
        hitting_stats_against_pitchers.rename(columns={
            'Team': 'Pitching Team',
            'Player': 'Pitcher'
        }, inplace=True)
    return hitting_stats_against_pitchers

def apply_filters(df, teams, opps, p_hands, b_hands, bases, outs, innings, counts, game_locs, start_date, end_date, team_col, opp_col, pinch_hitter_selection=None):
    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered['Parsed Date'] >= start_date]
    if end_date:
        filtered = filtered[filtered['Parsed Date'] <= end_date]

    if teams:
        expanded_teams = []
        for t in teams:
            if t == 'MOUNTAIN NORTH': expanded_teams.extend(['GARDEN CITY', 'TRINIDAD', 'GRAND JUNCTION', 'NORTH PLATTE', 'BLACKWELL'])
            elif t == 'MOUNTAIN SOUTH': expanded_teams.extend(['TUCSON', 'SANTA FE', 'ALPINE', 'PECOS', 'ROSWELL'])
            elif t == 'PACIFIC NORTH': expanded_teams.extend(['DUBLIN', 'AUSTIN', 'BAKERSFIELD', 'SAN RAFAEL', 'MARTINEZ'])
            else: expanded_teams.append(t)
        filtered = filtered[filtered[team_col].isin(expanded_teams)]

    if opps:
        expanded_opps = []
        for t in opps:
            if t == 'MOUNTAIN NORTH': expanded_opps.extend(['GARDEN CITY', 'TRINIDAD', 'GRAND JUNCTION', 'NORTH PLATTE', 'BLACKWELL'])
            elif t == 'MOUNTAIN SOUTH': expanded_opps.extend(['TUCSON', 'SANTA FE', 'ALPINE', 'PECOS', 'ROSWELL'])
            elif t == 'PACIFIC NORTH': expanded_opps.extend(['DUBLIN', 'AUSTIN', 'BAKERSFIELD', 'SAN RAFAEL', 'MARTINEZ'])
            else: expanded_opps.append(t)
        filtered = filtered[filtered[opp_col].isin(expanded_opps)]

    if p_hands: filtered = filtered[filtered['Pitcher Handedness'].isin(p_hands)]
    if b_hands: filtered = filtered[filtered['Batter Handedness'].isin(b_hands)]

    if bases:
        expanded_bases = []
        for b in bases:
            if b == 'RISP': expanded_bases.extend(['_2_', '__3', '_23', '12_', '1_3', '123'])
            else: expanded_bases.append(b)
        filtered = filtered[filtered['Previous BaseOut State'].isin(expanded_bases)]

    if outs: filtered = filtered[filtered['Outs'].isin(outs)]
    if innings: filtered = filtered[filtered['Inning Number'].astype(str).isin(innings)]

    if counts:
        expanded_counts = set()
        if '2 Strikes' in counts: expanded_counts.update(['0-2', '1-2', '2-2', '3-2'])
        if 'Hitter Ahead' in counts: expanded_counts.update(['1-0', '2-0', '3-0', '2-1', '3-1'])
        if 'Pitcher Ahead' in counts: expanded_counts.update(['0-1', '0-2', '1-2'])
        for c in counts:
            if c not in ['2 Strikes', 'Hitter Ahead', 'Pitcher Ahead']:
                expanded_counts.add(c)
        if expanded_counts:
            filtered = filtered[filtered['All Counts'].apply(lambda c_list: bool(expanded_counts.intersection(set(c_list))))]

    if game_locs:
        loc_col = 'Game Location (Hitting)' if team_col == 'Hitting Team' else 'Game Location (Pitching)'
        if loc_col in filtered.columns:
            filtered = filtered[filtered[loc_col].isin(game_locs)]

    if pinch_hitter_selection and team_col == 'Hitting Team' and 'Is Pinch Hitter' in filtered.columns:
        if 'Yes' in pinch_hitter_selection and 'No' not in pinch_hitter_selection:
            filtered = filtered[filtered['Is Pinch Hitter'] == True]
        elif 'No' in pinch_hitter_selection and 'Yes' not in pinch_hitter_selection:
            filtered = filtered[filtered['Is Pinch Hitter'] == False]

    return filtered

st.title("Stat Dashboard")
st.subheader("Created by Adam Lipman")

tab1, tab2 = st.tabs(["Batting Splits", "Pitching Splits"])

all_hitting_teams = sorted(df_all_games['Hitting Team'].dropna().unique().tolist())
all_pitching_teams = sorted(df_all_games['Pitching Team'].dropna().unique().tolist())
all_p_hands = sorted(df_all_games['Pitcher Handedness'].dropna().unique().tolist())
all_b_hands = sorted(df_all_games['Batter Handedness'].dropna().unique().tolist())
all_bases = ['RISP'] + sorted(df_all_games['Previous BaseOut State'].dropna().unique().tolist())
all_outs = sorted(df_all_games['Outs'].dropna().unique().tolist())
all_innings = sorted(df_all_games['Inning Number'].dropna().unique().astype(int).astype(str).tolist())
all_counts = ['2 Strikes', 'Hitter Ahead', 'Pitcher Ahead'] + sorted(list(set(c for clist in df_all_games['All Counts'] for c in clist)))
all_locs = sorted(df_all_games['Game Location (Hitting)'].dropna().unique().tolist())
all_p_locs = sorted(df_all_games['Game Location (Pitching)'].dropna().unique().tolist())
all_pinch_hitter_options = ['Yes', 'No']

with tab1:
    st.header("Batting Splits")
    with st.expander("Batting Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        b_teams = c1.multiselect("Hitting Team", all_hitting_teams, key='b_teams')
        b_opps = c2.multiselect("Opponent", all_pitching_teams, key='b_opps')
        b_p_hands = c3.multiselect("Pitcher Hand", all_p_hands, key='b_p_hands')
        b_b_hands = c4.multiselect("Batter Hand", all_b_hands, key='b_b_hands')

        c5, c6, c7, c8 = st.columns(4)
        b_bases = c5.multiselect("Bases (1,2,3)", all_bases, key='b_bases')
        b_outs = c6.multiselect("Outs", all_outs, key='b_outs')
        b_innings = c7.multiselect("Inning", all_innings, key='b_innings')
        b_counts = c8.multiselect("Count", all_counts, key='b_counts')

        c9, c10, c11, c12 = st.columns(4)
        b_locs = c9.multiselect("Game Location", all_locs, key='b_locs')
        b_pinch_hitter = c10.multiselect("Pinch Hitter", all_pinch_hitter_options, key='b_pinch_hitter')
        b_start_date = c11.date_input("Start Date", value=None, key='b_start')
        b_end_date = c12.date_input("End Date", value=None, key='b_end')

        c13, c14, c15, c16 = st.columns(4)
        stat_columns = ['G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'RBI', 'AVG', 'OBP', 'SLG', 'OPS', 'K%', 'BB%', 'P/PA']
        b_sort_col = c13.selectbox("Sort By", stat_columns, index=stat_columns.index('OPS'), key='b_sort_col')
        b_sort_order = c14.selectbox("Order", ['Descending', 'Ascending'], key='b_sort_order')
        b_qual_col = c15.selectbox("Qualifier Col", ['None'] + stat_columns, index=1, key='b_qual_col') # Default PA
        b_qual_val = c16.number_input("Qualifier Min Val", value=0, key='b_qual_val')

    if st.button("Calculate Batting Stats", type="primary"):
        filtered_df = apply_filters(df_all_games, b_teams, b_opps, b_p_hands, b_b_hands, b_bases, b_outs, b_innings, b_counts, b_locs, b_start_date, b_end_date, 'Hitting Team', 'Pitching Team', b_pinch_hitter)

        if not filtered_df.empty:
            stats = get_split_stats(filtered_df)
            if not stats.empty:
                if b_qual_col != 'None':
                    stats = stats[stats[b_qual_col] >= b_qual_val]
                stats = stats.sort_values(b_sort_col, ascending=(b_sort_order == 'Ascending'))
                st.dataframe(stats.style.format({'AVG': '{:.3f}', 'OBP': '{:.3f}', 'SLG': '{:.3f}', 'OPS': '{:.3f}', 'BABIP': '{:.3f}', 'K%': '{:.1%}', 'BB%': '{:.1%}', 'CSW%': '{:.1%}', 'FPS%': '{:.1%}', 'P/PA': '{:.2f}'}), use_container_width=True)
            else:
                st.warning("No qualifying players found for these filters.")
        else:
            st.warning("No plays match the selected criteria.")

with tab2:
    st.header("Pitching Splits")
    with st.expander("Pitching Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        p_teams = c1.multiselect("Pitcher's Team", all_pitching_teams, key='p_teams')
        p_opps = c2.multiselect("Hitting Team", all_hitting_teams, key='p_opps')
        p_p_hands = c3.multiselect("Pitcher Hand", all_p_hands, key='p_p_hands')
        p_b_hands = c4.multiselect("Batter Hand", all_b_hands, key='p_b_hands')

        c5, c6, c7, c8 = st.columns(4)
        p_bases = c5.multiselect("Bases (1,2,3)", all_bases, key='p_bases')
        p_outs = c6.multiselect("Outs", all_outs, key='p_outs')
        p_innings = c7.multiselect("Inning", all_innings, key='p_innings')
        p_counts = c8.multiselect("Count", all_counts, key='p_counts')

        c9, c10, c11 = st.columns(3)
        p_locs = c9.multiselect("Game Location", all_p_locs, key='p_locs')
        p_start_date = c10.date_input("Start Date", value=None, key='p_start')
        p_end_date = c11.date_input("End Date", value=None, key='p_end')

        c12, c13, c14, c15 = st.columns(4)
        p_stat_columns = ['G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'HBP', 'AVG', 'OBP', 'SLG', 'OPS', 'K%', 'BB%', 'P/PA']
        p_sort_col = c12.selectbox("Sort By", p_stat_columns, index=p_stat_columns.index('OPS'), key='p_sort_col')
        p_sort_order = c13.selectbox("Order", ['Descending', 'Ascending'], index=1, key='p_sort_order')
        p_qual_col = c14.selectbox("Qualifier Col", ['None'] + p_stat_columns, index=p_stat_columns.index('G') + 1, key='p_qual_col')
        p_qual_val = c15.number_input("Qualifier Min Val", value=0, key='p_qual_val')

    if st.button("Calculate Pitching Stats", type="primary", key="calc_pitching"):
        filtered_p_df = apply_filters(df_all_games, p_teams, p_opps, p_p_hands, p_b_hands, p_bases, p_outs, p_innings, p_counts, p_locs, p_start_date, p_end_date, 'Pitching Team', 'Hitting Team')

        if not filtered_p_df.empty:
            p_stats = get_pitching_split_stats(filtered_p_df)
            if not p_stats.empty:
                if p_qual_col != 'None':
                    p_stats = p_stats[p_stats[p_qual_col] >= p_qual_val]
                p_stats = p_stats.sort_values(p_sort_col, ascending=(p_sort_order == 'Ascending'))

                if not p_stats.empty:
                    total_pa = p_stats['PA'].sum()
                    total_ab = p_stats['AB'].sum()
                    total_row = pd.DataFrame([{
                        'Pitcher': 'TOTAL',
                        'Pitching Team': 'Combined',
                        'G': p_stats['G'].max(),
                        'PA': total_pa,
                        'AB': total_ab,
                        'R': p_stats['R'].sum(),
                        'H': p_stats['H'].sum(),
                        '1B': p_stats['1B'].sum(),
                        '2B': p_stats['2B'].sum(),
                        '3B': p_stats['3B'].sum(),
                        'HR': p_stats['HR'].sum(),
                        'RBI': p_stats['RBI'].sum(),
                        'BB': p_stats['BB'].sum(),
                        'SO': p_stats['SO'].sum(),
                        'HBP': p_stats['HBP'].sum(),
                        'AVG': p_stats['H'].sum() / total_ab if total_ab > 0 else 0,
                        'SLG': (p_stats['1B'].sum() + 2*p_stats['2B'].sum() + 3*p_stats['3B'].sum() + 4*p_stats['HR'].sum()) / total_ab if total_ab > 0 else 0,
                        'OBP': (p_stats['H'].sum() + p_stats['BB'].sum() + p_stats['HBP'].sum()) / total_pa if total_pa > 0 else 0,
                        'K%': (p_stats['SO'].sum()) / total_pa if total_pa > 0 else 0,
                        'BB%': (p_stats['BB'].sum()) / total_pa if total_pa > 0 else 0,
                        'CSW%': (p_stats['CSW%'] * p_stats['PA']).sum() / total_pa if total_pa > 0 else 0,
                        'FPS%': (p_stats['FPS%'] * p_stats['PA']).sum() / total_pa if total_pa > 0 else 0,
                        'P/PA': (p_stats['P/PA'] * p_stats['PA']).sum() / total_pa if total_pa > 0 else 0
                    }])
                    total_row['OPS'] = total_row['OBP'] + total_row['SLG']
                    p_stats = pd.concat([p_stats, total_row], ignore_index=True)

                st.dataframe(p_stats.style.format({'AVG': '{:.3f}', 'OBP': '{:.3f}', 'SLG': '{:.3f}', 'OPS': '{:.3f}', 'BABIP': '{:.3f}', 'K%': '{:.1%}', 'BB%': '{:.1%}', 'CSW%': '{:.1%}', 'FPS%': '{:.1%}', 'P/PA': '{:.2f}'}), use_container_width=True)
            else:
                st.warning("No qualifying pitchers found for these filters.")
        else:
            st.warning("No plays match the selected criteria.")
