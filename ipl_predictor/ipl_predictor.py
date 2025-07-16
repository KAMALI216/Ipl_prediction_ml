import streamlit as st
import pandas as pd
import pickle
import io

# Load model and encoders
model = pickle.load(open('xgb_model.pkl', 'rb'))
team_encoder = pickle.load(open('team_encoder.pkl', 'rb'))
venue_encoder = pickle.load(open('venue_encoder.pkl', 'rb'))
team_batting_strength = pd.read_csv('team_batting_strength.csv')
home_away = pd.read_csv('home_away.csv')
head2head = pd.read_csv('head_to_head.csv')  # Optional: for head-to-head stats

# Aliases
team_aliases = {
    'MI': 'Mumbai Indians',
    'RCB': 'Royal Challengers Bangalore',
    'CSK': 'Chennai Super Kings',
    'DC': 'Delhi Capitals',
    'KKR': 'Kolkata Knight Riders',
    'SRH': 'Sunrisers Hyderabad',
    'GT': 'Gujarat Titans',
    'LSG': 'Lucknow Super Giants',
    'RR': 'Rajasthan Royals',
    'PBKS': 'Punjab Kings',
}

# Team logos
team_logos = {
    'MI': 'ipl_images/mi.jpg',
    'RCB': 'ipl_images/RCB.png',
    'CSK': 'ipl_images/csk.jpg',
    'DC': 'ipl_images/dc.jpg',
    'KKR': 'ipl_images/kkr.jpg',
    'SRH': 'ipl_images/srh.jpg',
    'GT': 'ipl_images/GT.png',
    'LSG': 'ipl_images/lsg.jpg',
    'RR': 'ipl_images/rr.jpg',
    'PBKS': 'ipl_images/pbks.jpg',
}

def show_team_logo(team_code):
    logo_path = team_logos.get(team_code)
    if logo_path:
        st.image(logo_path, width=100)

def get_head_to_head_stats(t1, t2):
    match = head2head[((head2head['team1'] == t1) & (head2head['team2'] == t2)) |
                      ((head2head['team1'] == t2) & (head2head['team2'] == t1))]
    if not match.empty:
        row = match.iloc[0]
        team1_wins = row['team1_wins'] if row['team1'] == t1 else row['team2_wins']
        team2_wins = row['team2_wins'] if row['team2'] == t2 else row['team1_wins']
        st.markdown(f"📊 **Head-to-Head Stats**")
        st.write(f"Total Matches: {row['matches']}")
        st.write(f"{team_aliases[t1]} Wins: {team1_wins}")
        st.write(f"{team_aliases[t2]} Wins: {team2_wins}")


# Main prediction logic
def predict_winner(team1, team2, toss_winner, toss_decision, venue):
    team1_full = team_aliases[team1]
    team2_full = team_aliases[team2]
    toss_winner_full = team_aliases[toss_winner]

    t1 = team_encoder.transform([team1_full])[0]
    t2 = team_encoder.transform([team2_full])[0]
    toss = team_encoder.transform([toss_winner_full])[0]
    toss_dec = 0 if toss_decision == 'bat' else 1
    venue_encoded = venue_encoder.transform([venue])[0]

    t1_bat = team_batting_strength[team_batting_strength['team'] == t1]['batting_strength'].values[0] if t1 in team_batting_strength['team'].values else 130
    t2_bat = team_batting_strength[team_batting_strength['team'] == t2]['batting_strength'].values[0] if t2 in team_batting_strength['team'].values else 130
    t1_home = home_away[home_away['team'] == t1]['home_win_percentage'].values[0] if t1 in home_away['team'].values else 50
    t2_away = home_away[home_away['team'] == t2]['away_win_percentage'].values[0] if t2 in home_away['team'].values else 50

    input_df = pd.DataFrame([[
        t1, t2, toss, toss_dec, venue_encoded,
        t1_home, t2_away, t1_bat, t2_bat
    ]], columns=[
        'team1', 'team2', 'toss_winner', 'toss_decision', 'venue',
        'team1_home_win_pct', 'team2_away_win_pct',
        'team1_batting_strength', 'team2_batting_strength'
    ])

    pred_label = model.predict(input_df)[0]
    pred_team = team_encoder.inverse_transform([pred_label])[0]
    confidence = round(100 * model.predict_proba(input_df)[0][pred_label], 2)

    return pred_team, confidence


# Streamlit UI
st.set_page_config(page_title="IPL Predictor", page_icon="🏏")
st.markdown("<h1 style='text-align: center;'>🏏 IPL Winning Team Predictor</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

team_options = list(team_aliases.keys())
venue_options = sorted(venue_encoder.classes_)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("🔷 Select Team 1", team_options)
        toss_winner = st.selectbox("🎲 Toss Winner", team_options)
    with col2:
        team2 = st.selectbox("🔶 Select Team 2", team_options)
        toss_decision = st.radio("🧠 Toss Decision", ["bat", "field"])

venue = st.selectbox("📍 Venue", venue_options)

if st.button("🎯 Predict Winner"):
    if team1 == team2:
        st.error("Please select different teams.")
    else:
        winner, confidence = predict_winner(team1, team2, toss_winner, toss_decision, venue)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            show_team_logo(team1)
            st.markdown(f"**{team_aliases[team1]}**")
        with col2:
            show_team_logo(team2)
            st.markdown(f"**{team_aliases[team2]}**")

        get_head_to_head_stats(team1, team2)

        st.success(f"🏆 **Predicted Winner:** {winner}")
        st.info(f"🤖 **Model Confidence:** {confidence}%")

        # Export button
        if st.button("📥 Export Result (Excel)"):
            result_df = pd.DataFrame([{
                'Team 1': team_aliases[team1],
                'Team 2': team_aliases[team2],
                'Toss Winner': team_aliases[toss_winner],
                'Toss Decision': toss_decision,
                'Venue': venue,
                'Predicted Winner': winner,
                'Confidence': f"{confidence}%"
            }])

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, index=False, sheet_name='Prediction')
                writer.save()

            st.download_button(
                label="📊 Download Excel File",
                data=buffer,
                file_name='prediction_result.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
