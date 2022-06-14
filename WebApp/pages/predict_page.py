

import streamlit as st
import pickle

import numpy as np
import pandas as pd
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import accuracy_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from PIL import Image



#def load_model():
#    with open('saved_steps.pkl', 'rb') as file:
#        data = pickle.load(file)
#    return data

#data = load_model()

#model_loaded = data['model']
#scaler_loaded = data['scaler']
#lblEncoder_state = data['lblEncoder_state']
#lblEncoder_cons = data['lblEncoder_cons']
#lblEncoder_name = data['lblEncoder_name']
#lblEncoder_party = data['lblEncoder_party']
#lblEncoder_symbol = data['lblEncoder_symbol']
#lblEncoder_gender = data['lblEncoder_gender']


def show_predict_page():

    #title = st.title("U.S. Presidential Election Prediction Model (1976 - 2020)")
    results = st.container()
    actualResults = st.expander("View the actual election results")

    data = pd.read_csv('1976-2020-president.csv')
    electoral_df = pd.read_csv('Electoral_College.csv')

    

    years = [1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]


    states = ['ALABAMA',
    'ALASKA',
    'ARIZONA',
    'ARKANSAS',
    'CALIFORNIA',
    'COLORADO',
    'CONNECTICUT',
    'DELAWARE',
    'DISTRICT OF COLUMBIA',
    'FLORIDA',
    'GEORGIA',
    'HAWAII',
    'IDAHO',
    'ILLINOIS',
    'INDIANA',
    'IOWA',
    'KANSAS',
    'KENTUCKY',
    'LOUISIANA',
    'MAINE',
    'MARYLAND',
    'MASSACHUSETTS',
    'MICHIGAN',
    'MINNESOTA',
    'MISSISSIPPI',
    'MISSOURI',
    'MONTANA',
    'NEBRASKA',
    'NEVADA',
    'NEW HAMPSHIRE',
    'NEW JERSEY',
    'NEW MEXICO',
    'NEW YORK',
    'NORTH CAROLINA',
    'NORTH DAKOTA',
    'OHIO',
    'OKLAHOMA',
    'OREGON',
    'PENNSYLVANIA',
    'RHODE ISLAND',
    'SOUTH CAROLINA',
    'SOUTH DAKOTA',
    'TENNESSEE',
    'TEXAS',
    'UTAH',
    'VERMONT',
    'VIRGINIA',
    'WASHINGTON',
    'WEST VIRGINIA',
    'WISCONSIN',
    'WYOMING']


    parties = ['DEMOCRAT',
    'REPUBLICAN',
    'AMERICAN INDEPENDENT PARTY',
    'PROHIBITION',
    'COMMUNIST PARTY USE',
    'LIBERTARIAN',
    'INDEPENDENT',
    'SOCIALIST WORKERS',
    'AMERICAN',
    'PEACE & FREEDOM',
    'U.S. LABOR',
    'NO PARTY AFFILIATION',
    'SOCIALIST LABOR',
    'AMERICAN PARTY OF IOWA',
    'SOCIALIST U.S.A.',
    'CONSERVATIVE',
    'HUMAN RIGHTS',
    "MCCARTHY '76",
    "PEOPLE'S",
    'INTERNATIONAL DEVELOPMENT BANK',
    'INDUSTRIAL GOVERNMENT PARTY',
    'SOCIALIST',
    'LIBERAL PARTY',
    'FREE LIBERTARIAN',
    'LABOR',
    'CONSTITUTION PARTY',
    'CONCERNED CITIZENS',
    'STATESMAN',
    'CITIZENS',
    'WORKERS WORLD',
    'NATIONAL UNITY CAMPAIGN',
    'ANDERSON COALITION',
    'NOMINATED BY PETITION',
    'RESPECT FOR LIFE',
    'RIGHT-TO-LIFE',
    'MIDDLE CLASS CANDIDATE',
    'DOWN WITH LAWYERS',
    "NATURAL PEOPLE'S LEAGUE",
    'POPULIST',
    'ALLIANCE',
    'UNITED SOVEREIGN CITIZENS',
    'WORKERS LEAGUE',
    'BIG DEAL PARTY',
    'NATIONAL UNITY',
    'NEW ALLIANCE',
    'NATIONAL ECONOMIC RECOVERY',
    'THIRD WORLD ASSEMBLY',
    'SOLIDARITY',
    'PATRIOTIC PARTY',
    'OTHER',
    'PROGRESSIVE',
    'GRASSROOTS',
    'CONSUMER',
    'WRITE-IN',
    'UNITED CITIZENS',
    'LIBERTY UNION PARTY',
    'AMERICA FIRST',
    'INDEPENDENTS FOR ECONOMIC RECOVERY',
    'NATURAL LAW',
    'TAXPAYERS PARTY',
    'DEMOCRAT/REPUBLICAN',
    'UNAFFILIATED',
    'CAMPAIGN FOR A NEW TOMORROW',
    'EQUAL JUSTICE AND OPPORTUNITY',
    'MORE PERFECT DEMOCRACY',
    'JUSTICE, INDUSTRY, AND AGRICULTURE',
    'INDEPENDENT VOTERS',
    'LAROUCHE FOR PRESIDENT PARTY',
    'SOCIALIST PARTY USA',
    'TISCH INDEPENDENT CITIZENS',
    'INDEPENDENT AMERICAN',
    '6 MILLION JOBS',
    'RON DANIELS INDEPENDENT',
    'AMERICA FIRST POPULIST',
    'INDEPENDENTS FOR LAROUCHE',
    'FREEDOM FOR LAROUCHE',
    'LABOR-FARM-LABORISTA-AGRARIO',
    'THIRD PARTY',
    'REFORM PARTY',
    'GREEN',
    'U.S. TAXPAYERS PARTY',
    'LOOKING BACK PARTY',
    'LIBERTY, ECOLOGY, COMMUNITY',
    'SOCIALIST EQUALITY PARTY',
    'INDEPENDENT GRASSROOTS',
    'INDEPENDENCE',
    'FREEDOM',
    'INDEPENDENT NOMINATION',
    'PATRIOT PARTY',
    'UNENROLLED',
    'DEMOCRATIC-FARMER-LABOR',
    'REFORM PARTY MINNESOTA',
    'CITIZENS FIRST',
    'WORKING FAMILIES',
    'PROGRESSIVE/GREEN',
    'VERMONT GRASSROOTS',
    'ALASKAN INDEPENDENCE PARTY',
    'AMERICAN CONSTITUTION PARTY',
    'CONCERNS OF PEOPLE',
    'PETITIONING CANDIDATE',
    'D.C. STATEHOOD GREEN',
    'CONSTITUTION PARTY OF FLORIDA',
    'SOCIALIST PARTY OF FLORIDA',
    'PROTECTING WORKING FAMILIES',
    'GREEN-RAINBOW',
    'BETTER LIFE',
    'CHRISTIAN FREEDOM PARTY',
    'NEBRASKA PARTY',
    'PEACE AND JUSTICE',
    'PACIFIC GREEN',
    'NONPARTISAN',
    'PERSONAL CHOICE',
    'WISCONSIN GREEN',
    'SOCIALISM AND LIBERATION PARTY',
    "AMERICA'S INDEPENDENT PARTY",
    'BOSTON TEA PARTY',
    "HEARTQUAKE '08",
    'OBJECTIVIST PARTY',
    'U.S. PACIFIST PARTY',
    'ECOLOGY PARTY OF FLORIDA',
    'NEW',
    'LOUISIANA TAXPAYERS PARTY',
    'VOTE HERE',
    'PEACE PARTY',
    'INDEPENDENT GREEN',
    'MOUNTAIN PARTY',
    'WE THE PEOPLE',
    'JUSTICE',
    "AMERICA'S PARTY",
    'AMERICAN THIRD POSITION',
    'AMERICAN INDEPENDENT',
    'CONSTITUTIONAL GOVERNMENT',
    'NSA DID 911',
    'NEW MEXICO INDEPENDENT PARTY',
    'AMERICAN DELTA PARTY',
    'BETTER FOR AMERICA',
    'VETERANS PARTY OF AMERICA',
    'INDEPENDENT PEOPLE OF COLORADO',
    'AMERICAN SOLIDARITY PARTY',
    'NUTRITION PARTY',
    'KOTLIKOFF FOR PRESIDENT',
    'NONVIOLENT RESISTANCE/PACIFIST',
    'APPROVAL VOTING PARTY',
    'NEW INDEPENDENT PARTY IOWA',
    'LEGAL MARIJUANA NOW',
    'WORKERS WORLD PARTY',
    "WOMEN'S EQUALITY",
    'PARTY FOR SOCIALISM AND LIBERATION',
    'LIFE AND LIBERTY PARTY',
    'UNITY PARTY',
    'PROHIBITION PARTY',
    'PROGRESSIVE PARTY',
    'SOCIALIST WORKERS PARTY',
    'INDEPENDENT AMERICAN PARTY',
    'NON-AFFILIATED',
    'DC STATEHOOD GREEN',
    'AMERICAN SHOPPING PARTY',
    'GENEOLOGY KNOW YOUR FAMILY HISTORY PARTY',
    'BECOMING ONE NATION',
    'C.U.P',
    'FREEDOM AND PROSPERITY',
    'LIFE , LIBERTY, CONSTITUTION',
    'THE BIRTHDAY PARTY',
    'BREAD AND ROSES',
    'US TAXPAYERS PARTY',
    'GRUMPY OLD PATRIOTS',
    'AMERICAN SOLIDARITY',
    'CONSTITUTION',
    'LIBERTY UNION',
    'BOILING FROG',
    'BULL MOOSE',
    'APPROVAL VOTING']

    writeinops = [False, True]

    officeop = ['US PRESIDENT']

    partyChoices = ['DEMOCRAT','REPUBLICAN','OTHER']


    year = st.slider("Election Year",1992,2020,step=4)

    #state = st.selectbox("State",states)

    party = st.selectbox("Party",partyChoices)

    #writein = st.selectbox("Write In", writeinops)

    office = st.selectbox("Office", officeop)

    yes = st.button("Predict")

    if yes:

        X = ['state','state_cen','state_ic','party_detailed','office','writein']
        y = ['winner']

        #data['state'] = lblEncoder_state.transform(data['state'])

        #data['state_cen'] =lblEncoder_cons.transform(data['state_cen'])

        #data['state_ic'] =lblEncoder_name.transform(data['state_ic'])

        #data['party_detailed'] =lblEncoder_party.transform(data['party_detailed'])

        #data['office'] =lblEncoder_symbol.transform(data['office'])

        #data['writein'] =lblEncoder_gender.transform(data['writein'])

        lblEncoder_state = LabelEncoder()
        lblEncoder_state.fit(data['state'])
        data['state'] = lblEncoder_state.transform(data['state'])

        lblEncoder_cons = LabelEncoder()
        lblEncoder_cons.fit(data['state_cen'])
        data['state_cen'] = lblEncoder_cons.transform(data['state_cen'])

        lblEncoder_name = LabelEncoder()
        lblEncoder_name.fit(data['state_ic'])
        data['state_ic'] = lblEncoder_name.transform(data['state_ic'])

        lblEncoder_party = LabelEncoder()
        lblEncoder_party.fit(data['party_detailed'])
        data['party_detailed'] = lblEncoder_party.transform(data['party_detailed'])

        lblEncoder_symbol = LabelEncoder()
        lblEncoder_symbol.fit(data['office'])
        data['office'] = lblEncoder_symbol.transform(data['office'])

        lblEncoder_gender = LabelEncoder()
        lblEncoder_gender.fit(data['writein'])
        data['writein'] = lblEncoder_gender.transform(data['writein'])

        y = ['winner']
        X = ['state','state_cen','state_ic','party_detailed','office','writein']
        # split dataset into train and test data

        #beg = dataf[0:2730]#dataf[0:2044]
        #end =dataf[2730:] #dataf[2413:]
        #frames = [beg,end]
        #train_df = pd.concat(frames)
        
        #train_df = data[data['year'] <= year]
        train_df = data[data['year'] != year]

        #if year >= 2000:
        #    train_df = data[data['year'] != year]
        #else:
        #    otherdata = data[data['year'] < 2000]
        #    train_df = otherdata[otherdata['year'] != year]

        #loyear = year - 12
        #add_data= data[data['year'] >= loyear]
        #train_df = add_data[add_data['year'] != year]

        test_df = data[data['year'] == year]
        
        #dataf[2044:2413]
        #train_test_split(dataf, test_size=0.3)

        X_train = train_df[X]
        y_train = train_df[y]
        X_test = test_df[X]
        y_test = test_df[y]

        #preData = data[data['year'] == year]

        #inpDataScaled = scaler_loaded.transform(preData[X])

        #Xtrain_scaled = scaler_loaded.fit_transform(X_train)
        #Xtest_scaled = scaler_loaded.transform(X_test)
        #model_loaded.fit(Xtrain_scaled, y_train)

        scaler = StandardScaler()
        Xtrain_scaled = scaler.fit_transform(X_train)
        Xtest_scaled = scaler.transform(X_test)
        rf_model = RandomForestRegressor(n_estimators = 200, min_samples_split = 10, max_features = 'auto')
        rf_model.fit(Xtrain_scaled, y_train)

        prediction = rf_model.predict(Xtest_scaled)

        alteredPred = []

        for x in prediction: 
            alteredPred.append(int(x.round()))


        df = data[data['year']== year]
        df['pred_winners'] = alteredPred
        df2 = df[df['winner']==1]

        df3 = df[df['pred_winners']==1]

        predDems = df3[df3['party'] == 'DEMOCRAT']
        predReps = df3[df3['party'] == 'REPUBLICAN']

        frames = [predDems, predReps]
        fPreds = pd.concat(frames)

        actDems = df2[df2['party'] == 'DEMOCRAT']
        actReps = df2[df2['party'] == 'REPUBLICAN']

        frames = [actDems, actReps]
        fActual = pd.concat(frames)

        import plotly.express as px

        fig = px.choropleth(fActual,
                            locations='state_po', 
                            locationmode="USA-states", 
                            scope="usa",
                            color='candidate'
                            #color_continuous_scale="magma", 
                            )

        fig.update_layout(
                title_text = str(year) + ' U.S. Presidential Election Actual Results',
                title_font_family="Tahoma",
                title_font_size = 22,
                title_font_color="white", 
                title_x=0.45, 
                )

        fig2 = px.choropleth(fPreds,
                            locations='state_po', 
                            locationmode="USA-states", 
                            scope="usa",
                            color='candidate',
                            color_continuous_scale="Viridis.reverse", 
                            )

        fig2.update_layout(
                title_text = str(year) + ' U.S. Presidential Election Prediction Results',
                title_font_family="Tahoma",
                title_font_size = 22,
                title_font_color="white", 
                title_x=0.45, 
                )
        
        fPreds2 = fPreds[fPreds['winner']==1]
        fPreds3 = fPreds2[fPreds2['party_detailed']=='DEMOCRAT']

        average = 80
        val = round(100*accuracy_score(y_test, alteredPred),2)

        col1,col2 = st.sidebar.columns(2)

        with col1:
            st.metric(label='Accuracy',value=str(val)+'%',delta = str(round(val-average,2)) + '%'+ ' from AVG')
        with col2:
            st.metric(label="Mean Absolute Error",value= round(mean_absolute_error(y_test, alteredPred),3))
        
        #st.sidebar.title(str(year) + ' U.S. Presidential Election State Predictions')
        
        st.sidebar.metric(label='Year',value=year)

        #st.sidebar.metric(label='Office',value=officeop[0])
        #st.sidebar.markdown('<font color="#dab844">'+str(year) + ' U.S. Presidential Election State Predictions</font>',unsafe_allow_html=True)
        #st.sidebar.title(str(round(100*accuracy_score(y_test, alteredPred),2)) + '% Accuracy')
        
        
        demCandNamePred = np.array(predDems['candidate'])[1]
        repCandNamePred = np.array(predReps['candidate'])[1]

        #st.sidebar.subheader(str(len(predDems))+' states won', anchor = demCandNamePred)

        
        #st.sidebar.subheader(str(len(predReps))+' states won', anchor = repCandNamePred)
        

        
        organizedPreds = fPreds[['state_po','candidate']].sort_values('state_po',axis=0,ascending=True)

        organizedActual = fActual[['state_po','candidate']].sort_values('state_po',axis=0,ascending=True)

        
        electoralvotes = electoral_df['electoral_votes']

        arr = []
        for x in electoral_df['electoral_votes']:
            arr.append(x)

        organizedPreds['electoral_votes'] = arr
        organizedActual['electoral_votes'] = arr

        common = 0

        for x,y in zip(organizedPreds['candidate'],organizedActual['candidate']):
            if x==y:
                common = common +1

        demVotesPred = 0
        repVotesPred = 0
        demVotesActual = 0
        repVotesActual = 0

        for x,y in zip(organizedPreds['candidate'],organizedPreds['electoral_votes']):
            if x == demCandNamePred:
                demVotesPred = demVotesPred + y 
            if x == repCandNamePred:
                repVotesPred = repVotesPred + y 

        for x,y in zip(organizedActual['candidate'],organizedActual['electoral_votes']):
            if x == demCandNamePred:
                demVotesActual = demVotesActual + y 
            if x == repCandNamePred:
                repVotesActual = repVotesActual + y 

        st.sidebar.metric(label=demCandNamePred ,value=str(demVotesPred)+' Electoral Votes', delta = str(demVotesPred-demVotesActual) + ' from actual')
        st.sidebar.metric(label = repCandNamePred ,value = str(repVotesPred)+' Electoral Votes', delta= str(repVotesPred-repVotesActual) + ' from actual')

        
        if demVotesPred > repVotesPred:
            winner = demCandNamePred
        else:
            winner = repCandNamePred

        if demVotesActual > repVotesActual:
            winnerActual = demCandNamePred
        else:
            winnerActual = repCandNamePred

   
        st.sidebar.metric(label='Winner', value = winner, delta = bool(winner == winnerActual),delta_color= "off")

        #st.sidebar.metric(label=demCandNamePred,value=str(len(predDems))+' states won', delta=str(len(predDems)-len(actDems))+' from actual')
    
        #st.sidebar.metric(label=repCandNamePred,value=str(len(predReps))+' states won',delta=str(len(predReps)-len(actReps))+' from actual')
        
        st.sidebar.metric(label='Incorrect Predictions',value=str(len(organizedPreds)-common) + ' states')

        expander = st.sidebar.expander('View State Predictions')

        expander.table(organizedPreds[['state_po','candidate']])
        #results.subheader(str(year)+' U.S. Presidential Election Prediction')
        results.plotly_chart(fig2, use_container_width=True, sharing="streamlit")
        
        #actualResults.subheader(str(year)+' U.S. Presidential Election Actual Results')
        actualResults.plotly_chart(fig, use_container_width=False, sharing="streamlit")

        #st.sidebar.header(str(year) + ' U.S. Presidential Election Actual')
        #demCandNameAct = st.sidebar.header(np.array(actDems['candidate'])[1])
        #st.sidebar.subheader(str(len(predDems))+' states won', anchor = demCandNameAct)

        #repCandNameAct = st.sidebar.header(np.array(actReps['candidate'])[1])
        #st.sidebar.subheader(str(len(actReps))+' states won', anchor = repCandNameAct)

       # st.plotly_chart(fig2, use_container_width=False, sharing="streamlit")
       # st.plotly_chart(fig, use_container_width=False, sharing="streamlit")


