from h2o_wave import main, app, Q, ui
import pandas as pd
import numpy as np


@app('/decisiontree')
async def serve(q: Q):
    if not q.client.initialized:
        initialize_app(q)
    
    id = q.args['#']
       
    if id == 'dt':
        q.page['decision'] = ui.form_card(box='1 12 5 5', items=[ui.progress('Running...')])
        value = await q.run(dt_training,train,test)
        ui.message_bar(type='success', text="Model successfully trained!"),

        q.page['decision'] = ui.form_card(box='1 12 5 5', items=[
            ui.message_bar('info', message),
            ui.text(make_markdown_table(fields=value.columns.tolist(),rows=value.values.tolist()))
            ])
    await q.page.save()   
    
    if id:
        if id == 'show_data':
            q.page['show_data'].items=[
                ui.text(make_markdown_table(fields=df.columns.tolist(),rows=df.values.tolist()))
            ]
        elif id == 'preprocess':
            q.page['preprocess'].items=[
                 ui.text(make_markdown_table(fields=train.columns.tolist(),rows=train.values.tolist()))
            ]
             
        elif id == 'train':
            q.page['train'] = ui.form_card(
            box='4 10 4 2',
            items=[
                ui.message_bar(type='warning', text="Click on here to train the model"),
                ui.button(name='#dt', label='Train with Decision Tree', primary=True),
            ])
         
    else:
        q.page['nav'] = ui.tab_card(
            box='5 2 7 1',
            items=[
                ui.tab(name='#show_data', label='Show Raw Data'),
                ui.tab(name='#preprocess', label='Preprocessed Data'),
                ui.tab(name='#train', label='Train & Predict with DecisionTreeClassifier'),
               
            ],
        )  
        q.page['show_data'] = ui.form_card(
        box='1 3 9 4',
        items=[
            ui.text('Display Raw data here !'),
            ui.message_bar(type='info', text="To display the raw data Click on Show Raw Data"),
            ])    

        q.page['preprocess'] = ui.form_card(
        box='1 7 9 4',
        items=[
            ui.text('Preprocessing of data'),
            ui.message_bar(type='info', text="To display the preprocessed data Click on preprocessed Data"),
            
        ])
         
        await q.page.save()


def initialize_app(q):
 

    q.page['header'] = ui.header_card(
        box='1 1 11 1',
        title='Health Insurance Lead Prediction',
        subtitle='Tain and test your data with decisiontree!',
    )
    
#load trainning data
def load_data():
    data = pd.read_csv('train.csv')
    return data

#load testing data
def test_data():
    data = pd.read_csv('test.csv')
    return data
    
    
data = load_data()
test_data = test_data()

df =  data.loc[:200,:]

#Data preprocessing 
def preprocessing(data):
    data = data.drop(['ID'],axis=1)
    data = pd.get_dummies(data, columns=['Accomodation_Type','Reco_Insurance_Type','Is_Spouse'],drop_first=True) 
    data['City_Code'] = data['City_Code'].replace(to_replace='C',value='',regex=True)
    data['City_Code'] = data['City_Code'].astype(np.int64)
    
    data['Holding_Policy_Duration'] = data['Holding_Policy_Duration'].replace(to_replace='14+',value=15,regex=True)
    data['Holding_Policy_Duration'] = pd.to_numeric(data['Holding_Policy_Duration'])
    data['Holding_Policy_Duration'] = data['Holding_Policy_Duration'].fillna(data['Holding_Policy_Duration'].median())
    
    data['Health Indicator'] = data['Health Indicator'].replace(to_replace='X',value='',regex=True)
    data['Health Indicator'] = data['Health Indicator'].astype(np.float64)
    
    data  = data.fillna(data.mean())
    data['Health Indicator']  = data['Health Indicator'].fillna(data['Health Indicator'].mean())
    data = data.drop(['Upper_Age'],axis=1)
    return data


#preprocessing the both train and test data
train = preprocessing(data)
test = preprocessing(test_data)



def dt_training(train,test):
    y = train.iloc[:,10].values
    train = train.drop(['Response'],axis=1)
    train = train.iloc[:,:].values
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state = 0)
    dt.fit(train, y)  
    y_pred = dt.predict(test)
    y_pred = pd.DataFrame(y_pred,columns=['DT Predictions'])
    return y_pred


def make_markdown_row(values):
    return f"| {' | '.join([str(x) for x in values])} |"


def make_markdown_table(fields, rows):
    return '\n'.join([
        make_markdown_row(fields),
        make_markdown_row('---' * len(fields)),
        '\n'.join([make_markdown_row(row) for row in rows]),
    ])



 