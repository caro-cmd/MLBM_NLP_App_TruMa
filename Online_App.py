# import packages
import zipfile
import streamlit as st
import pandas as pd
import numpy as np

# nlp packages
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import string
import emoji
import nltk 
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup


# machine learning packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

#visualization packages
import plotly.express as px
import altair as alt
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt


# title of application
st.title('Machine Learning Web Application for Text')

# ----------- underlying functions for streamlit application --------------
# function to return dataframe
def return_df(file):
    name = file.name
    extension = name.split('.')[-1]  # get the extension of the file

    if extension == 'csv':
        df = pd.read_csv(file)
    elif extension == 'tsv':
        df = pd.read_csv(file, sep='\t')
    elif extension == 'json':
        df = pd.read_json(file)
    elif extension == 'xlsx':
        df = pd.read_excel(file)
    elif extension == 'xml':
        df = pd.read_xml(file)
    elif extension == 'zip':
        # handle ZIP file containing .txt files
        with zipfile.ZipFile(file, 'r') as zip:
            documents = []
            filenames = []

            # traverse through the ZIP file recursively to get all .txt files
            for file_info in zip.infolist():
                # skip the file if its a macosx file or a directory
                if file_info.is_dir() or file_info.filename.startswith('__MACOSX') or file_info.filename.startswith('.'):
                    continue 
                if file_info.filename.endswith('.txt'):
                    with zip.open(file_info.filename) as txt:
                        try:
                            # try reading as utf-8 first
                            content = txt.read().decode('utf-8')
                        except UnicodeDecodeError:
                            # if utf-8 fails, try reading as utf-8 with errors ignored
                            content = txt.read().decode('utf-8', errors='ignore')
                                
                        documents.append(content)
                        filenames.append(file_info.filename)
                

            # create a DataFrame with filenames and corresponding document content
            df = pd.DataFrame({
                'Filename': filenames,
                'Content': documents
            })

    return df 

# function to upload single file for for single dataset which will be used to perform train-test split
def file_upload_single(upload_type='single'):
    file = st.file_uploader('Please upload the dataset', type=['csv', 'tsv', 'json', 'xlsx', 'xml', 'zip'], key='single', accept_multiple_files=False)  # upload of single file
    df = None
    
    if file:
        df = return_df(file)
        st.success('File uploaded successfully')

    return df

# function to upload multiple files for train and test datasets seperately
def file_upload_multiple(upload_type='multiple'):
    files = st.file_uploader('Please upload the dataset ', type=['csv', 'tsv', 'json', 'xlsx', 'xml', 'zip'], key='multiple', accept_multiple_files=True)  # user can use file uploader
    train_df = None
    test_df = None

    if files:
        #df = return_df(file)
        st.success('Files uploaded successfully')

        files_dict = {file.name: file for file in files}

        #let user choose which file is the train and which is the test
        train = st.selectbox('Select Train Data', options=['Select File'] +list(files_dict.keys()))
        test = st.selectbox('Select Test Data', options=['Select File'] +list(files_dict.keys()))

        if train != 'Select File' and test != 'Select File':
            train_df = return_df(files_dict[train])
            test_df = return_df(files_dict[test])
    
    return train_df, test_df    

# function to fetch example data from URLs with a loading spinner
def fetch_data_with_progress(urls):
    df = []
    
    for i, url in enumerate(urls):
        st.write(f'Fetching data from: {url}')
        with st.spinner(f'Loading data from {url}...'):
            data = pd.read_csv(url)
            df.append(data)
    return df


# function to select target variable
def target_selection(df):
    target = None
    # check that the dataframe is not empty
    if df is not None:
        # select target
        target = st.selectbox('Select Target Variable', df.columns,help='Select the target variable for the machine learning model. This is the variable that the model will predict.')
    
    return target

def text_column_selection(df):
    text_col = None
    # check that the dataframe is not empty
    if df is not None:
        # select target
        text_col = st.selectbox('Select Text Column', df.columns,index=1,help='Select the text column that contains the text data for the machine learning model. This is the column that will be used to train the model.')
    
    return text_col

# function to split data into train and test sets
def split_data(data, target, test_size, ran_state):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=ran_state)
    return X_train, X_test, y_train, y_test

# function to remove HTML tags from text
def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()


# function to preprocess text
def preprocess_text(df, col, clean_text, lowercase, remove_punctuation, remove_numbers, remove_stopwords, remove_emojis, remove_spaces,remove_html,remove_html_entities, stemming, tokenize, tokenization_method):
    
    # drop rows with missing values in the selected column
    df = df.dropna(subset=[col])
    df = df[df[col].str.strip().astype(bool)]
    
    df[col] = df[col].astype(str)
    
    # text cleaning
    if clean_text:
        if lowercase:
            df[col] = df[col].str.lower() # convert to lowercase
            print(df[col])
            
        if remove_html:
            df[col] = df[col].apply(lambda x: remove_html_tags(x)) # remove html tags
        
        if remove_html_entities:
            import html
            df[col] = df[col].apply(lambda x: html.unescape(x)) # remove html entities
            
        if remove_punctuation:
            df[col] = df[col].str.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
            
        if remove_numbers:
            df[col] = df[col].str.replace(r'\d+', '', regex=True) # remove numbers
            
        if remove_stopwords:
            nltk.download('stopwords')
            stopwords_list = set(stopwords.words('english'))
            df[col] = df[col].apply(lambda x: ' '.join(word for word in x.split() if word not in stopwords_list)) # remove stopwords
            
        
        if remove_emojis:
            df[col] = df[col].apply(lambda x: ''.join(c for c in x if not emoji.is_emoji(c))) # remove emojis

        if remove_spaces:
            df[col] = df[col].apply(lambda x: ' '.join(x.split())) # remove extra spaces

        if stemming:
            stemmer = PorterStemmer()
            df[col] = df[col].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split())) # apply stemming

    # tokenization 
    if tokenize:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        if tokenization_method == 'Word Tokenization':
            df[col] = df[col].apply(lambda x: word_tokenize(x))
        elif tokenization_method == 'Sentence Tokenization':
            df[col] = df[col].apply(lambda x: sent_tokenize(x))
        elif tokenization_method == 'Tweet Tokenization':
            tknzr = TweetTokenizer()
            df[col] = df[col].apply(lambda x: tknzr.tokenize(x))
        
        # convert list of tokens to string
        df[col] = df[col].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    
    return df


# function to vectorize text data
def vectorizer(X_train, X_test, col, vectorizer_method, model_name=None):
    
    # check which vectorizer method is choosen
    if vectorizer_method == 'Bag-of-Words':
        vect = CountVectorizer()
        X_train_vec = vect.fit_transform(X_train[col])
        X_test_vec = vect.transform(X_test[col])
        
    elif vectorizer_method == 'TF-IDF':
        vect = TfidfVectorizer()
        X_train_vec = vect.fit_transform(X_train[col])
        X_test_vec = vect.transform(X_test[col])
        
    elif vectorizer_method == 'Sentence Transformer':
        if model_name is None:
            raise ValueError('Model name must be provided for Sentence Transformer')
        # load the Sentence Transformer model
        model = SentenceTransformer(model_name)
        # encode the text data to get embeddings
        X_train_vec = model.encode(X_train[col].tolist())
        X_test_vec = model.encode(X_test[col].tolist())

    return X_train_vec, X_test_vec


# function for augmenting text data
def augmentation(X_train, y_train, target, aug_syn_repl, aug_rand_char):

    augmented_datasets = []

    # augmentation of both synonyms and random characters
    if aug_syn_repl and aug_rand_char:
        from nltk.corpus import wordnet
        # synonym Augmentation
        aug_syn = naw.SynonymAug(aug_src='wordnet')
        X_train_syn = X_train.apply(lambda x: aug_syn.augment(x) if isinstance(x, str) else x)
        augmented_datasets.append(pd.concat([X_train_syn, y_train], axis=1))
        
        # random Character Augmentation
        aug_char = nac.RandomCharAug(action='swap')
        X_train_char = X_train.apply(lambda x: aug_char.augment(x) if isinstance(x, str) else x)
        augmented_datasets.append(pd.concat([X_train_char, y_train], axis=1))

    # augmentation of synonyms only
    elif aug_syn_repl:
        aug_syn = naw.SynonymAug(aug_src='wordnet')
        X_train_syn = X_train.apply(lambda x: aug_syn.augment(x) if isinstance(x, str) else x)
        augmented_datasets.append(pd.concat([X_train_syn, y_train], axis=1))

    # augmentation of random characters only
    elif aug_rand_char:
        aug_char = nac.RandomCharAug(action='swap')  
        X_train_char = X_train.apply(lambda x: aug_char.augment(x) if isinstance(x, str) else x)
        augmented_datasets.append(pd.concat([X_train_char, y_train], axis=1))

    # include original data
    augmented_datasets.append(pd.concat([X_train, y_train], axis=1))

    # combine all augmented datasets
    Train_full = pd.concat(augmented_datasets)

    # shuffle the combined dataset
    Train_full_shuffle = Train_full.sample(frac=1, random_state=42).reset_index(drop=True)

    # separate features and target
    X_train_all = Train_full_shuffle.drop(columns=[target])
    y_train_all = Train_full_shuffle[target]
    
    return X_train_all, y_train_all
   
#function that combines all text in the dataframe into a single string for WordCloud generation
def create_wordcloud(df, column):
    text = " ".join(df[column].dropna().values)
    wordcloud = WordCloud(width=800, height=400, background_color='#373737',colormap='Greens').generate(text)

    # Create a figure object to return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off') 
    return fig

# function to generate WordCloud for each label category and return figures
def create_wordcloud_by_label(df, text_column, target_column):
    wordclouds = {}
    # get unique labels from the target column and sort them numerically so that its ordered by label 0, label 1, etc.
    unique_labels = sorted(df[target_column].unique())
    
    # loop through each label and create a WordCloud
    for label in unique_labels:
        label_data = df[df[target_column] == label]
        fig = create_wordcloud(label_data, text_column)
        wordclouds[f"Label {label}"] = fig
    return wordclouds

# ----------- main streamlit application ---------------

# global variables
X_train = None
X_test = None
y_train = None
y_test = None
df = None
text_col = None
target = None
train = None
test = None
lowercase = None
remove_punctuation = None
remove_numbers = None
remove_stopwords = None
remove_emojis = None
remove_spaces = None
stemming = None
tokenize_text = None
tokenization_method = None
augment_text_options = None
synonym_replacement = None
random_char = None
vectorizing = None
model_name = None
apply_scaling = None
remove_html = None
remove_html_entities = None
df_clustering = None
parameter_grid = {}
param_cluster = {}

# create tabs for the two tasks 
tab1, tab2 = st.tabs(['Machine Learning', 'Clustering'])

# -------------------------------- Machine Learning Tab 1 --------------------------------

with tab1:
    # ................................. Data Upload ...............................
    with st.expander('Data Upload', expanded=True):
        data_upload = st.radio('Choose your Data ', ['Upload file', 'Use example data'],help='Select the data upload option. You can either upload your own data or use example data provided by the app.')

        st.divider()
        
        # ……………………………………………………………………… User Uploads Data ………………………………………………………………………
        if data_upload == 'Upload file':
            data_split = st.selectbox('Choose Data splitting Option ', options=['Select Option','Upload Separated Data', 'Perform Train-Test Split'],help='Select the data splitting option. You can either upload train and test data separately as two different files or perform train-test split on a single dataset.')
            
            # ~~~~~~~~~~~~~~~~ Upload Separated Data ~~~~~~~~~~~~~~~~~~
            if data_split == 'Upload Separated Data': 
                
                # file upload for train and test data
                train, test = file_upload_multiple(upload_type='multiple')

                if train is not None and test is not None:
                    # let user select target and text column
                    target = target_selection(train)
                    text_col = text_column_selection(train)

                    # make sure target and text column are different
                    if target == text_col:
                        st.error('Target and Text Column should be different')
            
            # ~~~~~~~~~~~~~~ Perform Train-Test Split ~~~~~~~~~~~~~~~~~~
            elif data_split == 'Perform Train-Test Split':
                
                # file upload for single file
                df = file_upload_single(upload_type='single')
                
                if df is not None:
                    # let user select target and text column
                    target = target_selection(df)
                    text_col = text_column_selection(df)

                    # make sure target and text column are different
                    if target == text_col:
                        st.error('Target and Text Column should be different')

                    # let user select train and testing
                    test_size = st.slider('Test Size', min_value=0.0, max_value=1.0, value=0.2, help='Select the proportion of the dataset to include in the test split.')
                    ran_state = st.number_input('Random State', min_value=0, max_value=100, value=42, help='The seed used by the random number generator for train-test split.')

        # ……………………………………………………………………… Use Example Data ………………………………………………………………………
        if data_upload == 'Use example data':
            
            st.write('Using example data')
    
            # URLs for example data
            url = [
                'https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv'
            ]
            
            # cache example data in session state in order to avoid fetching it multiple times when changing parameters
            if 'example_data' not in st.session_state:
                st.session_state.example_data = fetch_data_with_progress([
                    'https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv'
                ])[0]

            # access the cached example data
            df = st.session_state.example_data
            
            st.write('Dataframe Preview')
            st.dataframe(df.head())
            target = 'condition_label'
            text_col = 'medical_abstract'
            
            st.write('Target: condition_label')


            
    # ........................ Data Preprocessing Paramters ..........................
    
    with st.expander('Data Preprocessing', expanded=True):
        
        st.write('Choose the Data Preprocessing')
            
        # ……………………………………………………………………… Text Cleaning ………………………………………………………………………
        clean_text = st.checkbox('Clean Text', value=True,help='Select this option to clean the text data before training the model. This includes converting text to lowercase, removing punctuation, numbers, stopwords, emojis, extra spaces, HTML tags, and applying stemming.')
        if clean_text:
            st.write('Options for text cleaning:')
            
            # columns in order to be able to indent the following checkboxes
            col1, col2 = st.columns([0.01, 0.9]) 
            
            with col2:
                lowercase = st.checkbox('Convert to Lowercase', value=True, help='Select this option to convert all text to lowercase, e.g., "Hello" will be converted to "hello".')
                remove_punctuation = st.checkbox('Remove Punctuation', value=True, help='Select this option to remove all punctuation marks from the text, e.g., "Hello, World!" will be converted to "Hello World".')
                remove_numbers = st.checkbox('Remove Numbers', value=True, help='Select this option to remove all numbers from the text, e.g., "Hello 123" will be converted to "Hello".')
                remove_stopwords = st.checkbox('Remove Stopwords', value=True, help='Select this option to remove common stopwords from the text, e.g., "The quick brown fox" will be converted to "quick brown fox".')
                remove_emojis = st.checkbox('Remove Emojis', value=True, help='Select this option to remove emojis from the text, e.g., "Hello 😊" will be converted to "Hello".')
                remove_spaces = st.checkbox('Remove Extra Spaces', value=True, help='Select this option to remove extra spaces from the text, e.g., "Hello    World" will be converted to "Hello World".')
                remove_html = st.checkbox('Remove HTML Tags', value=True, help='Select this option to remove HTML tags from the text, e.g., "<p>Hello</p>" will be converted to "Hello".')
                remove_html_entities = st.checkbox('Decode HTML Entities', value=True, help='Select this option to decode HTML entities from the text, e.g., "&amp;" will be converted to "&".')
                stemming = st.checkbox('Apply Stemming', value=True, help='Select this option to apply stemming to the text, e.g., "running" will be converted to "run".')
                
            
        st.divider()
        
        # ……………………………………………………………………… Tokenization …………………………………………………………………………
        tokenize_text = st.checkbox('Tokenization', value=True, help='Select this option to tokenize the text data. This will split the text into words or sentences.')
        if tokenize_text:
            st.write('Options:')
            tokenization_method = st.selectbox('Select Tokenization Method', ['Word Tokenization', 'Sentence Tokenization', 'Tweet Tokenization'], help='Select the tokenization method to use for the text data. Word tokenization splits the text into words, sentence tokenization splits the text into sentences, and tweet tokenization is a special tokenization method for tweets.')
        else:
            tokenization_method = None
        st.divider()

        # ……………………………………………………………………… Data Augmentation ………………………………………………………………
        augment_text_options = st.checkbox('Data Augmentation', value=True, help='Select this option to augment the text data. This will generate new data samples by applying various augmentation techniques. This can help improve the model performance.')
        if augment_text_options:
            st.write('Data Augmentation Options:')
            synonym_replacement = st.checkbox('Synonym Replacement', value=False, help='Select this option to replace words in the text with their synonyms.')
            random_char = st.checkbox('Random Character', value=True, help='Select this option to randomly swap characters in the text.')
            
        st.divider()
        
        # ……………………………………………………………………… Vectorization ………………………………………………………………………
        vectorizing = st.selectbox('Select Vectorizer', ['Bag-of-Words', 'TF-IDF', 'Sentence Transformer'], index=1, help='Select the vectorization method to convert the text data into numerical features. Bag-of-Words and TF-IDF are traditional methods, while Sentence Transformer uses pre-trained models to generate embeddings for the text data.')
        if vectorizing == 'Sentence Transformer':
            st.write('Choose a Sentence Transformer model')
            model_name = st.selectbox('Model Name', ['all-MiniLM-L6-v2','bert-base-nli-mean-tokens','roberta-base-nli-stsb-mean-tokens','distilbert-base-nli-stsb-mean-tokens'], index=0, help='Select a Sentence Transformer model to use for generating embeddings for the text data.')
        
        st.divider()
        
        # ……………………………………………………………………… Scaling ………………………………………………………………………………………
        apply_scaling = st.checkbox('Apply Scaling', value=False, help='Select this option to apply scaling to the numerical features. This is important for some machine learning models that require scaled features. The Scaling method used is StandardScaler.')
        
        
    # .........................Model Selection & their Parameters .................
    
    with st.expander('Model Selection', expanded=True):

        # ……………………………………………………………………… Task Type ………………………………………………………………………
        task_type = st.radio('Select Task Type', ['Classification', 'Regression'], help='Select the type of machine learning task. Classification is used for predicting categories or labels, while regression is used for predicting continuous values.')

        # multiselect models depending on task type
        if task_type == 'Classification':
            model_options = ['Random Forest Classifier', 'Logistic Regression', 'Support Vector Machine']
        else:
            model_options = ['Random Forest Regressor', 'Support Vector Regressor (SVR)']

        # ……………………………………………………………………… Model Selection ………………………………………………………………………
        # user can choose models based on the task type 
        selected_models = st.multiselect('Select Models to Train', model_options, default=model_options[0], help='Select the machine learning models to train on the data. You can select multiple models for comparison. ')

        # ………………………………………choose parameters manually or use GridSearch……………………………
        param_config_method = st.radio('Select Parameter Configuration Method', ['Manual', 'GridSearch'], help='Select the method to configure the parameters for the selected models. Manual allows you to set the parameters manually, while GridSearch performs an exhaustive search over specified parameter values for the models.')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Manual Parameter Configuration ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if param_config_method == 'Manual':
            
            # Random Forest Parameters
            if 'Random Forest Classifier' in selected_models or 'Random Forest Regressor' in selected_models:
                st.write('Random Forest Parameters')
        
                n_estimators = st.slider('Number of Trees (n_estimators) for Random Forest', min_value=10, max_value=300, value=100,help='Select the number of trees to use in the Random Forest model.')
                max_depth = st.slider('Max Depth of the Tree for Random Forest', min_value=1, max_value=30, value=10, help='Select the maximum depth of the tree in the Random Forest model.')
                
                for model in ['Random Forest Classifier', 'Random Forest Regressor']:
                    if model in selected_models:
                        parameter_grid[model] = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth
                        }
               
            # Logistic Regression Parameters
            if 'Logistic Regression' in selected_models:
                st.divider()
                st.write('Logistic Regression Parameters')
                C = st.slider('Regularization Strength (C) for Logistic Regression', min_value=0.01, max_value=10.0, value=1.0, help='Select the regularization strength for the Logistic Regression model.')
                
                parameter_grid['Logistic Regression'] = {
                    'C': C
                }
                
            # Support Vector Machine Parameters
            if 'Support Vector Machine' in selected_models or 'Support Vector Regressor (SVR)' in selected_models:
                st. divider()
                st.write('Support Vector Machine/Regressor Parameters')
                C_svm = st.slider('Regularization Parameter (C) for SVM/SVR', min_value=0.01, max_value=10.0, value=1.0, help='Select the regularization parameter C for the SVM/SVR model.')
                kernel = st.selectbox('Kernel for SVM/SVR', ['linear', 'poly', 'rbf', 'sigmoid'], index=2, help='Select the kernel to use in the SVM/SVR model. The kernel type can be linear, polynomial, radial basis function (rbf), or sigmoid.')
                
                for model in ['Support Vector Machine', 'Support Vector Regressor (SVR)']:
                    if model in selected_models:
                        parameter_grid[model] = {
                            'C': C_svm,
                            'kernel': kernel
                        }
                
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GridSearch Parameter Configuration ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif param_config_method == 'GridSearch':
            
            st.write('Parameter Grid for Selected Models:')

            param_grids = {
                'Random Forest Classifier': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20]
                },
                'Random Forest Regressor': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20]
                },
                'Logistic Regression': {
                    'C': [0.1, 1, 10]
                },
                'Support Vector Machine': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                },
                'Support Vector Regressor (SVR)': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            }
            
            parameter_grid = param_grids
            
            # display the parameter grid for the selected models
            for model in selected_models:
                if model in param_grids:
                    st.write(f'**{model}**:')
                    st.write(param_grids[model])
            
    # ................................. Run Button ...............................
    
    # here all steps will be executed when user clicks on Run button
    if st.button('**Run**'):
        
        st.divider()
        
        st.write('## Results')
        
        
        if df is not None or (train is not None and test is not None):
            
            # checks if df is not none, meaning user uploaded a single file or uses example data
            if df is not None:
                
                # ………………………………………………… Visualization Data Distribution …………………………………………
                
                with st.expander('Data Balance - Distribution of Labels', expanded=True):
                    # get the distribution of the target labels
                    label_counts = df[target].value_counts()
                    
                    # save it as df for visualization
                    label_counts_df = pd.DataFrame({
                        'Label': label_counts.index,
                        'Count': label_counts.values
                    })
                    
                    # sort the labels by natural order (e.g., Label 0, Label 1, ...)
                    label_counts_df['Label'] = pd.Categorical(label_counts_df['Label'], ordered=True)
                    label_counts_df = label_counts_df.sort_values('Label')

                    # generate a Seaborn color palette with as many colors as there are labels
                    palette = sns.light_palette("seagreen", n_colors=len(label_counts), as_cmap=False).as_hex()


                    # create an Altair chart with custom colors
                    chart = alt.Chart(label_counts_df).mark_bar().encode(
                        x=alt.X('Label:N', sort=None),
                        y='Count:Q',
                        color=alt.Color('Label:N', scale=alt.Scale(range=palette))
                    ).properties(
                        title='Label Counts'
                    )

                    st.altair_chart(chart, use_container_width=True)
                    
                    # create a pie chart for label distribution with Plotly
                    
                    fig = px.pie(
                    values=label_counts_df['Count'],  
                    names=label_counts_df['Label'],   
                    color_discrete_sequence=palette,
                    title='Label Distribution'
                    )
                    st.plotly_chart(fig)
                
                # ………………………………………………… Visualization WordClouds …………………………………………                                        
                with st.spinner('Creating WordClouds...'):
                    # create WordClouds for the dataset and for each label
                    cloud_df = create_wordcloud(df, text_col)
                    clouds_labels = create_wordcloud_by_label(df, text_col, target)
                    
                    # combine overall WordCloud and label-based WordClouds into a single dictionary
                    all_wordclouds = {"Overall Dataset": cloud_df}
                    all_wordclouds.update(clouds_labels)

                    with st.expander('WordClouds for Clusters', expanded=True):
                    # create tabs for each WordCloud
                        if all_wordclouds:
                            tab_list = list(all_wordclouds.keys())
                            tabs = st.tabs(tab_list)
                            
                            for i, tab_name in enumerate(tab_list):
                                with tabs[i]:
                                    st.write(f"WordCloud for {tab_name}")
                                    st.pyplot(all_wordclouds[tab_name])
                
        
                
            # user uploaded train and test data separately
            if train is not None and test is not None:
                
                # ………………………………………………… Visualization Data Distribution …………………………………………
                
                with st.expander('Data Balance - Distribution of Labels', expanded=True):
                    # get the distribution of the target labels
                    label_counts = train[target].value_counts()
                    
                    # save it as df for visualization
                    label_counts_df = pd.DataFrame({
                        'Label': label_counts.index,
                        'Count': label_counts.values
                    })
                    
                    # sort the labels by natural order (e.g., Label 0, Label 1, ...)
                    label_counts_df['Label'] = pd.Categorical(label_counts_df['Label'], ordered=True)
                    label_counts_df = label_counts_df.sort_values('Label')

                    # generate a Seaborn color palette with as many colors as there are labels
                    palette = sns.light_palette("seagreen", n_colors=len(label_counts), as_cmap=False).as_hex()


                    # create an Altair chart with custom colors
                    chart_split = alt.Chart(label_counts_df).mark_bar().encode(
                        x=alt.X('Label:N', sort=None),
                        y='Count:Q',
                        color=alt.Color('Label:N', scale=alt.Scale(range=palette))
                    ).properties(
                        title='Label Counts'
                    )

                    st.altair_chart(chart_split, use_container_width=True)
                    
                    # create a pie chart for label distribution with Plotly
                    
                    fig = px.pie(
                    values=label_counts_df['Count'],  
                    names=label_counts_df['Label'],   
                    color_discrete_sequence=palette,
                    title='Label Distribution'
                    )
                    st.plotly_chart(fig)
                
                # ………………………………………………… Visualization WordClouds …………………………………………                                     
                with st.spinner('Creating WordClouds...'):
                    # create WordClouds for the dataset and for each label
                    cloud_df = create_wordcloud(train, text_col)
                    clouds_labels = create_wordcloud_by_label(train, text_col, target)
                    
                    # combine overall WordCloud and label-based WordClouds into a single dictionary
                    all_wordclouds = {"Overall Dataset": cloud_df}
                    all_wordclouds.update(clouds_labels)

                    with st.expander('WordClouds for Clusters', expanded=True):
                    # create tabs for each WordCloud
                        if all_wordclouds:
                            tab_list = list(all_wordclouds.keys())
                            tabs = st.tabs(tab_list)
                            
                            for i, tab_name in enumerate(tab_list):
                                with tabs[i]:
                                    st.write(f"WordCloud for {tab_name}")
                                    st.pyplot(all_wordclouds[tab_name])
                    
                
                
                
            # user uses example data
            if data_upload == 'Use example data':
                
                #………………………………………………………… Data Preprocessing ……………………………………………………………
                with st.spinner('Preprocessing data...'):
                    df_cleaned = preprocess_text(df, text_col, clean_text, lowercase, remove_punctuation, remove_numbers, remove_stopwords, remove_emojis, remove_spaces,remove_html, remove_html_entities, stemming, tokenize_text, tokenization_method)
                    X_train, X_test, y_train, y_test = split_data(df, target, 0.25, 42) 

                    
                    with st.expander('Preprocessed Data', expanded=True):
                        st.dataframe(df_cleaned)

                # ………………………………………………… Data Augmentation ……………………………………………………………
                if augment_text_options:
                    with st.spinner('Augmentation of data...'):
                        X_train, y_train = augmentation(X_train, y_train, target, synonym_replacement, random_char)
                       

                # ………………………………………………… Vectorization ……………………………………………………………            
                if vectorizing:
                    with st.spinner('Vectorizing data...'):
                        if vectorizing == 'Sentence Transformer':
                            X_train, X_test = vectorizer(X_train, X_test, text_col, vectorizing, model_name)
                        else:
                            X_train, X_test = vectorizer(X_train, X_test, text_col, vectorizing)
                        
                # ………………………………………………… Scaling ……………………………………………………………
                if apply_scaling:
                    with st.spinner('Scaling data...'):
                        scaler = StandardScaler(with_mean=False) 
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
            
            # user uploads train and test data separately            
            else:
                if data_split == 'Upload Separated Data':
                    if train is not None and test is not None and target is not None and text_col is not None:
                        
                        # ………………………………………………… Data Preprocessing ……………………………………………………………
                        with st.spinner('Preprocessing data...'):
                            if clean_text:
                                train_cleaned = preprocess_text(train, text_col, clean_text, lowercase, remove_punctuation, remove_numbers, remove_stopwords, remove_emojis, remove_spaces,remove_html ,remove_html_entities,stemming, tokenize_text, tokenization_method)
                                test_cleaned = preprocess_text(test, text_col, clean_text, lowercase, remove_punctuation, remove_numbers, remove_stopwords, remove_emojis, remove_spaces,remove_html,remove_html_entities, stemming, tokenize_text, tokenization_method)
                                
                                X_train = train_cleaned.drop(columns=[target])
                                y_train = train_cleaned[target]

                                X_test = test_cleaned.drop(columns=[target])
                                y_test = test_cleaned[target]
                            else:
                                X_train = train.drop(columns=[target])
                                y_train = train[target]

                                X_test = test.drop(columns=[target])
                                y_test = test[target]
                            
                            
                            
                            with st.expander('Preprocessed Data', expanded=True):
                                st.write('**Train Data**')
                                st.dataframe(train_cleaned)
                                st.write('**Test Data**')
                                st.dataframe(test_cleaned)

                        # ………………………………………………… Data Augmentation ……………………………………………………………
                        if augment_text_options:
                            with st.spinner('Augmentation of data...'):
                                X_train, y_train = augmentation(X_train, y_train, target, synonym_replacement, random_char)
                                
                    
                        # ………………………………………………… Vectorization ……………………………………………………………
                        if vectorizing:
                            with st.spinner('Vectorizing data...'):
                                if vectorizing == 'Sentence Transformer':
                                    X_train, X_test = vectorizer(X_train, X_test, text_col, vectorizing, model_name)
                                else:
                                    X_train, X_test = vectorizer(X_train, X_test, text_col, vectorizing)
                            
                        # ………………………………………………… Scaling ……………………………………………………………
                        if apply_scaling:
                            with st.spinner('Scaling data...'):
                                scaler = StandardScaler(with_mean=False) 
                                X_train = scaler.fit_transform(X_train)
                                X_test = scaler.transform(X_test)
                              
                        
                    else:
                        st.error('Please upload a file and select target and text column or use example data')

                # user uploads a single file and performs train-test split
                elif data_split == 'Perform Train-Test Split':
                    if df is not None and target is not None and test_size is not None and ran_state is not None:
                        
                        # ………………………………………………… Data Preprocessing ……………………………………………………………
                        with st.spinner('Preprocessing data...'):
                            if clean_text:
                                df = preprocess_text(df, text_col, clean_text, lowercase, remove_punctuation, remove_numbers, remove_stopwords, remove_emojis, remove_spaces, remove_html,remove_html_entities, stemming, tokenize_text, tokenization_method)

                            X_train, X_test, y_train, y_test = split_data(df, target, test_size, ran_state)    
                            
                            
                            
                            with st.expander('Preprocessed Data', expanded=True):
                                st.write('Shape of the Train and Test Data')
                                st.write(f'Shape of Trainset: {X_train.shape}')
                                st.write(f'Shape of Testset: {X_test.shape}')
                                
                                st.write('**Train Data**')
                                st.dataframe(X_train)
                                st.write('**Test Data**')
                                st.dataframe(X_test)
                            
                        # ………………………………………………… Data Augmentation ……………………………………………………………   
                        if augment_text_options:
                            with st.spinner('Augmentation of data...'):
                                X_train, y_train = augmentation(X_train, y_train, target, synonym_replacement, random_char)
                                
                            
                        # ………………………………………………… Vectorization ……………………………………………………………
                        if vectorizing:
                            with st.spinner('Vectorizing data...'):
                                if vectorizing == 'Sentence Transformer':
                                    X_train, X_test = vectorizer(X_train, X_test, text_col, vectorizing, model_name)
                                else:
                                    X_train, X_test = vectorizer(X_train, X_test, text_col, vectorizing)
                                
                        # ………………………………………………… Scaling ……………………………………………………………
                        if apply_scaling:
                            with st.spinner('Scaling data...'):
                                scaler = StandardScaler(with_mean=False) 
                                X_train = scaler.fit_transform(X_train)
                                X_test = scaler.transform(X_test)
                                
                    else:
                        st.error('Please upload a file and select target and text column or use example data')

            # ------------------- Model Training -------------------
            
            accuracy_scores = {}
            precision_scores = {}
            recall_scores = {}
            f1_scores = {}
            
            mse_scores = {}
            mae_scores = {}
            r2_scores = {}
            
            y_pred_dict = {}
            
            # loop through each selected model           
            for model in selected_models:
                
                with st.spinner(f'Training {model}...'):
                    
                    # ~~~~~~~~~~~~~~~~~~~~~~ Manual Parameter Configuration~~~~~~~~~~~~~~~~~~~~~~~~
                    if param_config_method == 'Manual':
                        if model == 'Random Forest Classifier':
                            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                        elif model == 'Random Forest Regressor':
                            clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                        elif model == 'Logistic Regression':
                            clf = LogisticRegression(C=C)
                        elif model == 'Support Vector Machine':
                            clf = SVC(C=C_svm, kernel=kernel)
                        elif model == 'Support Vector Regressor (SVR)':
                            clf = SVR(C=C_svm, kernel=kernel)
                        
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        
                        y_pred_dict[model] = y_pred
                        
                        # metrics for Classification
                        if task_type == 'Classification':
                            report = classification_report(y_test, y_pred, output_dict=True)
                                    
                            # scores
                            accuracy = report['accuracy']
                            precision = report['weighted avg']['precision']
                            recall = report['weighted avg']['recall']
                            f1_score = report['weighted avg']['f1-score']
                            
                            
                            # save scores to their respective dictionaries
                            accuracy_scores[model] = accuracy
                            precision_scores[model] = precision
                            recall_scores[model] = recall
                            f1_scores[model] = f1_score
                        
                            
                            # dataFrame to store the collected metrics
                            metrics_df = pd.DataFrame({
                                'Model': list(accuracy_scores.keys()),
                                'Accuracy': [accuracy_scores[m] for m in accuracy_scores],
                                'Precision': [precision_scores[m] for m in precision_scores],
                                'Recall': [recall_scores[m] for m in recall_scores],
                                'F1-Score': [f1_scores[m] for m in f1_scores]
                            })

                            metrics_df.set_index('Model', inplace=True)
                            
                        # metrics for Regression
                        else:
                            mse = mean_squared_error(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)

                            # save the scores to their dictionaries
                            mse_scores[model] = mse
                            mae_scores[model] = mae
                            r2_scores[model] = r2

                            regression_metrics_df = pd.DataFrame({
                                'Model': list(mse_scores.keys()),
                                'MSE': [mse_scores[m] for m in mse_scores],
                                'MAE': [mae_scores[m] for m in mae_scores],
                                'R²': [r2_scores[m] for m in r2_scores]
                            })
                            
                            
                    # ~~~~~~~~~~~~~~~~~~~~~~ GridSearch Parameter Configuration~~~~~~~~~~~~~~~~~~~~~~~~
                    elif param_config_method == 'GridSearch':
                        
                        with st.spinner(f' Performing Gridsearch for {model}...'):
                            
                            if model in param_grids:
                                param_grid = param_grids[model]
                                if model == 'Random Forest Classifier':
                                    clf = RandomForestClassifier()
                                elif model == 'Random Forest Regressor':
                                    clf = RandomForestRegressor()
                                elif model == 'Logistic Regression':
                                    clf = LogisticRegression()
                                elif model == 'Support Vector Machine':
                                    clf = SVC()
                                elif model == 'Support Vector Regressor (SVR)':
                                    clf = SVR()
                                
                                # perform GridSearch
                                grid_search = GridSearchCV(clf, param_grid, cv=5)
                                grid_search.fit(X_train, y_train)
                                best_model = grid_search.best_estimator_
                                y_pred = best_model.predict(X_test)
                                
                                # save the predictions
                                y_pred_dict[model] = y_pred
                                
                                # metrics for classification
                                if task_type == 'Classification':
                                    report = classification_report(y_test, y_pred, output_dict=True)
                                    
                                    # scores
                                    accuracy = report['accuracy']
                                    precision = report['weighted avg']['precision']
                                    recall = report['weighted avg']['recall']
                                    f1_score = report['weighted avg']['f1-score']
                                    
                                    
                                    # save scores to their respective dictionaries
                                    accuracy_scores[model] = accuracy
                                    precision_scores[model] = precision
                                    recall_scores[model] = recall
                                    f1_scores[model] = f1_score
                                    
                                    # dataFrame to store the collected metrics
                                    metrics_df = pd.DataFrame({
                                        'Model': list(accuracy_scores.keys()),
                                        'Accuracy': [accuracy_scores[m] for m in accuracy_scores],
                                        'Precision': [precision_scores[m] for m in precision_scores],
                                        'Recall': [recall_scores[m] for m in recall_scores],
                                        'F1-Score': [f1_scores[m] for m in f1_scores]
                                    })

                                    # Set the Model column as the index for the DataFrame to use in the bar chart
                                    metrics_df.set_index('Model', inplace=True)

                                    
                                # metrics for regression 
                                else:
                                    mse = mean_squared_error(y_test, y_pred)
                                    mae = mean_absolute_error(y_test, y_pred)
                                    r2 = r2_score(y_test, y_pred)

                                    # save the scores to their dictionaries
                                    mse_scores[model] = mse
                                    mae_scores[model] = mae
                                    r2_scores[model] = r2

                                    regression_metrics_df = pd.DataFrame({
                                        'Model': list(mse_scores.keys()),
                                        'MSE': [mse_scores[m] for m in mse_scores],
                                        'MAE': [mae_scores[m] for m in mae_scores],
                                        'R²': [r2_scores[m] for m in r2_scores]
                                    })

            with st.expander('Model Parameter', expanded=True):
                st.write(parameter_grid)
            
            
            if task_type == 'Classification':
                
                # ----------------- Model Performance Metrics -------------------
                
                with st.expander('Model Performance Metrics', expanded=True):
                    
                    st.dataframe(metrics_df)

                    # generate a sns color palette based on the number of unique metrics
                    palette = sns.light_palette("seagreen", n_colors=4, as_cmap=False).as_hex()

                    
                    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                    charts = []
                    
                    # create bar charts for each metric 
                    for i, metric in enumerate(metrics):
                        chart = alt.Chart(metrics_df.reset_index()).mark_bar().encode(
                            x=alt.X('Model:N', title="Model"),
                            y=alt.Y(f'{metric}:Q', title=metric),
                            color=alt.value(palette[i])
                        ).properties(
                            width=90, 
                            height=200, 
                            title=f'{metric} per Model'
                        )
                        charts.append(chart)

                    # concatenate the charts horizontally for comparison
                    combined_chart = alt.hconcat(*charts)
                    
                    st.altair_chart(combined_chart, use_container_width=True)
                                                            
                    
                
                # ----------------- Confusion Matrix -------------------
                with st.expander('Confusion Matrix', expanded=True):
                    
                    # custom colormap so that it matches the color scheme
                    from matplotlib.colors import LinearSegmentedColormap
                    colors = ['#fafafa', '#669977'] 
                    custom_cmap = LinearSegmentedColormap.from_list('green_white_cmap', colors)
                    
                    # columns for displaying confusion matrices side by side
                    num_columns = len(selected_models)
                    columns = st.columns(num_columns)

                    # loop through each model and its respective column
                    for i, model in enumerate(selected_models):
                        with columns[i]:
                            st.write(f'Confusion Matrix for {model}')
                            
                            # create confusion matrix
                            cm = confusion_matrix(y_test, y_pred_dict[model])
                            
                            # set up the figure and axis for seaborn
                            fig_cm, ax = plt.subplots()
                            sns.set_style("whitegrid")
                            heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap, ax=ax)
                            
                            colorbar = heatmap.collections[0].colorbar
                            colorbar.ax.tick_params(colors='#fafafa')
                            colorbar.outline.set_edgecolor('#fafafa')
                            ax.set_xlabel('Predicted', color='#fafafa')
                            ax.set_ylabel('Actual', color='#fafafa')
                            ax.set_title(f'Confusion Matrix - {model}', color='#fafafa')
                            fig_cm.patch.set_facecolor('#373737')
                            ax.tick_params(axis='x', colors='#fafafa')
                            ax.tick_params(axis='y', colors='#fafafa')
                            
                            
                            st.pyplot(fig_cm)
                
            # ----------------- Regression Metrics -------------------
            else:
                
                with st.expander('Regression Metrics', expanded=True):
                    st.dataframe(regression_metrics_df)
                    
                    palette = sns.light_palette("seagreen", n_colors=3, as_cmap=False).as_hex()
                    
                    metrics = ['MSE', 'MAE', 'R²']
                    
                    charts = []
                    for i, metric in enumerate(metrics):
                        chart = alt.Chart(regression_metrics_df.reset_index()).mark_bar().encode(
                            x=alt.X('Model:N', title="Model"),
                            y=alt.Y(f'{metric}:Q', title=metric),
                            color=alt.value(palette[i])
                        ).properties(
                            width=90, 
                            height=200, 
                            title=f'{metric} per Model'
                        )
                        charts.append(chart)

                    combined_chart = alt.hconcat(*charts)
                    st.altair_chart(combined_chart, use_container_width=True)
                    
                    # columns for displaying confusion matrices side by side
                    num_columns = len(selected_models)
                    columns = st.columns(num_columns)

                    # loop through each model and its respective column
                    for i, model in enumerate(selected_models):
                        with columns[i]:
                            st.write(f'### Predicted vs Actual for {model}')
                            
                            fig_pred, ax_pred = plt.subplots()
                            
                            # scatter plot for predicted vs actual values
                            ax_pred.scatter(y_test, y_pred_dict[model], color='#ECDFCC', label='Predicted Values')
                            
                            # diagonal line for reference
                            ax_pred.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='#669977', lw=2, label='Ideal Prediction')
                            
                            ax_pred.set_xlabel('Actual Values')
                            ax_pred.set_ylabel('Predicted Values')
                            ax_pred.set_title(f'Predicted vs Actual - {model}')
                            ax_pred.legend()
                            
                            st.pyplot(fig_pred)
                
                        
        else:
            st.error('Please upload a file and select target and text column or use example data')
             


# -------------------------------- Clustering Tab 2 --------------------------------------

with tab2:
    
    st.header('Clustering')
    
    # -------------------- Data Upload ------------------------
    with st.expander('Data Upload', expanded=True):

        # upload single file for clustering
        file_cluster = st.file_uploader('Please upload the dataset for Clustering ', type=['csv', 'tsv', 'json', 'xlsx', 'xml', 'zip'], key='cluster', accept_multiple_files=False) 
        
        if file_cluster is not None:
            df_clustering = return_df(file_cluster)
            st.success('File uploaded successfully')
            st.write('Data preview:')
            st.dataframe(df_clustering.head())
            text_col_cluster = text_column_selection(df_clustering)
        
            
    # ------------------- Data Preprocessing -------------------
    with st.expander('Data Preprocessing', expanded=True):
            
        # clean text checkbox
        clean_text_cluster = st.checkbox('Clean Text ', value=True, help='Clean the text data by applying various text preprocessing steps like converting to lowercase, removing punctuation, removing numbers, removing stopwords, etc.')
        if clean_text_cluster:
            st.write('Options for text cleaning:')
            
             # columns in order to be able to indent the following checkboxes
            col1, col2 = st.columns([0.01, 0.9]) 
            
            with col2:
                lowercase_cluster = st.checkbox('Convert to Lowercase ', value=True, help='Convert all text to lowercase, e.g., "Hello" to "hello"')
                remove_punctuation_cluster = st.checkbox('Remove Punctuation ', value=True, help='Remove all punctuation marks, e.g., "Hello, world!" to "Hello world"')
                remove_numbers_cluster = st.checkbox('Remove Numbers ', value=True, help='Remove all numbers, e.g., "Hello 123" to "Hello"')
                remove_stopwords_cluster = st.checkbox('Remove Stopwords ', value=True, help='Remove all stopwords, e.g., "The quick brown fox" to "quick brown fox"')
                remove_emojis_cluster = st.checkbox('Remove Emojis ', value=False, help='Remove all emojis, e.g., "I am happy 😊" to "I am happy"')
                remove_spaces_cluster = st.checkbox('Remove Extra Spaces ', value=True, help='Remove extra spaces, e.g., "Hello    world" to "Hello world"')
                remove_html_cluster = st.checkbox('Remove HTML Tags ', value=True, help='Remove all HTML tags, e.g., "<p>Hello</p>" to "Hello"')
                remove_html_entities_cluster = st.checkbox('Decode HTML Entities ', value=True, help='Decode all HTML entities, e.g., "&amp;" to "&"')
                stemming_cluster = st.checkbox('Apply Stemming ', value=False, help='Apply stemming to the text, e.g., "running" to "run"')
            
        st.divider()
        
        # tokenization options
        tokenize_text_cluster = st.checkbox('Tokenization ', value=True, help='Tokenize the text data into words, sentences, or tweets, e.g., "Hello world" to ["Hello", "world"]')
        if tokenize_text_cluster:
            st.write('Options: ')
            tokenization_method_cluster = st.selectbox('Select Tokenization Method ', ['Word Tokenization', 'Sentence Tokenization', 'Tweet Tokenization'],help='Select the tokenization method to use for tokenizing the text data. Word tokenization splits the text into words, sentence tokenization splits the text into sentences, and tweet tokenization tokenizes tweets.')
        else:
            tokenization_method_cluster = None
        st.divider()
        
        # vectorization options
        vectorizing_cluster = st.selectbox('Select Vectorizer ', ['Bag-of-Words', 'TF-IDF', 'Sentence Transformer'], index=1, help='Select the vectorization method to use for converting the text data into numerical features. Bag-of-Words and TF-IDF are traditional methods, while Sentence Transformer uses pre-trained transformer models to encode the text data.')
        if vectorizing_cluster == 'Sentence Transformer':
            st.write('Choose a Sentence Transformer model')
            model_name_cluster = st.selectbox('Model Name ', ['all-MiniLM-L6-v2','bert-base-nli-mean-tokens','roberta-base-nli-stsb-mean-tokens','distilbert-base-nli-stsb-mean-tokens'], index=0, help='Select the Sentence Transformer model to use for encoding the text data. ')

        st.divider()
        
        # scaling options
        apply_scaling_cluster = st.checkbox('Apply Scaling ', value=False, help='Apply scaling to the numerical features in the data. This step is necessary if the features have different scales. The used method is StandardScaler.')

        
    # ----- Clustering Parameters ------------
    with st.expander('Clustering Parameters', expanded=True):

        clustering_method = st.selectbox('Select Clustering Method', ['K-Means', 'DBSCAN'],help='Select the clustering method to use for clustering the text data. K-Means is a centroid-based clustering algorithm that partitions the data into K clusters. DBSCAN is a density-based clustering algorithm that groups together points that are closely packed.')

        if clustering_method == 'K-Means':
            num_clusters = st.slider('Number of Clusters', min_value=2, max_value=10, value=3,help='Select the number of clusters to create using the K-Means clustering algorithm.')
            
            param_cluster['K-Means'] = {
                'Number of Clusters': num_clusters
            }

        elif clustering_method == 'DBSCAN':
            eps = st.slider('Maximum Distance Between Points', min_value=0.0, max_value=1.0, value=0.5, help='Select the maximum distance between two samples for one to be considered as in the neighborhood of the other.')
            min_samples = st.slider('Minimum Samples for Core Point', min_value=1, max_value=50, value=5, help='Select the number of samples in a neighborhood for a point to be considered as a core point.')
            
            param_cluster['DBSCAN'] = {
                'Epsilon': eps,
                'Minimum Samples': min_samples
            }


    # ------------------- Run Clustering -------------------
    if st.button('Run Clustering'):
        # preprocess the data
        
        if df_clustering is not None:

            with st.spinner('Preprocessing data...'):
                if clean_text_cluster:
                    df_cleaned_cluster = preprocess_text(df_clustering, text_col_cluster, clean_text_cluster, lowercase_cluster, 
                                                        remove_punctuation_cluster, remove_numbers_cluster, remove_stopwords_cluster, 
                                                        remove_emojis_cluster, remove_spaces_cluster,remove_html_cluster ,remove_html_entities, stemming_cluster, tokenize_text_cluster, 
                                                        tokenization_method_cluster)
                
            
                    # save the original dataframe in the cleaned dataframe for wordcloud later
                    df_cleaned_cluster['Original Text'] = df_clustering[text_col_cluster]
                    
                    with st.expander('Preprocessed Data', expanded=True):
                        st.write('Preprocessed Data:')
                        st.dataframe(df_cleaned_cluster.head())
            
            # vectorization
            if vectorizing_cluster:
                with st.spinner('Vectorizing data...'):
                    if vectorizing_cluster == 'Sentence Transformer':
                        X_train_cluster, X_test_cluster = vectorizer(df_cleaned_cluster, df_cleaned_cluster, text_col_cluster, vectorizing_cluster, model_name_cluster)
                    else:
                        X_train_cluster, X_test_cluster= vectorizer(df_cleaned_cluster, df_cleaned_cluster, text_col_cluster, vectorizing_cluster)
                    

            if apply_scaling_cluster:
                with st.spinner('Scaling data...'):

                    # ensure that the data is 2D (shape should be (n_samples, n_features))
                    if X_train_cluster.ndim == 1:
                        X_train_cluster = X_train_cluster.reshape(-1, 1)
                    if X_test_cluster.ndim == 1:
                        X_test_cluster = X_test_cluster.reshape(-1, 1)
                    
                    scaler = StandardScaler(with_mean=False)
                    X_train_cluster = scaler.fit_transform(X_train_cluster)
                    X_test_cluster = scaler.transform(X_test_cluster)
                    

            # perform clustering
            with st.spinner('Running Clustering...'):
                if clustering_method == 'K-Means':
                    st.write('Running K-Means Clustering...')
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X_train_cluster)
                elif clustering_method == 'DBSCAN':
                    st.write('Running DBSCAN Clustering...')
                    from sklearn.cluster import DBSCAN
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    clusters = dbscan.fit_predict(X_train_cluster)
               
                
            
            # add cluster labels to the dataframe
            df_cleaned_cluster['Cluster'] = clusters
            
            # ----------------- Clustering Results -------------------
            
            # display results
            
            with st.expander('Dataframe with Clusters', expanded=True):
                st.dataframe(df_cleaned_cluster.head())
                
                csv = df_cleaned_cluster.to_csv(index=False).encode('utf-8')
                
                st.download_button(label = 'Download Dataframe as CSV', 
                                   data = csv, 
                                   file_name='clustered_data.csv', 
                                   mime='text/csv')
            
            with st.expander('Model Parameter', expanded=True):
                st.write(param_cluster)
            
            
            # ----------------- WordClouds -------------------
            
            with st.spinner('Creating WordClouds...'):
            
                # wordcloud for the whole dataframe
                cloud_df_cluster = create_wordcloud(df_cleaned_cluster, 'Original Text')
                
                # wordclouds for each cluster
                clouds_labels = create_wordcloud_by_label(df_cleaned_cluster, 'Original Text', 'Cluster')
                
        
                # combine overall WordCloud and label-based WordClouds 
                all_wordclouds = {"Overall Dataset": cloud_df_cluster}
                all_wordclouds.update(clouds_labels)

                with st.expander('WordClouds for Clusters', expanded=True):
                    
                    # create tabs for each WordCloud
                    if all_wordclouds:
                        tab_list = list(all_wordclouds.keys())
                        tabs = st.tabs(tab_list)
                        
                        for i, tab_name in enumerate(tab_list):
                            with tabs[i]:
                                st.write(f"WordCloud for {tab_name}")
                                st.pyplot(all_wordclouds[tab_name])
                            
            
            # ----------------- Visualize Clusters -------------------
            with st.spinner('Visualizing Clusters...'):
                with st.expander('Visualized Cluster with PCA', expanded=True):
                    # visualize clusters with PCA 
                    from sklearn.decomposition import PCA
                    
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_train_cluster)

                    # create a DataFrame with PCA results and cluster labels
                    pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
                    pca_df['Cluster'] = clusters

                    # number of unique clusters to create a color palette
                    num_clusters = len(pca_df['Cluster'].unique())

                    # generate a palette based on the number of clusters
                    palette = sns.light_palette("seagreen", n_colors=num_clusters, as_cmap=False).as_hex()

                    # convert the cluster labels to strings for plotting
                    pca_df['Cluster'] = pca_df['Cluster'].astype(str)

                    # plot the interactive scatter plot using Plotly
                    fig = px.scatter(
                        pca_df,
                        x='PCA1',
                        y='PCA2',
                        color='Cluster',
                        title=f'{clustering_method} Clustering Results',
                        color_discrete_sequence=palette,  # Use the Seaborn-generated palette
                        labels={'PCA1': '', 'PCA2': ''},  # Remove axis labels
                        width=800,
                        height=600
                    )

                    # layout settings to match color theme
                    fig.update_layout(
                        plot_bgcolor='#373737',  
                        paper_bgcolor='#373737',  
                        font=dict(color='#fafafa'), 
                        title_font=dict(size=20, color='#fafafa', family="Arial"),
                        showlegend=True,
                        xaxis=dict(showticklabels=True, title='', zeroline=False, gridcolor='rgba(255, 255, 255, 0.1)'),
                        yaxis=dict(showticklabels=True, title='', zeroline=False, gridcolor='rgba(255, 255, 255, 0.1)'),   
                    )
                   
                    st.plotly_chart(fig)   
            
        else:
            st.error('Please upload a file and select text column or use example data')    

                        

