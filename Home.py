import streamlit as st

# create title

st.markdown(
    "<h3 style='color:#669977; font-size:50px;'>Welcome!</h3>",
    unsafe_allow_html=True
)


# create info text
st.markdown(
    '''
    This Application provides an interface to perform supervised machine learning tasks and Clustering on text data. 
    Wether you're analyzing text for classification or regression, this tool simplifies the process from data upload 
    to model evaluation.
    
    ''')

st.markdown(
    "<div style='background-color:#669977; padding:6px; border-radius:5px;'>Key Features</div>",
    unsafe_allow_html=True
)

st.markdown(
    '''
    -	Data Upload: Upload your datasets in various formats, including CSV, TSV, JSON, XML, and ZIP files containing text documents. Alternatively, fetch example data.
    -	Data Preprocessing: Clean and preprocess your text data with options to:
        -	Convert text to lowercase
        -	Remove punctuation, numbers, stopwords, and emojis
        -	Apply stemming
        -	Tokenize text using different methods
        -	Text Augmentation: Enhance your training data with techniques like synonym replacement and random character augmentation to improve model performance.
        -	Vectorization: Transform your text data into numerical representations using methods such as Bag-of-Words, TF-IDF, or advanced Sentence Transformers for better feature extraction.
    -	Model Training and Evaluation: Select from a range of machine learning models for classification or regression, including:
        -	Random Forest Classifier/Regressor
        -	Logistic Regression
        -	Support Vector Machine (SVM)
        -	Support Vector Regressor (SVR)
        Configure model parameters manually or use GridSearch to find the best hyperparameters for your models.
    -	Results and Insights: View evaluation and performance metrics to understand how well your models are performing.
    ''')

st.markdown(
    "<div style='background-color:#669977; padding:6px; border-radius:5px;'>How to use:</div>",
    unsafe_allow_html=True
)

st.markdown(
    'Choose either supervised machine learning or clustering from the tabs at the top to get started. The process for both tasks is similar and involves the following steps:'
    )

col1, col2 = st.columns([0.01, 0.9]) 
with col2:
    st.markdown('''
    1.	Upload Your Data: Choose to upload files or use example datasets.
    2.	Preprocess and Augment: Clean and augment your text data as needed.
    3.	Select Vectorization Method: Pick the vectorization method that suits your needs.
    4.	Train and Evaluate Models: Choose the machine learning models you wish to train and configure their parameters.
    5.	Review Results: Analyze the results and model performance.
    ''')

st.markdown(
    '''
    Feel free to explore and experiment with different settings to achieve the best results for your text data analysis tasks. If you have any questions or need assistance, please refer to the documentation or reach out for support.

    :rainbow[Happy Analyzing!] 
    '''
    )

