import streamlit as st

# create title
st.title('Example')

# create text
st.subheader('**Get started with our Machine Learning Application:**')

st.info('If you have any questions regarding the machine learning or clustering options, please refer to the information provided below or hover over the question marks next to the options.')

st.markdown('''
            The main online Application is seperated in two tabs: 
            - **Supervised Machine Learning:** Use this tab to perform classification or regression tasks on your text data.
            - **Clustering:** Use this tab to group similar text data into clusters.
            ''')

with st.expander('**Supervised Machine Learning**', expanded=True):
    st.markdown(
        "<h3 style='color:#669977; font-size:20px;'>Data</h3>",
        unsafe_allow_html=True
    )

    st.markdown('To get started you can use the example data provided in the application or you can download the data from the following:')

    col1, col2 = st.columns([0.01, 0.9])
    with col2:
        st.markdown('1. [Medical Abstracts: CSV files](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)')
        st.markdown('2. [Medical Notes: Data folder as zip folder](https://www.kaggle.com/competitions/medicalnotes-2019/data)')

    st.markdown('''
                The data can be uploaded in various formats, including CSV, TSV, JSON, XML, and ZIP files containing text documents.
                Also, you can fetch example data.
                It is also possible to either upload the train and test set separately or upload a single file containing both which will be split into train and test set.
                After uploading the data, you will be asked to choose the target column and the text column. Depending on if you uploaded a single file or two files, you will also be asked to clarify which file contains the train and test set.
                ''')

    tab1,tab2 = st.tabs(['Upload seperated data', 'Perform Train-Test-Split on single file'])
    with tab1:
        col1, col2 = st.columns([0.1, 0.9])
        with col2:
            st.image('Upload_separated_data.png', width=500, caption='Upload separated data and choose train data, test data, target and text column',)
    with tab2:
        col1, col2 = st.columns([0.1, 0.9])
        with col2:
            st.image('train_split.png', width=500, caption='After uploading the file, choose target and text column and adjust parameter for Train_Test_Split',)

    st.markdown(
        "<h3 style='color:#669977; font-size:20px;'>Text Preprocessing</h3>",
        unsafe_allow_html=True
    )

    st.markdown('''
                After uploading the data, the text preprocessing step will be performed. 
                The following preprocessing steps are available:
                - Convert text to lowercase
                - Remove punctuation, numbers, stopwords, and emojis
                - Apply stemming
                - Tokenize text using different methods
                - Text Augmentation: Enhance your training data with techniques like synonym replacement and random character augmentation to improve model performance.
                - Vectorization: Transform your text data into numerical representations using methods such as Bag-of-Words, TF-IDF, or advanced Sentence Transformers for better feature extraction.
                - Scaling: Scale the features of the dataset so that the model can be trained more efficiently.
                ''')
    
    st.markdown(
        "<h3 style='color:#669977; font-size:20px;'>Machine Learning Models</h3>",
        unsafe_allow_html=True
    )
    st.markdown('''
                Also there are various model selection options available:
                - Random Forest Classifier/Regressor
                - Logistic Regression
                - Support Vector Machine (SVM)
                - Support Vector Regressor (SVR)
                Configure model parameters manually or use GridSearch to find the best hyperparameters for your models.
                ''')
    
    st.markdown('''
                After choosing all the necessary parameters, the model will be trained and evaluated. To start the training process, click on the 'Run' button.
                With the loading bar, you can track the progress of the training process. After the training is completed, the evaluation metrics will be displayed.''')
    
    
    st.markdown(
        "<h3 style='color:#669977; font-size:20px;'>Evaluation</h3>",
        unsafe_allow_html=True
    )
    
    st.markdown('''
                The evaluation metrics include:
                - Data Balance
                - Accuracy
                - Precision
                - Recall
                - F1-Score
                - Confusion Matrix
                - Mean Squared Error
                - R2 Score
                - Mean Absolute Error
                ''')
    
    tab1,tab2, tab3, tab4, tab5, tab6, = st.tabs(['Data Balance - Label Counts', 'Data Balance - Label Distribution', 'Word Cloud', 'Metrics - Classification','Metrics - Regression','Confusion Matrix'])
    
    with tab1:
        col1, col2 = st.columns([0.1, 0.9])
        with col2:
            st.image('visualization.png', width=500, caption='Data Balance - Label Counts',)
 
    with tab2:
        col1, col2 = st.columns([0.1, 0.9])
        with col2:
            st.image('newplot.png', width=500, caption='Data Balance - Label Distribution',)
    
    with tab3:
        col1, col2 = st.columns([0.1, 0.9])
        with col2:
            st.image('wordcloud.png', width=500, caption='Word Cloud',)
    
    with tab4:
        col1, col2 = st.columns([0.1, 0.9])
        with col2:
            st.image('met_clas.png', width=500, caption='Metrics - Classification',)
        
    with tab5:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image('met_reg.png', width=300, caption='Metrics - Regression',)
        with col2:
            st.image('regr.png', width=300, caption='Predicted vs. Actual',)

    with tab6:
        col1, col2 = st.columns([0.1, 0.9])
        with col2:
            st.image('confmat.png', width=500, caption='Confusion Matrix',)

with st.expander('**Clustering**', expanded=True):
    st.markdown(
        "<h3 style='color:#669977; font-size:20px;'>Data</h3>",
        unsafe_allow_html=True
    )

    st.markdown('To get started you can use download the data from the following:')

    col1, col2 = st.columns([0.01, 0.9])
    with col2:
        st.markdown('1. [Medical Notes: Data folder as zip folder](https://www.kaggle.com/competitions/medicalnotes-2019/data)')
        st.markdown('In the data folder, you will find different files. The one you need is called "data", zip it and upload it here.')
        
    st.markdown('''
                The data can be uploaded in various formats, including CSV, TSV, JSON, XML, and ZIP files containing text documents.
                Also, you can fetch example data.
                After uploading the data, you will be asked to choose the text column.
                ''')
    
    st.markdown(
        "<h3 style='color:#669977; font-size:20px;'>Text Preprocessing</h3>",
        unsafe_allow_html=True
    )

    st.markdown('''
                After uploading the data, the text preprocessing step will be performed. 
                The following preprocessing steps are available:
                - Convert text to lowercase
                - Remove punctuation, numbers, stopwords, and emojis
                - Apply stemming
                - Tokenize text using different methods
                - Text Augmentation: Enhance your training data with techniques like synonym replacement and random character augmentation to improve model performance.
                - Vectorization: Transform your text data into numerical representations using methods such as Bag-of-Words, TF-IDF, or advanced Sentence Transformers for better feature extraction.
                - Scaling: Scale the features of the dataset so that the model can be trained more efficiently.
                ''')
    
    st.markdown(
        "<h3 style='color:#669977; font-size:20px;'>Clustering Models</h3>",
        unsafe_allow_html=True
    )
    
    st.markdown('''
                Also there are different model selection options available:
                - KMeans
                - DBSCAN
                You can configure model parameters manually.
                ''')
    
    st.markdown('''
                After choosing all the necessary parameters, the model will be trained and evaluated. To start the training process, click on the 'Run Clustering' button.
                With the loading bar, you can track the progress of the training process. After the training is completed, the evaluation metrics will be displayed.''')
    
    st.markdown(
        "<h3 style='color:#669977; font-size:20px;'>Evaluation</h3>",
        unsafe_allow_html=True
    )
    
    tab1,tab2 = st.tabs(['Wordclouds', 'Cluster Visualization'])
 
    with tab1:
        col1, col2 = st.columns([0.1, 0.9])
        with col2:
            st.image('wordcloud_cluster.png', width=500, caption='Word Cloud',)
    
    with tab2:
        col1, col2 = st.columns([0.1, 0.9])
        with col2:
            st.image('cluster.png', width=500, caption='Cluster Visualization',)
      