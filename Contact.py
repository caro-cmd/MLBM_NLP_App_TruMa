import streamlit as st

# create title
st.markdown(
    "<h3 style='color:#669977; font-size:50px;'>Contact</h3>",
    unsafe_allow_html=True
)

# create info text

col1, col2 = st.columns(2)

with col2:
    st.image("Logo_ML_transparent.png", use_column_width = True)
    
    

with col1:
    
    st.markdown(
        """
        <br><br>
        If you experience any problems with our application, have questions, or need more
        information regarding copyright regulations, please don't hesitate to contact us!
        
        **TRUMA Technology**  
        Caroline Trust and Lisa Mathes \\
        Grindelberg 22  
        20144 Hamburg  
        Tel: 0402764182  
        E-Mail: truma@technology.com
        """, 
        unsafe_allow_html=True
    )