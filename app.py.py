import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="Course Recommendation System", layout="wide")

@st.cache_data
def load_data():
    # Load your dataset here
    # For deployment, you might want to upload this file through Streamlit or host it online
    df = pd.read_excel('course_data.xlsx')
    
    # Preprocessing
    le = LabelEncoder()
    df['course_name_enc'] = le.fit_transform(df['course_name'])
    df['instructor_enc'] = le.fit_transform(df['instructor'])
    df['difficulty_enc'] = le.fit_transform(df['difficulty_level'])
    df['material_enc'] = le.fit_transform(df['study_material_available'])

    # Normalize numeric features
    scaler = MinMaxScaler()
    features_to_scale = ['course_duration_hours', 'course_price', 'rating',
                        'feedback_score', 'time_spent_hours', 'previous_courses_taken', 'enrollment_numbers']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return df

df = load_data()

# Popularity-Based Recommendation
def get_popular_courses(df, top_n=5):
    top_courses = df.sort_values(by='enrollment_numbers', ascending=False)
    return top_courses[['course_name', 'enrollment_numbers']].head(top_n)

# Content-Based Filtering
def get_similar_courses_based_on_input(enrollment, interest, engagement, top_n=5):
    # Create a dummy input vector
    user_input = pd.DataFrame([[enrollment, interest, engagement]],
                            columns=['enrollment_numbers', 'previous_courses_taken', 'time_spent_hours'])
    # Scale user input to match dataset scale
    scaler_input = MinMaxScaler()
    temp = df[['enrollment_numbers', 'previous_courses_taken', 'time_spent_hours']]
    scaler_input.fit(temp)
    user_input_scaled = scaler_input.transform(user_input)

    # Calculate cosine similarity between input and existing courses
    data_scaled = scaler_input.transform(temp)
    similarities = cosine_similarity(user_input_scaled, data_scaled).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    return df.iloc[top_indices][['course_id', 'course_name', 'enrollment_numbers', 'rating']]

# Recommendation Based on Existing Enrollment IDs
def recommend_based_on_enrollment_ids(enrollment_ids, df):
    results = {}
    
    # Get popular courses
    results['popular_courses'] = get_popular_courses(df)
    
    # Get personalized recommendations
    personalized_recs = {}
    for eid in enrollment_ids:
        if eid in df['course_id'].values:
            course_row = df[df['course_id'] == eid][['enrollment_numbers', 'previous_courses_taken', 'time_spent_hours']].iloc[0]
            enrollment, interest, engagement = course_row
            personalized_recs[eid] = get_similar_courses_based_on_input(enrollment, interest, engagement)
        else:
            personalized_recs[eid] = None
    
    results['personalized'] = personalized_recs
    return results

# Streamlit UI
st.title("Course Recommendation System")

# Input section
st.header("Input Course IDs")
st.write("Enter one or more course IDs to get personalized recommendations")

course_ids_input = st.text_input("Enter course IDs (comma separated)", "101")

# Process input
course_ids = [int(id.strip()) for id in course_ids_input.split(",") if id.strip().isdigit()]

if st.button("Get Recommendations"):
    if not course_ids:
        st.warning("Please enter at least one valid course ID")
    else:
        with st.spinner("Generating recommendations..."):
            results = recommend_based_on_enrollment_ids(course_ids, df)
            
            # Display popular courses
            st.subheader("Top 5 Popular Courses")
            st.dataframe(results['popular_courses'])
            
            # Display personalized recommendations
            st.subheader("Personalized Recommendations")
            for eid, recs in results['personalized'].items():
                if recs is not None:
                    st.write(f"Recommendations based on course ID {eid}:")
                    st.dataframe(recs)
                else:
                    st.warning(f"Course ID {eid} not found in dataset")

# Optional: Show sample course IDs
st.sidebar.header("Sample Course IDs")
st.sidebar.write("Try these sample course IDs:")
sample_ids = [101, 4787, 2603, 3350, 9635]
st.sidebar.write(", ".join(map(str, sample_ids)))

# Optional: Show raw data
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(df)