import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Sleep & Health Decision Dashboard",
    page_icon="💤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard Components
# Header Section
def create_header_section():
    """Create the header section of the dashboard"""
    st.title("🛌 Sleep & Health Decision Dashboard")
    
    st.markdown("""
    This dashboard helps you understand the relationship between sleep patterns, 
    physical activity, stress, and overall health metrics. Use it to:
    
    * Identify key health factors affecting sleep quality
    * Discover relationships between physical activity, stress, and sleep
    * Compare health metrics across different demographic groups
    * Get personalized recommendations based on data patterns
    
    **Use the sidebar to filter data and navigate between dashboard sections.**
    """)

# Overview Metrics Section
def create_overview_metrics(df):
    """Create key metrics overview section"""
    st.header("📊 Key Metrics Overview")
    
    # Create three columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_sleep = df['Sleep Duration'].mean()
        st.metric(
            label="Average Sleep Duration",
            value=f"{avg_sleep:.2f} hrs",
            delta=f"{avg_sleep - 7:.2f}" if avg_sleep - 7 != 0 else "0",
            delta_color="normal" if avg_sleep >= 7 else "inverse"
        )
    
    with col2:
        avg_quality = df['Quality of Sleep'].mean()
        st.metric(
            label="Average Sleep Quality",
            value=f"{avg_quality:.1f}/10",
            delta=f"{avg_quality - 7:.1f}" if avg_quality - 7 != 0 else "0",
            delta_color="normal" if avg_quality >= 7 else "inverse"
        )
    
    with col3:
        avg_stress = df['Stress Level'].mean()
        st.metric(
            label="Average Stress Level",
            value=f"{avg_stress:.1f}/10",
            delta=f"{5 - avg_stress:.1f}" if 5 - avg_stress != 0 else "0",
            delta_color="inverse" if avg_stress > 5 else "normal"
        )
    
    with col4:
        avg_activity = df['Physical Activity Level'].mean()
        st.metric(
            label="Avg Physical Activity",
            value=f"{avg_activity:.0f} min/day",
            delta=f"{avg_activity - 60:.0f}" if avg_activity - 60 != 0 else "0",
            delta_color="normal" if avg_activity >= 60 else "inverse"
        )
    
    # Show population distribution
    st.subheader("Population Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution
        gender_counts = df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        
        fig = px.pie(
            gender_counts, 
            values='Count', 
            names='Gender',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
            title_text="Gender Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age distribution
        fig = px.histogram(
            df, 
            x="Age",
            color_discrete_sequence=['#3498db'],
            nbins=10
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
            title_text="Age Distribution",
            xaxis_title="Age",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)

# Sleep Analysis Section
def create_sleep_analysis_section(df, summary=False):
    """Create sleep analysis section"""
    if not summary:
        st.header("😴 Sleep Analysis")
        st.markdown("""
        This section helps you understand sleep patterns and identify factors affecting sleep quality.
        """)
    else:
        st.subheader("😴 Sleep Analysis Summary")
    
    # If this is a summary view, keep it concise
    if summary:
        # Show one key chart for the summary
        fig = px.scatter(
            df, 
            x="Sleep Duration", 
            y="Quality of Sleep",
            color="Sleep Disorder",
            size="Stress Level",
            hover_data=["Age", "Gender", "BMI Category"],
            color_discrete_sequence=px.colors.qualitative.Safe,
            opacity=0.7,
            title="Sleep Duration vs Quality"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        return
    
    # For full view, show multiple analyses
    col1, col2 = st.columns(2)
    
    with col1:
        # Sleep duration by disorder type
        fig = px.box(
            df, 
            x="Sleep Disorder", 
            y="Sleep Duration",
            color="Sleep Disorder",
            color_discrete_sequence=px.colors.qualitative.Safe,
            points="all"
        )
        fig.update_layout(
            title_text="Sleep Duration by Disorder Type",
            xaxis_title="Sleep Disorder",
            yaxis_title="Sleep Duration (hours)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Average sleep metrics by occupation
        sleep_by_occupation = df.groupby('Occupation')[['Sleep Duration', 'Quality of Sleep']].mean().reset_index()
        sleep_by_occupation = sleep_by_occupation.sort_values('Sleep Duration')
        
        if len(sleep_by_occupation) > 6:
            sleep_by_occupation = sleep_by_occupation.iloc[-6:]  # Show top 6 for readability
            
        fig = px.bar(
            sleep_by_occupation,
            x="Occupation",
            y=["Sleep Duration", "Quality of Sleep"],
            barmode="group",
            title="Sleep Metrics by Occupation",
            color_discrete_sequence=['#3498db', '#2ecc71']
        )
        fig.update_layout(
            xaxis_title="Occupation",
            yaxis_title="Value",
            legend_title="Metric"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot of sleep duration vs quality
        fig = px.scatter(
            df, 
            x="Sleep Duration", 
            y="Quality of Sleep",
            color="Sleep Disorder",
            symbol="Gender",
            size="Stress Level",
            hover_data=["Age", "BMI Category", "Occupation"],
            title="Sleep Duration vs Quality Correlation"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sleep quality distribution
        fig = px.histogram(
            df,
            x="Quality of Sleep",
            color="Sleep Disorder",
            marginal="box",
            nbins=10,
            title="Sleep Quality Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sleep patterns by age and gender
    st.subheader("Sleep Patterns by Age and Gender")
    
    # Create pivot tables
    sleep_by_age_gender = df.pivot_table(
        index="Gender",
        columns=pd.cut(df["Age"], bins=[25, 35, 45, 55, 65]),
        values="Sleep Duration",
        aggfunc="mean"
    ).round(2)
    
    quality_by_age_gender = df.pivot_table(
        index="Gender",
        columns=pd.cut(df["Age"], bins=[25, 35, 45, 55, 65]),
        values="Quality of Sleep",
        aggfunc="mean"
    ).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Average Sleep Duration by Age Group and Gender**")
        st.dataframe(sleep_by_age_gender, use_container_width=True)
    
    with col2:
        st.markdown("**Average Sleep Quality by Age Group and Gender**")
        st.dataframe(quality_by_age_gender, use_container_width=True)

# Physical Activity Section (simplified version)
def create_physical_activity_section(df, summary=False):
    """Create physical activity analysis section"""
    if not summary:
        st.header("🏃‍♂️ Physical Activity Analysis")
        st.markdown("""
        This section examines the relationship between physical activity and other health metrics.
        """)
    else:
        st.subheader("🏃‍♂️ Physical Activity Summary")
    
    # If this is summary view, keep it concise
    if summary:
        # Show relationship between physical activity and sleep quality
        fig = px.scatter(
            df,
            x="Physical Activity Level",
            y="Quality of Sleep",
            color="BMI Category",
            size="Daily Steps",
            size_max=15,
            opacity=0.7,
            title="Physical Activity vs Sleep Quality",
            color_discrete_sequence=px.colors.qualitative.G10
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        return
    
    # Full analysis with multiple charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Physical activity by gender and BMI
        fig = px.box(
            df,
            x="BMI Category",
            y="Physical Activity Level",
            color="Gender",
            notched=True,
            title="Physical Activity by BMI Category and Gender"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Relationship between physical activity and sleep quality
        fig = px.scatter(
            df,
            x="Physical Activity Level",
            y="Quality of Sleep",
            color="BMI Category",
            size="Daily Steps",
            trendline="ols",
            title="Physical Activity vs Sleep Quality"
        )
        st.plotly_chart(fig, use_container_width=True)

# Health Metrics Section (simplified version)
def create_health_metrics_section(df, summary=False):
    """Create health metrics analysis section"""
    if not summary:
        st.header("💓 Health Metrics Analysis")
        st.markdown("""
        This section analyzes key health indicators and their relationships.
        """)
    else:
        st.subheader("💓 Health Metrics Summary")
    
    # If this is summary view, keep it concise
    if summary:
        # Show heart rate and blood pressure distributions
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Heart Rate Distribution", "BMI Distribution"])
        
        # Heart rate histogram
        fig.add_trace(
            go.Histogram(
                x=df["Heart Rate"],
                name="Heart Rate",
                marker_color="#3498db"
            ),
            row=1, col=1
        )
        
        # BMI Category count
        bmi_counts = df["BMI Category"].value_counts().reset_index()
        bmi_counts.columns = ["BMI Category", "Count"]
        
        fig.add_trace(
            go.Bar(
                x=bmi_counts["BMI Category"],
                y=bmi_counts["Count"],
                marker_color="#2ecc71"
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        return
    
    # Full analysis with multiple charts
    col1, col2 = st.columns(2)
    
    with col1:
        # BMI distribution by gender
        fig = px.histogram(
            df,
            x="BMI Category",
            color="Gender",
            barmode="group",
            title="BMI Distribution by Gender"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Heart rate by sleep disorder
        fig = px.box(
            df,
            x="Sleep Disorder",
            y="Heart Rate",
            color="Sleep Disorder",
            notched=True,
            points="all",
            title="Heart Rate by Sleep Disorder"
        )
        st.plotly_chart(fig, use_container_width=True)

# Stress Analysis Section (simplified version)
def create_stress_analysis_section(df, summary=False):
    """Create stress analysis section"""
    if not summary:
        st.header("😓 Stress Analysis")
        st.markdown("""
        This section explores stress levels and their impact on sleep and health.
        """)
    else:
        st.subheader("😓 Stress Analysis Summary")
    
    # If this is summary view, keep it concise
    if summary:
        # Show stress level impact on sleep quality
        fig = px.scatter(
            df, 
            x="Stress Level", 
            y="Quality of Sleep",
            color="Sleep Disorder",
            size="Physical Activity Level",
            opacity=0.7,
            trendline="ols",
            title="Stress Level vs Sleep Quality"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        return
    
    # Full analysis with multiple charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Stress levels by occupation
        stress_by_occupation = df.groupby('Occupation')['Stress Level'].mean().sort_values(ascending=False).reset_index()
        
        if len(stress_by_occupation) > 6:
            stress_by_occupation = stress_by_occupation.head(6)  # Show top 6 for readability
            
        fig = px.bar(
            stress_by_occupation,
            x="Occupation",
            y="Stress Level",
            color="Stress Level",
            title="Average Stress Level by Occupation",
            color_continuous_scale=px.colors.sequential.Reds
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stress level impact on sleep quality
        fig = px.scatter(
            df, 
            x="Stress Level", 
            y="Quality of Sleep",
            color="Sleep Disorder",
            size="Sleep Duration",
            hover_data=["Age", "Gender", "Occupation"],
            trendline="ols",
            title="Stress Level vs Sleep Quality"
        )
        st.plotly_chart(fig, use_container_width=True)

# Recommendations Section (simplified version)
def create_recommendations_section(df):
    """Create personalized recommendations section"""
    st.header("🎯 Recommendations & Insights")
    st.markdown("""
    Based on the data analysis, here are key insights and recommendations to improve sleep and health.
    """)
    
    # Calculate insights
    avg_sleep = df['Sleep Duration'].mean()
    avg_quality = df['Quality of Sleep'].mean()
    avg_activity = df['Physical Activity Level'].mean()
    avg_stress = df['Stress Level'].mean()
    
    # Sleep disorder prevalence
    sleep_disorder_counts = df['Sleep Disorder'].value_counts(normalize=True) * 100
    has_disorder = 100 - sleep_disorder_counts.get('None', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Insights")
        
        st.markdown(f"""
        * **Sleep Duration**: Average sleep duration is **{avg_sleep:.2f} hours** per night
        * **Sleep Quality**: Average sleep quality rating is **{avg_quality:.1f}/10**
        * **Sleep Disorders**: **{has_disorder:.1f}%** of individuals have some form of sleep disorder
        * **Physical Activity**: Average physical activity is **{avg_activity:.1f} minutes** per day
        * **Stress Level**: Average stress level rating is **{avg_stress:.1f}/10**
        """)
        
        # Create summary for BMI categories
        st.markdown("### BMI Category Analysis")
        
        bmi_summary = df.groupby('BMI Category').agg({
            'Sleep Duration': 'mean',
            'Quality of Sleep': 'mean',
            'Stress Level': 'mean',
            'Physical Activity Level': 'mean'
        }).round(2)
        
        st.dataframe(bmi_summary, use_container_width=True)
    
    with col2:
        st.subheader("Recommendations")
        
        if avg_sleep < 7:
            st.markdown("""
            * 😴 **Increase Sleep Duration**: The average sleep duration is below the recommended 7-8 hours. 
              Consider establishing a regular sleep schedule and bedtime routine.
            """)
        
        if avg_activity < 60:
            st.markdown("""
            * 🏃‍♀️ **Boost Physical Activity**: Data shows a positive correlation between physical activity and sleep quality. 
              Aim for at least 30-60 minutes of moderate activity daily.
            """)
        
        if avg_stress > 5:
            st.markdown("""
            * 🧘‍♂️ **Manage Stress Levels**: Higher stress is associated with poorer sleep quality. 
              Consider stress-reduction techniques like meditation, deep breathing, or mindfulness.
            """)
        
        # BMI-specific recommendations
        if 'Overweight' in df['BMI Category'].unique() or 'Obese' in df['BMI Category'].unique():
            st.markdown("""
            * ⚖️ **Weight Management**: Individuals in higher BMI categories show higher prevalence of sleep disorders. 
              A balanced diet and regular exercise may help improve sleep quality.
            """)
        
        # General health recommendations
        st.markdown("""
        * 📱 **Digital Detox**: Limit screen time before bed to improve sleep quality.
          The blue light from screens can interfere with melatonin production.
        
        * 🥤 **Limit Stimulants**: Reduce caffeine and alcohol consumption, especially in the afternoon and evening.
        """)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('sleep_health_data.csv')
        # Convert blood pressure to numeric for analysis
        df['Systolic_BP'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
        df['Diastolic_BP'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return a sample dataframe if file not found
        return pd.DataFrame({
            'Person ID': list(range(1, 11)),
            'Gender': ['Male', 'Female'] * 5,
            'Age': [30, 35, 40, 45, 50, 28, 33, 38, 43, 48],
            'Occupation': ['Engineer', 'Doctor', 'Teacher', 'Nurse', 'Manager'] * 2,
            'Sleep Duration': [7, 6.5, 8, 7.2, 6, 7.5, 6.8, 7.8, 6.2, 6.5],
            'Quality of Sleep': [8, 7, 9, 6, 5, 8, 7, 8, 6, 5],
            'Physical Activity Level': [60, 45, 30, 75, 40, 65, 50, 70, 35, 55],
            'Stress Level': [4, 7, 3, 6, 8, 3, 6, 5, 8, 7],
            'BMI Category': ['Normal', 'Normal', 'Overweight', 'Normal', 'Overweight', 'Normal', 'Normal', 'Normal', 'Overweight', 'Obese'],
            'Blood Pressure': ['120/80', '125/85', '130/90', '120/75', '140/95', '115/75', '125/80', '120/80', '135/90', '145/95'],
            'Heart Rate': [70, 75, 80, 68, 88, 65, 72, 70, 84, 90],
            'Daily Steps': [8000, 7000, 6000, 10000, 5000, 9000, 7500, 11000, 5500, 4000],
            'Sleep Disorder': ['None', 'None', 'Sleep Apnea', 'None', 'Insomnia', 'None', 'None', 'None', 'Insomnia', 'Sleep Apnea'],
            'Systolic_BP': [120, 125, 130, 120, 140, 115, 125, 120, 135, 145],
            'Diastolic_BP': [80, 85, 90, 75, 95, 75, 80, 80, 90, 95]
        })

# Main function
def main():
    try:
        # Create sidebar
        st.sidebar.title("Dashboard Navigation")
        
        # Add filter options to sidebar
        st.sidebar.subheader("Filter Data")
        
        # Load data
        df = load_data()
        
        # Create filters in sidebar
        gender_filter = st.sidebar.multiselect(
            "Select Gender",
            options=df['Gender'].unique(),
            default=df['Gender'].unique()
        )
        
        age_range = st.sidebar.slider(
            "Age Range",
            min_value=int(df['Age'].min()),
            max_value=int(df['Age'].max()),
            value=[int(df['Age'].min()), int(df['Age'].max())]
        )
        
        occupation_filter = st.sidebar.multiselect(
            "Select Occupation",
            options=df['Occupation'].unique(),
            default=df['Occupation'].unique()[:3]  # Default to showing top 3
        )
        
        sleep_disorder_filter = st.sidebar.multiselect(
            "Sleep Disorder",
            options=df['Sleep Disorder'].unique(),
            default=df['Sleep Disorder'].unique()
        )
        
        # Filter data based on selections
        filtered_df = df[
            (df['Gender'].isin(gender_filter)) &
            (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
            (df['Occupation'].isin(occupation_filter)) &
            (df['Sleep Disorder'].isin(sleep_disorder_filter))
        ]
        
        # Show filter information
        st.sidebar.markdown(f"**Showing data for {len(filtered_df)} individuals**")
        
        # Include navigation in sidebar
        pages = ["Dashboard Overview", "Sleep Analysis", "Physical Activity", 
                "Health Metrics", "Stress Analysis", "Recommendations"]
        
        selected_page = st.sidebar.radio("Navigate to", pages)
        
        # Create header section
        create_header_section()
        
        # Display specific sections based on navigation selection
        if selected_page == "Dashboard Overview":
            create_overview_metrics(filtered_df)
            # Show summary of all sections
            col1, col2 = st.columns(2)
            with col1:
                create_sleep_analysis_section(filtered_df, summary=True)
                create_health_metrics_section(filtered_df, summary=True)
            with col2:
                create_physical_activity_section(filtered_df, summary=True)
                create_stress_analysis_section(filtered_df, summary=True)
        
        elif selected_page == "Sleep Analysis":
            create_sleep_analysis_section(filtered_df)
        
        elif selected_page == "Physical Activity":
            create_physical_activity_section(filtered_df)
        
        elif selected_page == "Health Metrics":
            create_health_metrics_section(filtered_df)
        
        elif selected_page == "Stress Analysis":
            create_stress_analysis_section(filtered_df)
        
        elif selected_page == "Recommendations":
            create_recommendations_section(filtered_df)

        # Add information about data source and dashboard purpose at the bottom
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: gray; font-size: 0.8em;'>
            Data based on sleep and health metrics. Dashboard designed to support decision-making by minimizing cognitive load and highlighting key relationships.
            </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Try reloading the page or check if all required files are in place.")

if __name__ == "__main__":
    main()