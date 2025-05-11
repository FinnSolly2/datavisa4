import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Sleep Health & Lifestyle Dashboard",
    page_icon="😴",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    # Assuming data is read from a CSV file

    df = pd.read_csv("sleep_health_data.csv")
    
    # Split blood pressure into separate columns for analysis
    df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    
    return df

# Load the data
df = load_data()

# Define color palette for consistent visual language
primary_color = "#1E88E5"  # Blue
secondary_color = "#FFC107"  # Amber
tertiary_color = "#4CAF50"  # Green
warning_color = "#F44336"  # Red

# Disorder color mapping for consistent visual encoding
disorder_colors = {
    'None': tertiary_color,
    'Insomnia': warning_color,
    'Sleep Apnea': secondary_color
}

# Custom CSS for better data-ink ratio and visual hierarchy
st.markdown("""
<style>
    /* Improve visual hierarchy with typography */
    h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    h2 {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1rem;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    h3 {
        font-size: 1.3rem;
        font-weight: 500;
    }
    /* Remove unnecessary decoration for better data-ink ratio */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    /* Improve separation between sections */
    .section-divider {
        margin: 1.5rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard Title - Clear visual hierarchy starts here
st.title("😴 Sleep Health & Lifestyle Analysis")

# Top-level navigation tabs for progressive disclosure
tab1, tab2, tab3 = st.tabs([
    "🔍 Key Insights", 
    "🔄 Comparative Analysis", 
    "🧪 Advanced Analytics"
])

# -- TAB 1: KEY INSIGHTS --
with tab1:
    # Brief introduction (minimizing cognitive load with clear explanation)
    st.markdown("""
    This dashboard helps you understand the relationships between sleep patterns, lifestyle factors, 
    and health conditions. The analysis focuses on identifying key factors affecting sleep quality 
    and potential interventions to improve sleep health.
    """)
    
    # First row - Key metrics and distribution for situational awareness
    st.subheader("Population Overview")
    
    # Create 4 columns for key metrics to fit within working memory limits
    col1, col2, col3, col4 = st.columns(4)
    
    # Metric 1 - Average Sleep Duration
    avg_sleep = df['Sleep Duration'].mean()
    with col1:
        st.metric(
            "Avg. Sleep Duration", 
            f"{avg_sleep:.1f} hrs",
            delta=f"{avg_sleep - 7:.1f}" if avg_sleep - 7 != 0 else None,
            delta_color="normal" if avg_sleep >= 7 else "off"
        )
    
    # Metric 2 - Average Sleep Quality
    avg_quality = df['Quality of Sleep'].mean()
    with col2:
        st.metric(
            "Avg. Sleep Quality", 
            f"{avg_quality:.1f}/10",
            delta=f"{avg_quality - 5:.1f}" if avg_quality - 5 != 0 else None,
            delta_color="normal" if avg_quality >= 5 else "off"
        )
    
    # Metric 3 - Sleep Disorder Prevalence  
    disorder_prevalence = (df['Sleep Disorder'] != 'None').mean() * 100
    with col3:
        st.metric(
            "Sleep Disorder Prevalence", 
            f"{disorder_prevalence:.1f}%",
            delta=f"{-disorder_prevalence:.1f}%" if disorder_prevalence > 0 else None,
            delta_color="off" if disorder_prevalence > 0 else "normal"
        )
    
    # Metric 4 - Average Physical Activity 
    avg_activity = df['Physical Activity Level'].mean()
    with col4:
        st.metric(
            "Avg. Physical Activity", 
            f"{avg_activity:.0f} min/day",
            delta=f"{avg_activity - 60:.0f}" if avg_activity - 60 != 0 else None,
            delta_color="normal" if avg_activity >= 60 else "off"
        )
    
    # Second row - Exception identification (sleep disorders distribution) and age distribution
    st.subheader("Population Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of sleep disorders - for identifying exceptions
        disorder_count = df['Sleep Disorder'].value_counts().reset_index()
        disorder_count.columns = ['Sleep Disorder', 'Count']
        
        fig = px.pie(
            disorder_count, 
            values='Count', 
            names='Sleep Disorder',
            color='Sleep Disorder',
            color_discrete_map=disorder_colors,
            hole=0.4,
            title="Sleep Disorder Distribution"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=-0.2)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Age distribution - for understanding the population
        age_bins = [18, 30, 40, 50, 60, 70, 80]
        age_labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
        df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
        
        age_dist = df.groupby(['Age Group', 'Gender']).size().reset_index(name='Count')
        
        fig = px.bar(
            age_dist, 
            x='Age Group', 
            y='Count', 
            color='Gender',
            barmode='group',
            title="Age and Gender Distribution",
            color_discrete_sequence=[primary_color, secondary_color]
        )
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=-0.2)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Third row - Key relationship: Sleep Duration vs Quality
    st.subheader("Key Relationship: Sleep Duration vs Quality")
    
    # Create columns for the chart and key insights to emphasize comparative analysis
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Scatter plot with sleep duration vs quality with disorders highlighted
        fig = px.scatter(
            df, 
            x='Sleep Duration', 
            y='Quality of Sleep',
            color='Sleep Disorder',
            color_discrete_map=disorder_colors,
            size='Stress Level',
            hover_data=['Age', 'Gender', 'Occupation', 'Stress Level'],
            opacity=0.7,
            title="Sleep Duration vs. Quality by Sleep Disorder",
        )
        
        # Add reference lines for recommended sleep (7-9 hours) and good quality (>7)
        fig.add_shape(
            type="rect",
            x0=7, x1=9, y0=7, y1=10,
            line=dict(color="rgba(0,100,0,0.2)", width=2),
            fillcolor="rgba(0,100,0,0.1)",
            layer="below"
        )
        
        # Add annotation for the optimal zone
        fig.add_annotation(
            x=8, y=8.5,
            text="Optimal Sleep Zone",
            showarrow=False,
            font=dict(size=12, color="green")
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=-0.15)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Key insights callout box for causal investigation
        st.markdown("""
        ### Key Insights
        
        * **Clear Separation:** Sleep disorder cases cluster in the lower-left (short duration, poor quality)
        
        * **Threshold Effect:** Most healthy sleepers get >7 hours and rate quality >7/10
        
        * **Stress Impact:** Larger bubbles (higher stress) tend to correlate with poorer sleep
        """)
        
        # Simple action metrics for action determination
        st.markdown("### Recommended Targets")
        st.metric("Target Sleep Duration", "7-9 hours")
        st.metric("Target Sleep Quality", ">7/10")
    
    # Fourth row - Interactive exploration
    st.subheader("Explore Factors Affecting Sleep Quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select which factor to explore
        factor = st.selectbox(
            "Select factor to explore:",
            ["Physical Activity Level", 
             "Stress Level", 
             "BMI Category",
             "Heart Rate (bpm)",
             "Daily Steps"]
        )
        
        # Adapt chart type to data type for chart appropriateness
        if factor == "BMI Category":
            # For categorical data, use boxplot
            fig = px.box(
                df, 
                x=factor, 
                y="Quality of Sleep",
                color="Sleep Disorder",
                color_discrete_map=disorder_colors,
                title=f"Sleep Quality by {factor}",
                category_orders={"BMI Category": ["Underweight", "Normal", "Overweight", "Obese"]}
            )
        else:
            # For continuous data, use scatter plot
            fig = px.scatter(
                df, 
                x=factor, 
                y="Quality of Sleep",
                color="Sleep Disorder",
                color_discrete_map=disorder_colors,
                opacity=0.7,
                trendline="ols",
                title=f"Sleep Quality vs {factor}"
            )
        
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Similar chart for sleep duration to allow comparison
        if factor == "BMI Category":
            fig = px.box(
                df, 
                x=factor, 
                y="Sleep Duration",
                color="Sleep Disorder",
                color_discrete_map=disorder_colors,
                title=f"Sleep Duration by {factor}",
                category_orders={"BMI Category": ["Underweight", "Normal", "Overweight", "Obese"]}
            )
        else:
            fig = px.scatter(
                df, 
                x=factor, 
                y="Sleep Duration",
                color="Sleep Disorder",
                color_discrete_map=disorder_colors,
                opacity=0.7,
                trendline="ols",
                title=f"Sleep Duration vs {factor}"
            )
        
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

# -- TAB 2: COMPARATIVE ANALYSIS --
with tab2:
    st.subheader("Comparative Analysis")
    
    # Filters for comparative analysis - at the top for context
    st.markdown("### Filter Data for Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Gender filter
        gender_filter = st.multiselect(
            "Gender:",
            options=df['Gender'].unique(),
            default=df['Gender'].unique()
        )
    
    with col2:
        # Age group filter
        age_group_filter = st.multiselect(
            "Age Group:",
            options=df['Age Group'].unique(),
            default=df['Age Group'].unique()
        )
    
    with col3:
        # Occupation filter with select all option
        all_occupations = df['Occupation'].unique()
        occupation_filter = st.multiselect(
            "Occupation:",
            options=['All'] + list(all_occupations),
            default=['All']
        )
        
        if 'All' in occupation_filter:
            occupation_filter = list(all_occupations)
    
    # Apply filters
    filtered_df = df[
        (df['Gender'].isin(gender_filter)) &
        (df['Age Group'].isin(age_group_filter)) &
        (df['Occupation'].isin(occupation_filter))
    ]
    
    # Show sample size for context
    st.info(f"Selected sample: {len(filtered_df)} out of {len(df)} individuals")
    
    # Comparative Analysis Section - Sleep Metrics by Disorder
    st.markdown("### Sleep Metrics by Disorder Type")
    
    # Calculate mean values for each disorder type
    disorder_metrics = filtered_df.groupby('Sleep Disorder').agg({
        'Sleep Duration': 'mean',
        'Quality of Sleep': 'mean',
        'Physical Activity Level': 'mean',
        'Stress Level': 'mean',
        'Heart Rate': 'mean',
        'Daily Steps': 'mean'
    }).reset_index()
    
    # Create a selection for which metrics to compare
    metrics_options = [
        'Sleep Duration', 
        'Quality of Sleep', 
        'Physical Activity Level',
        'Stress Level',
        'Heart Rate',
        'Daily Steps'
    ]
    
    selected_metrics = st.multiselect(
        "Select metrics to compare:",
        options=metrics_options,
        default=metrics_options[:3]  # Default to first 3 for reduced cognitive load
    )
    
    if selected_metrics:
        # Create a long-format dataframe for the selected metrics
        plot_data = pd.melt(
            disorder_metrics, 
            id_vars=['Sleep Disorder'],
            value_vars=selected_metrics,
            var_name='Metric',
            value_name='Value'
        )
        
        # Create a bar chart with facets for each metric
        fig = px.bar(
            plot_data,
            x='Sleep Disorder',
            y='Value',
            color='Sleep Disorder',
            color_discrete_map=disorder_colors,
            facet_col='Metric',
            title="Comparison of Metrics by Sleep Disorder",
            barmode='group',
            facet_col_wrap=min(len(selected_metrics), 3),  # At most 3 metrics per row
            height=300 * (len(selected_metrics) // 3 + (1 if len(selected_metrics) % 3 > 0 else 0))
        )
        
        # Improve layout
        fig.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=True
        )
        
        # Update facet titles
        for i, metric in enumerate(selected_metrics):
            fig.layout.annotations[i].text = metric.split('(')[0].strip()
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Occupation Analysis - categorical comparison
    st.markdown("### Sleep Quality by Occupation")
    
    # Group data by occupation
    occupation_data = filtered_df.groupby('Occupation').agg({
        'Sleep Duration': 'mean',
        'Quality of Sleep': 'mean',
        'Sleep Disorder': lambda x: (x != 'None').mean() * 100
    }).reset_index()
    
    occupation_data.columns = ['Occupation', 'Avg. Sleep Duration', 'Avg. Sleep Quality', 'Disorder Prevalence (%)']
    
    # Sort by sleep quality for better comparison
    occupation_data = occupation_data.sort_values('Avg. Sleep Quality', ascending=False)
    
    # Create a horizontal bar chart for occupations
    fig = px.bar(
        occupation_data,
        y='Occupation',
        x='Avg. Sleep Quality',
        orientation='h',
        title="Average Sleep Quality by Occupation",
        color='Avg. Sleep Quality',
        color_continuous_scale='blues',
        height=400
    )
    
    # Add a reference line for the overall average
    overall_avg = filtered_df['Quality of Sleep'].mean()
    fig.add_vline(
        x=overall_avg, 
        line_width=2, 
        line_dash="dash", 
        line_color="gray",
        annotation_text=f"Overall Avg: {overall_avg:.1f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Show the full data table
        st.markdown("### Occupation Data")
        st.dataframe(occupation_data, hide_index=True)
    
    # Age Group Analysis
    st.markdown("### Age-Related Sleep Patterns")
    
    # Group data by age group
    age_data = filtered_df.groupby(['Age Group', 'Gender']).agg({
        'Sleep Duration': 'mean',
        'Quality of Sleep': 'mean',
        'Sleep Disorder': lambda x: (x != 'None').mean() * 100
    }).reset_index()
    
    age_data.columns = ['Age Group', 'Gender', 'Avg. Sleep Duration', 'Avg. Sleep Quality', 'Disorder Prevalence (%)']
    
    # Create tabs for different metrics
    age_tab1, age_tab2, age_tab3 = st.tabs(["Sleep Duration", "Sleep Quality", "Disorder Prevalence"])
    
    with age_tab1:
        fig = px.line(
            age_data, 
            x='Age Group', 
            y='Avg. Sleep Duration',
            color='Gender',
            markers=True,
            title="Average Sleep Duration by Age Group and Gender",
            color_discrete_sequence=[primary_color, secondary_color]
        )
        
        # Add reference range for recommended sleep
        fig.add_hrect(
            y0=7, y1=9,
            line_width=0,
            fillcolor="rgba(0,100,0,0.1)",
            annotation_text="Recommended Range",
            annotation_position="top right"
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with age_tab2:
        fig = px.line(
            age_data, 
            x='Age Group', 
            y='Avg. Sleep Quality',
            color='Gender',
            markers=True,
            title="Average Sleep Quality by Age Group and Gender",
            color_discrete_sequence=[primary_color, secondary_color]
        )
        
        # Add reference line for good sleep quality
        fig.add_hline(
            y=7,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text="Good Quality Threshold",
            annotation_position="top right"
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with age_tab3:
        fig = px.line(
            age_data, 
            x='Age Group', 
            y='Disorder Prevalence (%)',
            color='Gender',
            markers=True,
            title="Sleep Disorder Prevalence by Age Group and Gender",
            color_discrete_sequence=[primary_color, secondary_color]
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

# -- TAB 3: ADVANCED ANALYTICS --
with tab3:
    st.subheader("Advanced Analytics")
    
    # Correlation Analysis
    st.markdown("### Correlation Heatmap")
    
    # Select numerical columns for correlation
    numeric_cols = [
        'Age', 
        'Sleep Duration', 
        'Quality of Sleep',
        'Physical Activity Level', 
        'Stress Level',
        'Heart Rate', 
        'Daily Steps',
        'Systolic',
        'Diastolic'
    ]
    
    # Calculate correlation matrix
    corr = df[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title="Correlation Between Metrics",
        aspect="auto"
    )
    
    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk Factor Analysis
    st.markdown("### Sleep Disorder Risk Factors")
    
    # Create three groups for risk analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Physical Activity Analysis
        activity_bins = [0, 30, 60, 90, 120]
        activity_labels = ['Low (0-30)', 'Moderate (30-60)', 'High (60-90)', 'Very High (90+)']
        
        df['Activity Level'] = pd.cut(
            df['Physical Activity Level'], 
            bins=activity_bins, 
            labels=activity_labels
        )
        
        activity_risk = df.groupby('Activity Level')['Sleep Disorder'].apply(
            lambda x: (x != 'None').mean() * 100
        ).reset_index()
        activity_risk.columns = ['Activity Level', 'Disorder Risk (%)']
        
        fig = px.bar(
            activity_risk,
            x='Activity Level',
            y='Disorder Risk (%)',
            title="Sleep Disorder Risk by Physical Activity Level",
            color='Disorder Risk (%)',
            color_continuous_scale='reds_r',
            text_auto='.1f'
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stress Level Analysis
        stress_bins = [0, 3, 6, 10]
        stress_labels = ['Low (0-3)', 'Moderate (4-6)', 'High (7-10)']
        
        df['Stress Category'] = pd.cut(
            df['Stress Level'], 
            bins=stress_bins, 
            labels=stress_labels
        )
        
        stress_risk = df.groupby('Stress Category')['Sleep Disorder'].apply(
            lambda x: (x != 'None').mean() * 100
        ).reset_index()
        stress_risk.columns = ['Stress Category', 'Disorder Risk (%)']
        
        fig = px.bar(
            stress_risk,
            x='Stress Category',
            y='Disorder Risk (%)',
            title="Sleep Disorder Risk by Stress Level",
            color='Disorder Risk (%)',
            color_continuous_scale='reds',
            text_auto='.1f'
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # BMI and Sleep Patterns
    st.markdown("### BMI Category Analysis")
    
    # Create columns for BMI analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # BMI and Sleep Quality
        bmi_quality = df.groupby('BMI Category')['Quality of Sleep'].mean().reset_index()
        bmi_quality.columns = ['BMI Category', 'Avg. Sleep Quality']
        
        # Sort categories in logical order
        bmi_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
        bmi_quality['BMI Category'] = pd.Categorical(
            bmi_quality['BMI Category'], 
            categories=bmi_order, 
            ordered=True
        )
        bmi_quality = bmi_quality.sort_values('BMI Category')
        
        fig = px.bar(
            bmi_quality,
            x='BMI Category',
            y='Avg. Sleep Quality',
            title="Average Sleep Quality by BMI Category",
            color='BMI Category',
            color_discrete_sequence=['#FFC107', '#4CAF50', '#FF9800', '#F44336'],
            text_auto='.1f'
        )
        
        # Add a reference line for good sleep quality
        fig.add_hline(
            y=7,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text="Good Quality Threshold",
            annotation_position="top right"
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # BMI and Sleep Disorder Prevalence
        bmi_disorder = df.groupby('BMI Category')['Sleep Disorder'].apply(
            lambda x: (x != 'None').mean() * 100
        ).reset_index()
        bmi_disorder.columns = ['BMI Category', 'Disorder Prevalence (%)']
        
        # Sort categories in logical order
        bmi_disorder['BMI Category'] = pd.Categorical(
            bmi_disorder['BMI Category'], 
            categories=bmi_order, 
            ordered=True
        )
        bmi_disorder = bmi_disorder.sort_values('BMI Category')
        
        fig = px.bar(
            bmi_disorder,
            x='BMI Category',
            y='Disorder Prevalence (%)',
            title="Sleep Disorder Prevalence by BMI Category",
            color='BMI Category',
            color_discrete_sequence=['#FFC107', '#4CAF50', '#FF9800', '#F44336'],
            text_auto='.1f'
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Blood Pressure Analysis
    st.markdown("### Blood Pressure and Sleep Health")
    
    # Create BP categories
    df['BP Category'] = 'Normal'
    df.loc[(df['Systolic'] >= 120) & (df['Systolic'] < 130) & (df['Diastolic'] < 80), 'BP Category'] = 'Elevated'
    df.loc[(df['Systolic'] >= 130) | (df['Diastolic'] >= 80), 'BP Category'] = 'Hypertensive'
    
    # Count individuals in each category
    bp_counts = df['BP Category'].value_counts().reset_index()
    bp_counts.columns = ['BP Category', 'Count']
    
    # BP and Sleep Quality relationship
    bp_sleep = df.groupby('BP Category').agg({
        'Sleep Duration': 'mean',
        'Quality of Sleep': 'mean',
        'Sleep Disorder': lambda x: (x != 'None').mean() * 100
    }).reset_index()
    
    bp_sleep.columns = ['BP Category', 'Avg. Sleep Duration', 'Avg. Sleep Quality', 'Disorder Prevalence (%)']
    
    # Ensure order makes sense
    bp_order = ['Normal', 'Elevated', 'Hypertensive']
    bp_sleep['BP Category'] = pd.Categorical(bp_sleep['BP Category'], categories=bp_order, ordered=True)
    bp_sleep = bp_sleep.sort_values('BP Category')
    
    # Column layout for BP analysis
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Distribution of BP categories
        colors = {'Normal': tertiary_color, 'Elevated': secondary_color, 'Hypertensive': warning_color}
        fig = px.pie(
            bp_counts,
            values='Count',
            names='BP Category',
            title="Blood Pressure Category Distribution",
            color='BP Category',
            color_discrete_map=colors,
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # BP and sleep metrics relationship
        fig = px.bar(
            bp_sleep,
            x='BP Category',
            y=['Avg. Sleep Duration', 'Avg. Sleep Quality', 'Disorder Prevalence (%)'],
            barmode='group',
            title="Blood Pressure Category vs Sleep Metrics",
            color_discrete_sequence=[primary_color, secondary_color, warning_color]
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Final section - Intervention recommendation based on data
    st.markdown("### Recommendations for Improving Sleep Health")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Lifestyle Interventions
        
        Based on the data analysis, these factors show the strongest association with improved sleep:
        
        1. **Physical Activity**: Aim for 60-90 minutes of daily physical activity
        2. **Stress Management**: Keep stress levels below 4/10 for optimal sleep quality
        3. **Weight Management**: Maintain a BMI in the normal range
        4. **Consistent Sleep Schedule**: Those with better sleep quality have more consistent patterns
        """)
        
    with col2:
        st.markdown("""
        #### Monitoring Recommendations
        
        For individuals with sleep issues, monitor these key metrics:
        
        1. **Blood Pressure**: Higher BP correlates with sleep disorders
        2. **Stress Levels**: Track daily stress using a 1-10 scale
        3. **Physical Activity**: Use a step counter to ensure sufficient daily movement
        4. **Sleep Duration**: Use sleep tracking to ensure 7-9 hours per night
        """)
    
