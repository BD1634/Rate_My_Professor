import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, ks_2samp, norm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.svm import SVC

# Set page configuration
st.set_page_config(
    page_title="RateMyProfessor Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and introduction
st.title("🎓 Rate My Professor Data Analysis")
st.markdown("""
This application explores patterns and insights from RateMyProfessor data, 
analyzing differences in ratings based on gender, tags, and other factors.
""")

# Helper functions
def root_mean_squared_error(y_true, y_pred, sample_weight=None):
    output_errors = np.sqrt(
        mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput="raw_values"
        )
    )
    return output_errors

def calculate_vif(X):
    """Calculate Variance Inflation Factor (VIF) for each feature in the dataset."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def remove_high_vif_features(X, threshold=5.0):
    """Iteratively removes features with VIF above the given threshold."""
    results = []
    while True:
        vif_data = calculate_vif(X)
        max_vif = vif_data["VIF"].max()
        if max_vif > threshold:
            feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
            results.append(f"Removing '{feature_to_remove}' with VIF = {max_vif:.2f}")
            X = X.drop(columns=[feature_to_remove])
        else:
            break
    return X, results

def compute_classification_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (
        confusion_matrix,
        roc_auc_score,
        accuracy_score,
        f1_score,
        roc_curve,
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc_roc
    }

def cohen_d(x, y):
    """Calculate Cohen's d effect size"""
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    n_x = len(x)
    n_y = len(y)
    
    # Pooled standard deviation
    s_pooled = np.sqrt(((n_x - 1) * std_x**2 + (n_y - 1) * std_y**2) / (n_x + n_y - 2))
    return (mean_x - mean_y) / s_pooled

# Data loading sidebar
st.sidebar.header("⚙️ Data Configuration")

# In a real app, you'd upload the files here
# For now we'll mock the file upload
uploaded_data = st.sidebar.checkbox("Use uploaded data files", value=False)

if uploaded_data:
    num_file = st.sidebar.file_uploader("Upload Numerical Data (CSV)", type="csv")
    qual_file = st.sidebar.file_uploader("Upload Qualitative Data (CSV)", type="csv")
    tags_file = st.sidebar.file_uploader("Upload Tags Data (CSV)", type="csv")
    
    if num_file and qual_file and tags_file:
        cap_num_columns = [
            "Average Rating", "Average Difficulty", "Number of Ratings", 
            "Received a Pepper", "Proportion Would Take Again", 
            "Online Class Ratings", "Male", "Female"
        ]
        cap_qual_columns = ["Major", "University", "StateCode"]
        cap_tags_columns = [
            "tough_grader", "good_feedback", "respected", "lots_to_read",
            "participation_matters", "dont_skip_class", "lots_of_homework",
            "inspirational", "pop_quizzes", "accessible", "so_many_papers",
            "clear_grading", "hilarious", "test_heavy", "graded_by_few_things",
            "amazing_lectures", "caring", "extra_credit", "group_projects", "lecture_heavy"
        ]
        
        cap_num = pd.read_csv(num_file, header=None, names=cap_num_columns)
        cap_qual = pd.read_csv(qual_file, header=None, names=cap_qual_columns)
        cap_tags = pd.read_csv(tags_file, header=None, names=cap_tags_columns)
        data_loaded = True
    else:
        data_loaded = False
        st.warning("Please upload all three data files to continue.")
else:
    # For demo purposes, generate mock data
    st.sidebar.info("Using demonstration data since no files were uploaded.")
    
    # Mock data generation
    np.random.seed(42)
    
    # Sample size for demo
    sample_size = 1000
    
    # Generate cap_num data
    cap_num_columns = [
        "Average Rating", "Average Difficulty", "Number of Ratings", 
        "Received a Pepper", "Proportion Would Take Again", 
        "Online Class Ratings", "Male", "Female"
    ]
    
    cap_num = pd.DataFrame({
        "Average Rating": np.random.uniform(1, 5, sample_size),
        "Average Difficulty": np.random.uniform(1, 5, sample_size),
        "Number of Ratings": np.random.randint(1, 100, sample_size),
        "Received a Pepper": np.random.randint(0, 2, sample_size),
        "Proportion Would Take Again": np.random.uniform(0, 1, sample_size),
        "Online Class Ratings": np.random.uniform(1, 5, sample_size),
        "Male": np.random.choice([0, 1], sample_size, p=[0.4, 0.6]),
        "Female": np.random.choice([0, 1], sample_size, p=[0.6, 0.4])
    })
    
    # Make sure male and female are mutually exclusive
    for i in range(sample_size):
        if cap_num.loc[i, "Male"] == 1:
            cap_num.loc[i, "Female"] = 0
        elif cap_num.loc[i, "Female"] == 1:
            cap_num.loc[i, "Male"] = 0
    
    # Generate cap_qual data
    cap_qual_columns = ["Major", "University", "StateCode"]
    
    universities = ["New York University", "Harvard University", "Stanford University", 
                   "MIT", "UCLA", "UC Berkeley", "University of Michigan", 
                   "University of Washington", "Georgia Tech", "Cornell University"]
    
    majors = ["Computer Science", "Biology", "Chemistry", "Physics", "Mathematics", 
             "English", "History", "Psychology", "Business", "Economics"]
    
    state_codes = ["NY", "CA", "MA", "MI", "WA", "GA", "IL", "TX", "PA", "FL"]
    
    cap_qual = pd.DataFrame({
        "Major": np.random.choice(majors, sample_size),
        "University": np.random.choice(universities, sample_size),
        "StateCode": np.random.choice(state_codes, sample_size)
    })
    
    # Generate cap_tags data
    cap_tags_columns = [
        "tough_grader", "good_feedback", "respected", "lots_to_read",
        "participation_matters", "dont_skip_class", "lots_of_homework",
        "inspirational", "pop_quizzes", "accessible", "so_many_papers",
        "clear_grading", "hilarious", "test_heavy", "graded_by_few_things",
        "amazing_lectures", "caring", "extra_credit", "group_projects", "lecture_heavy"
    ]
    
    cap_tags = pd.DataFrame()
    for tag in cap_tags_columns:
        cap_tags[tag] = np.random.choice([0, 1], sample_size, p=[0.7, 0.3])
    
    data_loaded = True

if data_loaded:
    # Combine datasets
    combined_dataset = pd.concat([cap_num, cap_qual, cap_tags], axis=1)
    
    # Data preprocessing
    st.sidebar.header("🔍 Data Filtering")
    min_ratings = st.sidebar.slider(
        "Minimum Number of Ratings", 
        min_value=int(combined_dataset["Number of Ratings"].min()),
        max_value=int(combined_dataset["Number of Ratings"].max()),
        value=int(np.median(combined_dataset["Number of Ratings"].dropna()))
    )
    
    filtered_combined_dataset = combined_dataset[combined_dataset["Number of Ratings"] > min_ratings]
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Gender Analysis", 
        "Tag Analysis", 
        "Rating Models",
        "Difficulty Analysis",
        "Pepper Classification"
    ])
    
    # TAB 1: Gender Analysis
    with tab1:
        st.header("Gender Analysis")
        st.subheader("Comparing ratings between male and female professors")
        
        # Create masks for gender filtering
        male_mask = (filtered_combined_dataset["Male"] == 1).values & (filtered_combined_dataset["Female"] == 0).values
        male_ratings = filtered_combined_dataset[male_mask]["Average Rating"]
        non_male_ratings = filtered_combined_dataset[~male_mask]["Average Rating"]
        
        # Display basic statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Male Professors", len(male_ratings))
            st.metric("Avg. Rating (Male)", f"{male_ratings.mean():.2f}")
            st.metric("Std. Dev (Male)", f"{male_ratings.std():.2f}")
        
        with col2:
            st.metric("Female Professors", len(non_male_ratings))
            st.metric("Avg. Rating (Female)", f"{non_male_ratings.mean():.2f}")
            st.metric("Std. Dev (Female)", f"{non_male_ratings.std():.2f}")
        
        # KS test for ratings
        ks_statistic, p_value = ks_2samp(male_ratings, non_male_ratings)
        
        st.subheader("Statistical Test Results")
        st.write(f"Kolmogorov-Smirnov Test: KS Statistic = {ks_statistic:.4f}, P-value = {p_value:.4f}")
        st.write(f"Conclusion: {'Significant difference' if p_value < 0.005 else 'No significant difference'}")
        
        # Plot ratings distribution
        st.subheader("Ratings Distribution by Gender")
        fig, ax = plt.subplots(figsize=(10, 6))
        temp_df = pd.DataFrame({
            "ratings": list(male_ratings) + list(non_male_ratings),
            "gender": ["Male"] * male_ratings.shape[0] + ["Female"] * non_male_ratings.shape[0]
        })
        sns.kdeplot(data=temp_df, x="ratings", hue="gender", fill=True, ax=ax)
        ax.set_title("Density Distribution of Ratings by Gender", fontsize=15)
        ax.set_xlabel("Average Rating", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        st.pyplot(fig)
        
        # CDF plot
        st.subheader("Cumulative Distribution of Ratings")
        fig, ax = plt.subplots(figsize=(10, 6))
        group_m_sorted = np.sort(male_ratings)
        group_f_sorted = np.sort(non_male_ratings)
        cdf_m = np.linspace(0, 1, len(group_m_sorted))
        cdf_f = np.linspace(0, 1, len(group_f_sorted))
        ax.plot(group_m_sorted, cdf_m, label='Male', color='blue', linewidth=2)
        ax.plot(group_f_sorted, cdf_f, label='Female', color='green', linewidth=2)
        ax.set_title('Cumulative Distribution Function (CDF)', fontsize=15)
        ax.set_xlabel('Rating Value', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.legend(title='Gender', fontsize=12)
        ax.grid(True)
        st.pyplot(fig)
        
        # Effect size calculation
        mean_male = np.mean(male_ratings)
        mean_non_male = np.mean(non_male_ratings)
        std_male = np.std(male_ratings, ddof=1)
        std_non_male = np.std(non_male_ratings, ddof=1)
        n_male = len(male_ratings)
        n_non_male = len(non_male_ratings)
        
        s_pooled = np.sqrt(((n_male - 1) * std_male**2 + (n_non_male - 1) * std_non_male**2) / (n_male + n_non_male - 2))
        cohens_d = (mean_male - mean_non_male) / s_pooled
        
        se_d = np.sqrt((1/n_male) + (1/n_non_male) + (cohens_d**2 / (2 * (n_male + n_non_male))))
        ci_d = (cohens_d - 1.96 * se_d, cohens_d + 1.96 * se_d)
        
        st.subheader("Effect Size Analysis")
        st.write(f"Cohen's d: {cohens_d:.4f}")
        st.write(f"95% Confidence Interval: ({ci_d[0]:.4f}, {ci_d[1]:.4f})")
        st.write(f"Interpretation: {'Small' if abs(cohens_d) < 0.2 else 'Medium' if abs(cohens_d) < 0.5 else 'Large'} effect size")
        
    # TAB 2: Tag Analysis
    with tab2:
        st.header("Professor Tag Analysis")
        st.subheader("Comparing tags between male and female professors")
        
        # Scale tag data
        scaled_filtered_combined_dataset = filtered_combined_dataset.copy()
        scaler = StandardScaler()
        
        for i in cap_tags_columns:
            scaled_filtered_combined_dataset[i] = scaler.fit_transform(scaled_filtered_combined_dataset[i].values.reshape(-1,1))
        
        scaled_filtered_combined_dataset_male = scaled_filtered_combined_dataset[male_mask]
        scaled_filtered_combined_dataset_non_male = scaled_filtered_combined_dataset[~male_mask]
        
        # Calculate p-values for each tag
        p_value_map = {"Sig": [], "Non-Sig": []}
        for col in cap_tags_columns:
            ks_statistic, ks_p_value = ks_2samp(scaled_filtered_combined_dataset_male[col], scaled_filtered_combined_dataset_non_male[col])
            if ks_p_value > 0.005:
                p_value_map["Non-Sig"].append((col, ks_p_value))
            else:
                p_value_map["Sig"].append((col, ks_p_value))
        
        # Sort by p-value
        T1 = sorted(p_value_map["Sig"], key=lambda x: x[1])
        
        # Display the results
        st.subheader("Most Gendered Tags (Lowest p-values)")
        if T1:
            most_gendered_df = pd.DataFrame(T1[:5], columns=["Tag", "p-value"])
            st.table(most_gendered_df)
            
            # Tag frequency comparison
            st.subheader("Tag Frequency by Gender")
            selected_tags = st.multiselect(
                "Select tags to compare:", 
                options=cap_tags_columns,
                default=most_gendered_df["Tag"].tolist()[:3] if len(most_gendered_df) >= 3 else most_gendered_df["Tag"].tolist()
            )
            
            if selected_tags:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Calculate the frequency for selected tags
                male_freq = []
                female_freq = []
                
                for tag in selected_tags:
                    male_freq.append((filtered_combined_dataset[male_mask][tag] > 0).mean())
                    female_freq.append((filtered_combined_dataset[~male_mask][tag] > 0).mean())
                
                x = np.arange(len(selected_tags))
                width = 0.35
                
                rects1 = ax.bar(x - width/2, male_freq, width, label='Male')
                rects2 = ax.bar(x + width/2, female_freq, width, label='Female')
                
                ax.set_ylabel('Frequency')
                ax.set_title('Tag Frequency by Gender')
                ax.set_xticks(x)
                ax.set_xticklabels(selected_tags, rotation=45, ha='right')
                ax.legend()
                
                # Add value labels
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(f'{height:.2f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                
                autolabel(rects1)
                autolabel(rects2)
                
                fig.tight_layout()
                st.pyplot(fig)
        else:
            st.info("No statistically significant gender differences in tags were found.")
    
    # TAB 3: Rating Models  
with tab3:
    st.header("Rating Prediction Models")
    st.subheader("Analyzing factors that influence ratings")
    
    target_column = "Average Rating"
    
    # Handle missing values more thoroughly
    # First check percentage of missing values
    missing_percentage = cap_num["Average Difficulty"].isna().mean() * 100
    st.info(f"Missing values in 'Average Difficulty': {missing_percentage:.2f}%")
    
    # Handle missing values
    missing_removal_mask = cap_num["Average Difficulty"].isna()
    filtered_cap_num = cap_num[~missing_removal_mask].reset_index(drop=True)
    
    # Handle missing values in Proportion Would Take Again
    temp_target_column = "Proportion Would Take Again"
    na_mask = filtered_cap_num[temp_target_column].isna()
    na_percentage = na_mask.mean() * 100
    
    if sum(na_mask) > 0:
        st.info(f"Imputing missing values for 'Proportion Would Take Again' ({na_percentage:.2f}% missing)")
        sub_train_data = filtered_cap_num[~na_mask]
        
        # Use a simple imputer first to avoid issues with the SVR
        simple_imputer_value = sub_train_data[temp_target_column].median()
        filtered_cap_num.loc[na_mask, temp_target_column] = simple_imputer_value
        
        # Now use SVR for more sophisticated imputation if there are enough samples
        if len(sub_train_data) > 20:  # Only use SVR if we have enough data
            try:
                missing_imputer = SVR()
                # Use only numeric columns for prediction
                numeric_cols = sub_train_data.select_dtypes(include=['float64', 'int64']).columns
                numeric_cols = [col for col in numeric_cols if col != temp_target_column]
                
                missing_imputer.fit(sub_train_data[numeric_cols], sub_train_data[temp_target_column])
                temp_indices = filtered_cap_num[na_mask].index
                filtered_cap_num.loc[temp_indices, temp_target_column] = missing_imputer.predict(
                    filtered_cap_num.loc[temp_indices, numeric_cols]
                )
                st.success("Successfully imputed missing values using SVR")
            except Exception as e:
                st.error(f"Error during SVR imputation: {e}. Using median imputation instead.")
    
    # Examine target distribution
    st.subheader("Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_cap_num[target_column], kde=True, ax=ax)
    ax.set_title(f"Distribution of {target_column}", fontsize=15)
    ax.set_xlabel(target_column, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    st.pyplot(fig)
    
    # Check for outliers
    Q1 = filtered_cap_num[target_column].quantile(0.25)
    Q3 = filtered_cap_num[target_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = filtered_cap_num[(filtered_cap_num[target_column] < lower_bound) | 
                               (filtered_cap_num[target_column] > upper_bound)]
    
    if len(outliers) > 0:
        st.warning(f"Detected {len(outliers)} outliers in the target variable")
        
        # Option to remove outliers
        remove_outliers = st.checkbox("Remove outliers for model training")
        if remove_outliers:
            filtered_cap_num = filtered_cap_num[(filtered_cap_num[target_column] >= lower_bound) & 
                                              (filtered_cap_num[target_column] <= upper_bound)]
            st.success(f"Removed {len(outliers)} outliers. Working with {len(filtered_cap_num)} data points.")
    
    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = filtered_cap_num.corr()
    
    # Correlation with target
    target_correlations = correlation_matrix[target_column].drop(target_column).sort_values(ascending=False)
    st.write("Correlations with target variable:")
    st.write(target_correlations)
    
    # Plot heatmap
    sns.heatmap(correlation_matrix, center=0, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Feature selection options
    st.subheader("Feature Selection")
    min_correlation = st.slider("Minimum absolute correlation with target", 0.0, 1.0, 0.1, 0.05)
    
    # Select features based on correlation
    selected_features = target_correlations[abs(target_correlations) >= min_correlation].index.tolist()
    
    if not selected_features:
        st.error("No features meet the correlation threshold. Using all features.")
        selected_features = filtered_cap_num.drop(columns=[target_column]).columns.tolist()
    
    st.write(f"Selected features: {', '.join(selected_features)}")
    
    # Train model with selected features
    st.subheader("Linear Regression Model")
    
    RANDOM_SEED_VALUE = 13369770
    np.random.seed(RANDOM_SEED_VALUE)
    
    X = filtered_cap_num[selected_features]
    y = filtered_cap_num[target_column]
    
    # Check if we have enough data
    if len(X) < 10:
        st.error("Not enough data points for model training after filtering.")
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED_VALUE
        )
        
        # Option to normalize features
        normalize_features = st.checkbox("Normalize features before training", value=True)
        if normalize_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_train_df = pd.DataFrame(X_train, columns=selected_features)
            X_test_df = pd.DataFrame(X_test, columns=selected_features)
        else:
            X_train_df = X_train
            X_test_df = X_test
        
        # Train model
        try:
            lr_model = LinearRegression()
            lr_model.fit(X_train_df, y_train)
            test_preds_lr = lr_model.predict(X_test_df)
            train_preds_lr = lr_model.predict(X_train_df)
            
            # Model metrics
            r2_train = r2_score(y_train, train_preds_lr)
            r2_test = r2_score(y_test, test_preds_lr)
            rmse_train = root_mean_squared_error(y_train, train_preds_lr)
            rmse_test = root_mean_squared_error(y_test, test_preds_lr)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Train R²", f"{r2_train:.4f}")
                st.metric("Test R²", f"{r2_test:.4f}")
            with col2:
                st.metric("Train RMSE", f"{rmse_train[0]:.4f}")
                st.metric("Test RMSE", f"{rmse_test[0]:.4f}")
            
            # Check if R² is negative
            if r2_test < 0:
                st.error("""
                Negative R² detected! This indicates the model performs worse than a horizontal line.
                Possible reasons:
                - The data might not have a linear relationship
                - Features don't have predictive power
                - The model is overfitting
                - Outliers are affecting performance
                
                Try different features, removing outliers, or a different model type.
                """)
            
            # Feature importance visualization
            coefficients = lr_model.coef_
            importance_df = pd.DataFrame({
                "Feature": selected_features,
                "Importance": coefficients
            }).sort_values(by="Importance", ascending=False)
            
            st.subheader("Feature Importance (Linear Regression Coefficients)")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
            ax.set_title("Feature Importance for Predicting Average Rating", fontsize=15)
            ax.set_xlabel("Coefficient Value", fontsize=12)
            ax.set_ylabel("Feature", fontsize=12)
            st.pyplot(fig)
            
            # Residual plot
            st.subheader("Residual Analysis")
            residuals = y_test - test_preds_lr
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=test_preds_lr, y=residuals, ax=ax)
            ax.axhline(y=0, color='r', linestyle='-')
            ax.set_title("Residual Plot", fontsize=15)
            ax.set_xlabel("Predicted Values", fontsize=12)
            ax.set_ylabel("Residuals", fontsize=12)
            st.pyplot(fig)
            
            # Actual vs Predicted
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=test_preds_lr, ax=ax)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_title("Actual vs Predicted Values", fontsize=15)
            ax.set_xlabel("Actual Values", fontsize=12)
            ax.set_ylabel("Predicted Values", fontsize=12)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error during Linear Regression training: {e}")
        
        # Try Random Forest as an alternative model
        st.subheader("Random Forest Model as Alternative")
        
        try:
            # Option to tune hyperparameters
            n_estimators = st.slider("Number of trees", 10, 200, 100, 10)
            max_depth = st.slider("Maximum tree depth", 2, 20, 10, 1)
            
            rf_model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                random_state=RANDOM_SEED_VALUE
            )
            
            rf_model.fit(X_train_df, y_train)
            
            test_preds_rf = rf_model.predict(X_test_df)
            train_preds_rf = rf_model.predict(X_train_df)
            
            # RF Model metrics
            r2_train_rf = r2_score(y_train, train_preds_rf)
            r2_test_rf = r2_score(y_test, test_preds_rf)
            rmse_train_rf = root_mean_squared_error(y_train, train_preds_rf)
            rmse_test_rf = root_mean_squared_error(y_test, test_preds_rf)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RF Train R²", f"{r2_train_rf:.4f}")
                st.metric("RF Test R²", f"{r2_test_rf:.4f}")
            with col2:
                st.metric("RF Train RMSE", f"{rmse_train_rf[0]:.4f}")
                st.metric("RF Test RMSE", f"{rmse_test_rf[0]:.4f}")
            
            # RF Feature importance
            feature_importances = rf_model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": selected_features,
                "Importance": feature_importances
            }).sort_values(by="Importance", ascending=False)
            
            st.subheader("Feature Importance (Random Forest)")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
            ax.set_title("Feature Importance for Predicting Average Rating (Random Forest)", fontsize=15)
            ax.set_xlabel("Importance", fontsize=12)
            ax.set_ylabel("Feature", fontsize=12)
            st.pyplot(fig)
            
            # Actual vs Predicted for RF
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=test_preds_rf, ax=ax)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_title("Actual vs Predicted Values (Random Forest)", fontsize=15)
            ax.set_xlabel("Actual Values", fontsize=12)
            ax.set_ylabel("Predicted Values", fontsize=12)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error during Random Forest training: {e}")
        
    # TAB 4: Difficulty Analysis
    with tab4:
        st.header("Professor Difficulty Analysis")
        st.subheader("Analyzing factors that influence difficulty ratings")
        
        male_avg_difficulty = filtered_combined_dataset[male_mask]["Average Difficulty"]
        non_male_avg_difficulty = filtered_combined_dataset[~male_mask]["Average Difficulty"]
        
        # Display basic statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Male Professors", len(male_avg_difficulty))
            st.metric("Avg. Difficulty (Male)", f"{male_avg_difficulty.mean():.2f}")
            st.metric("Std. Dev (Male)", f"{male_avg_difficulty.std():.2f}")
        
        with col2:
            st.metric("Female Professors", len(non_male_avg_difficulty))
            st.metric("Avg. Difficulty (Female)", f"{non_male_avg_difficulty.mean():.2f}")
            st.metric("Std. Dev (Female)", f"{non_male_avg_difficulty.std():.2f}")
        
        # KS test for difficulty
        ks_statistic, p_value = ks_2samp(male_avg_difficulty, non_male_avg_difficulty)
        
        st.subheader("Statistical Test Results")
        st.write(f"Kolmogorov-Smirnov Test: KS Statistic = {ks_statistic:.4f}, P-value = {p_value:.4f}")
        st.write(f"Conclusion: {'Significant difference' if p_value < 0.005 else 'No significant difference'}")
        
        # Plot difficulty distribution
        st.subheader("Difficulty Distribution by Gender")
        fig, ax = plt.subplots(figsize=(10, 6))
        temp_df = pd.DataFrame({
            "difficulty": list(male_avg_difficulty) + list(non_male_avg_difficulty),
            "gender": ["Male"] * male_avg_difficulty.shape[0] + ["Female"] * non_male_avg_difficulty.shape[0]
        })
        sns.kdeplot(data=temp_df, x="difficulty", hue="gender", fill=True, ax=ax)
        ax.set_title("Density Distribution of Difficulty Ratings by Gender", fontsize=15)
        ax.set_xlabel("Average Difficulty", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        st.pyplot(fig)
        
        # Bootstrap for Cohen's d
        st.subheader("Effect Size Analysis (Bootstrapped)")
        
        n_bootstrap = st.slider("Number of Bootstrap Samples", 1000, 10000, 5000, 1000)
        
        if st.button("Run Bootstrap Analysis"):
            with st.spinner("Running bootstrap analysis..."):
                bootstrap_results = []
                
                # Perform bootstrapping
                for _ in range(n_bootstrap):
                    male_sample = np.random.choice(male_avg_difficulty, size=len(male_avg_difficulty), replace=True)
                    non_male_sample = np.random.choice(non_male_avg_difficulty, size=len(non_male_avg_difficulty), replace=True)
                    bootstrap_results.append(cohen_d(male_sample, non_male_sample))
                
                # Convert bootstrap results to a NumPy array
                bootstrap_results = np.array(bootstrap_results)
                
                # Calculate the mean and 95% confidence interval for Cohen's d
                mean_cohens_d = np.mean(bootstrap_results)
                ci_lower = np.percentile(bootstrap_results, 2.5)
                ci_upper = np.percentile(bootstrap_results, 97.5)
                
                # Display results
                st.write(f"Mean Cohen's d: {mean_cohens_d:.4f}")
                st.write(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
                st.write(f"Interpretation: {'Small' if abs(mean_cohens_d) < 0.2 else 'Medium' if abs(mean_cohens_d) < 0.5 else 'Large'} effect size")
                
                # Visualization of the bootstrapped distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(bootstrap_results, bins=50, color='skyblue', edgecolor='black', density=True)
                ax.axvline(mean_cohens_d, color='red', linestyle='--', label=f"Mean Cohen's d = {mean_cohens_d:.2f}")
                ax.axvline(ci_lower, color='green', linestyle='--', label=f"2.5% CI = {ci_lower:.2f}")
                ax.axvline(ci_upper, color='green', linestyle='--', label=f"97.5% CI = {ci_upper:.2f}")
                ax.set_title("Bootstrapped Distribution of Cohen's d", fontsize=15)
                ax.set_xlabel("Cohen's d", fontsize=12)
                ax.set_ylabel("Density", fontsize=12)
                ax.legend()
                st.pyplot(fig)
    
    # TAB 5: Pepper Classification
    # TAB 5: Pepper Classification
with tab5:
    st.header("'Received a Pepper' Classification")
    st.subheader("Analyzing what factors predict receiving a pepper")
    
    target_column = "Received a Pepper"
    pepper_dist = filtered_combined_dataset[target_column].value_counts()
    
    # Display class distribution
    st.subheader("Class Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        pepper_dist.values, 
        labels=[f"No Pepper ({pepper_dist.index[0]}): {pepper_dist.values[0]}", 
               f"Received Pepper ({pepper_dist.index[1]}): {pepper_dist.values[1]}"],
        autopct='%1.1f%%',
        colors=['lightgray', 'red'],
        startangle=90
    )
    ax.set_title("Distribution of 'Received a Pepper'", fontsize=15)
    st.pyplot(fig)
    
    # Prepare data for the classification model
    X = filtered_combined_dataset.drop(columns=[target_column])
    y = filtered_combined_dataset[target_column]
    
    # Select only numerical features for simplicity
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove non-predictive features
    if 'Male' in numerical_features and 'Female' in numerical_features:
        features_to_use = [col for col in numerical_features if col not in ['Number of Ratings']]
    else:
        features_to_use = [col for col in numerical_features if col != 'Number of Ratings']
    
    X_selected = X[features_to_use]
    
    # Handle missing values
    X_selected = X_selected.fillna(X_selected.mean())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, 
        y,
        test_size=0.2,
        random_state=RANDOM_SEED_VALUE,
        stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    st.subheader("SVM Classification Model")
    
    # Add hyperparameter tuning options
    col1, col2 = st.columns(2)
    with col1:
        C_value = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0, 0.1)
    with col2:
        kernel_type = st.selectbox("Kernel Type", ["linear", "rbf", "poly"])
    
    if st.button("Train SVM Model"):
        with st.spinner("Training model..."):
            # Train the SVM classifier
            svm_model = SVC(C=C_value, kernel=kernel_type, probability=True, random_state=RANDOM_SEED_VALUE)
            svm_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = svm_model.predict(X_test_scaled)
            y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
            
            # Compute metrics
            metrics = compute_classification_metrics(y_test, y_pred, y_prob)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
            with col3:
                st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
            
            # Display confusion matrix
            st.subheader("Confusion Matrix")
            cm = metrics['confusion_matrix']
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['No Pepper', 'Received Pepper'],
                yticklabels=['No Pepper', 'Received Pepper'],
                ax=ax
            )
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            # Feature importance (coefficients for linear kernel)
            if kernel_type == "linear":
                st.subheader("Feature Importance (SVM Coefficients)")
                importance_df = pd.DataFrame({
                    "Feature": features_to_use,
                    "Importance": np.abs(svm_model.coef_[0])
                }).sort_values(by="Importance", ascending=False)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
                ax.set_title("Feature Importance for Predicting 'Received a Pepper'", fontsize=15)
                ax.set_xlabel("Absolute Coefficient Value", fontsize=12)
                ax.set_ylabel("Feature", fontsize=12)
                st.pyplot(fig)
            
            # Cross-validation scores
            st.subheader("Cross-Validation Performance")
            cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED_VALUE)
            cv_scores = cross_val_score(
                SVC(C=C_value, kernel=kernel_type, probability=True, random_state=RANDOM_SEED_VALUE),
                X_selected, 
                y, 
                cv=cv, 
                scoring='accuracy'
            )
            
            st.write(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Plot ROC curve
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'AUC = {metrics["auc_roc"]:.4f}')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc='lower right')
            st.pyplot(fig)
    
    # Gender analysis for peppers
    st.subheader("Gender Analysis for 'Received a Pepper'")
    
    # Calculate statistics by gender
    male_pepper = filtered_combined_dataset[filtered_combined_dataset["Male"] == 1][target_column].mean()
    female_pepper = filtered_combined_dataset[filtered_combined_dataset["Female"] == 1][target_column].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Male Professors with Pepper (%)", f"{male_pepper*100:.2f}%")
    with col2:
        st.metric("Female Professors with Pepper (%)", f"{female_pepper*100:.2f}%")
    
    # Chi-square test for independence
    male_with_pepper = (filtered_combined_dataset["Male"] == 1) & (filtered_combined_dataset[target_column] == 1)
    male_without_pepper = (filtered_combined_dataset["Male"] == 1) & (filtered_combined_dataset[target_column] == 0)
    female_with_pepper = (filtered_combined_dataset["Female"] == 1) & (filtered_combined_dataset[target_column] == 1)
    female_without_pepper = (filtered_combined_dataset["Female"] == 1) & (filtered_combined_dataset[target_column] == 0)
    
    contingency_table = np.array([
        [sum(male_with_pepper), sum(male_without_pepper)],
        [sum(female_with_pepper), sum(female_without_pepper)]
    ])
    
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    st.write(f"Chi-square test for independence: χ² = {chi2:.4f}, p-value = {p:.4f}")
    st.write(f"Conclusion: {'Significant association between gender and receiving a pepper' if p < 0.05 else 'No significant association between gender and receiving a pepper'}")
    
    # Tags associated with peppers
    st.subheader("Tags Associated with 'Received a Pepper'")
    
    # Calculate point-biserial correlation between tags and pepper
    tag_correlations = []
    for tag in cap_tags_columns:
        correlation, p_value = stats.pointbiserialr(filtered_combined_dataset[tag], filtered_combined_dataset[target_column])
        tag_correlations.append((tag, correlation, p_value))
    
    # Sort by absolute correlation
    tag_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Display top correlations
    top_correlations_df = pd.DataFrame(tag_correlations[:10], columns=["Tag", "Correlation", "p-value"])
    st.table(top_correlations_df.style.format({"Correlation": "{:.4f}", "p-value": "{:.4f}"}))
    
    # Plot top positive and negative correlations
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter significant correlations
    significant_tags = [(tag, corr) for tag, corr, p in tag_correlations if p < 0.05]
    if significant_tags:
        top_positive = sorted(significant_tags, key=lambda x: x[1], reverse=True)[:5]
        top_negative = sorted(significant_tags, key=lambda x: x[1])[:5]
        
        # Combine lists
        top_tags = top_positive + top_negative
        tags, correlations = zip(*top_tags)
        
        # Create colors based on correlation sign
        colors = ['green' if c > 0 else 'red' for c in correlations]
        
        sns.barplot(x=list(correlations), y=list(tags), palette=colors, ax=ax)
        ax.set_title("Tag Correlations with 'Received a Pepper'", fontsize=15)
        ax.set_xlabel("Correlation Coefficient", fontsize=12)
        ax.set_ylabel("Tag", fontsize=12)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("No statistically significant correlations between tags and receiving a pepper were found.")
