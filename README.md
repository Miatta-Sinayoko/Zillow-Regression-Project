# Zillow-Regression Model Project

## Project Description
The Zillow Model Project aims to develop a robust predictive model for accurately estimating the value of residential properties. By leveraging various features, the model will provide valuable insights into the factors that influence home values.

## Project Goals
1. Identify Key Features: Determine the features within the dataset that exhibit strong correlations with home values.
2. Build Predictive Model: Develop a high-performing model that can accurately estimate property values based on the selected features.
3. Share Insights: Communicate the findings and model insights to the Zillow data science team for further analysis and decision-making.

## Initial Thoughts
Preliminary analysis suggests that the county where a property is located significantly impacts its value. The assessed value of a home closely aligns with its tax-assessed value, making county a crucial factor in determining property worth.

## The Plan
1. Data Acquisition: Obtain the necessary dataset containing relevant information about residential properties.
2. Data Preparation:
   - Column Renaming: Rename columns for clarity and ease of interpretation.
   - Data Type Adjustment: Ensure appropriate data types for each column.
   - FIPS Renaming: Rename FIPS values to county names to enhance readability.
   - Handling Null Values: Drop null values, as they account for only a negligible percentage of the dataset.
   - Outlier Removal: Apply the Interquartile Range (IQR) method to remove outliers that could skew the data.
   - Data Split: Divide the dataset into training, validation, and testing sets using a 50-30-20 split ratio.
3. Data Exploration: Conduct exploratory data analysis to answer key questions, including:
   - Relationship between county and property value.
   - Correlation between bedroom count and home value.
   - Correlation between bathroom count and home value.
   - Impact of square footage on property value.
4. Model Development:
   - Evaluation Metric: Utilize Root Mean Squared Error (RMSE) as the primary evaluation metric for model performance.
   - Baseline Estimation: Establish a baseline using the mean value of home prices.
   - Model Building: Develop and refine a predictive model that accurately estimates home values.
5. Draw Conclusions: Summarize the key findings and insights obtained from the project.
6. Data Dictionary: Provide a clear and concise description of the dataset's features and their respective meanings.
7. Steps to Reproduce: Outline the step-by-step process required to replicate the project.
8. Conclusions: Provide a comprehensive summary of the project's outcomes and key takeaways.
9. Next Steps: Suggest potential future directions and areas for further exploration based on the project's findings.
10. Recommendations: Offer specific suggestions and recommendations based on the insights gained from the project.

## Data Dictionary
The data dictionary defines the features included in the dataset and their corresponding descriptions:

| Feature         | Description                                       |
|-----------------|---------------------------------------------------|
| Bedroom         | Number of bedrooms in the property                |
| Bathroom        | Number of bathrooms in the property               |
| Square Footage  | Total calculated square footage of the property   |
| Home Value      | Assessed property value                           |
| County          | County of property location in California         |

Note: The "Home Value" feature represents the target variable, while "County" is a categorical variable.

## Conclusions
- The LassoLars model demonstrated superior performance compared to the baseline model when evaluated on unseen test data.
- The current model's accuracy is limited to homes falling within the upper and lower quartiles of the dataset.

## Next Steps
- Further analyze the data by splitting it into individual county subsets and building separate models for each.
- Investigate the potential impact of scaling the data on the performance of the existing model.
- Explore the correlation between bathroom and bedroom counts and consider creating a combined feature to enhance model performance.
- Explore additional features within the dataset to uncover their relationship with home values.

## Recommendations
- Continue refining and improving the predictive model to enhance the accuracy and reliability of home value predictions.
