 ---

# **Movie Recommendation System**  
### *A Hybrid Approach Using Collaborative and Content-Based Filtering*  

---

## **Overview**  
This project builds a personalized movie recommendation system using the Movielens small dataset. The system combines **collaborative filtering** (CF) and **content-based filtering** (CBF) to provide accurate and diverse recommendations, even for new users (cold start problem).  

---

## **Key Features**  
1. **Collaborative Filtering**: Uses Singular Value Decomposition (SVD) to predict user preferences based on past ratings.  
2. **Content-Based Filtering**: Recommends movies based on genre similarity using TF-IDF and cosine similarity.  
3. **Hybrid Model**: Dynamically switches between CF and CBF based on user activity.  
4. **Cold Start Handling**: Provides genre-based recommendations for users with few or no ratings.  

---

## **Dataset**  
- **Source**: [Movielens Small Dataset](https://grouplens.org/datasets/movielens/)  
- **Contents**:  
  - **Ratings**: 100,000 ratings from 610 users on 9,724 movies.  
  - **Movies**: Metadata including title and genres.  
- **Key Stats**:  
  - Average rating: 3.5/5.  
  - Most popular genres: Drama, Comedy, Action.  

---

## **Notebook Structure**  
1. **Data Exploration**:  
   - Load and inspect the dataset.  
   - Visualize rating distribution and genre popularity.  

2. **Data Preparation**:  
   - Handle missing values.  
   - Transform genres into numerical features using TF-IDF.  

3. **Modeling**:  
   - Train a collaborative filtering model using SVD.  
   - Optimize hyperparameters with GridSearchCV.  
   - Build a hybrid recommendation system.  

4. **Evaluation**:  
   - Measure model performance using RMSE and MAE.  
   - Validate results with cross-validation.  

5. **Recommendations**:  
   - Generate top 5 movie recommendations for any user.  
   - Handle cold start users with genre-based suggestions.  

---

## **How to Use**  
1. **Install Dependencies**:  
```bash
pip install -r requirements.txt
```

2. **Run the Notebook**:  
   - Open `movie_recommendation.ipynb` in Jupyter Notebook or Google Colab.  
   - Execute cells sequentially to load data, train models, and generate recommendations.  

3. **Test Recommendations**:  
   - Use the `recommend_movies()` function to get personalized recommendations for any user:  
     ```python
     recommendations = recommend_movies(user_id=1, model=optimized_svd, movies_df=movies_df, ratings_df=ratings_df, n=5)
     print(recommendations)
     ```

---

## **Results**  
- **Model Performance**:  
  - Hybrid Model: RMSE = 0.8324, MAE = 0.6211.  
  - Baseline SVD Model: RMSE = 0.8812, MAE = 0.6772.  
- **Cold Start Handling**: New users receive relevant recommendations based on popular genres.  

---

## **Limitations**  
1. **Sparsity**: The dataset is sparse (1.5% density), limiting prediction accuracy.  
2. **Scalability**: The current implementation is optimized for small datasets. For larger datasets, consider using distributed computing frameworks like Apache Spark.  
3. **Feature Limitations**: Recommendations rely heavily on genres. Adding metadata (e.g., cast, director) could improve results.  

---

## **Future Work**  
1. **Deep Learning**: Explore Neural Collaborative Filtering (NCF) for better accuracy.  
2. **Real-Time Recommendations**: Integrate with a streaming platform for live suggestions.  
3. **User Feedback**: Incorporate thumbs up/down ratings to refine recommendations over time.  

---

## **Dependencies**  
- Python 3.8+  
- Libraries:  
  - `pandas`, `numpy`: Data manipulation.  
  - `scikit-learn`: TF-IDF and cosine similarity.  
  - `surprise`: Collaborative filtering with SVD.  
  - `nltk`, `langdetect`: Text preprocessing.  
  - `matplotlib`, `seaborn`: Data visualization.  

---

## **Installation and Setup**  
1. **Clone the repository**:  
```bash
git clone https://github.com/<your-username>/<repo-name>.git
```

2. **Navigate to the project directory**:  
```bash
cd <repo-name>
```

3. **Create a virtual environment** (optional but recommended):  
```bash
python -m venv venv
source venv/bin/activate  # On Linux/MacOS
.\venv\Scripts\activate   # On Windows
```

4. **Install dependencies**:  
```bash
pip install -r requirements.txt
```

---

## **Reproducibility Tips**  
- Ensure the Movielens dataset is correctly placed in the `data` directory.  
- The `requirements.txt` file lists all the dependencies to avoid version conflicts.  
- Use the same Python version (3.8+) for compatibility.  

---

## **Contact**  
For questions or feedback, please contact:  
- **Name**: [Daniel]  
- **Email**: [muigaimunyua@gmail.com]]  
- **GitHub**: [Daniel-MM24]  

---

**Enjoy discovering movies! 🎬🍿**    

---

## **Acknowledgments**  
- Movielens for the dataset.  
- scikit-learn and Surprise libraries for model implementation.  

---

### **Note:**  
Since the dataset is too large for GitHub, please download it from [Movielens](https://grouplens.org/datasets/movielens/) and place it in the `data` directory.  

---

