# 🎬 Movie Recommendation System using K-Nearest Neighbors (KNN)

This project is a **Movie Recommendation System** built with Python using **Collaborative Filtering** and **K-Nearest Neighbors (KNN)**. It suggests similar movies based on user ratings data, leveraging the power of machine learning with the `scikit-learn` library.


## 📁 Project Structure

├── movie_recommender_app.py       # Main application file

├── icon.ico                       # App icon (optional)

├── DATA/

│   ├── movies.csv                 # Movie metadata

│   └── ratings.csv                # Ratings by users

└── README.md


## 🚀 Features

- Collaborative filtering with user-item interaction matrix.
- Memory-efficient sparse matrix implementation.
- KNN model using cosine similarity.
- Filters for minimum user engagement:
  - Movies rated by more than 10 users.
  - Users who rated more than 50 movies.
- Returns top 10 similar movies with similarity scores.


## 🧠 Tech Stack

- **Python 3.x**
- **Pandas** — data manipulation
- **scikit-learn** — machine learning (KNN)
- **SciPy** — sparse matrix utilities
- **GUI** - Tkinter for Graphical User Interface 


## 🧪 How It Works

1. **Load Data**: Import movie and ratings datasets.
2. **Preprocess**:
   - Create a pivot table of movieId vs. userId.
   - Fill missing values with 0.
   - Apply filters to retain quality data.
3. **Build KNN Model**:
   - Convert to sparse matrix.
   - Fit using cosine distance metric.
4. **Get Recommendations**:
   - Search the input movie title.
   - Find 10 nearest neighbors using KNN.
   - Return movie titles and distances.


## 📦 Installation
Clone the Repository
 - git clone https://github.com/yourusername/movie-recommender-knn.git
 - cd movie-recommender-knn
   
Install Dependencies
 - pip install pandas scikit-learn scipy tkinter
   
Add Dataset
 - Place movies.csv and ratings.csv inside the DATA/ folder.
   
Run the Recommender
 - python movie_recommender.py
   
📊 Dataset Source
   This application uses the **[Kaggle Dataset](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset)**, specifically:
   
    - movies.csv
    - ratings.csv

## 🛠 Future Improvements
 - Web UI using Flask or Streamlit
 - Hybrid filtering (content + collaborative)
 - Save model for deployment
 - User-based recommendations

## 👨‍💻 Author
Mratyunjay Saxena

Python Developer & Data Analyst
