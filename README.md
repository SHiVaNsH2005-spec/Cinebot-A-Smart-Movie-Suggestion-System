
# 🎬 CineBot: A Smart Movie Suggestion System

CineBot is a smart content-based movie recommendation system that suggests movies similar to the one you enter, using only content information — no user ratings needed.


## What it does

- Uses movie **overview, genres, keywords, main cast, and director**.
- Combines and cleans these features into one **tags** column.
- Processes text using **NLTK PorterStemmer**.
- Converts text to numbers with **CountVectorizer** (Bag of Words).
- Calculates **cosine similarity** between movies.
- Recommends **top 5 similar movies** for any input.
- Includes a simple **HTML frontend** for easy use.


## Tech Stack

- **Python 3**
- **pandas**
- **NLTK**
- **scikit-learn**
- **Jupyter Notebook**
- **HTML**


## How to Run

1️⃣ Clone or download the repo.  
2️⃣ Open the `CineBot.ipynb` notebook in **Jupyter Notebook**.  
3️⃣ Run all the cells to build the model.  
4️⃣ Use the `recommend("Movie Name")` function to get recommendations.  
5️⃣ Or run the **HTML page** to enter a movie name and see similar movies.


## Example

  Input:  `The Dark Knight Rises`  
  Output: `The Dark Knight`, `Batman Begins`, `Man of Steel`, `Avengers`, `Iron Man`


## Contributors

  Group 11:
- Shivansh Porwal — Data merging, feature extraction  
- Utkarsh Singh   — Testing & debugging 
- Aashi Sharma    — HTML And App Development
- Aaryan Dhama    — Integration & PPT
- Kabir Arora     — Graphs and Similarity


