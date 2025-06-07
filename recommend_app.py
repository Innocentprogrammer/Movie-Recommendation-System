# Import necessary libraries
import tkinter as tk
from tkinter import ttk, messagebox, font
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import threading
from tkinter import scrolledtext

class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_styles()
        self.load_data()
        self.create_widgets()
        
    def setup_window(self):
        self.root.title("Movie Recommendation System")
        self.root.iconbitmap('icon.ico')
        self.root.geometry("1000x700")
        self.root.configure(bg='#1a1a1a')
        self.root.resizable(True, True)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1000 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"1000x700+{x}+{y}")
        
    def setup_styles(self):
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', 
                       background='#1a1a1a', 
                       foreground='#ffffff', 
                       font=('Helvetica', 24, 'bold'))
        
        style.configure('Subtitle.TLabel', 
                       background='#1a1a1a', 
                       foreground='#cccccc', 
                       font=('Helvetica', 12))
        
        style.configure('Modern.TEntry',
                       fieldbackground='#2d2d2d',
                       borderwidth=0,
                       insertcolor='#ffffff',
                       foreground='#ffffff',
                       font=('Helvetica', 12))
        
        style.configure('Modern.TButton',
                       background='#4CAF50',
                       foreground='#ffffff',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Helvetica', 11, 'bold'))
        
        style.map('Modern.TButton',
                 background=[('active', '#45a049'),
                           ('pressed', '#3d8b40')])
        
    def load_data(self):
        """Load and preprocess the movie data"""
        try:
            # Update these paths as needed
            self.movies = pd.read_csv("C:/Projects/Advance Project/Data Analysis Projects/DATA/movies.csv")
            ratings = pd.read_csv("C:/Projects/Advance Project/Data Analysis Projects/DATA/ratings.csv")
            
            # Create pivot table
            final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
            final_dataset.fillna(0, inplace=True)
            
            # Filter: Minimum 10 users per movie
            no_user_voted = ratings.groupby('movieId')['rating'].count()
            final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
            
            # Filter: Minimum 50 movies rated per user
            no_movies_voted = ratings.groupby('userId')['rating'].count()
            final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]
            
            # Convert to sparse matrix
            csr_data = csr_matrix(final_dataset.values)
            
            # Reset index
            final_dataset.reset_index(inplace=True)
            
            # Build KNN model
            self.knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
            self.knn.fit(csr_data)
            
            self.final_dataset = final_dataset
            self.csr_data = csr_data
            
            self.data_loaded = True
            
        except Exception as e:
            self.data_loaded = False
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎬 Movie Recommendation System", style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, text="Discover movies similar to your favorites using AI", style='Subtitle.TLabel')
        subtitle_label.pack(pady=(0, 30))
        
        # Search frame
        search_frame = tk.Frame(main_frame, bg='#1a1a1a')
        search_frame.pack(fill='x', pady=(0, 20))
        
        # Search label
        search_label = ttk.Label(search_frame, text="Enter Movie Name:", style='Subtitle.TLabel')
        search_label.pack(anchor='w', pady=(0, 5))
        
        # Entry frame for better styling
        entry_frame = tk.Frame(search_frame, bg='#2d2d2d', relief='flat', bd=1)
        entry_frame.pack(fill='x', pady=(0, 15))
        
        self.movie_entry = tk.Entry(entry_frame, 
                                   bg='#2d2d2d', 
                                   fg='#ffffff', 
                                   insertbackground='#ffffff',
                                   border=0,
                                   font=('Helvetica', 12))
        self.movie_entry.pack(fill='x', padx=10, pady=8)
        self.movie_entry.bind('<Return>', lambda e: self.search_movie())
        
        # Button frame
        button_frame = tk.Frame(search_frame, bg='#1a1a1a')
        button_frame.pack(fill='x')
        
        # Search button
        self.search_btn = tk.Button(button_frame,
                                   text="🔍 Get Recommendations",
                                   bg='#4CAF50',
                                   fg='#ffffff',
                                   border=0,
                                   font=('Helvetica', 11, 'bold'),
                                   cursor='hand2',
                                   command=self.search_movie)
        self.search_btn.pack(side='left')
        
        # Clear button
        clear_btn = tk.Button(button_frame,
                             text="🗑️ Clear",
                             bg='#f44336',
                             fg='#ffffff',
                             border=0,
                             font=('Helvetica', 11, 'bold'),
                             cursor='hand2',
                             command=self.clear_results)
        clear_btn.pack(side='left', padx=(10, 0))
        
        # Loading label
        self.loading_label = ttk.Label(main_frame, text="", style='Subtitle.TLabel')
        self.loading_label.pack(pady=10)
        
        # Results frame
        results_frame = tk.Frame(main_frame, bg='#1a1a1a')
        results_frame.pack(fill='both', expand=True, pady=(20, 0))
        
        # Results title
        self.results_title = ttk.Label(results_frame, text="", style='Title.TLabel')
        self.results_title.pack(anchor='w', pady=(0, 10))
        
        # Treeview for results
        columns = ('Rank', 'Movie Title', 'Similarity Score')
        self.tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        self.tree.heading('Rank', text='#')
        self.tree.heading('Movie Title', text='Movie Title')
        self.tree.heading('Similarity Score', text='Similarity Score')
        
        self.tree.column('Rank', width=50, anchor='center')
        self.tree.column('Movie Title', width=600, anchor='w')
        self.tree.column('Similarity Score', width=150, anchor='center')
        
        # Configure treeview colors
        style = ttk.Style()
        style.configure("Treeview",
                       background="#2d2d2d",
                       foreground="#ffffff",
                       rowheight=25,
                       fieldbackground="#2d2d2d",
                       font=('Helvetica', 10))
        style.map('Treeview', background=[('selected', '#4CAF50')])
        
        style.configure("Treeview.Heading",
                       background="#333333",
                       foreground="#ffffff",
                       font=('Helvetica', 11, 'bold'))
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#333333', height=25)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, 
                                    text="Ready" if self.data_loaded else "Failed to load data",
                                    bg='#333333', 
                                    fg='#ffffff',
                                    font=('Helvetica', 9))
        self.status_label.pack(side='left', padx=10, pady=3)
        
    def search_movie(self):
        if not self.data_loaded:
            messagebox.showerror("Error", "Data not loaded. Please check your file paths.")
            return
            
        movie_name = self.movie_entry.get().strip()
        if not movie_name:
            messagebox.showwarning("Warning", "Please enter a movie name!")
            return
        
        # Disable search button and show loading
        self.search_btn.config(state='disabled', text="Searching...")
        self.loading_label.config(text="🔍 Finding similar movies...")
        self.status_label.config(text="Processing...")
        
        # Run search in separate thread to prevent GUI freezing
        thread = threading.Thread(target=self.perform_search, args=(movie_name,))
        thread.daemon = True
        thread.start()
    
    def perform_search(self, movie_name):
        try:
            recommendations = self.get_movie_recommendation(movie_name)
            
            # Update GUI in main thread
            self.root.after(0, self.display_results, recommendations, movie_name)
            
        except Exception as e:
            self.root.after(0, self.handle_error, str(e))
    
    def get_movie_recommendation(self, movie_name):
        n_movies_to_recommend = 10
        movie_list = self.movies[self.movies['title'].str.contains(movie_name, case=False, na=False)]
        
        if movie_list.empty:
            return "No movies found. Please check your input."
        
        movie_id = movie_list.iloc[0]['movieId']
        
        if movie_id not in self.final_dataset['movieId'].values:
            return "Movie is not present in the final dataset (possibly due to filtering). Try another movie."
        
        movie_idx = self.final_dataset[self.final_dataset['movieId'] == movie_id].index[0]
        
        distances, indices = self.knn.kneighbors(self.csr_data[movie_idx], n_neighbors=n_movies_to_recommend + 1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                  key=lambda x: x[1])[1:]
        
        recommend_frame = []
        for val in rec_movie_indices:
            rec_movie_id = self.final_dataset.iloc[val[0]]['movieId']
            movie_title = self.movies[self.movies['movieId'] == rec_movie_id]['title'].values[0]
            similarity_score = f"{(1 - val[1]):.3f}"  # Convert distance to similarity
            recommend_frame.append({'Title': movie_title, 'Score': similarity_score})
        
        return recommend_frame
    
    def display_results(self, recommendations, movie_name):
        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if isinstance(recommendations, str):
            # Error message
            self.results_title.config(text="❌ No Results Found")
            messagebox.showinfo("No Results", recommendations)
        else:
            # Display recommendations
            self.results_title.config(text=f"🎯 Recommendations for '{movie_name}'")
            
            for i, rec in enumerate(recommendations, 1):
                self.tree.insert('', 'end', values=(i, rec['Title'], rec['Score']))
        
        # Reset button and labels
        self.search_btn.config(state='normal', text="🔍 Get Recommendations")
        self.loading_label.config(text="")
        self.status_label.config(text=f"Found {len(recommendations) if isinstance(recommendations, list) else 0} recommendations")
    
    def handle_error(self, error_msg):
        self.search_btn.config(state='normal', text="🔍 Get Recommendations")
        self.loading_label.config(text="")
        self.status_label.config(text="Error occurred")
        messagebox.showerror("Error", f"An error occurred: {error_msg}")
    
    def clear_results(self):
        self.movie_entry.delete(0, tk.END)
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.results_title.config(text="")
        self.loading_label.config(text="")
        self.status_label.config(text="Ready" if self.data_loaded else "Failed to load data")

def main():
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()