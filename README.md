🍽️ Restaurant Success Predictor & Recommender

Hey there! Welcome to our awesome restaurant analysis project that combines the power of NLP, machine learning, and some cool recommendation algorithms to help foodies find their next favorite spot!

🌟 What's This All About?

This project does two super cool things:
Figures out how successful a restaurant might be (on a scale of 0-10) by analyzing Yelp reviews
Recommends restaurants to users based on their location, food preferences, and our fancy success scores

🛠️ Tech Stack
Machine Learning: Hugging Face Transformers, NMF (Non-negative Matrix Factorization)
Backend: FastAPI, Flask
Data Processing: pandas, scikit-learn
Frontend: HTML, Bootstrap, JavaScript
Database: Handles 1.8M+ Yelp reviews like a champ!

🎯 Features

Restaurant Success Score
Uses state-of-the-art NLP models to analyze review sentiment
Implements NMF for smart feature extraction
Generates a reliable success score between 0-10
Smart Recommendations
Location-aware suggestions
Cuisine preference matching
Minimum success score filtering
Fast and efficient for large datasets

🤝 Contributing

Got ideas? We'd love to hear them! Here's how you can help:
Fork it

Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 Notes
The system is designed to handle 1.8M+ reviews efficiently
Uses batch processing for large datasets
Implements proper error handling and rate limiting

🐛 Known Issues
Heavy CPU usage during initial model loading
Might be slow on first recommendation request due to feature matrix building

🎉 Acknowledgments
Shoutout to Yelp for the amazing dataset
Thanks to the Hugging Face team for their awesome transformers library
High five to all the contributors who made this possible!

📫 Questions?
Feel free to open an issue or reach out if you have any questions. We're here to help! Made with ❤️ and lots of ☕
