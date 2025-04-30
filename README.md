# fliptastic
Final project for Brown's CSCI 1430 Computer Vision course

# Clone
To clone the repository, run the following command:
```
git clone https://github.com/chris100904/fliptastic.git
```

# Environment Setup
This project is compiled in Python 3.9. On Mac, you can run `brew install python@3.9` if it is not already installed.

To create your virtual environment, run the following command in a new terminal:

```
python3.9 -m venv fliptastic
source fliptastic/bin/activate
```

To install the packages required to run this project, run the following command in your terminal:
```
pip install -r requirements.txt
```
# Run
To run the frontend server:
- Navigate to `fliptastic-ui` and run `npm start`:

```
cd fliptastic-ui
npm start
```

To run the backend server:
```
cd backend
uvicorn main:app --reload --port 8000
```


