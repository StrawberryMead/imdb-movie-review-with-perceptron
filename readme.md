Needed packages are:
1. Numpy
2. scikit-learn
3. Flask

Or

Just run this bad boi: 
pip install Flask scikit-learn numpy

you can get the data set from here:
http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

Just extract it in the root folder (there should be aclImdb folder inside it) next to app.py etc etc
Like this:
root_folder/
│
├── aclImdb/
│   ├── train/
│   │   ├── pos/
│   │   └── neg/
│   └── test/
│       ├── pos/
│       └── neg/
├── static/
│       └── css/
│           └── style.css
├── templates/
│   └── index.html
│
├── .gitignore
├── app.py
└── readme.md