# CineVerse AI backend
pip install -r requirements.txt
uvicorn main:app --reload
# POST /recommend {"genres":["Sci-Fi","Thriller"],"debias":0.6}  (new user)
# POST /recommend {"user_id":12}                                  (known user)
