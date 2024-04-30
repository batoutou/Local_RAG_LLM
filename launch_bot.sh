# source the environment
source venv/bin/activate

# delete the previous pdf DB
if [ -d "data/" ] ; then
    rm -Rf "data"
fi

if [ -d "vec_db" ] ; then
    rm -Rf "vec_db"
fi

# run the web app 
streamlit run main.py --server.fileWatcherType none