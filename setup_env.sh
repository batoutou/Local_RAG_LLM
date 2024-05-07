sudo apt install python3-virtualenv

virtualenv -p python3 venv
source venv/bin/activate

pip install -r requirements.txt

curl https://ollama.ai/install.sh | sh
ollama pull llama3

https://medium.com/@ingridwickstevens/langchain-chat-with-your-data-qdrant-ollama-openai-913020ec504b