<h1><p align="center">ALTHERA</p></h1>

<h2><p align="center">System Overview</p></h2>
<p align="justify">ALTHERA is an AI-powered medicine recommendation system that uses text-based similarity search. It integrates <strong>TF-IDF</strong>, <strong>Vector Databases</strong>, and <strong>LLM APIs</strong> to enhance accuracy and efficiency.</p>

<h2><p align="center">System Components</p></h2>

<h3><p align="justify">1. Data Preprocessing</p></h3>
<p align="justify">The system loads structured medical data, processes text by cleaning and tokenizing, and converts it into numerical feature vectors using <strong>TF-IDF</strong> for efficient similarity matching.</p>

<h3><p align="justify">2. User Interface</p></h3>
<p align="justify">Users enter symptoms or conditions as queries, and the system retrieves and presents relevant medicine recommendations in an easy-to-read format.</p>

<h3><p align="justify">3. Recommendation Engine</p></h3>
<p align="justify">ALTHERA identifies key symptoms from user input, performs similarity searches using vector-based comparisons, and fetches detailed medicine information from the database.</p>

<h3><p align="justify">4. Vector Database (LanceDB)</p></h3>
<p align="justify">LanceDB stores indexed medical data, enabling high-speed retrieval and efficient real-time query matching for precise medicine recommendations.</p>

<h3><p align="justify">5. Query Storage & Retrieval</p></h3>
<p align="justify">The system logs past queries for future analysis and retrieves similar previously stored queries to optimize and refine recommendations over time.</p>

<h3><p align="justify">6. LLM Enhancement (Mistral API)</p></h3>
<p align="justify">By structuring queries intelligently, the system utilizes the <strong>Mistral API</strong> to refine recommendations using AI and presents responses in a well-structured, user-friendly format.</p>



<img src = 'Readme Pics/LLM Architecture.jpg'>

<h2>FUTURE MODIFICATIONS</h2>

- Integration of a Retrieval Augmentated Generation (RAG) model using LangChain and LangGraph for smart context to context response.


⚒️Setup and Installation
1. Create a Python virtual environment.
```
python -m venv venv
```
2. Create a config file for storing the API keys.
```
config.env
```

Python --version : 3.11.0 or better.




