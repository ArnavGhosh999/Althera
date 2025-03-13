<h1 align = "center">ALTHERA</h1>
<p align="justify"> ALTHERA is an advanced AI system that utilizes the MISTRAL LLM to provide smart medicine recommendations and optimization. It is built using a combination of powerful technologies to ensure accurate, efficient, and intelligent suggestions. The system is trained on real medical data using a Mistral LLM backend, allowing it to understand complex queries. A Random Forest Classifier is used to identify common patterns in infections, medications, and allergies, helping to provide better treatment recommendations. Additionally, a BERT-based NLP model enhances the system's ability to process natural language queries, and with Google Translate API, it supports multilingual users worldwide. ALTHERA also uses LanceDB, a vector database that efficiently stores and retrieves medical queries, ensuring fast and relevant responses. With these technologies, ALTHERA aims to revolutionize medicine optimization through AI-driven insights.</h2>

<h2>FUTURE MODIFICATIONS</h2>

- Integration of a Retrieval Augmentated Generation (RAG) model using LangChain and LangGraph for smart context to context response.

<img src = 'Readme Pics/LLM Architecture.jpg'>



<h1><p align="center">ALTHERA</p></h1>

<h2><p align="justify">System Overview</p></h2>
<ul>
    <li><p align="justify">AI-driven medicine recommendation system using text-based similarity search.</p></li>
    <li><p align="justify">Integrates <strong>TF-IDF</strong>, <strong>Vector Databases</strong>, and <strong>LLM APIs</strong> for enhanced recommendations.</p></li>
</ul>

<h2><p align="justify">System Components</p></h2>

<h3><p align="justify">1. Data Preprocessing</p></h3>
<ul>
    <li><p align="justify"><strong>Load Dataset:</strong> Imports structured medicine-related data.</p></li>
    <li><p align="justify"><strong>Text Processing:</strong> Cleans and tokenizes textual information.</p></li>
    <li><p align="justify"><strong>TF-IDF Vectors:</strong> Converts processed text into numerical feature vectors for similarity matching.</p></li>
</ul>

<h3><p align="justify">2. User Interface</p></h3>
<ul>
    <li><p align="justify"><strong>Query Input:</strong> Users enter symptoms or conditions.</p></li>
    <li><p align="justify"><strong>Display Results:</strong> Retrieves and presents recommended medicines.</p></li>
</ul>

<h3><p align="justify">3. Recommendation Condition</p></h3>
<ul>
    <li><p align="justify"><strong>Extract Condition:</strong> Identifies medical symptoms from user input.</p></li>
    <li><p align="justify"><strong>Similarity Search:</strong> Finds relevant medicine recommendations using vector-based comparison.</p></li>
    <li><p align="justify"><strong>Get Details:</strong> Fetches detailed medicine data from the database.</p></li>
</ul>

<h3><p align="justify">4. Vector Database (LanceDB)</p></h3>
<ul>
    <li><p align="justify">Stores indexed medical data for fast retrieval.</p></li>
    <li><p align="justify">Enables high-speed searching for similar queries.</p></li>
</ul>

<h3><p align="justify">5. Query Storage & Retrieval</p></h3>
<ul>
    <li><p align="justify"><strong>Query Storage:</strong> Logs past queries for analysis.</p></li>
    <li><p align="justify"><strong>Searching Similar Query:</strong> Retrieves previously stored queries to optimize recommendations.</p></li>
</ul>

<h3><p align="justify">6. LLM Enhancement (Mistral API)</p></h3>
<ul>
    <li><p align="justify"><strong>Create Prompt:</strong> Generates a structured query for LLM-based enhancement.</p></li>
    <li><p align="justify"><strong>MISTRAL API:</strong> Uses a large language model for refining medicine suggestions.</p></li>
    <li><p align="justify"><strong>Format Responses:</strong> Returns a user-friendly structured response.</p></li>
</ul>


