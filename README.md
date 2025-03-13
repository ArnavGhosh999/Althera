<h1 align = "center">ALTHERA</h1>
<p align="justify"> ALTHERA is an advanced AI system that utilizes the MISTRAL LLM to provide smart medicine recommendations and optimization. It is built using a combination of powerful technologies to ensure accurate, efficient, and intelligent suggestions. The system is trained on real medical data using a Mistral LLM backend, allowing it to understand complex queries. A Random Forest Classifier is used to identify common patterns in infections, medications, and allergies, helping to provide better treatment recommendations. Additionally, a BERT-based NLP model enhances the system's ability to process natural language queries, and with Google Translate API, it supports multilingual users worldwide. ALTHERA also uses LanceDB, a vector database that efficiently stores and retrieves medical queries, ensuring fast and relevant responses. With these technologies, ALTHERA aims to revolutionize medicine optimization through AI-driven insights.</h2>

<h2>FUTURE MODIFICATIONS</h2>

- Integration of a Retrieval Augmentated Generation (RAG) model using LangChain and LangGraph for smart context to context response.

<img src = 'Readme Pics/LLM Architecture.jpg'>



<h1>ALTHERA</h1>

<h2>System Overview</h2>
<ul>
    <li>AI-driven medicine recommendation system using text-based similarity search.</li>
    <li>Integrates <strong>TF-IDF</strong>, <strong>Vector Databases</strong>, and <strong>LLM APIs</strong> for enhanced recommendations.</li>
</ul>

<h2>System Components</h2>

<h3>1. Data Preprocessing</h3>
<ul>
    <li><strong>Load Dataset:</strong> Imports structured medicine-related data.</li>
    <li><strong>Text Processing:</strong> Cleans and tokenizes textual information.</li>
    <li><strong>TF-IDF Vectors:</strong> Converts processed text into numerical feature vectors for similarity matching.</li>
</ul>

<h3>2. User Interface</h3>
<ul>
    <li><strong>Query Input:</strong> Users enter symptoms or conditions.</li>
    <li><strong>Display Results:</strong> Retrieves and presents recommended medicines.</li>
</ul>

<h3>3. Recommendation Condition</h3>
<ul>
    <li><strong>Extract Condition:</strong> Identifies medical symptoms from user input.</li>
    <li><strong>Similarity Search:</strong> Finds relevant medicine recommendations using vector-based comparison.</li>
    <li><strong>Get Details:</strong> Fetches detailed medicine data from the database.</li>
</ul>

<h3>4. Vector Database (LanceDB)</h3>
<ul>
    <li>Stores indexed medical data for fast retrieval.</li>
    <li>Enables high-speed searching for similar queries.</li>
</ul>

<h3>5. Query Storage & Retrieval</h3>
<ul>
    <li><strong>Query Storage:</strong> Logs past queries for analysis.</li>
    <li><strong>Searching Similar Query:</strong> Retrieves previously stored queries to optimize recommendations.</li>
</ul>

<h3>6. LLM Enhancement (Mistral API)</h3>
<ul>
    <li><strong>Create Prompt:</strong> Generates a structured query for LLM-based enhancement.</li>
    <li><strong>MISTRAL API:</strong> Uses a large language model for refining medicine suggestions.</li>
    <li><strong>Format Responses:</strong> Returns a user-friendly structured response.</li>
</ul>
</body>
</html>

