Why this project🎯
Most LLM applications send every request to the same model, regardless of whether the question actually needs it This system first measures the query's meaning — using semantic embeddings and cosine similarity to gauge how complex or simple it actually is, by comparing it against reference examples rather than counting words — then routes it to the right Gemini version accordingly, from Gemini Pro (for complex questions) down to Flash (for simpler ones) aiming to pick whatever is most cost-appropriate for that specific query.
To avoid hitting the API on every single request the system checks a semantic cache first: if a similar question was already asked and stored it returns that cached answer instead of calling the API again — cutting cost further and avoiding unnecessary pressure on the API.


How it works⚙️

Preprocessing: The text is cleaned first — punctuation marks and extra whitespace are stripped, so only the meaningful content remains.

Vectorization: The cleaned text is converted into an embedding — a real vector representation of its meaning not just superficial text features.

Similarity Matching: Cosine similarity is used to find the closest match between the query and reference examples (greetings, complexity, and cached queries).

Error Handling: Errors are handled clearly rather than hidden — for example hitting Gemini's rate limit returns a clear message instead of a raw technical error Every query whether successful or failed is still logged.

Logging: Each query's route tokens real cost and latency are recorded in SQLite.

Efficiency Over Time: As usage accumulates the cache hit rate increases reducing API load for repeated or similar questions — making the system more efficient and reliable over time in terms of cost and speed Routing decision accuracy itself is currently static (based on predefined reference examples); improving it further would require updating these examples using real accumulated data.