# Financial-chatbot
Features of the chat bot are,
1. PII Anonymization - The program includes a function (anonymize_data) to detect and anonymize personally identifiable information (PII) such as email addresses, phone numbers, Social Security numbers, and credit card numbers using regular expressions. This protects user privacy during interactions.
2. Response Generation - It uses a fine-tuned GPT-2 model (generate_response) to generate contextual and relevant responses to user queries. The model can provide natural language responses based on the user input.
3. Word Pattern Analysis - The integration of spaCy allows for basic word pattern analysis, breaking the text into sentences and counting words. This can help in understanding user input better and improving responses.
4. Speculative Financial Advice Handling - The chatbot has a built-in mechanism (handle_financial_advice) to identify and prevent giving speculative financial advice. It recognizes specific phrases and responds appropriately, directing users to consult certified financial advisors instead.
5. Sensitive Information Escalation - The program includes a feature (escalate_to_human) to detect sensitive topics, such as fraud or financial disputes, and escalate these conversations to a human support agent. This ensures sensitive matters are handled appropriately.
6. Explainable Financial Decisions - The chatbot can explain its recommendations regarding financial decisions (explain_decision). It uses keywords related to loans and eligibility to provide context for its responses, promoting transparency.
7. User Interaction Management - The chatbot interacts with users through a command-line interface, allowing them to input questions and receive responses. It handles a "close" command to terminate the conversation.
