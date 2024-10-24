!pip install transformers

!pip install detoxify

pip install transformers torch



# GPT-2 model training

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import Dataset, DataLoader

# Sample financial questions and answers for fine-tuning
financial_data = [
    {"question": "What is the interest rate on savings accounts?", "answer": "The interest rate on savings accounts is currently 1.5%."},
    {"question": "How do I apply for a loan?", "answer": "You can apply for a loan online through our website or visit your nearest branch."},
    {"question": "What is the minimum balance required?", "answer": "The minimum balance required is $500 for standard accounts."},
    {"question": "How can I dispute a transaction?", "answer": "To dispute a transaction, please contact our customer support and provide the transaction details."},
    {"question": "What are the fees for international transfers?", "answer": "The fees for international transfers vary by amount and destination. Please refer to our fee schedule."},
    {"question": "What is a savings account?", "answer": "A savings account is a deposit account that allows individuals to store money securely while earning interest on their balance. It typically limits withdrawals and is meant for long-term savings."},
    {"question": "How does compound interest work?", "answer": "Compound interest is calculated on the initial principal as well as the accumulated interest from previous periods. This leads to exponential growth of the balance over time."},
    {"question": "What is the minimum credit score for a personal loan?", "answer": "The minimum credit score required for a personal loan varies by lender, but generally, a score of at least 600 is needed to qualify for most personal loans."},
    {"question": "How can I apply for a home loan?", "answer": "You can apply for a home loan by contacting a bank or lender directly, submitting your financial documents such as proof of income, credit reports, and personal identification, and completing an application form."},
    {"question": "What is the difference between a fixed-rate and an adjustable-rate mortgage?", "answer": "A fixed-rate mortgage has an interest rate that remains constant for the entire loan term, while an adjustable-rate mortgage has an interest rate that may fluctuate periodically based on market conditions."},
    {"question": "What is an emergency fund?", "answer": "An emergency fund is a savings account that is set aside for unexpected expenses such as medical emergencies, car repairs, or loss of income. It typically covers 3 to 6 months' worth of living expenses."},
    {"question": "What is a credit report?", "answer": "A credit report is a detailed record of your credit history, including your borrowing and repayment behavior. It includes information such as your credit accounts, outstanding debts, and payment history."},
    {"question": "What does it mean to refinance a loan?", "answer": "Refinancing a loan means replacing your current loan with a new one, often to get a lower interest rate, extend the loan term, or change other loan conditions."},
    {"question": "What is a 401(k) retirement account?", "answer": "A 401(k) is a retirement savings plan offered by employers in the U.S. It allows employees to save and invest a portion of their paycheck before taxes are deducted, helping to grow their retirement fund."},
    {"question": "What is the purpose of a financial advisor?", "answer": "A financial advisor provides personalized advice on managing your money, including budgeting, investing, retirement planning, and saving for specific goals."},
    {"question": "How can I improve my credit score?", "answer": "You can improve your credit score by paying your bills on time, reducing your credit card balances, avoiding opening multiple new credit accounts, and checking your credit report for errors."},
    {"question": "What is a certificate of deposit (CD)?", "answer": "A certificate of deposit (CD) is a savings product offered by banks that provides a fixed interest rate for a specified term, such as six months or five years. CDs typically offer higher interest rates than regular savings accounts."},
    {"question": "What are the tax benefits of contributing to an IRA?", "answer": "Contributions to a traditional Individual Retirement Account (IRA) are tax-deductible, and the earnings grow tax-deferred until you withdraw the funds in retirement. Roth IRA contributions are made with after-tax dollars, but qualified withdrawals are tax-free."},
    {"question": "What are the risks of investing in the stock market?", "answer": "Investing in the stock market carries risks such as market volatility, where the value of your investments can rise or fall due to economic conditions, company performance, and other factors. There is also the risk of losing your principal investment."},
    {"question": "What is an annuity?", "answer": "An annuity is a financial product that provides a steady income stream, typically used for retirement. You can invest a lump sum or make periodic payments, and in return, you receive regular payouts over a specified period."},
    {"question": "What is the difference between term life insurance and whole life insurance?", "answer": "Term life insurance provides coverage for a specific period, such as 10 or 20 years, while whole life insurance offers coverage for your entire lifetime. Whole life insurance also includes a savings component that builds cash value over time."},
    {"question": "What is the difference between a checking account and a savings account?", "answer": "A checking account is designed for frequent transactions such as deposits and withdrawals, while a savings account is meant for longer-term savings and earns interest. Checking accounts typically come with debit cards, while savings accounts have limited transaction allowances."},
    {"question": "What are overdraft fees?", "answer": "Overdraft fees are charges imposed by banks when you withdraw more money from your checking account than you have available. Banks may cover the transaction, but they will charge a fee for the overdraft service."},
    {"question": "What is a mutual fund?", "answer": "A mutual fund pools money from multiple investors to invest in a diversified portfolio of stocks, bonds, or other securities. It is managed by a professional portfolio manager, and investors share in the gains or losses of the fund."},
    {"question": "What is a balance transfer on a credit card?", "answer": "A balance transfer allows you to move debt from one credit card to another, often to take advantage of lower interest rates or promotional offers. It can help you save on interest and pay off your debt faster."}
]

# Convert the financial data into a single text dataset
def create_training_data(data):
    training_texts = []
    for entry in data:
        qa_pair = f"Question: {entry['question']}\nAnswer: {entry['answer']}\n\n"
        training_texts.append(qa_pair)
    return "".join(training_texts)

# Prepare the training text
training_text = create_training_data(financial_data)

# Create a custom dataset class
class FinancialDataset(Dataset):
    def __init__(self, tokenizer, text, max_length=512):
        self.input_ids = []
        self.attn_masks = []

        # Encode the text
        encodings = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        self.input_ids.append(encodings['input_ids'])
        self.attn_masks.append(encodings['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[0][idx], self.attn_masks[0][idx]

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set eos_token as the pad_token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Use the entire dataset for training without splitting
train_dataset = FinancialDataset(tokenizer, training_text)

# DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=2)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Fine-tuning the model
def train(model, train_loader, epochs=3):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        # Training loop
        for batch in train_loader:
            input_ids, attn_masks = batch
            outputs = model(input_ids=input_ids, attention_mask=attn_masks, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss}")

    return model

# Fine-tune the model
fine_tuned_model = train(model, train_loader)

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
print("Model fine-tuned and saved!")

# Chatbot

import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from spacy.lang.en import English

# Load the fine-tuned GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")

# Set up spaCy for word pattern analysis
nlp = English()
nlp.add_pipe("sentencizer")

# Function to anonymize PII data
def anonymize_data(text):
    pii_patterns = {
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'phone': r'\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b(?:\d[ -]*?){13,16}\b'
    }

    for pattern_name, pattern in pii_patterns.items():
        text = re.sub(pattern, f'<{pattern_name}>', text)  # Replace PII with a placeholder

    return text

# Function to generate a response using the fine-tuned GPT-2 model
def generate_response(text):
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to perform basic word pattern analysis with spaCy
def analyze_text_patterns(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    word_count = len(text.split())
    return sentences, word_count

# Function to check for speculative financial advice
def handle_financial_advice(question):
    speculative_keywords = ["should I invest", "is it a good idea", "loan approval", "guaranteed return"]

    for keyword in speculative_keywords:
        if keyword in question.lower():
            return "I cannot provide speculative financial advice. Please contact a certified financial advisor."

    return None

# Escalate sensitive conversations to human support agent
def escalate_to_human(text):
    sensitive_keywords = ["fraud", "dispute", "illegal", "financial loss", "court case"]
    for keyword in sensitive_keywords:
        if keyword in text.lower():
            return "This query involves sensitive information. I will connect you to a human agent for further assistance."

    return None

# Function to explain decisions in financial conversations
def explain_decision(question):
    loan_related_keywords = ["loan approved", "loan rejected", "eligibility", "approval", "denial"]

    for keyword in loan_related_keywords:
        if keyword in question.lower():
            # Placeholder for explaining the recommendation
            return "Based on your credit score, income, and debt-to-income ratio, we were able to determine your loan eligibility. If you'd like more clarification, feel free to ask."

    return None

# Main Chatbot Function
def chatbot():
    print("Chatbot: Hello! How can I assist you today with your financial queries?")
    print("Type 'close' to end the conversation.")

    while True:
        user_input = input("You: ")

        # Step 1: Check for 'close' command
        if user_input.lower() == "close":
            print("Chatbot: Thank you for chatting with me. Goodbye!")
            break

        # Step 2: Anonymize user input to protect privacy
        sanitized_input = anonymize_data(user_input)

        # Step 3: Perform basic text pattern analysis
        sentences, word_count = analyze_text_patterns(sanitized_input)
        print(f"Chatbot (internal): Analyzed {len(sentences)} sentences and {word_count} words.")

        # Step 4: Check if input contains speculative or unsafe financial advice questions
        financial_advice_response = handle_financial_advice(sanitized_input)
        if financial_advice_response:
            print(f"Chatbot: {financial_advice_response}")
            continue

        # Step 5: Escalate sensitive conversations to a human agent
        escalation_response = escalate_to_human(sanitized_input)
        if escalation_response:
            print(f"Chatbot: {escalation_response}")
            continue

        # Step 6: Provide explainable recommendations for transparency
        explanation_response = explain_decision(sanitized_input)
        if explanation_response:
            print(f"Chatbot: {explanation_response}")
            continue

        # Step 7: Generate a response using fine-tuned GPT-2
        gpt2_response = generate_response(sanitized_input)
        print(f"Chatbot: {gpt2_response}")

# Start the chatbot
chatbot()
