import tiktoken
import openai

def count_tokens(text):
  # Choose the appropriate tokenizer for your model
  encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
  tokens = encoding.encode(text)

  # Return the token count
  return len(tokens)

if __name__ == "__main__":
  input_text = """const expectedPrompt = `Given a Golden Retriever dog aged 8 months, exhibiting the following symptoms:
    Gut:
    Symptom: Vomiting; Duration: 2023-10-04; Condition: Getting worse.
    Symptom: No appetite; Duration: 2023-10-02; Condition: Getting worse.
    Skin:
    Symptom: Redness; Duration: 2023-10-02; Condition: Getting worse.
    Provide your assessment in the following format: 'Assessment: [diagnosis]. Reasoning: [brief reasoning]. Suggested Actions: [general recommended next steps, personalized for breed and age]. Accuracy Score: [score out of 100]. Urgency Score: [score out of 100].' Note: This is a general suggestion and may not replace professional veterinary advice. Provide a concise assessment not exceeding 10 sentences and make it personalized to breed and age.`;
    """

  # Count the tokens in the input text
  token_count = count_tokens(input_text)

  # Token price for gpt-3.5-turbo with a 16K context
  token_price = 0.004  # $0.004 per 1K tokens

  # Calculate the total price
  total_price = token_count * (token_price / 1000)  # Convert token price to per token

  print(f"Token count: {token_count}")
  print(f"Token price for gpt-3.5-turbo (16K context): ${token_price} per 1K tokens")
  print(f"Total price: ${total_price:.4f}")

  # Calculate the number of queries for $1, $10, and $100
  queries_1_dollar = 1 / total_price
  queries_10_dollars = 10 / total_price
  queries_100_dollars = 100 / total_price

  print(f"Number of Queries for $1: {queries_1_dollar:.2f}")
  print(f"Number of Queries for $10: {queries_10_dollars:.2f}")
  print(f"Number of Queries for $100: {queries_100_dollars:.2f}")
