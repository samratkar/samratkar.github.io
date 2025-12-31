import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hey There! My name is Piyush Garg"
tokens = enc.encode(text)

# Tokens [25216, 3274, 0, 3673, 1308, 382, 398, 3403, 1776, 170676]
print("Tokens", tokens)

decoded = enc.decode([25216, 3274, 0, 3673, 1308, 382, 398, 3403, 1776, 170676])
print("Decoded", decoded)