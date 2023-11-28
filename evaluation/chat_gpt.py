import re
import openai


class ChatGPT:
    def __init__(self):
        pass

    def rate_prompt(self, prompt):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        response = completion.choices[0].message['content']
        return self.parse_response(response)

    def parse_response(self, response):
        print(response)
        match = re.search("Serendipity: (\S+)", response)
        rating = float(match[1])
        return rating

