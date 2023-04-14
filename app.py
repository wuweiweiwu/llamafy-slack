import os
import re
import time
import json


from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import openai

# Load default environment variables (.env)
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")

openai.api_key = OPENAI_API_KEY


def openai_call(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.5,
    max_tokens: int = 100,
):
    while True:
        try:
            # Use chat completion API
            messages = [{"role": "system", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )

            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break


# Initializes your app with your bot token and socket mode handler
app = App(token=SLACK_BOT_TOKEN)


# Listens to incoming messages that contain "hello"
@app.event("app_mention")
def handle_mentions(event, client, say):
    thread_ts = event.get("thread_ts", None) or event["ts"]
    prompt = re.sub("\\s<@[^, ]*|^<@[^, ]*", "", event["text"])

    # say() sends a message to the channel where the event was triggered
    say(
        # blocks=[
        #     {
        #         "type": "section",
        #         "text": {"type": "mrkdwn", "text": f"Hey there <@{event['user']}>!"},
        #         "accessory": {
        #             "type": "button",
        #             "text": {"type": "plain_text", "text": "Click Me"},
        #             "action_id": "button_click",
        #             "value": thread_ts,
        #         },
        #     }
        # ],
        text=prompt,
        thread_ts=thread_ts,
    )


@app.event("message")
def handle_message_events(body, logger):
    logger.info(body)


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
