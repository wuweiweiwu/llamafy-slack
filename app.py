import os
import re


from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

# Initializes your app with your bot token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))


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


# @app.action("button_click")
# def action_button_click(body, ack, say):
#     # Acknowledge the action
#     ack()
#     say(f"<@{body['user']['id']}> clicked the button", thread_ts=body["value"])


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
