import logging
import os
from key import TOKEN
import model
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import time
import random
from groups import *

net = model.create_net('model.pth')

chats_dict = {}

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Bot started! Type /play to play')
    chat = update.message.chat
    chats_dict[chat] = Group()


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('This bot plays Gugu. Type /start and /play to try.')


def photo(update: Update, context: CallbackContext) -> int:
    try:
        group = chats_dict[update.message.chat]
    except:
        logging.info("Group has not started the bot yet.")
        return

    if(group.playing):
        user = update.message.from_user
        photo_file = update.message.photo[-1].get_file()
        photo_file.download('user_photo.jpg')
        logging.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
        tensor = model.img_tensor('user_photo.jpg')
        outputs = net(tensor)
        # predictions = model.torch.max(outputs)
        std = model.torch.std(outputs)
        predictions = model.torch.argmax(outputs)
        if(std > 8 and predictions == group.p_class):
            update.message.reply_text(f'Yeah! Thats exactly what I was looking for!')
            update.message.reply_text(f'You won this round, {user.name}.')
            group.playing = False
        elif(std > 6.5):
            update.message.reply_text(f'I am not sure if this is a {model.classes[group.p_class]}.')
        else:
            update.message.reply_text("What is this?")

def gugu(update: Update, context: CallbackContext):
    try:
        group = chats_dict[update.message.chat]
    except:
        update.message.reply_text('Please start the bot with /start.')
        return

    user = update.message.from_user
    group.p_class = random.randint(0,9)
    if(group.playing):
        update.message.reply_text(f'This one was too dificult, huh? Show me a {model.classes[group.p_class]}!')
    else:
        update.message.reply_text(f'Whats up? I want to see a {model.classes[group.p_class]}!')
        group.playing = True

def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater(TOKEN, use_context=True)
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("play", gugu))


    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, photo))
    updater.start_polling()
    logging.info("=== Bot running! ===")
    

    # updater.start_webhook(listen="0.0.0.0",
    #                   port=PORT,
    #                   url_path=TOKEN)
    # updater.bot.set_webhook("https://yourapp.herokuapp.com/" + TOKEN)
    updater.idle()
    logging.info("=== Bot shutting down! ===")

if __name__ == '__main__':
    main()


