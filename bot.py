import telebot
import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request

load_dotenv()

TELEGRAM_API_KEY = os.getenv('TELEGRAM_API_KEY')
SECRET = os.getenv('SECRET')
url = os.getenv('URL') + SECRET

bot = telebot.TeleBot(TELEGRAM_API_KEY, threaded=False)
bot.remove_webhook()
bot.set_webhook(url=url)

app = Flask(__name__)
@app.route('/'+SECRET, methods=['POST'])
def webhook():
    update = telebot.types.Update.de_json(request.stream.read().decode('utf-8'))
    bot.process_new_updates([update])

    return 'OK', 200

@bot.message_handler(commands=['start', 'help'])
def hello(message):
    bot.send_message(message.chat.id, "Hello! I am MoodAI."
                    " Send me your selfie and I'll send you emojis"
                    " that best describe the mood that it gives off!")

@bot.message_handler(func=lambda message: True)
def text_respond(message):
    bot.send_message(message.chat.id, "Sorry I only analyze selfie.")
    
@bot.message_handler(content_types=['photo'])
def photo_input(message):
    bot.send_message(message.chat.id, "Analyzing photo...")

    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)
    
    img = cv2.imread('image.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    if len(faces) == 0:
        bot.send_message(message.chat.id, "No face detected. Sorry I only analyze selfie.")

    else:
        f = faces[0]
        face = img[f[1]:f[1]+f[3], f[0]:f[0]+f[2]]
        
        cv2.imwrite(f'face.jpg', face)

        #lst = os.listdir('faces')
        #number_files = len(lst)
        #cv2.imwrite(f'faces/{number_files}.jpg', face)

        face = cv2.imread('face.jpg')
        os.remove('face.jpg')
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'face.jpg', face)

        os.remove('image.jpg')

        image = Image.open('face.jpg')
        os.remove('face.jpg')
        data = np.asarray(image)
        data = data.reshape(1,48,48,1)
        data = data / 255

        model = tf.keras.models.load_model('fer13.h5')
        model_weights = np.load('fer13_weights.npy', allow_pickle=True)
        model.set_weights(model_weights)

        result = model.predict(data)
        result = 100 * result[0]
        emotions = ["\U0001F621 (angry):", "\U0001F922 (disgust):", "\U0001F631 (fear):", "\U0001F60D (happy):", "\U0001F97A (sad):", "\U0001F62F (surprise):", "\U0001F610 (neutral):"]

        print(result)
        max_emo = np.argmax(result)

        reply = 'MoodAI feels you are \n'
        for i in range(len(emotions)):
            if i == max_emo:
                reply += '<b>' + emotions[i] + f'{(result[i]): .2f}%' + '</b>'
            else:
                reply += emotions[i] + f'{(result[i]): .2f}%'
            reply += '\n'

        bot.send_message(message.chat.id, reply, parse_mode='HTML')

        button_0 = telebot.types.InlineKeyboardButton('\U0001F621 (angry)', callback_data='0')
        button_1 = telebot.types.InlineKeyboardButton('\U0001F922 (disgust)', callback_data='1')
        button_2 = telebot.types.InlineKeyboardButton('\U0001F631 (fear)', callback_data='2')
        button_3 = telebot.types.InlineKeyboardButton('\U0001F60D (happy)', callback_data='3')
        button_4 = telebot.types.InlineKeyboardButton('\U0001F97A (sad)', callback_data='4')
        button_5 = telebot.types.InlineKeyboardButton('\U0001F62F (surprise)', callback_data='5')
        button_6 = telebot.types.InlineKeyboardButton('\U0001F610 (neutral)', callback_data='6')

        keyboard = telebot.types.InlineKeyboardMarkup()
        keyboard.add(button_0)
        keyboard.add(button_1)
        keyboard.add(button_2)
        keyboard.add(button_3)
        keyboard.add(button_4)
        keyboard.add(button_5)
        keyboard.add(button_6)

        bot.send_message(message.chat.id, text='lmk your feeling!', reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    #with open("true_feelings.txt", "a") as myfile:
        #myfile.write(call.data+',')
    bot.delete_message(call.message.chat.id, call.message.message_id)
    bot.send_message(call.message.chat.id, "Thank you! \U0001F618")
