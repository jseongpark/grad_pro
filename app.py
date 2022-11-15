
from flask import Flask, request, render_template, json, jsonify

import os
import torch

default_dir = "C:\proj/final/test1/cnn/"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # GPU 할당

CFG = {
    'IMG_SIZE': 32,  # 이미지 사이즈
}

labels = []
word_to_num = dict()
f = open(default_dir + "ko.txt", 'r', encoding='utf-8')
index = 0

while True:
    line = f.readline()
    if not line: break
    labels.append(line[-2])
    word_to_num[line[-2]] = index
    index = index + 1
f.close()


def invert_dictionary(obj):
    return {value: key for key, value in obj.items()}


num_to_word = invert_dictionary(word_to_num)

from glob import glob


def get_test_data(data_dir):
    img_path_list = []

    # get image path
    img_path_list.extend(glob(os.path.join(data_dir, '*.png')))
    img_path_list.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))

    return img_path_list


import torchvision.transforms as transforms  # 이미지 변환 툴

from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, train_mode=True, transforms=None):  # 필요한 변수들을 선언
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list

    def __getitem__(self, index):  # index번째 data를 return
        img_path = self.img_path_list[index]
        # Get image data
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):  # 길이 return
        return len(self.img_path_list)


import cv2
import torch.nn as nn

class CNNclassification(nn.Module):
    def __init__(self):
        super(CNNclassification, self).__init__()
        self.layer1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)

        self.layer2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)

        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.layer6 = torch.nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer7 = torch.nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer8 = torch.nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc_layer = nn.Sequential(
            nn.Linear(2048, 1030)
        )

    def forward(self, x):
        x = self.layer1(x)  # 1층

        x = self.layer2(x)  # 2층

        x = self.layer3(x)  # 3층

        x = self.layer4(x)  # 4층

        x = self.layer5(x)  # 5층

        x = self.layer6(x)  # 6층

        x = self.layer7(x)  # 7층

        x = self.layer8(x)  # 8층

        x = torch.flatten(x, start_dim=1)  # N차원 배열 -> 1차원 배열

        out = self.fc_layer(x)
        return out


def predict(cnn_model, test_loader, device):
    cnn_model.eval()
    model_pred = []
    with torch.no_grad():
        for img in iter(test_loader):
            img = img.to(device)

            pred_logit = cnn_model(img)
            pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

            model_pred.extend(pred_logit.tolist())
    return model_pred


print("loading CNN model...")
checkpoint = torch.load(default_dir + 'best_model.pth', map_location=device)
cnn_model = CNNclassification().to(device)
cnn_model.load_state_dict(checkpoint)


def getSize(txt, font):
    testImg = Image.new('RGB', (1, 1))
    testDraw = ImageDraw.Draw(testImg)
    return testDraw.textsize(txt, font)


from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

colorBackground = "white"
colorText = "black"

from hangul_utils import join_jamos
from jamo import h2j, j2hcj


def cnn(text):
    # text=u"ひらがなㄱㄴㄷㄹ"

    bool = []
    text_length = len(text)
<<<<<<< HEAD
    for i in range(text_length):
        ch = text[i]
        if ord('가') <= ord(ch) <= ord('힣') or ord('a') <= ord(ch.lower()) <= ord('z'):
            bool.append(False)
        else:
            bool.append(True)
            font = ImageFont.truetype(default_dir + "ARIALUNI.ttf", 14)
            width, height = getSize(ch, font)
            img = Image.new('L', (width + 8, height + 8), colorBackground)
            d = ImageDraw.Draw(img)
            d.text((4, height / 2 - 4), ch, fill=colorText, font=font)

            img_dir = default_dir + 'imgs/'
=======
    img_dir = default_dir + 'imgs/'

    for i in range(text_length):
        ch = text[i]
        if ch == ' ' or ord('가') <= ord(ch) <= ord('힣') or ord('1') <= ord(ch) <= ord('9'): # ord('a') <= ord(ch.lower()) <= ord('z'):
            bool.append(False)
        else:
            bool.append(True)
            font = ImageFont.truetype(default_dir + "ARIALUNI.ttf", 14)
            width, height = getSize(ch, font)
            img = Image.new('L', (width + 8, height + 8), colorBackground)
            d = ImageDraw.Draw(img)
            d.text((4, height / 2 - 4), ch, fill=colorText, font=font)
>>>>>>> 1b649c283be3c980b6dc4f96425fa62490b90a6d
            img.save(img_dir + str(i) + ".png")

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    test_img_path = get_test_data(img_dir)
    test_dataset = CustomDataset(test_img_path, None, train_mode=False, transforms=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=len(text), shuffle=False, num_workers=0)

    preds = predict(cnn_model, test_loader, device)

    chs = list(map(lambda pred: num_to_word[pred], preds))
    

    flag = False
    if len(chs) > 1 and chs[0] == chs[1] and (bool[0] == True and bool[1] == True):
        flag = True
        del chs[0]
        text_length = text_length - 1
        if chs[0] == 'ㄱ':
            chs[0] = 'ㄲ'
        elif chs[0] == 'ㄷ':
            chs[0] = 'ㄸ'
        elif chs[0] == 'ㅂ':
            chs[0] = 'ㅃ'
        elif chs[0] == 'ㅅ':
            chs[0] = 'ㅆ'
        elif chs[0] == 'ㅈ':
            chs[0] = 'ㅉ'

<<<<<<< HEAD
    result = []
    j = 0
    for i in range(text_length):
=======


    result = []
    j = 0

    

    start = 1 if flag else 0
    for i in range(start, text_length):
>>>>>>> 1b649c283be3c980b6dc4f96425fa62490b90a6d
        if bool[i]:
            result.append(chs[j])
            j = j + 1
        else:
            result.append(text[i])
    return result
<<<<<<< HEAD

=======
>>>>>>> 1b649c283be3c980b6dc4f96425fa62490b90a6d

# example
# chat = u"^^l발ひらがなㄱㄴㄷㄹ"
# chs = cnn(chat)
# chat = ''.join(chs)
# result = join_jamos(chat) # <- 결과물

import tensorflow as tf
from tensorflow import keras
from transformers import TFRobertaModel
from RoBERTa_predict import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)
<<<<<<< HEAD
=======

roberta_model = tf.keras.models.load_model('./static/abusing_detection_1.h5',
custom_objects={'TFRobertaModel': TFRobertaModel})

roberta_tokenizer = get_tokenizer()
>>>>>>> 1b649c283be3c980b6dc4f96425fa62490b90a6d


@app.route("/")
def root():
    return render_template('index.html')

@app.route("/model", methods=['POST'])
def model():
    if request.method == 'POST':
        if os.path.exists(default_dir + 'imgs/'):
            for file in os.scandir(default_dir + 'imgs/'):
                # os.remove('C:\proj/final/test1/cnn/imgs/0.png')
                os.remove(file)
            print('img 폴더 초기화 완료')

        chat = request.form['chat']

        text1 = cnn(chat)
        print('CNN 처리 이후 : ')
        print(text1)
        text2 = ''.join(text1)

        text3 = join_jamos(text2)
        print('한글 Automata 처리 이후 : ')
        print(text3)
<<<<<<< HEAD
        roberta_model = tf.keras.models.load_model('./static/abusing_detection_1.h5',
                                                   custom_objects={'TFRobertaModel': TFRobertaModel})

        roberta_tokenizer = get_tokenizer()
=======

>>>>>>> 1b649c283be3c980b6dc4f96425fa62490b90a6d

        prediction = get_predict_by_model(roberta_model, roberta_tokenizer, text3)
        print('욕설 확률 : ')
        print(prediction)
        prediction2 = round(prediction, 1)
        jsondata = json.dumps(prediction2)

        return jsondata


if __name__ == "__main__":
    print("runnig server")
    app.run(host="127.0.0.1", port=5000, debug=False)
