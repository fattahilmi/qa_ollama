import sys
import openai
from threading import Thread
#from multiprocessing import Process
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel

from PyQt6.QtWidgets import (
    QMainWindow, QApplication,
    QLabel, QToolBar, QStatusBar, QSpinBox,
    QLineEdit, QPushButton, QMessageBox, QVBoxLayout,
    QFormLayout, QPlainTextEdit, QFileDialog
)

from PyQt6 import QtCore, QtGui
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt #, QObject, pyqtSignal, QRunnable, pyqtSlot, QThreadPool
from gtts import gTTS
from playsound import playsound

import numpy as np
from transformers import BertForQuestionAnswering, GPT2Tokenizer, TFGPT2Model
#from transformers import BertTokenizer
from transformers import BertTokenizerFast, pipeline, set_seed
import requests

from pvrecorder import PvRecorder
import speech_recognition as sr
import wave
import struct
from time import time, sleep
#import traceback, sys

model_PATH = 'train_log/model-indolem-16-4-new/model/'


#model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
#model.save_pretrained("/BertModel/")
#tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
#tokenizer = BertTokenizer.from_pretrained("/BertModel/.")

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Sistem Tanya Jawab")
        self.left = 10
        self.top = 10
        self.width = 620
        self.height = 500
        self.konteks = []
        self.language = 'id'
        self.jawaban = ""
        # self.record

        

        #self.model = BertForQuestionAnswering.from_pretrained("Rifky/Indobert-QA") #from_pt=True
        #self.tokenizer = BertTokenizerFast.from_pretrained('Rifky/Indobert-QA')
        #self.model = BertForQuestionAnswering.from_pretrained("Wikidepia/indobert-lite-squad")
        #self.tokenizer = BertTokenizerFast.from_pretrained('Wikidepia/indobert-lite-squad')
        self.model = BertForQuestionAnswering.from_pretrained(model_PATH)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_PATH)
        # self.tokenizerGPT = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
        # self.modelGPT = OpenAIGPTModel.from_pretrained("openai-gpt")
        # self.generator = pipeline('text-generation', model='cahya/gpt2-small-indonesian-522M')

        #question answer
        self.qa_pipeline = pipeline("question-answering",
        model=self.model,
        tokenizer=self.tokenizer)

        # #question answer GPT
        # self.qa_pipelineGPT = pipeline("question-answering",
        # model=self.modelGPT,
        # tokenizer=self.tokenizerGPT)

        '''
        #summarization
        self.qa_pipeline = pipeline("summarization",
        model=self.model,
        tokenizer=self.tokenizer)
        '''

        self.setGeometry(self.left, self.top, self.width, self.height)


        toolbar = QToolBar("My main toolbar")
        self.addToolBar(toolbar)
        self.labelRec = QLabel('')
        self.labelRec.setFixedSize(130, 15)
        self.konteks = []


        #self.konteks.append("""MRT Jakarta dikelola oleh PT MRT Jakarta yang berstatus BUMD (Badan Usaha Milik Daerah), yang dimiliki oleh Pemerintah Provinsi DKI Jakarta. Pendanaan proyek berasal dari pinjaman Pemerintah Jepang melalui Japan International Cooperation Agency (JICA) ke Pemerintah Indonesia. Dana tersebut kemudian diterushibahkan ke Pemerintah Provinsi DKI Jakarta sebesar 49% dan diserahpinjamkan sebesar 51%.MRT Jakarta dirancang sebagai salah satu pilihan moda raya terpadu untuk mengatasi masalah kemacetan di Jakarta. PT MRT Jakarta memiliki misi untuk menjadi penyedia jasa transportasi publik terdepan yang berkomitmen untuk mendorong pertumbuhan ekonomi melalui peningkatan mobilitas, pengurangan kemacetan, dan pengembangan sistem transit perkotaan. Selayaknya instrumen transportasi publik terpenting adalah kemauan masyarakat untuk berubah. Kehadiran MRT Jakarta yang aman dan nyaman serta dapat diandalkan diharapkan menjadi magnet bagi masyarakat yang selama ini menggunakan kendaraan pribadi, terlebih dengan akan adanya interkoneksi dengan KRL, Transjakarta, LRT, dan kereta bandara.""")

        button_action = QAction("Lihat Kalimat Konteks", self)
        button_action.setStatusTip("This is your button")
        button_action.triggered.connect(self.lihatKalimatKonteks)
        toolbar.addAction(button_action)
        button_action.setCheckable(True)
        toolbar.addAction(button_action)

        button_action2 = QAction("Ganti Kalimat Konteks", self)
        button_action2.setStatusTip("Button Ganti Context")
        button_action2.triggered.connect(self.loadContext)
        toolbar.addAction(button_action2)

        button_action3 = QAction("Keluar", self)
        button_action3.setStatusTip("Button Keluar")
        button_action3.triggered.connect(self.keluar)
        toolbar.addAction(button_action3)

        #self.boxPertanyaan = QLineEdit(self)
        self.boxPertanyaan = QPlainTextEdit(self)
        self.boxPertanyaan.setFixedWidth(570)
        self.boxPertanyaan.setFixedHeight(50)
        self.boxPertanyaan.move(20, 80)
        label_1 = QLabel("Pertanyaan",self)
        label_1.move(20, 40)



        self.BertBtn = QPushButton('Jawab', self)
        # self.BertBtn.setGeometry(20,185,140,40) #posisi (x,y) ukuran (w,h)
        self.BertBtn.setGeometry(20,140,140,40) #posisi (x,y) ukuran (w,h)
        #layout.addWidget(button)
        # connect button to function on_click
        self.BertBtn.clicked.connect(self.jawab)

        #boxJawaban = QLineEdit(self)
        self.boxJawaban = QPlainTextEdit(self)
        self.boxJawaban.setFixedWidth(570)
        self.boxJawaban.setFixedHeight(200)
        self.boxJawaban.move(20, 270)
        label_2 = QLabel("Jawaban :",self)
        label_2.move(20, 235) #posisi (x,y)

        self.tambahBtn = QPushButton("Tambah Kalimat Konteks",self)
        self.tambahBtn.clicked.connect(self.tambahKonteks)
        self.tambahBtn.setGeometry(180,140,180,40)
        print(self.konteks)

        self.hpsBtn = QPushButton("Hapus Pertanyaan",self)
        self.hpsBtn.clicked.connect(self.hapus)
        self.hpsBtn.setGeometry(380,140,150,40)

        self.recBtn = QPushButton("",self)
        self.recBtn.setIcon(QtGui.QIcon('mic.png'))
        self.recBtn.setIconSize(QtCore.QSize(24,24))
        self.recBtn.clicked.connect(self.rekamSuara)
        self.recBtn.setGeometry(550,140,40,40)
        

        # Create a button in the window
        self.procBtn = QPushButton('Jawaban Ollama', self)
        self.procBtn.setGeometry(20,185,140,40) #posisi (x,y) ukuran (w,h)
        #layout.addWidget(button)
        # connect button to function on_click
        self.procBtn.clicked.connect(self.jawab)

        # self.recBtn = QPushButton("",self)
        # self.recBtn.setIcon(QtGui.QIcon('mic.png'))
        # self.recBtn.setIconSize(QtCore.QSize(24,24))
        # # self.recBtn.clicked.connect(self.rekamSuara)
        # self.recBtn.setGeometry(550,140,40,40)

    def setStop (self):
        self.berhenti = 1

    def rekamSuara(self):
        r = sr.Recognizer()
        self.berhenti = 0

        print("===> Begin recording. Press Ctrl+C to stop the recording ")
        recorder = PvRecorder(device_index=-1, frame_length=512)
        audio = []

        path = "pertanyaan.wav"
        recorder.start()
        #self.labelRec.setText('◉ recording...')
        self.repaint()
        start = time()

        while (time()-start < 4):
            frame = recorder.read()
            audio.extend(frame)
        #except (self.berhenti == 1) :

        recorder.stop()
        with wave.open(path, 'w') as f:
            f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
            f.writeframes(struct.pack("h" * len(audio), *audio))
        # Extract data and sampling rate from file
        print('Converting audio transcripts into text ...')
        try:
            with sr.AudioFile(path) as source:
                audio_text = r.listen(source)
                text = r.recognize_google(audio_text, language = "id-ID")
                print(text)
                self.boxPertanyaan.clear()
                self.boxPertanyaan.insertPlainText(text)
        except sr.UnknownValueError:
            print("Silakan Mengulangi Lagi")

        recorder.delete()

    def cetakJawaban(self):
        self.boxJawaban.insertPlainText(self.jawaban)

    def ucapkanJawaban(self):
        try:
            myobj = gTTS(text=self.jawaban, lang=self.language, slow=False)
            myobj.save('jawaban.mp3')
            playsound('jawaban.mp3')
        except Exception as e:
            print("Tidak ada koneksi Internet di ucapkanJawaban")

    def setelahEnamDetik(self):
        self.boxJawaban.insertPlainText("\nSetelah 6 detik")

    def jawab(self):
        self.boxJawaban.clear()

        self.jawaban = self.jawabOllama()
        print("kalimat jawaban: ", self.jawaban)

        # Execute

        #Using threadpool
        self.threadlist = []
        self.threadlist.append(Thread(target=self.cetakJawaban()))
        self.threadlist.append(Thread(target=self.ucapkanJawaban()))

        #self.processlist = []
        #self.processlist.append(Process(target=self.ucapkanJawaban(jawaban)))
        #self.processlist.append(Process(target=self.cetakJawaban(jawaban)))


        for t in self.threadlist:
            t.start()

        for t in self.threadlist:
            t.join()

        #for t in self.processlist:
        #    t.start()

        #for t in self.processlist:
        #    t.join()

    # def jawabOpenAI(self):
    #     self.boxJawaban.clear()

    #     jawaban = self.question_answerOpenAI()
    #     self.jawaban = ""

    #     for jwb in jawaban :
    #         self.jawaban = self.jawaban+"\n"+jwb
    #         #print("jawaban ke: ",i," jwb")
    #     print("kalimat jawaban: ", self.jawaban)
    #     #self.boxJawaban.insertPlainText(self.jawaban)
    #     self.threadlist = []
    #     self.threadlist.append(Thread(target=self.cetakJawaban()))
    #     self.threadlist.append(Thread(target=self.ucapkanJawaban()))

    #     #self.processlist = []
    #     #self.processlist.append(Process(target=self.ucapkanJawaban(jawaban)))
    #     #self.processlist.append(Process(target=self.cetakJawaban(jawaban)))


    #     for t in self.threadlist:
    #         t.start()

    #     for t in self.threadlist:
    #         t.join()

    def hapus(self):
        self.boxPertanyaan.clear()
        print("Hapus")

    def lihatKalimatKonteks(self, s):
            print("click", s)

    def loadContext(self):
        #print("Ganti Kalimat Konteks")
        fname = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "${HOME}",
            "All Files (*);; Text Files (*.txt)",
        )
        print("Nama File: ",fname[0])
        f = open(fname[0],'r')
        kalimat = f.read()
        self.konteks = []
        self.konteks.append(kalimat)
        f.close()
        print(self.konteks)

    def tambahKonteks(self):
        print("Masuk tambah kalimat konteks")
        try:
            fname = QFileDialog.getOpenFileName(
                self,
                "Open File",
                "${HOME}",
                "All Files (*);; Text Files (*.txt)",
            )
            print("Nama File: ",fname[0])
            f = open(fname[0],'r')
            kalimat = f.read()
            self.konteks.append(kalimat)
            print(kalimat)
            f.close()
            print(self.konteks)
        except FileNotFoundError:
            print("File tidak ditemukan")
    def value_changed(self, i):
        print(i)

    def value_changed_str(self, s):
        print(s)

    def keluar(self):
        sys.exit()

    def question_answer(self):
        question = self.boxPertanyaan.toPlainText()
        print("Pertanyaan: ",question)
        #tokenize question and text as a pair

        j = 0
        #ketemu = 0

        answer = []

        print("panjang list konteks: ",len(self.konteks))
        maxSkor = 0
        jawab = ""

        while (j<len(self.konteks)):
            jawaban = self.qa_pipeline({'question': question, \
            'context': self.konteks[j]})

            jwb = jawaban.get('answer')
            skor = jawaban.get('score')
            print("jawaban ke: ",j," ",jwb," dengan skor: ",skor)
            if (maxSkor < skor):
                #answer.append(jwb)
                jawab = jwb
                maxSkor = skor
                #answer = "Unable to find the answer to your question."

            j = j+1
        #answer.append(jawab)
        #print("jawaban: ",answer)
        return jawab

    def question_answerOpenAI(self):
        question = self.boxPertanyaan.toPlainText()
        print("Pertanyaan: ",question)
        #answer = question
        #tokenize question and text as a pair
        openai.api_key = "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"

        j = 0
        #ketemu = 0

        answer = []

        print("panjang list konteks: ",len(self.konteks))
        jawab = ""
        konteks = ""

        for k in self.konteks:
            konteks = konteks + k

        message=[{"role":"assistant","content": konteks, "role": "user", "content": question}]

        #response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages= [{ \
        try:
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo",\
            #messages= [{ \
            #'role':'assistant','content': konteks, \
            #'role':'user','content': question}],
            messages = message,\
            temperature=0.2, \
            max_tokens=1000, \
            frequency_penalty=0.0)
            print("Konteks: ",konteks)
            jawab = response["choices"][0]["message"]["content"]
            print(jawab)
                #jwb = jawaban.get('answer')
                #skor = jawaban.get('score')
                #if (maxSkor < skor):
                    #answer.append(jwb)
                #    jawab = jwb
                #    maxSkor = skor
                    #answer = "Unable to find the answer to your question."
            answer.append(jawab)
        except Exception as e:
            print("Tidak ada koneksi Internet di question_answerOpenAI")
        return answer

    #def textGenertor(self):
        #question = self.boxPertanyaan.toPlainText()
        #set_seed(42
        #gen = generator(question, max_length=30, num_return_sequences=5, num_beams=10)




    def question_answerOpenAI_GPT(self):
        question = self.boxPertanyaan.toPlainText()
        print("Pertanyaan: ",question)
        #answer = question
        #tokenize question and text as a pair
        openai.api_key = "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"

        j = 0
        #ketemu = 0

        answer = []

        print("panjang list konteks: ",len(self.konteks))
        jawab = ""
        konteks = ""

        for k in self.konteks:
            konteks = konteks + k

        message=[{"role":"assistant","content": konteks, "role": "user", "content": question}]

        #response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages= [{ \
        try:
            response = openai.ChatCompletion.create(model="open-ai",\
            #messages= [{ \
            #'role':'assistant','content': konteks, \
            #'role':'user','content': question}],
            messages = message,\
            temperature=0.2, \
            max_tokens=1000, \
            frequency_penalty=0.0)
            print("Konteks: ",konteks)
            jawab = response["choices"][0]["message"]["content"]
            print(jawab)
                #jwb = jawaban.get('answer')
                #skor = jawaban.get('score')
                #if (maxSkor < skor):
                    #answer.append(jwb)
                #    jawab = jwb
                #    maxSkor = skor
                    #answer = "Unable to find the answer to your question."
            answer.append(jawab)
        except Exception as e:
            print("Tidak ada koneksi Internet di question_answerOpenAI")
        return answer

    def jawabOllama(self):
        question = self.boxPertanyaan.toPlainText()
        print("Pertanyaan: ",question)
        url = "http://localhost:11434/api/generate"  # Replace with your actual Ollama API endpoint
        input_text = question
        payload = {
            "model": "abdul_mukti:latest",
            "prompt": input_text,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload)
            # print("Response text:", response.text)
            response_data = response.json()
            jawab = response_data.get('response', 'No response')
            print(jawab)
            # print(response_data)
            # output_label.config(text=response_data.get('response', 'No response'))
        except Exception as e:
            # output_label.config(text=f"Error: {e}")
            print("Error")
        
        self.boxJawaban.insertPlainText(jawab)
        return jawab


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
