import cv2 # Biblioteca de Visão Computacional
import os # Biblioteca que integra comandos ao sistema operacional
import numpy as np 
import pyttsx3 # biblioteca que faz o assistente falar
import serial # Biblioteca que comunica com o arduino via porta serial
import threading # Blbioteca que permite o codigo fazer mais de uma ação ao mesmo tempo
import time # Biblioteca para gerenciar o tempo

# Inicializa a comunicação com o Arduino
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)  # Ajuste o nome da porta conforme necessário

# Inicializa o mecanismo de fala
engine = pyttsx3.init()

# Função para falar as mensagens
def speak(message):
    engine.say(message)
    engine.runAndWait()

# Carrega Haar cascade para detecção de faces (Rede neural já treinada)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Carrega dataset de rostos conhecidos
def load_known_faces(dataset_path):
    known_face_encodings = []
    known_face_names = []

    for image_name in os.listdir(dataset_path):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):  # Formatos de imagens aceitas
            # Pré-processamento das imagens
            image_path = os.path.join(dataset_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face_encodings = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

            if len(face_encodings) > 0:
                # Usa o primeiro rosto encontrado na imagem
                x, y, w, h = face_encodings[0]
                face_roi = image[y:y+h, x:x+w]
                # Ajuste do tamanho para o mesmo da variável face_roi_resized (100x100)
                face_roi_resized = cv2.resize(face_roi, (100, 100))
                known_face_encodings.append(face_roi_resized)
                known_face_names.append(os.path.splitext(image_name)[0])

    return known_face_encodings, known_face_names

# Função que compara duas imagens por histograma
def compare_histograms(img1, img2):
    hist_img1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist_img2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist_img1 = cv2.normalize(hist_img1, hist_img1).flatten()
    hist_img2 = cv2.normalize(hist_img2, hist_img2).flatten()

    score = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    return score

# Função para capturar quadros em uma thread separada
def capture_frames(cam):
    global frame
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Falhou em capturar imagem!")
            break

# Função principal para processamento do vídeo e interface do usuário
def virtual_assistant():
    global frame
    cam = cv2.VideoCapture(0) # Seleciona a webcam, ou outra camera

    # Ajusta a resolução do frame para 320x240 para melhorar a performance e abrir mais rápido
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Pre-abre a câmera para garantir que ela esteja pronta antes de qualquer reconhecimento
    time.sleep(1)

    # Local do dataset no computador
    dataset_path = r'C:\Users\Robson C. Augusto\Downloads\Pyhton_games\AI Vision\known_faces'  # Caminho do dataset no seu computador
    known_face_encodings, known_face_names = load_known_faces(dataset_path)

    # Inicia o thread de captura de frames
    capture_thread = threading.Thread(target=capture_frames, args=(cam,))
    capture_thread.start()

    # Aguarda até que a captura de frames tenha começado
    while frame is None:
        time.sleep(0.1)

    while True:
        if frame is None:
            continue  # Espera até que a captura de frames tenha iniciado

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

        for (x, y, w, h) in face_locations:
            face_roi = gray_frame[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (100, 100))

            # Comparando faces com método de histograma
            name = "Desconhecido"
            highest_score = -1
            for known_face, known_name in zip(known_face_encodings, known_face_names):
                score = compare_histograms(known_face, face_roi_resized)
                if score > highest_score and score > 0.6:  # Ajuste do threshold
                    highest_score = score
                    name = known_name

            if name != "Desconhecido":
                speak(f"Bem vindo, {name}, Acesso permitido!")
                arduino.write(b"Acesso_permitido\n")  # Envia comando para acender o LED
            else:
                speak("Pessoa desconhecida, acesso negado!")
                arduino.write(b"Acesso_negado\n")  # Envia comando para desligar o LED

            # Desenhando um retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Desenha retangulo
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2) # Escolhe fonte e cor do texto

        # Redimensiona o frame mostrado na janela para uma menor resolução para abrir mais rápido
        small_frame = cv2.resize(frame, (280, 200))
        cv2.imshow("Assistente Virtual", small_frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:  # Pressiona a tecla ESC para sair
            break

        time.sleep(5)  # Aguarda 5 segundos antes de tentar detectar novamente

    cam.release()
    cv2.destroyAllWindows()
    arduino.close()

# Executa o assistente virtual
if __name__ == "__main__":
    frame = None  # Variável global para armazenar o frame capturado
    virtual_assistant()
