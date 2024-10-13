# Detectando faces
# Detectando pessoas de um dataset conhecido

import cv2
import os
import numpy as np

# Carrega Haar cascade para detecção de faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Carrega dataset de rostos conhecidos
def load_known_faces(dataset_path):
    known_face_encodings = []
    known_face_names = []

    for image_name in os.listdir(dataset_path):
        if image_name.endswith(('.jpg', '.jpeg', '.png')): # Formatos de imagens aceitas

            # Pré processamento das imagens
            image_path = os.path.join(dataset_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face_encodings = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(face_encodings) > 0:
                # Usa o primeiro rosto encontrado na imagem
                x, y, w, h = face_encodings[0]
                face_roi = image[y:y+h, x:x+w]
                
                # Ajuste do tamanho para o mesmo da variavel face_roi_resized (100X100)
                face_roi_resized = cv2.resize(face_roi, (100, 100))
                known_face_encodings.append(face_roi_resized)
                known_face_names.append(os.path.splitext(image_name)[0])

    return known_face_encodings, known_face_names

# Função que compara duas iamgens por histograma
def compare_histograms(img1, img2):
    hist_img1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist_img2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist_img1 = cv2.normalize(hist_img1, hist_img1).flatten()
    hist_img2 = cv2.normalize(hist_img2, hist_img2).flatten()

    score = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    return score

# Função principal para processamento do video e interface do usuario
def virtual_assistant():
    cam = cv2.VideoCapture(0)

    # Local do dataset no computador
    dataset_path = r'C:\Users\Robson C. Augusto\Downloads\Pyhton_games\AI Vision\known_faces'  # Caminho de dataset no seu computador
    known_face_encodings, known_face_names = load_known_faces(dataset_path)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Falhou em capturar imagem!")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in face_locations:
            face_roi = gray_frame[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (500, 500)) # muda a sensibilidade de captura da area da camera 2D

            # Comparando faces com metodo de histograma

            name = "Desconhecido"
            highest_score = -1
            for known_face, known_name in zip(known_face_encodings, known_face_names):
                score = compare_histograms(known_face, face_roi_resized)
                if score > highest_score and score > 0.6:  # 0.6 threshold pode ser ajustado.
                    highest_score = score
                    name = known_name

            print(f"Olá, {name}!" if name != "Desconhecido" else "Pessoa desconhecida detectada.")

            # Desenhando um retangulo ao redor da face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Assistente Virtual", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:  # pressiona a tecla ESC para sair 
            break

    cam.release()
    cv2.destroyAllWindows()

# Executa o assistente virtual
if __name__ == "__main__":
    virtual_assistant()
