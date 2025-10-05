import cv2
import face_recognition
import numpy as np


class FacialRecognizer:
    def __init__(self, reference_image_path):
        
        # load reference image
        self.reference_image = cv2.imread(reference_image_path)
        if self.reference_image is None:
            raise Exception(f'Imagem de referência {reference_image_path} não encontrada!')

        # Load Haar Cascade FIRST
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Check if cascade was loaded correctly
        if self.face_cascade.empty():
            raise Exception('Não foi possível carregar o classificador Haar Cascade!')

        # detect and extract reference face
        self.reference_face = self._extract_face(self.reference_image)
        if self.reference_face is None:
            raise Exception(f'Nenhum rosto encontrado na imagem de referência {reference_image_path}!')

        # precompute encoding of reference
        self.encoding_reference = self._extract_encoding(self.reference_face)

        # general settings
        self.similarity_threshold = 0.6

        # History to smooth results
        self.similarity_history = []


    def _extract_face(self, image):
        """Extract the biggest face in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

        if len(faces) > 0:
            # catch the biggest face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # extract the face
            face = image[y: y + h, x: x + w]
            
            return cv2.resize(face, (200, 200))  # standardize size

        return None
    

    def _extract_encoding(self, face):
        """Extract facial encoding using face_recognition"""

        # Convert BGR to RGB (face_recognition uses RGB)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(face_rgb)

        return encodings[0] if len(encodings) > 0 else None
    

    def compare_with_reference(self, face_cam):
        """Compare camera face with reference"""

        if self.encoding_reference is None:
            return 0, False
        
        # Extract enconding from face cam
        encoding_cam = self._extract_encoding(face_cam)
        if encoding_cam is None:
            return 0, False
        

        # Calculate similarity
        distance = face_recognition.face_distance(
            [self.encoding_reference], encoding_cam,
        )

        similarity = 1 - distance[0]

        # Smooth with moving average
        self.similarity_history.append(similarity)
        if len(self.similarity_history) > 5:
            self.similarity_history.pop(0)

        smoothed_similarity = np.mean(self.similarity_history)
        same_face = smoothed_similarity >= self.similarity_threshold

        return smoothed_similarity, same_face


    def execute_recognizer(self):
        """Main recognition loop""" 

        webcam = cv2.VideoCapture(0)  # 0 = default camera

        # Check if the webcam is open
        if not webcam.isOpened():
            raise Exception('Não foi possível acessar a webcam!')

        # setup windows
        cv2.namedWindow('Reconhecimento Facil', cv2.WINDOW_AUTOSIZE)

        print('Pressione "Q" para sair!')

        while True:
            ret, frame = webcam.read()

            if not ret:
                print('Erro ao capturar frame da webcam.')
                break

            # Mirror the frame (more natural)
            frame = cv2.flip(frame, 1)

            # Detect faces in the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # extract face
                face_cam = frame[y: y + h, x: x + w]
                face_cam = cv2.resize(face_cam, (200, 200))

                try:
                    # Compare with reference
                    similarity, same_face = self.compare_with_reference(face_cam)

                    # Set color and text based on the result
                    color = (0, 255, 0) if same_face else (0, 0, 255)
                    status = 'John xD' if same_face else 'DESCONHECIDO'
                    text = f'{status} ({similarity:.2f})'

                    # Draw rectangle and text
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    )

                except Exception as e:
                    print(f'Erro: {e}')
                    cv2.putText(
                        frame, 'Erro na comparação', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )

            # Show threshold information
            cv2.putText(
                frame, f'Limiar: {self.similarity_threshold:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.putText(
                frame, f'Rostos reconhecidos: {len(faces)}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            
            # show frame
            cv2.imshow('Reconhecimento Facil', frame)

            # controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()


