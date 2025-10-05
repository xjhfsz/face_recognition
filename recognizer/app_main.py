from recognizer import FacialRecognizer


if __name__ == '__main__':
    try:
        recognizer = FacialRecognizer('faces/john1.jpg')
        recognizer.execute_recognizer()
    except Exception as e:
        print(f'Erro: {e}')
        