# Importar bibliotecas necesarias
import argparse
import os
import jiwer
import glob
import pandas as pd
from faster_whisper import WhisperModel
import time
import warnings

# Filtrar advertencias para mejorar la legibilidad del resultado
warnings.filterwarnings("ignore")

# Crear un DataFrame para almacenar los resultados
result = pd.DataFrame(columns=['Audio', 'WER'])


# Definir las normalizaciones que se aplicarán al texto
def text_transformation(text: str) -> str:
    """
    Aplica transformaciones al texto para normalizarlo.

    Args:
        text (str): Texto a transformar.

    Returns:
        str: Texto transformado.
    """
    text = jiwer.ToLowerCase()(text)
    text = jiwer.ExpandCommonEnglishContractions()(text)
    text = jiwer.RemovePunctuation()(text)
    text = jiwer.Strip()(text)
    text = jiwer.RemoveMultipleSpaces()(text)
    return text


# Función para obtener la transcripción correspondiente a un archivo de audio
def obtener_transcripcion(audio_path, text_files):
    """
    Obtiene la transcripción correspondiente a un archivo de audio.

    Args:
        audio_path (str): Ruta del archivo de audio.
        text_files (list): Lista de archivos de texto.

    Returns:
        str: Transcripción si se encuentra, None si no se encuentra.
    """
    nombre_audio = os.path.splitext(os.path.basename(audio_path))[0]
    for text_file in text_files:
        if nombre_audio in open(text_file).read():
            with open(text_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if nombre_audio in line:
                        transcripcion = line.split(' ', 1)[1].strip()
                        # Si tu archivo .txt tiene formato audio1: transcripcion1, modifica la línea anterior como sigue:
                        # transcripcion = line.split(':', 1)[1].strip()
                        return transcripcion
    return None


# Función principal que realiza el procesamiento de transcripciones
def main(args):
    """
    Función principal que realiza el procesamiento de transcripciones.

    Args:
        args: Argumentos proporcionados desde la línea de comandos.
    """
    global result
    print("Argumentos:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(" \n Cargando modelo... \n")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    # Obtener la lista de archivos de audio y archivos de texto en la carpeta especificada
    audio_files = glob.glob(os.path.join(args.audio_folder, '**', '*.' + args.extension_audio_file), recursive=True)
    text_files = glob.glob(os.path.join(args.audio_folder, '**', '*.txt'), recursive=True)

    # Iterar sobre los archivos de audio y obtener las transcripciones
    for audio_path in audio_files:
        transcripcion = obtener_transcripcion(audio_path, text_files)
        if transcripcion is not None:
            texto_de_referencia = text_transformation(transcripcion)
            segments, _ = model.transcribe(audio_path)
            segments = list(segments)
            texts = [segment.text for segment in segments]
            texto_inferencia = text_transformation(texts)
            texto_inferencia = ' '.join(texto_inferencia)
            wer_value = jiwer.wer(texto_de_referencia, texto_inferencia)
            print(f"Audio: {os.path.basename(audio_path)}, WER: {wer_value}")
            result = pd.concat([result, pd.DataFrame({'Audio': [os.path.basename(audio_path)], 'WER': [wer_value]})],
                               ignore_index=True)
            print("-------------------------------------------------------------------------------------------------")
            time.sleep(3)  # Esperar 3 segundos para evitar errores de memoria
        else:
            print(f"No se encontró transcripción para {os.path.basename(audio_path)}")

    # Guardar el DataFrame result en un archivo CSV
    result.to_csv('resultados.csv', index=False)
    print("Resultados guardados en 'resultados.csv'")


# Bloque principal para ejecutar el script desde la línea de comandos
if __name__ == "__main__":
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description="Calcular el Word Error Rate (WER) para un conjunto de archivos")
    parser.add_argument("--audio_folder", type=str, help="Ruta a la carpeta que contiene los archivos de audio")
    parser.add_argument("--model", type=str, help="Nombre del modelo a utilizar para el cálculo del WER (por ejemplo, 'tiny', 'base', 'base.en', 'medium)")
    parser.add_argument("--extension_audio_file", type=str, help="Extensión del archivo a utilizar para el cálculo del WER (por ejemplo, 'wav', 'mp3', 'flac')")
    args = parser.parse_args()

    # Llamar a la función principal con los argumentos proporcionados
    main(args)
