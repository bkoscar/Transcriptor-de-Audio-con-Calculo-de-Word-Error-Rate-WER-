# Transcriptor-de-Audio-con-Calculo-de-Word-Error-Rate-WER-

Este repositorio presenta un script en Python que emplea el modelo Whisper para transcribir archivos de audio y calcular el Word Error Rate (WER). Automáticamente compara las transcripciones generadas con las de referencia en archivos de texto asociados.

## Cómo Usar

### Organización de Carpetas:

Coloca los archivos de audio en una carpeta, por ejemplo, llamada `samples`, junto con el archivo .txt que contiene las transcripciones de referencia.

### Configuración del Entorno Virtual:

- **Clonar:**
  ```bash
  git clone https://github.com/bkoscar/Transcriptor-de-Audio-con-Calculo-de-Word-Error-Rate-WER-.git
  ```

Antes de ejecutar el script, activa tu entorno virtual:

- **Windows:**
  ```bash
  Enviroment\Scripts\activate
  ```
- **Linux:**
  ```bash
  source Enviroment/bin/activate
  ```
- **Como correr el script:**
  ```bash
  python Wer.py --audio_folder "Nombre_folder_audios" --model "models de whisper(tiny.en, base.en, medium.en, etc)" --extension_audio_file "extensión_de_tus_audios"
  ```
- **Ejemplo del script:**
  ```bash
  python Wer.py --audio_folder ./samples --model base.en --extension_audio_file flac
  ```
