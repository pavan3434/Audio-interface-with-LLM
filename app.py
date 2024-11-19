import os
import base64
import json
import time
from datetime import datetime
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, GenerationConfig
from flask import Flask, request, render_template, jsonify, send_file, send_from_directory
from pathlib import Path
import jinja2

app = Flask(__name__)

# Ensure output directory exists
UPLOAD_FOLDER = Path(app.root_path) / 'output'
UPLOAD_FOLDER.mkdir(exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.errorhandler(404)
def page_not_found(e):
    try:
        return render_template('404.html'), 404
    except jinja2.exceptions.TemplateNotFound:
        return "404 Error: Page Not Found", 404

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    audio_file = request.files['audioFile']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"analysis_{timestamp}"
    
    file_extension = os.path.splitext(audio_file.filename)[1].lower()
    is_recorded = audio_file.filename == 'recording.mp3'
    
    temp_path = os.path.join(app.root_path, f'temp_{timestamp}{file_extension}')
    audio_file.save(temp_path)
    
    try:
        mime_type = "audio/mpeg"
        if file_extension == '.wav':
            mime_type = "audio/wav"
            
        final_result = generate_transcript_and_analysis(temp_path, mime_type)
        
        output_path = UPLOAD_FOLDER / f"{base_filename}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_result)
        
        return jsonify({
            'result': final_result,
            'downloadPath': f'/download/{base_filename}.txt'
        })
    
    except Exception as e:
        print(f"Error details: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            UPLOAD_FOLDER / filename,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

def generate_transcript_and_analysis(audio_path, mime_type):
    attempt = 0
    max_retries = 5
    backoff_factor = 2
    while attempt < max_retries:
        try:
            vertexai.init(project="conv-436114", location="us-central1")
            model = GenerativeModel("gemini-1.5-pro")
            
            chunk_size = 1024 * 1024
            audio_content = b""
            
            with open(audio_path, "rb") as audio_file:
                while chunk := audio_file.read(chunk_size):
                    audio_content += chunk
                    
            audio_base64 = base64.b64encode(audio_content).decode("utf-8")
            
            audio_part = Part.from_data(
                data=audio_base64,
                mime_type=mime_type
            )
            
            prompt = """Please provide a detailed transcription of this audio recording, followed by a comprehensive analysis. Include:

            1. Transcription:
            - Exact words spoken
            - Natural pauses with "..." notation
            - Significant background sounds in [brackets]
            - Unclear segments marked with [unclear]
            - Speaker changes indicated with "Speaker 1:", "Speaker 2:", etc.

            2. Analysis:
            - Summary of main topics and key points
            - Context and purpose of the recording
            - Sentiment analysis (overall tone, emotion, speaker attitude)
            - Key insights, important quotes, and action items
            - Speech patterns and delivery characteristics
            - Any relevant background context or additional observations

            Provide the transcription first, followed by the analysis."""
            
            config = GenerationConfig(
                temperature=0.4,
                top_p=0.8,
                top_k=40,
                max_output_tokens=8192,
            )
            
            response = model.generate_content(
                [prompt, audio_part],
                generation_config=config,
                safety_settings=safety_settings
            )
            
            return response.text if hasattr(response, 'text') else "Analysis completed but no text was generated."
        
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                raise Exception(f"Failed to analyze audio after {max_retries} attempts: {str(e)}")
            wait_time = backoff_factor ** attempt
            print(f"Quota exceeded, retrying in {wait_time} seconds...")
            time.sleep(wait_time)

# Safety settings
safety_settings = [
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, 
                  threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, 
                  threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, 
                  threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, 
                  threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
]

if __name__ == '__main__':
    app.run(debug=True)
