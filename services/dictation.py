import openai
from pydub import AudioSegment
import requests

# Set your OpenAI API key
openai.api_key = 'your-openai-api-key'

# Function to process the dictation and structure the notes
def processDictation(transcription):
    sections = organizeIntoSections(transcription)
    summarizedSections = summarizeKeyPoints(sections)
    structuredTemplate = formatIntoTemplate(summarizedSections)
    finalNotes = reviewAndHighlight(structuredTemplate)
    return finalNotes

# Function to organize the transcription into predefined sections using flexible prompts
def organizeIntoSections(text):
    sections = {
        "Preparation and Introduction": extractSection(text, "describe the preparation and introduction of the consultation"),
        "History Taking": extractSection(text, "extract the history taking part of the consultation"),
        "Physical Examination": extractSection(text, "describe the physical examination part of the consultation"),
        "Diagnostic Testing": extractSection(text, "extract any diagnostic tests mentioned"),
        "Diagnosis": extractSection(text, "describe the diagnosis given during the consultation"),
        "Treatment Plan": extractSection(text, "extract the treatment plan discussed"),
        "Client Education and Instructions": extractSection(text, "describe the client education and instructions provided"),
        "Conclusion": extractSection(text, "summarize the conclusion of the consultation")
    }
    return sections

# Function to extract a specific section from the text using flexible contextual prompts
def extractSection(text, sectionDescription):
    prompt = f"""Identify and extract the part of the following veterinary consultation transcript that {sectionDescription}:
    Transcript: {text}
    Extracted Section:"""

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to summarize the given text using OpenAI API
def callOpenAISummarizeAPI(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Summarize the following content:\n\n{text}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to highlight critical information in the text using OpenAI API
def callOpenAIHighlightAPI(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Highlight important information in the following content:\n\n{text}\n\nHighlight the important parts within the original content without removing any information.",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to summarize key points in each section
def summarizeKeyPoints(sections):
    for key, value in sections.items():
        sections[key] = callOpenAISummarizeAPI(value)
    return sections

# Function to format the summarized sections into a structured template
def formatIntoTemplate(sections):
    template = f"""
    **Preparation and Introduction:**
    {sections['Preparation and Introduction']}
    
    **History Taking:**
    {sections['History Taking']}
    
    **Physical Examination:**
    {sections['Physical Examination']}
    
    **Diagnostic Testing:**
    {sections['Diagnostic Testing']}
    
    **Diagnosis:**
    {sections['Diagnosis']}
    
    **Treatment Plan:**
    {sections['Treatment Plan']}
    
    **Client Education and Instructions:**
    {sections['Client Education and Instructions']}
    
    **Conclusion:**
    {sections['Conclusion']}
    """
    return template

# Function to review and highlight important information using OpenAI API
def reviewAndHighlight(text):
    highlightedTemplate = callOpenAIHighlightAPI(text)
    return highlightedTemplate

# Function to transcribe audio file to text
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(file_path)
    audio.export("temp.wav", format="wav")
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

# Function to post consultation to Provet Cloud
def post_consultation_to_provet(finalNotes, patient_url, department_url, vet_url):
    url = "https://provetcloud.com/<provet_id>/api/0.1/consultation/"
    headers = {
        "Accept": "application/json",
        "Authorization": "Token your_api_token",
        "Content-Type": "application/json"
    }
    data = {
        "patient": patient_url,
        "date": "2024-03-21",
        "department": department_url,
        "status": 1,
        "type": 0,
        "supervising_veterinarian": vet_url,
        "consultation_items": [],
        "notes": finalNotes
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Example usage
audio_file_path = "path_to_audio_file.mp3"
transcription = transcribe_audio(audio_file_path)
finalNotes = processDictation(transcription)

patient_url = "https://provetcloud.com/<provet_id>/api/0.1/patient/2130/"
department_url = "https://provetcloud.com/<provet_id>/api/0.1/department/1/"
vet_url = "https://provetcloud.com/<provet_id>/api/0.1/user/1/"

response = post_consultation_to_provet(finalNotes, patient_url, department_url, vet_url)
print(response)
