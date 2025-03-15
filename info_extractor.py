import json
import os
from pathlib import Path
import ollama

def extract_resume_info_with_qwen(resume_text):
    """
    Use Qwen2.5:3b model via Ollama to extract resume information
    with comprehensive tech stack extraction
    
    Args:
        resume_text (str): The text content of a resume
        
    Returns:
        dict: Dictionary containing extracted information
    """
    # Construct a prompt that instructs the model to comprehensively extract tech stack
    prompt = f"""
You are an expert resume parser. Extract the following information from this resume text:

1. Full Name: The complete name of the individual
2. Phone Number: Any contact phone number(s)
3. Email Address: The email address used for professional contact
4. Location: City, state, country or full address

5. Tech Stack: IMPORTANT - Comprehensively extract ALL technical skills, programming languages, 
   frameworks, libraries, tools, platforms, and technologies mentioned ANYWHERE in the resume.
   Include skills from ALL sections including skills sections, project descriptions, work experience, 
   education, certifications, etc. Be thorough and don't miss any technology mentions.
   Include:
   - Programming languages (Python, Java, JavaScript, etc.)
   - Frameworks (React, Django, Spring, etc.)
   - Databases (MySQL, MongoDB, PostgreSQL, etc.)
   - Cloud platforms (AWS, Azure, GCP, etc.)
   - Tools (Git, Docker, Kubernetes, etc.)
   - Any other technical skills or technologies

IMPORTANT: Your response must be ONLY a valid JSON object with these exact keys:
{{
  "full_name": "Extracted full name",
  "phone_number": "Extracted phone number",
  "email": "Extracted email",
  "location": "Extracted location",
  "tech_stack": ["Skill 1", "Skill 2", "Skill 3", ...]
}}

For tech_stack, include ALL technical skills found anywhere in the resume.
Do not include any explanation, markdown formatting, or commentary outside of the JSON object.

Resume text:
{resume_text}
"""
    
    try:
        # Call the Qwen2.5:3b model through Ollama
        response = ollama.chat(
            model='qwen2.5:3b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.0,  # Zero temperature for most deterministic output
                'num_predict': 1024  # Allow for sufficient response length
            }
        )
        
        # Extract the model's response
        model_response = response['message']['content']
        
        # Find and clean up the JSON content in the response
        json_start = model_response.find('{')
        json_end = model_response.rfind('}') + 1
        
        if json_start >= 0 and json_end > 0:
            json_content = model_response[json_start:json_end]
            
            # Parse the JSON
            extracted_info = json.loads(json_content)
            
            # Ensure we have all expected fields with default values
            result = {
                "full_name": extracted_info.get("full_name"),
                "phone_number": extracted_info.get("phone_number"),
                "email": extracted_info.get("email"),
                "location": extracted_info.get("location"),
                "tech_stack": extracted_info.get("tech_stack", [])
            }
            
            return result
            
    except Exception as e:
        print(f"Error during extraction: {e}")
    
    # Default empty result if extraction fails
    return {
        "full_name": None,
        "phone_number": None,
        "email": None,
        "location": None,
        "tech_stack": [],
        "error": "Extraction failed"
    }

def process_resume_file(file_path):
    """
    Process a single resume text file using Qwen2.5:3b
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        dict: Extracted information from the resume
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        extracted_info = extract_resume_info_with_qwen(text)
        return extracted_info
    
    except Exception as e:
        return {"error": str(e)}

def process_all_resumes(directory_path):
    """
    Process all text files in a directory using Qwen2.5:3b
    
    Args:
        directory_path (str): Path to directory containing resume text files
        
    Returns:
        list: List of dictionaries with extracted information
    """
    results = []
    
    for file_path in Path(directory_path).glob('*.txt'):
        print(f"Processing: {file_path.name}")
        extracted_info = process_resume_file(str(file_path))
        extracted_info['file_name'] = file_path.name
        results.append(extracted_info)
    
    return results

def save_results_to_json(results, output_path):
    """
    Save extracted information to a JSON file
    
    Args:
        results (list or dict): Extracted information
        output_path (str): Path to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

# Example usage
if __name__ == "__main__":
    
    # Process a single file
    resume_text_path = "C:/Users/harsh/Downloads/extracted_Aryan-Rajpurkar_resume.txt"
    if os.path.exists(resume_text_path):
        print(f"Extracting information from {resume_text_path}")
        info = process_resume_file(resume_text_path)
        print(json.dumps(info, indent=2))
        
        # Save to file
        save_results_to_json(info, "extracted_resume_info.json")
    
    