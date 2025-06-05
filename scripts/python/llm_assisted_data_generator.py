import argparse
import json
import os
import random
import time
import boto3
import sagemaker # For default bucket

# --- Configuration ---
JOB_DESC_COLUMN_NAME = "job_description_text"
CATEGORY_COLUMN_NAME = "category_label"

# Categories for which to generate data
TARGET_CATEGORIES = [
    "Software Engineer", "Data Scientist", "Product Manager",
    "Sales Representative", "Marketing Specialist", "HR Manager",
    "UX Designer", "DevOps Engineer", "Business Analyst", "Accountant",
    "Customer Support Agent", "Operations Manager"
]

# Languages: {name_for_prompt: language_code_for_translate_or_llm_understanding}
TARGET_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr"
    # Add more as needed e.g. "German": "de"
}

# Bedrock Model ID for Llama 3 Instruct
# Choose 8B for faster/cheaper generation, 70B for potentially higher quality
# Ensure you have model access granted for the chosen model in Bedrock for your AWS region.
LLAMA3_MODEL_ID = 'meta.llama3-8b-instruct-v1:0'
# LLAMA3_MODEL_ID = 'meta.llama3-70b-instruct-v1:0'

def generate_prompt_for_llm(category, language_name, num_jds_to_request):
    """
    Crafts a detailed prompt for Llama 3 to generate job descriptions.
    """
    prompt = f"""You are an expert Human Resources content writer specializing in crafting compelling job descriptions.
Your task is to generate {num_jds_to_request} distinct and realistic job descriptions for the role of "{category}" in {language_name}.

For each job description, please ensure the following:
1.  A clear and concise job title that accurately reflects the "{category}" role.
2.  A brief, engaging introduction to a fictional company. Vary the company type (e.g., tech startup, established corporation, non-profit, e-commerce, financial services, healthcare tech).
3.  A section detailing 3-5 key responsibilities.
4.  A section listing 3-5 essential qualifications or skills. This could include specific technologies, software, years of experience, educational background, or soft skills.
5.  Optionally, a brief mention of company culture or unique benefits.
6.  Vary the overall length of each job description (some shorter, around 100-150 words; some medium, around 150-250 words; and some longer, around 250-350 words).
7.  Vary the tone and style (e.g., some formal and corporate, others more casual and startup-oriented, some highly technical, some more focused on impact).
8.  Ensure each of the {num_jds_to_request} job descriptions you generate for this request are significantly different from one another in terms of company, specific responsibilities, and phrasing. Avoid simple rephrasing of the same core points.

Please format your entire response as a single block of text.
Clearly separate each generated job description using the following delimiters:
--- JD START ---
(Job Description Content for JD 1)
--- JD END ---
--- JD START ---
(Job Description Content for JD 2)
--- JD END ---
...and so on for all {num_jds_to_request} job descriptions.
Do not include any other text, preamble, or explanation outside of these delimited job descriptions.
"""
    return f"<s>[INST] {prompt} [/INST]" # Llama 3 Instruct format

def parse_llm_output(llm_text_output):
    """
    Parses the LLM's raw text output to extract individual job descriptions.
    Assumes JDs are separated by '--- JD START ---' and '--- JD END ---'.
    """
    jds = []
    if not llm_text_output:
        return jds
        
    parts = llm_text_output.split("--- JD START ---")
    for part in parts:
        if "--- JD END ---" in part:
            jd_content = part.split("--- JD END ---")[0].strip()
            if jd_content: # Ensure it's not an empty string between delimiters
                jds.append(jd_content)
    return jds

def generate_llm_assisted_data(
    categories_list,
    languages_dict,
    num_jds_per_category_lang,
    job_desc_col,
    category_col,
    aws_region,
    bedrock_model_id,
    jds_per_llm_call=5 # How many JDs to ask the LLM for in a single API call
):
    all_raw_data = []
    print(f"Generating LLM-assisted raw data for categories: {categories_list}")
    print(f"Languages: {list(languages_dict.keys())}")
    print(f"Target JDs per category per language: {num_jds_per_category_lang}")
    print(f"Using Bedrock model: {bedrock_model_id} in region {aws_region}")

    bedrock_runtime = None
    try:
        bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=aws_region)
        print(f"Amazon Bedrock Runtime client initialized for region: {aws_region}")
    except Exception as e:
        print(f"FATAL: Error initializing Amazon Bedrock Runtime client: {e}")
        print("LLM-assisted data generation cannot proceed.")
        return []

    for category in categories_list:
        for lang_name, lang_code in languages_dict.items():
            print(f"  Processing Category: '{category}', Language: '{lang_name}'")
            generated_for_this_combo = 0
            attempts = 0
            max_attempts_per_combo = (num_jds_per_category_lang // jds_per_llm_call) + 3 # Allow some retries

            while generated_for_this_combo < num_jds_per_category_lang and attempts < max_attempts_per_combo:
                attempts += 1
                jds_to_request_this_call = min(jds_per_llm_call, num_jds_per_category_lang - generated_for_this_combo)
                if jds_to_request_this_call <= 0:
                    break

                llama_prompt = generate_prompt_for_llm(category, lang_name, jds_to_request_this_call)
                body = json.dumps({
                    "prompt": llama_prompt,
                    "max_gen_len": 512 * jds_to_request_this_call, # Generous length per JD
                    "temperature": 0.75,
                    "top_p": 0.9,
                })
                
                print(f"    Attempt {attempts}: Requesting {jds_to_request_this_call} JDs from LLM...")
                try:
                    response = bedrock_runtime.invoke_model(
                        body=body, modelId=bedrock_model_id, 
                        contentType='application/json', accept='application/json'
                    )
                    response_body = json.loads(response.get('body').read())
                    llm_output = response_body.get('generation')
                    
                    parsed_jds = parse_llm_output(llm_output)
                    print(f"    LLM call returned {len(parsed_jds)} processable JDs.")

                    for jd_text in parsed_jds:
                        if generated_for_this_combo < num_jds_per_category_lang:
                            all_raw_data.append({
                                job_desc_col: jd_text,
                                category_col: category
                                # You could add a 'language' field too if needed for later analysis
                                # "language": lang_code 
                            })
                            generated_for_this_combo += 1
                        else:
                            break # Reached target for this combo
                    
                    # Small delay to be kind to the API
                    time.sleep(1) # Adjust as needed, especially for larger models/more calls

                except Exception as e:
                    print(f"    ERROR during LLM call for {category}/{lang_name} (Attempt {attempts}): {e}")
                    time.sleep(5) # Longer delay on error
            
            print(f"    Generated a total of {generated_for_this_combo} JDs for Category: '{category}', Language: '{lang_name}'")

    random.shuffle(all_raw_data)
    return all_raw_data

def main():
    parser = argparse.ArgumentParser(description="Generate LLM-assisted synthetic raw job description data and upload to S3.")
    parser.add_argument("--s3_prefix", type=str, default="raw_job_description_data/llm_v1", help="S3 prefix.")
    parser.add_argument("--output_filename", type=str, default="llm_raw_jds_v1.jsonl", help="Output filename.")
    parser.add_argument("--num_jds_per_category_language", type=int, default=50, help="Target JDs per category per language.")
    parser.add_argument("--jds_per_llm_call", type=int, default=5, help="Number of JDs to request in a single LLM API call.")
    parser.add_argument("--languages", type=str, default="English:en,Spanish:es,French:fr", help="Comma-separated lang_name:lang_code pairs (e.g., English:en,German:de).")
    parser.add_argument("--aws_region", type=str, default=None, help="AWS region for Bedrock. Tries to auto-detect.")
    parser.add_argument("--bedrock_model_id", type=str, default=LLAMA3_MODEL_ID, help="Bedrock Model ID for Llama 3.")
    
    args = parser.parse_args()

    effective_aws_region = args.aws_region or boto3.Session().region_name or "us-east-1" # Default if not found
    if not effective_aws_region:
        parser.error("Could not determine AWS region. Please specify with --aws_region or configure AWS CLI default.")

    languages_dict = {}
    try:
        for pair in args.languages.split(','):
            name, code = pair.split(':')
            languages_dict[name.strip()] = code.strip()
    except ValueError:
        parser.error("Invalid format for --languages. Expected 'Name1:Code1,Name2:Code2'.")

    raw_data = generate_llm_assisted_data(
        TARGET_CATEGORIES,
        languages_dict,
        args.num_jds_per_category_language,
        JOB_DESC_COLUMN_NAME,
        CATEGORY_COLUMN_NAME,
        effective_aws_region,
        args.bedrock_model_id,
        args.jds_per_llm_call
    )

    if not raw_data:
        print("No data generated. Exiting.")
        return
        
    print(f"Generated a total of {len(raw_data)} LLM-assisted raw job description entries.")

    local_dir = "/tmp/llm_raw_data_output"
    os.makedirs(local_dir, exist_ok=True)
    local_file_path = os.path.join(local_dir, args.output_filename)

    with open(local_file_path, 'w', encoding='utf-8') as f:
        for item in raw_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Raw data saved locally to: {local_file_path}")

    try:
        sagemaker_session = sagemaker.Session()
        default_s3_bucket = sagemaker_session.default_bucket()
        s3_client = boto3.client("s3")
        s3_key = f"{args.s3_prefix.strip('/')}/{args.output_filename}"
        s3_client.upload_file(local_file_path, default_s3_bucket, s3_key)
        full_s3_path = f"s3://{default_s3_bucket}/{s3_key}"
        print(f"Successfully uploaded LLM-assisted raw data to: {full_s3_path}")
        print(f"This S3 URI can be used as the 'RawDatasetIdentifier' for the SageMaker pipeline.")
    except Exception as e:
        print(f"Error uploading to S3: {e}")
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

if __name__ == "__main__":
    main()