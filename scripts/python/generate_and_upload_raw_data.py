import argparse
import json
import os
import random
import time
import boto3
import sagemaker

# --- Configuration for Data Generation ---
POC_CATEGORIES = [ # This list still defines which categories we process
    "Software Engineer", "Data Scientist", "Product Manager",
    "Sales Representative", "Marketing Specialist", "HR Manager",
    "UX Designer", "DevOps Engineer", "Business Analyst", "Accountant",
    "Customer Support Agent", "Operations Manager"
]

# SAMPLE_JD_TEMPLATES will now be loaded from a JSON file
# Default path for the templates file
DEFAULT_TEMPLATES_FILE = "./data/jd_templates.json"

JOB_DESC_COLUMN_NAME = "job_description_text"
CATEGORY_COLUMN_NAME = "category_label"

def load_jd_templates(templates_file_path):
    """Loads JD templates from a JSON file."""
    try:
        with open(templates_file_path, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        print(f"Successfully loaded JD templates from: {templates_file_path}")
        return templates
    except FileNotFoundError:
        print(f"Error: Templates file not found at {templates_file_path}")
        print("Please ensure 'jd_templates.json' exists or provide the correct path via --templates-file.")
        return {} # Return empty dict to allow script to potentially continue or fail gracefully later
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {templates_file_path}. Please check its format.")
        return {}

def generate_jd_for_category(category, sample_jd_templates, num_variations_per_template=3):
    jds = []
    if category not in sample_jd_templates or not sample_jd_templates[category]:
        # Generic fallback if no specific template or empty list for category
        base_desc = f"This is a job description for a {category}. Responsibilities include tasks related to {category} and team collaboration. Requires expertise in {category}-specific tools."
        for i in range(num_variations_per_template * 2):
            jds.append(f"{base_desc} (Ref: JD{random.randint(1000,9999)})")
        return jds

    category_templates = sample_jd_templates[category]
    for template in category_templates:
        for i in range(num_variations_per_template):
            variant_jd = template.replace(".", f" (Internal ID: {random.randint(100,999)}).")
            if i % 2 == 0 and "Experience with" in variant_jd:
                variant_jd = variant_jd.replace("Experience with", "Strong experience with")
            jds.append(variant_jd)
    return jds

def generate_raw_data(
    categories_list,
    num_jds_per_category_lang,
    languages,
    job_desc_col,
    category_col,
    aws_region,
    sample_jd_templates # Pass the loaded templates
):
    all_raw_data = []
    print(f"Generating raw data for categories: {categories_list}")
    print(f"Languages: {languages}")
    print(f"JDs per category per language: {num_jds_per_category_lang}")

    try:
        translate_client = boto3.client(service_name='translate', region_name=aws_region, use_ssl=True)
        print(f"Amazon Translate client initialized for region: {aws_region}")
    except Exception as e:
        print(f"Error initializing Amazon Translate client: {e}. Translations will be by tagging.")
        translate_client = None

    for category in categories_list:
        num_base_templates_for_cat = len(sample_jd_templates.get(category, [1])) # Use 1 if category missing to avoid div by zero
        variations_needed = (num_jds_per_category_lang // num_base_templates_for_cat) + 1 if num_base_templates_for_cat > 0 else num_jds_per_category_lang

        base_jds_for_category_en = generate_jd_for_category(category, sample_jd_templates, num_variations_per_template=variations_needed)
        
        if not base_jds_for_category_en:
            print(f"    WARNING: No base JDs generated for category '{category}'. Skipping.")
            continue

        random.shuffle(base_jds_for_category_en)

        for lang_code in languages:
            print(f"  Processing Category: '{category}', Language: '{lang_code}'")
            jd_count_for_lang_category = 0
            
            for jd_text_en in base_jds_for_category_en:
                if jd_count_for_lang_category >= num_jds_per_category_lang:
                    break

                final_jd_text_for_lang = jd_text_en
                if lang_code != "en" and translate_client:
                    try:
                        response = translate_client.translate_text(
                            Text=jd_text_en,
                            SourceLanguageCode='en',
                            TargetLanguageCode=lang_code
                        )
                        final_jd_text_for_lang = response['TranslatedText']
                    except Exception as e:
                        print(f"    WARNING: Error translating to '{lang_code}' for category '{category}': {e}")
                        final_jd_text_for_lang = f"[{lang_code.upper()}] {jd_text_en}"
                elif lang_code != "en" and not translate_client:
                    final_jd_text_for_lang = f"[{lang_code.upper()}] {jd_text_en}"

                all_raw_data.append({
                    job_desc_col: final_jd_text_for_lang,
                    category_col: category
                })
                jd_count_for_lang_category += 1
            
            print(f"    Generated {jd_count_for_lang_category} JDs for Category: '{category}', Language: '{lang_code}'")

    random.shuffle(all_raw_data)
    return all_raw_data

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic raw job description data (with translation) and upload to S3.")
    parser.add_argument(
        "--s3_prefix", type=str, default="raw_job_description_data/v1_translated",
        help="S3 prefix within the SageMaker default bucket to upload the data."
    )
    parser.add_argument(
        "--output_filename", type=str, default="raw_jds_translated.jsonl",
        help="Filename for the output JSONL data."
    )
    parser.add_argument(
        "--num_jds_per_category_language", type=int, default=20,
        help="Number of job descriptions to generate for each category for each language."
    )
    parser.add_argument(
        "--languages", type=str, default="en,es,fr",
        help="Comma-separated list of language codes (e.g., en,es,fr,de,pt)."
    )
    parser.add_argument(
        "--aws_region", type=str, default=boto3.Session().region_name or "us-east-1",
        help="AWS region for the Amazon Translate client."
    )
    parser.add_argument(
        "--templates-file", type=str, default=DEFAULT_TEMPLATES_FILE,
        help=f"Path to the JSON file containing JD templates (default: {DEFAULT_TEMPLATES_FILE})."
    )
    args = parser.parse_args()

    langs_list = [lang.strip() for lang in args.languages.split(',')]
    if not args.aws_region:
        parser.error("Could not determine AWS region. Please specify with --aws_region.")

    # Load templates from JSON file
    sample_jd_templates = load_jd_templates(args.templates_file)
    if not sample_jd_templates: # If loading failed and returned empty dict
        print("Exiting due to issues loading JD templates.")
        return

    # Generate data
    raw_data = generate_raw_data(
        POC_CATEGORIES,
        args.num_jds_per_category_language,
        langs_list,
        JOB_DESC_COLUMN_NAME,
        CATEGORY_COLUMN_NAME,
        args.aws_region,
        sample_jd_templates # Pass loaded templates here
    )
    print(f"Generated a total of {len(raw_data)} raw job description entries (with translations where applicable).")

    local_dir = "/tmp/raw_data_output"
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
        print(f"Successfully uploaded raw data to: {full_s3_path}")
        print(f"This S3 URI can be used as the 'RawDatasetIdentifier' parameter for the SageMaker pipeline.")
    except Exception as e:
        print(f"Error uploading to S3: {e}")
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

if __name__ == "__main__":
    main()