import argparse
import json
import os
import random
import time
import boto3
import sagemaker
import time
from botocore.exceptions import ClientError

# --- Configuration for Data Generation ---
# This list still defines which categories we process.
# It should match the top-level keys in your jd_templates.json
POC_CATEGORIES = [
    "Software Engineer", "Data Scientist", "Product Manager",
    "Sales Representative", "Marketing Specialist", "HR Manager",
    "UX Designer", "DevOps Engineer", "Business Analyst", "Accountant",
    "Customer Support Agent", "Operations Manager"
]

DEFAULT_TEMPLATES_FILE = "./data/jd_templates.json" # Path to your templates

JOB_DESC_COLUMN_NAME = "job_description_text"
CATEGORY_COLUMN_NAME = "category_label"

# --- Data for Enhanced Variation Logic ---
SKILL_KEYWORDS = {
    "Software Engineer": ["Python", "Java", "C++", "JavaScript", "React", "Angular", "Node.js", "Spring Boot", "Django", "Flask", "microservices", "REST APIs", "GraphQL", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "SQL", "NoSQL", "Git", "Agile", "Scrum", "CI/CD pipelines"],
    "Data Scientist": ["Python", "R", "SQL", "Spark", "Hadoop", "machine learning", "deep learning", "NLP", "computer vision", "TensorFlow", "PyTorch", "scikit-learn", "statistics", "data mining", "A/B testing", "Tableau", "PowerBI", "causal inference"],
    "Product Manager": ["roadmapping", "user stories", "agile methodologies", "market analysis", "product strategy", "A/B testing", "JIRA", "Confluence", "stakeholder management", "UX principles"],
    "Sales Representative": ["B2B sales", "B2C sales", "lead generation", "CRM software (Salesforce, HubSpot)", "negotiation", "closing deals", "cold calling", "account management", "sales quotas"],
    "Marketing Specialist": ["digital marketing", "SEO/SEM", "content creation", "social media marketing", "email marketing", "Google Analytics", "PPC campaigns", "brand management", "market research"],
    "HR Manager": ["recruitment", "employee relations", "HRIS systems", "performance management", "compensation and benefits", "labor law", "onboarding", "talent development"],
    "UX Designer": ["Figma", "Sketch", "Adobe XD", "user research", "wireframing", "prototyping", "usability testing", "user personas", "interaction design", "information architecture"],
    "DevOps Engineer": ["CI/CD", "Docker", "Kubernetes", "Terraform", "Ansible", "AWS", "Azure", "GCP", "Linux administration", "scripting (Bash, Python)", "monitoring tools (Prometheus, Grafana)"],
    "Business Analyst": ["requirements gathering", "process mapping", "data analysis", "SQL", "Excel", "UML", "stakeholder communication", "problem-solving", "Agile/Scrum"],
    "Accountant": ["GAAP", "IFRS", "financial reporting", "bookkeeping", "QuickBooks", "SAP", "Excel", "auditing", "tax preparation", "budgeting"],
    "Customer Support Agent": ["Zendesk", "Salesforce Service Cloud", "communication skills", "problem-solving", "empathy", "technical troubleshooting", "multitasking", "de-escalation"],
    "Operations Manager": ["process improvement", "logistics", "supply chain management", "Six Sigma", "Lean principles", "project management", "budget management", "team leadership", "inventory control"]
}

RESPONSIBILITY_PHRASES = {
    "Software Engineer": ["design and develop robust", "build and maintain scalable", "collaborate with cross-functional teams on", "test and deploy high-quality", "optimize and refactor existing"],
    "Data Scientist": ["analyze complex datasets for", "build and validate predictive models for", "develop data-driven insights for", "communicate findings to stakeholders regarding", "implement machine learning algorithms for"],
    "Product Manager": ["define the product vision and strategy for", "conduct user and market research for", "lead the go-to-market strategy for", "work closely with engineering and design teams on", "manage the product backlog and prioritize features for"],
    # ... Add more for other categories or keep generic if not specific ...
}

COMPANY_TYPES = ["a fast-growing startup", "a leading enterprise company", "a dynamic scale-up", "an innovative tech firm", "a well-established organization", "a mission-driven non-profit", "a cutting-edge research lab"]
SENIORITY_HINTS = ["an entry-level", "a junior", "a mid-level", "a senior", "a lead", "a staff", "a principal", "an experienced"]
ACTION_VERBS = ["Seeking", "Hiring", "Looking for", "We need", "Opportunity for"]

def load_jd_templates(templates_file_path):
    try:
        with open(templates_file_path, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        print(f"Successfully loaded JD templates from: {templates_file_path}")
        return templates
    except FileNotFoundError:
        print(f"Error: Templates file not found at {templates_file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {templates_file_path}. Please check its format.")
        return {}

def generate_jd_for_category(category, sample_jd_templates, num_variations_per_template=3):
    jds = []
    category_templates = sample_jd_templates.get(category)

    if not category_templates:
        base_desc = f"Standard job description for a {category}. Key responsibilities involve core {category} tasks and team collaboration. Expertise in {category}-specific tools and methodologies is required."
        for i in range(num_variations_per_template * (len(sample_jd_templates.get(random.choice(list(sample_jd_templates.keys())), [1,2,3])) if sample_jd_templates else 2) ): # Generate a few generic ones
            jds.append(f"{base_desc} (Ref: GenericJD{i}-{random.randint(1000,9999)})")
        return jds

    for template in category_templates:
        for i in range(num_variations_per_template):
            variant_jd = template
            original_sentences = variant_jd.split(". ") # Simple sentence split

            # Variation 1: Add a random skill
            if category in SKILL_KEYWORDS and random.random() < 0.6:
                skill = random.choice(SKILL_KEYWORDS[category])
                # Try to insert it naturally or append
                if "Experience with" in variant_jd and skill not in variant_jd:
                    variant_jd = variant_jd.replace("Experience with", f"Experience with {skill} and", 1)
                elif "skills" in variant_jd.lower() and skill not in variant_jd:
                     variant_jd = variant_jd.replace("skills", f"skills including {skill}", 1)
                elif skill not in variant_jd :
                    variant_jd += f" Proficiency in {skill} is beneficial."

            # Variation 2: Prepend with context (company type or seniority)
            if random.random() < 0.4:
                prefix_type = random.choice(["company", "seniority", "action_verb"])
                if prefix_type == "company" and len(original_sentences) > 0:
                    company_type = random.choice(COMPANY_TYPES)
                    original_sentences[0] = f"Join {company_type}. {random.choice(ACTION_VERBS)} {original_sentences[0]}"
                elif prefix_type == "seniority" and len(original_sentences) > 0:
                    seniority = random.choice(SENIORITY_HINTS)
                    original_sentences[0] = f"{random.choice(ACTION_VERBS)} {seniority} {category}. {original_sentences[0]}"
                elif prefix_type == "action_verb" and len(original_sentences) > 0:
                     original_sentences[0] = f"{random.choice(ACTION_VERBS)} {original_sentences[0]}"
                variant_jd = ". ".join(original_sentences)


            # Variation 3: Shuffle some sentences (if more than 2)
            if len(original_sentences) > 2 and random.random() < 0.3:
                # Keep first and last, shuffle middle ones
                first_sentence = original_sentences[0]
                last_sentence = original_sentences[-1]
                middle_sentences = original_sentences[1:-1]
                random.shuffle(middle_sentences)
                variant_jd = first_sentence + ". " + ". ".join(middle_sentences) + ". " + last_sentence
            
            # Ensure a unique-ish element
            variant_jd = variant_jd.strip() + f" (Job ID: {random.randint(10000,99999)})"
            jds.append(variant_jd)
    return jds

def generate_raw_data(
    categories_list,
    num_jds_per_category_lang,
    languages,
    job_desc_col,
    category_col,
    aws_region,
    sample_jd_templates
):
    all_raw_data = []
    print(f"Generating raw data for categories: {categories_list}")
    print(f"Languages: {languages}")
    print(f"JDs per category per language: {num_jds_per_category_lang}")
    print(f"Target AWS region: {aws_region}")

    # UPDATED: Better translation client initialization
    translate_client = None
    if any(lang != "en" for lang in languages):
        try:
            translate_client = boto3.client('translate', region_name=aws_region)
            
            # Test translation capability
            test_response = translate_client.translate_text(
                Text="Test",
                SourceLanguageCode='en',
                TargetLanguageCode='fr'
            )
            print(f"✅ Translation test successful: 'Test' -> '{test_response['TranslatedText']}'")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDeniedException':
                print(f"❌ AWS Translate permission denied. Please check IAM permissions.")
            elif error_code == 'UnsupportedLanguagePairException':
                print(f"❌ Translation not supported in region {aws_region}")
            else:
                print(f"❌ AWS Translate error: {e}")
            
            # Try alternative region
            print("Trying us-east-1 as fallback region...")
            try:
                translate_client = boto3.client('translate', region_name='us-east-1')
                test_response = translate_client.translate_text(
                    Text="Test", SourceLanguageCode='en', TargetLanguageCode='fr'
                )
                print(f"✅ Fallback region works: 'Test' -> '{test_response['TranslatedText']}'")
                aws_region = 'us-east-1'
            except Exception as e2:
                print(f"❌ Fallback region also failed: {e2}")
                translate_client = None
                
        except Exception as e:
            print(f"❌ Failed to initialize translate client: {e}")
            translate_client = None

    # Track translation statistics
    translation_stats = {
        "attempted": 0,
        "successful": 0, 
        "failed": 0,
        "error_types": {}
    }

    for category in categories_list:
        # Your existing JD generation logic
        num_base_templates_for_cat = len(sample_jd_templates.get(category, []))
        if num_base_templates_for_cat == 0:
            print(f"    WARNING: No templates found for category '{category}'. Generating generic JDs.")
            variations_needed = num_jds_per_category_lang
        else:
            variations_needed = (num_jds_per_category_lang // num_base_templates_for_cat) + 1
        
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
                    translation_stats["attempted"] += 1
                    
                    # UPDATED: Robust translation with retries
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Add small delay to avoid rate limiting
                            if translation_stats["attempted"] > 1:
                                time.sleep(0.2)
                            
                            response = translate_client.translate_text(
                                Text=jd_text_en,
                                SourceLanguageCode='en',
                                TargetLanguageCode=lang_code
                            )
                            final_jd_text_for_lang = response['TranslatedText']
                            translation_stats["successful"] += 1
                            break  # Success, exit retry loop
                            
                        except ClientError as e:
                            error_code = e.response['Error']['Code']
                            
                            if error_code == 'ThrottlingException' and attempt < max_retries - 1:
                                # Rate limit, wait longer and retry
                                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                                print(f"    Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                                time.sleep(wait_time)
                                continue
                            else:
                                # Other error or max retries reached
                                error_msg = f"{error_code}: {str(e)}"
                                translation_stats["error_types"][error_msg] = translation_stats["error_types"].get(error_msg, 0) + 1
                                final_jd_text_for_lang = f"[{lang_code.upper()}_FAILED_{error_code}] {jd_text_en}"
                                translation_stats["failed"] += 1
                                break
                                
                        except Exception as e:
                            error_msg = f"UnknownError: {str(e)}"
                            translation_stats["error_types"][error_msg] = translation_stats["error_types"].get(error_msg, 0) + 1
                            final_jd_text_for_lang = f"[{lang_code.upper()}_ERROR] {jd_text_en}"
                            translation_stats["failed"] += 1
                            break
                            
                elif lang_code != "en" and not translate_client:
                    final_jd_text_for_lang = f"[{lang_code.upper()}_NO_CLIENT] {jd_text_en}"

                all_raw_data.append({
                    job_desc_col: final_jd_text_for_lang,
                    category_col: category,
                    "source_language": "en",
                    "target_language": lang_code
                })
                jd_count_for_lang_category += 1
            
            print(f"    Generated {jd_count_for_lang_category} JDs for Category: '{category}', Language: '{lang_code}'")

    # Print translation statistics
    print(f"\n=== Translation Statistics ===")
    print(f"Translation attempts: {translation_stats['attempted']}")
    print(f"Successful: {translation_stats['successful']}")
    print(f"Failed: {translation_stats['failed']}")
    
    if translation_stats['successful'] > 0:
        success_rate = (translation_stats['successful'] / translation_stats['attempted']) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    if translation_stats['error_types']:
        print("\nError breakdown:")
        for error, count in translation_stats['error_types'].items():
            print(f"  {error}: {count}")
    
    random.shuffle(all_raw_data)
    return all_raw_data

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic raw job description data (with translation) and upload to S3.")
    parser.add_argument("--s3_prefix", type=str, default="raw_job_description_data/v2_translated", help="S3 prefix.")
    parser.add_argument("--output_filename", type=str, default="raw_jds_translated_v2.jsonl", help="Output filename.")
    parser.add_argument("--num_jds_per_category_language", type=int, default=30, help="Number of JDs per category per language.")
    parser.add_argument("--languages", type=str, default="en,es,fr", help="Comma-separated language codes.")
    parser.add_argument("--aws_region", type=str, default=None, help="AWS region for Translate. Tries to auto-detect if not set.")
    parser.add_argument("--templates-file", type=str, default=DEFAULT_TEMPLATES_FILE, help=f"Path to JD templates JSON file.")
    args = parser.parse_args()

    effective_aws_region = args.aws_region or boto3.Session().region_name or "us-east-1"
    if not effective_aws_region:
        parser.error("Could not determine AWS region. Please specify with --aws_region or configure AWS CLI default.")

    langs_list = [lang.strip() for lang in args.languages.split(',')]
    
    sample_jd_templates = load_jd_templates(args.templates_file)
    if not sample_jd_templates:
        print("Exiting due to issues loading JD templates.")
        return

    raw_data = generate_raw_data(
        POC_CATEGORIES,
        args.num_jds_per_category_language,
        langs_list,
        JOB_DESC_COLUMN_NAME,
        CATEGORY_COLUMN_NAME,
        effective_aws_region,
        sample_jd_templates
    )
    print(f"Generated a total of {len(raw_data)} raw job description entries.")

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