#!/bin/bash

# Default values - can be overridden by command line arguments
DEFAULT_NUM_JDS=20
DEFAULT_LANGUAGES="en,es,fr"
DEFAULT_S3_PREFIX="raw_job_data/poc_multilingual_set_$(date +%Y%m%d_%H%M%S)" # Unique prefix by default
DEFAULT_AWS_REGION=$(aws configure get region) # Tries to get from AWS CLI config
DEFAULT_TEMPLATES_FILE="data/jd_templates.json" # Default path to the templates JSON file

# --- Usage Function ---
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Generates synthetic job description data and uploads it to S3."
    echo ""
    echo "Options:"
    echo "  -n, --num-jds <number>         Number of JDs per category per language (default: $DEFAULT_NUM_JDS)"
    echo "  -l, --languages <langs>        Comma-separated language codes (e.g., en,es,fr) (default: $DEFAULT_LANGUAGES)"
    echo "  -p, --s3-prefix <prefix>       S3 prefix for output (default: $DEFAULT_S3_PREFIX)"
    echo "  -r, --aws-region <region>      AWS region for Amazon Translate (default: attempts to get from AWS CLI config, or us-east-1 if not found)"
    echo "  -t, --templates-file <path>    Path to JD templates JSON file (default: $DEFAULT_TEMPLATES_FILE)"
    echo "  -h, --help                     Display this help message"
    exit 1
}

# --- Parse Command Line Arguments ---
NUM_JDS=$DEFAULT_NUM_JDS
LANGUAGES=$DEFAULT_LANGUAGES
S3_PREFIX=$DEFAULT_S3_PREFIX
AWS_REGION=$DEFAULT_AWS_REGION
TEMPLATES_FILE=$DEFAULT_TEMPLATES_FILE

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--num-jds) NUM_JDS="$2"; shift ;;
        -l|--languages) LANGUAGES="$2"; shift ;;
        -p|--s3-prefix) S3_PREFIX="$2"; shift ;;
        -r|--aws-region) AWS_REGION="$2"; shift ;;
        -t|--templates-file) TEMPLATES_FILE="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# If AWS_REGION is still empty (e.g., aws configure get region failed or returned empty)
if [ -z "$AWS_REGION" ]; then
    AWS_REGION="us-east-1" # Fallback default region
    echo "Warning: Could not determine AWS region from CLI config. Using default: $AWS_REGION"
fi

# --- Output parameters being used ---
echo "--- Running Data Generation with the following parameters ---"
echo "Number of JDs per category/language: $NUM_JDS"
echo "Languages: $LANGUAGES"
echo "S3 Prefix: $S3_PREFIX"
echo "AWS Region: $AWS_REGION"
echo "Templates File: $TEMPLATES_FILE"
echo "-------------------------------------------------------------"

# --- Execute the Python script ---
# Assuming generate_and_upload_raw_data.py is in the same directory as this script
# If not, adjust the path to the python script.
PYTHON_SCRIPT_PATH="scripts/python/generate_and_upload_raw_data.py" 

if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT_PATH' not found."
    echo "Please ensure it's in the correct location or update PYTHON_SCRIPT_PATH in this bash script."
    exit 1
fi

if [ ! -f "$TEMPLATES_FILE" ]; then
    echo "Error: Templates JSON file '$TEMPLATES_FILE' not found."
    echo "Please ensure it exists or provide the correct path using --templates-file."
    exit 1
fi

python3 "$PYTHON_SCRIPT_PATH" \
    --num_jds_per_category_language "$NUM_JDS" \
    --languages "$LANGUAGES" \
    --s3_prefix "$S3_PREFIX" \
    --aws_region "$AWS_REGION" \
    --templates-file "$TEMPLATES_FILE"

# Check exit status of the python script
if [ $? -eq 0 ]; then
    echo "-------------------------------------------------------------"
    echo "Data generation script completed successfully."
else
    echo "-------------------------------------------------------------"
    echo "Error: Data generation script failed."
fi

echo "Done."