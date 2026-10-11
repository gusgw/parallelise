# Source this before running the agents:   . ./env.sh
#
# PRIMARY (verified 2026-06-09): Gemma 4 12B (QAT W4A16) on vLLM.
export MODEL_BASE_URL="http://localhost:4000/v1"
export MODEL_NAME="google/gemma-4-12B-it-qat-w4a16-ct"
export MODEL_API_KEY=""              # no key needed

# Alternative: Mistral-7B (llama.cpp --jinja), the earlier baseline:
#   MODEL_BASE_URL=http://localhost:4000/v1
#   MODEL_NAME=mistral
