#!/bin/bash
# Valence Pod Deployment Wrapper
# Orchestrates environment validation, deployment, and verification

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default options
CHECK_MODE=false
SKIP_VALIDATION=false
SKIP_VERIFICATION=false
VERBOSE=false
TAGS=""
LIMIT=""
EXTRA_VARS=""

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Valence Pod Deployment Script

Options:
  -c, --check           Dry-run mode (show what would change)
  -v, --verbose         Verbose output
  -t, --tags TAGS       Only run specific tags (comma-separated)
  -l, --limit HOST      Limit to specific host
  --skip-validation     Skip environment validation
  --skip-verification   Skip post-deployment verification
  --extra-vars VARS     Extra variables for Ansible
  -h, --help            Show this help message

Examples:
  $0                    Full deployment
  $0 --check            Preview changes
  $0 -t vkb             Only deploy VKB role
  $0 --skip-verification Quick deploy without verification

Environment:
  Required variables should be set in .env.pod (see .env.example)
  Run: source .env.pod before deploying
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--check)
            CHECK_MODE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--tags)
            TAGS="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --skip-verification)
            SKIP_VERIFICATION=true
            shift
            ;;
        --extra-vars)
            EXTRA_VARS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_step() {
    echo -e "\n${BLUE}==>${NC} $1"
}

log_success() {
    echo -e "${GREEN}SUCCESS:${NC} $1"
}

log_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

# Header
echo ""
echo "========================================"
echo "Valence Pod Deployment"
echo "========================================"
echo ""

if [ "$CHECK_MODE" = true ]; then
    echo -e "${YELLOW}DRY-RUN MODE${NC} - No changes will be made"
    echo ""
fi

# Step 1: Environment Validation
if [ "$SKIP_VALIDATION" = false ]; then
    log_step "Validating environment variables..."

    if [ -f "$SCRIPT_DIR/scripts/validate-env.sh" ]; then
        if ! "$SCRIPT_DIR/scripts/validate-env.sh"; then
            log_error "Environment validation failed"
            echo ""
            echo "Fix the issues above before deploying, or run with --skip-validation"
            exit 1
        fi
    else
        log_warn "validate-env.sh not found, skipping validation"
    fi
else
    log_step "Skipping environment validation (--skip-validation)"
fi

# Step 2: Check Ansible is installed
log_step "Checking Ansible installation..."

if ! command -v ansible-playbook &> /dev/null; then
    log_error "ansible-playbook not found"
    echo "Install with: pip install ansible"
    exit 1
fi

ANSIBLE_VERSION=$(ansible-playbook --version | head -1)
echo "Using: $ANSIBLE_VERSION"

# Step 3: Check inventory
log_step "Checking inventory..."

if [ ! -f "$SCRIPT_DIR/inventory.yml" ]; then
    log_error "inventory.yml not found"
    exit 1
fi

echo "Inventory: inventory.yml"

# Step 4: Build Ansible command
log_step "Preparing deployment..."

ANSIBLE_CMD="ansible-playbook -i inventory.yml site.yml"

if [ "$CHECK_MODE" = true ]; then
    ANSIBLE_CMD="$ANSIBLE_CMD --check --diff"
fi

if [ "$VERBOSE" = true ]; then
    ANSIBLE_CMD="$ANSIBLE_CMD -vv"
fi

if [ -n "$TAGS" ]; then
    ANSIBLE_CMD="$ANSIBLE_CMD --tags $TAGS"
fi

if [ -n "$LIMIT" ]; then
    ANSIBLE_CMD="$ANSIBLE_CMD --limit $LIMIT"
fi

if [ -n "$EXTRA_VARS" ]; then
    ANSIBLE_CMD="$ANSIBLE_CMD --extra-vars \"$EXTRA_VARS\""
fi

# Skip verification role in check mode
if [ "$CHECK_MODE" = true ] || [ "$SKIP_VERIFICATION" = true ]; then
    ANSIBLE_CMD="$ANSIBLE_CMD --extra-vars verify_deployment=false"
fi

echo "Command: $ANSIBLE_CMD"

# Step 5: Run deployment
log_step "Running deployment..."
echo ""

START_TIME=$(date +%s)

if eval $ANSIBLE_CMD; then
    DEPLOY_SUCCESS=true
else
    DEPLOY_SUCCESS=false
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
if [ "$DEPLOY_SUCCESS" = true ]; then
    log_success "Ansible playbook completed in ${DURATION}s"
else
    log_error "Ansible playbook failed after ${DURATION}s"
    exit 1
fi

# Step 6: Post-deployment verification (if not already done by Ansible)
if [ "$CHECK_MODE" = false ] && [ "$SKIP_VERIFICATION" = false ]; then
    # Only run shell verification if Ansible verification was skipped
    if [ -n "$TAGS" ] && [[ "$TAGS" != *"verify"* ]]; then
        log_step "Running post-deployment verification..."

        if [ -f "$SCRIPT_DIR/scripts/verify-deployment.sh" ]; then
            if "$SCRIPT_DIR/scripts/verify-deployment.sh"; then
                log_success "Verification passed"
            else
                log_warn "Verification had issues - check output above"
            fi
        else
            log_warn "verify-deployment.sh not found, skipping verification"
        fi
    fi
fi

# Summary
echo ""
echo "========================================"
echo "Deployment Summary"
echo "========================================"

if [ "$CHECK_MODE" = true ]; then
    echo "Mode: Dry-run (no changes made)"
else
    echo "Mode: Full deployment"
fi

echo "Duration: ${DURATION}s"

if [ -n "$VALENCE_DOMAIN" ]; then
    echo ""
    echo "Access your pod:"
    echo "  Matrix: https://$VALENCE_DOMAIN"
    echo "  SSH: ssh valence@$VALENCE_POD_IP"
fi

if [ "$DEPLOY_SUCCESS" = true ]; then
    echo ""
    log_success "Deployment completed successfully!"
else
    echo ""
    log_error "Deployment failed - check output above"
    exit 1
fi
