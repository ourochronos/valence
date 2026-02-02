#!/bin/bash
# Valence Pod Environment Variable Validation
# Validates all required environment variables before deployment

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

log_error() {
    echo -e "${RED}ERROR:${NC} $1"
    ((ERRORS++)) || true
}

log_warn() {
    echo -e "${YELLOW}WARNING:${NC} $1"
    ((WARNINGS++)) || true
}

log_ok() {
    echo -e "${GREEN}OK:${NC} $1"
}

log_info() {
    echo -e "INFO: $1"
}

# Check if a variable is set and non-empty
check_var() {
    local var_name="$1"
    local description="$2"
    local required="${3:-true}"

    if [ -z "${!var_name}" ]; then
        if [ "$required" = "true" ]; then
            log_error "$var_name is not set ($description)"
        else
            log_warn "$var_name is not set ($description)"
        fi
        return 1
    fi
    log_ok "$var_name is set"
    return 0
}

# Validate IP address format
validate_ip() {
    local ip="$1"
    if [[ $ip =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        return 0
    fi
    return 1
}

# Validate domain format
validate_domain() {
    local domain="$1"
    if [[ $domain =~ ^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$ ]]; then
        return 0
    fi
    return 1
}

# Validate email format
validate_email() {
    local email="$1"
    if [[ $email =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
        return 0
    fi
    return 1
}

# Validate SSH public key format
validate_ssh_pubkey() {
    local key="$1"
    if [[ $key =~ ^(ssh-rsa|ssh-ed25519|ecdsa-sha2-nistp256|ecdsa-sha2-nistp384|ecdsa-sha2-nistp521)\ [A-Za-z0-9+/]+[=]*( .*)?$ ]]; then
        return 0
    fi
    return 1
}

# Generate a random password
generate_password() {
    openssl rand -base64 32 | tr -d '\n'
}

echo "========================================"
echo "Valence Pod Environment Validation"
echo "========================================"
echo ""

# Parse arguments
GENERATE_SECRETS=false
EXPORT_VARS=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --generate) GENERATE_SECRETS=true ;;
        --export) EXPORT_VARS=true ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --generate  Generate missing secrets (passwords, tokens)"
            echo "  --export    Export validated variables (for sourcing)"
            echo "  -h, --help  Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# === INFRASTRUCTURE ===
echo "--- Infrastructure Variables ---"

# VALENCE_POD_IP
if check_var "VALENCE_POD_IP" "Droplet IP address"; then
    if ! validate_ip "$VALENCE_POD_IP"; then
        log_error "VALENCE_POD_IP is not a valid IP address: $VALENCE_POD_IP"
    fi
fi

# VALENCE_DOMAIN
if check_var "VALENCE_DOMAIN" "Domain name"; then
    if ! validate_domain "$VALENCE_DOMAIN"; then
        log_error "VALENCE_DOMAIN is not a valid domain: $VALENCE_DOMAIN"
    fi
fi

# LETSENCRYPT_EMAIL
if check_var "LETSENCRYPT_EMAIL" "Let's Encrypt email"; then
    if ! validate_email "$LETSENCRYPT_EMAIL"; then
        log_error "LETSENCRYPT_EMAIL is not a valid email: $LETSENCRYPT_EMAIL"
    fi
fi

echo ""
echo "--- SSH Configuration ---"

# VALENCE_SSH_PUBKEY or VALENCE_SSH_PUBKEY_FILE
if [ -n "$VALENCE_SSH_PUBKEY" ]; then
    log_ok "VALENCE_SSH_PUBKEY is set (inline key)"
    if ! validate_ssh_pubkey "$VALENCE_SSH_PUBKEY"; then
        log_error "VALENCE_SSH_PUBKEY is not a valid SSH public key"
    fi
elif [ -n "$VALENCE_SSH_PUBKEY_FILE" ]; then
    log_ok "VALENCE_SSH_PUBKEY_FILE is set: $VALENCE_SSH_PUBKEY_FILE"
    if [ -f "$VALENCE_SSH_PUBKEY_FILE" ]; then
        VALENCE_SSH_PUBKEY=$(cat "$VALENCE_SSH_PUBKEY_FILE")
        if ! validate_ssh_pubkey "$VALENCE_SSH_PUBKEY"; then
            log_error "File $VALENCE_SSH_PUBKEY_FILE does not contain a valid SSH public key"
        else
            log_ok "SSH public key loaded from file"
            if [ "$EXPORT_VARS" = true ]; then
                export VALENCE_SSH_PUBKEY
            fi
        fi
    else
        log_error "VALENCE_SSH_PUBKEY_FILE does not exist: $VALENCE_SSH_PUBKEY_FILE"
    fi
else
    # Try default locations
    for keyfile in ~/.ssh/valence_pod.pub ~/.ssh/id_ed25519.pub ~/.ssh/id_rsa.pub; do
        if [ -f "$keyfile" ]; then
            log_warn "Neither VALENCE_SSH_PUBKEY nor VALENCE_SSH_PUBKEY_FILE is set"
            log_info "Found SSH key at $keyfile - consider setting VALENCE_SSH_PUBKEY_FILE=$keyfile"
            break
        fi
    done
    if [ -z "$VALENCE_SSH_PUBKEY" ]; then
        log_error "VALENCE_SSH_PUBKEY or VALENCE_SSH_PUBKEY_FILE must be set"
    fi
fi

echo ""
echo "--- Database Secrets ---"

# VALENCE_DB_PASSWORD
if ! check_var "VALENCE_DB_PASSWORD" "PostgreSQL password"; then
    if [ "$GENERATE_SECRETS" = true ]; then
        export VALENCE_DB_PASSWORD=$(generate_password)
        log_info "Generated VALENCE_DB_PASSWORD"
        ((ERRORS--)) || true
    fi
fi

echo ""
echo "--- Matrix/Synapse Secrets ---"

# VALENCE_BOT_PASSWORD
if ! check_var "VALENCE_BOT_PASSWORD" "Matrix bot password"; then
    if [ "$GENERATE_SECRETS" = true ]; then
        export VALENCE_BOT_PASSWORD=$(generate_password)
        log_info "Generated VALENCE_BOT_PASSWORD"
        ((ERRORS--)) || true
    fi
fi

# SYNAPSE_ADMIN_PASSWORD
if ! check_var "SYNAPSE_ADMIN_PASSWORD" "Synapse admin password"; then
    if [ "$GENERATE_SECRETS" = true ]; then
        export SYNAPSE_ADMIN_PASSWORD=$(generate_password)
        log_info "Generated SYNAPSE_ADMIN_PASSWORD"
        ((ERRORS--)) || true
    fi
fi

# SYNAPSE_REGISTRATION_SECRET
if ! check_var "SYNAPSE_REGISTRATION_SECRET" "Synapse registration secret"; then
    if [ "$GENERATE_SECRETS" = true ]; then
        export SYNAPSE_REGISTRATION_SECRET=$(generate_password)
        log_info "Generated SYNAPSE_REGISTRATION_SECRET"
        ((ERRORS--)) || true
    fi
fi

# SYNAPSE_MACAROON_SECRET
if ! check_var "SYNAPSE_MACAROON_SECRET" "Synapse macaroon secret"; then
    if [ "$GENERATE_SECRETS" = true ]; then
        export SYNAPSE_MACAROON_SECRET=$(generate_password)
        log_info "Generated SYNAPSE_MACAROON_SECRET"
        ((ERRORS--)) || true
    fi
fi

# SYNAPSE_FORM_SECRET
if ! check_var "SYNAPSE_FORM_SECRET" "Synapse form secret"; then
    if [ "$GENERATE_SECRETS" = true ]; then
        export SYNAPSE_FORM_SECRET=$(generate_password)
        log_info "Generated SYNAPSE_FORM_SECRET"
        ((ERRORS--)) || true
    fi
fi

echo ""
echo "--- API Keys ---"

# OPENAI_API_KEY
if check_var "OPENAI_API_KEY" "OpenAI API key for embeddings"; then
    if [[ ! $OPENAI_API_KEY =~ ^sk- ]]; then
        log_warn "OPENAI_API_KEY does not start with 'sk-' - may be invalid"
    fi
fi

# ANTHROPIC_API_KEY (optional)
check_var "ANTHROPIC_API_KEY" "Anthropic API key (optional)" "false" || true

echo ""
echo "========================================"
echo "Validation Summary"
echo "========================================"

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}FAILED:${NC} $ERRORS errors, $WARNINGS warnings"
    echo ""
    echo "Fix the errors above before deploying."
    if [ "$GENERATE_SECRETS" = false ]; then
        echo "Tip: Run with --generate to auto-generate missing secrets"
    fi
    exit 1
else
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}PASSED WITH WARNINGS:${NC} $WARNINGS warnings"
    else
        echo -e "${GREEN}PASSED:${NC} All required variables are set and valid"
    fi
fi

# Export if requested
if [ "$EXPORT_VARS" = true ]; then
    echo ""
    echo "Exporting validated variables..."

    # Generate export commands that can be sourced
    cat << 'EXPORT_EOF'
# Source this output to set environment variables:
# eval "$(./validate-env.sh --export)"
EXPORT_EOF

    echo "export VALENCE_POD_IP=\"$VALENCE_POD_IP\""
    echo "export VALENCE_DOMAIN=\"$VALENCE_DOMAIN\""
    echo "export LETSENCRYPT_EMAIL=\"$LETSENCRYPT_EMAIL\""
    echo "export VALENCE_SSH_PUBKEY=\"$VALENCE_SSH_PUBKEY\""
    echo "export VALENCE_DB_PASSWORD=\"$VALENCE_DB_PASSWORD\""
    echo "export VALENCE_BOT_PASSWORD=\"$VALENCE_BOT_PASSWORD\""
    echo "export SYNAPSE_ADMIN_PASSWORD=\"$SYNAPSE_ADMIN_PASSWORD\""
    echo "export SYNAPSE_REGISTRATION_SECRET=\"$SYNAPSE_REGISTRATION_SECRET\""
    echo "export SYNAPSE_MACAROON_SECRET=\"$SYNAPSE_MACAROON_SECRET\""
    echo "export SYNAPSE_FORM_SECRET=\"$SYNAPSE_FORM_SECRET\""
    echo "export OPENAI_API_KEY=\"$OPENAI_API_KEY\""
    [ -n "$ANTHROPIC_API_KEY" ] && echo "export ANTHROPIC_API_KEY=\"$ANTHROPIC_API_KEY\""
fi

exit 0
