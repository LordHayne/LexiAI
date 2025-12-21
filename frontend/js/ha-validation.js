/**
 * Home Assistant Configuration Field Validation
 * Provides real-time validation for HA URL and Token fields
 */

// Home Assistant URL validation
function validateHAUrl(url) {
    if (!url) return { valid: false, message: '' };

    const urlPattern = /^https?:\/\/.+/;
    if (!urlPattern.test(url)) {
        return { valid: false, message: 'URL muss mit http:// oder https:// beginnen' };
    }

    // Check for trailing slash
    if (url.endsWith('/')) {
        return { valid: false, message: 'URL sollte keinen abschließenden Slash haben' };
    }

    // Check for port
    const hasPort = /:\d+/.test(url);
    if (!hasPort) {
        return { valid: true, message: '⚠️ Standard-Port 8123 wird verwendet' };
    }

    return { valid: true, message: '✓ URL-Format korrekt' };
}

// Home Assistant Token validation
function validateHAToken(token) {
    if (!token) return { valid: false, message: '' };

    // Basic length check
    if (token.length < 20) {
        return { valid: false, message: 'Token zu kurz (mindestens 20 Zeichen)' };
    }

    // Long-lived tokens are typically much longer
    if (token.length < 100) {
        return { valid: true, message: '⚠️ Ungewöhnlich kurzes Token' };
    }

    return { valid: true, message: '✓ Token-Format korrekt' };
}

// Initialize validation on DOM load
document.addEventListener('DOMContentLoaded', () => {
    const haUrlInput = document.getElementById('ha-url');
    const haTokenInput = document.getElementById('ha-token');

    if (haUrlInput) {
        haUrlInput.addEventListener('input', function() {
            const validation = validateHAUrl(this.value);
            const hint = this.parentElement.querySelector('.form-hint');

            if (this.value) {
                this.style.borderColor = validation.valid ? 'var(--success-color, #22c55e)' : 'var(--error-color, #ef4444)';
                if (hint && validation.message) {
                    // XSS-safe: Validation message from server
                    window.sanitize.setInnerHTML(hint, validation.message);
                    hint.style.color = validation.valid ? 'var(--success-color, #22c55e)' : 'var(--error-color, #ef4444)';
                }
            } else {
                this.style.borderColor = '';
                if (hint) {
                    // Safe: Static message
                    hint.innerHTML = 'URL zu deiner Home Assistant Instanz (ohne abschließenden Slash)';
                    hint.style.color = '';
                }
            }
        });
    }

    if (haTokenInput) {
        haTokenInput.addEventListener('input', function() {
            const validation = validateHAToken(this.value);

            if (this.value) {
                this.style.borderColor = validation.valid ? 'var(--success-color, #22c55e)' : 'var(--error-color, #ef4444)';
            } else {
                this.style.borderColor = '';
            }
        });
    }
});
