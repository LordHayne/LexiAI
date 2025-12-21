/**
 * DOM Sanitization Utility
 * Provides XSS-safe methods for DOM manipulation using DOMPurify
 *
 * Usage:
 *   // Instead of: element.innerHTML = userInput
 *   // Use: setInnerHTML(element, userInput)
 *
 * Security: All user-generated content MUST be sanitized before rendering
 */

// DOMPurify will be loaded from CDN
// Verify it's available
if (typeof DOMPurify === 'undefined') {
    console.error('❌ DOMPurify not loaded! Include DOMPurify CDN before this script.');
}

/**
 * Safely set innerHTML with XSS protection
 * @param {HTMLElement} element - Target element
 * @param {string} html - HTML content to set
 * @param {Object} config - Optional DOMPurify config
 */
function setInnerHTML(element, html, config = {}) {
    if (!element) {
        console.warn('setInnerHTML: Element is null or undefined');
        return;
    }

    if (typeof html !== 'string') {
        console.warn('setInnerHTML: Content is not a string, converting...');
        html = String(html);
    }

    // Default config: Allow safe HTML, remove scripts and dangerous attributes
    const defaultConfig = {
        ALLOWED_TAGS: [
            'a', 'b', 'br', 'code', 'div', 'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'i', 'li', 'ol', 'p', 'pre', 'span', 'strong', 'ul', 'img', 'table',
            'thead', 'tbody', 'tr', 'td', 'th', 'blockquote', 'hr', 'small'
        ],
        ALLOWED_ATTR: [
            'class', 'id', 'href', 'src', 'alt', 'title', 'style', 'data-*',
            'aria-*', 'role', 'target'
        ],
        ALLOW_DATA_ATTR: true,
        ALLOW_ARIA_ATTR: true,
        ...config
    };

    const clean = DOMPurify.sanitize(html, defaultConfig);
    element.innerHTML = clean;
}

/**
 * Create a sanitized DOM element from HTML string
 * @param {string} html - HTML string
 * @param {Object} config - Optional DOMPurify config
 * @returns {HTMLElement} Sanitized element
 */
function createSafeElement(html, config = {}) {
    const temp = document.createElement('div');
    setInnerHTML(temp, html, config);
    return temp.firstElementChild || temp;
}

/**
 * Append sanitized HTML to element
 * @param {HTMLElement} element - Target element
 * @param {string} html - HTML to append
 * @param {Object} config - Optional DOMPurify config
 */
function appendSafeHTML(element, html, config = {}) {
    if (!element) return;
    const safeElement = createSafeElement(html, config);
    element.appendChild(safeElement);
}

/**
 * Insert sanitized HTML before/after element
 * @param {HTMLElement} element - Reference element
 * @param {string} position - 'beforebegin', 'afterbegin', 'beforeend', 'afterend'
 * @param {string} html - HTML to insert
 * @param {Object} config - Optional DOMPurify config
 */
function insertSafeHTML(element, position, html, config = {}) {
    if (!element) return;

    const defaultConfig = {
        ALLOWED_TAGS: [
            'a', 'b', 'br', 'code', 'div', 'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'i', 'li', 'ol', 'p', 'pre', 'span', 'strong', 'ul', 'img', 'table',
            'thead', 'tbody', 'tr', 'td', 'th', 'blockquote', 'hr', 'small'
        ],
        ALLOWED_ATTR: [
            'class', 'id', 'href', 'src', 'alt', 'title', 'style', 'data-*',
            'aria-*', 'role', 'target'
        ],
        ALLOW_DATA_ATTR: true,
        ALLOW_ARIA_ATTR: true,
        ...config
    };

    const clean = DOMPurify.sanitize(html, defaultConfig);
    element.insertAdjacentHTML(position, clean);
}

/**
 * Sanitize HTML string and return clean version
 * @param {string} html - HTML to sanitize
 * @param {Object} config - Optional DOMPurify config
 * @returns {string} Sanitized HTML
 */
function sanitizeHTML(html, config = {}) {
    if (typeof html !== 'string') {
        html = String(html);
    }

    const defaultConfig = {
        ALLOWED_TAGS: [
            'a', 'b', 'br', 'code', 'div', 'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'i', 'li', 'ol', 'p', 'pre', 'span', 'strong', 'ul', 'img', 'table',
            'thead', 'tbody', 'tr', 'td', 'th', 'blockquote', 'hr', 'small'
        ],
        ALLOWED_ATTR: [
            'class', 'id', 'href', 'src', 'alt', 'title', 'style', 'data-*',
            'aria-*', 'role', 'target'
        ],
        ALLOW_DATA_ATTR: true,
        ALLOW_ARIA_ATTR: true,
        ...config
    };

    return DOMPurify.sanitize(html, defaultConfig);
}

/**
 * Strict sanitization for user-generated content (more restrictive)
 * @param {string} html - HTML to sanitize
 * @returns {string} Sanitized HTML with minimal allowed tags
 */
function sanitizeUserContent(html) {
    return DOMPurify.sanitize(html, {
        ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'br', 'p', 'span'],
        ALLOWED_ATTR: ['class'],
        KEEP_CONTENT: true
    });
}

// Export for use in HTML files
window.sanitize = {
    setInnerHTML,
    createSafeElement,
    appendSafeHTML,
    insertSafeHTML,
    sanitizeHTML,
    sanitizeUserContent
};

// Backward compatibility (deprecated, use window.sanitize instead)
window.setInnerHTML = setInnerHTML;
window.sanitizeHTML = sanitizeHTML;
