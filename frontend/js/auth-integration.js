/**
 * Auth Modal Integration
 * Handles modal control, form submission, and user menu interactions
 */

/**
 * Cleanup function for event listeners (if needed for SPA migration)
 * Currently not needed for multi-page app, but provided for future use
 */
window.cleanupAuthIntegration = function() {
    // Clear all active notifications
    activeNotifications.forEach(notification => {
        if (notification && notification.parentNode) {
            notification.remove();
        }
    });
    activeNotifications = [];
};

// Modal Control Functions
function openAuthModal(tab = 'login') {
    const modal = document.getElementById('authModal');
    if (!modal) {
        console.error('Auth modal not found');
        return;
    }

    modal.classList.add('active');
    switchAuthTab(tab);

    // Clear previous errors/success messages
    clearAuthMessages();
}

function closeAuthModal() {
    const modal = document.getElementById('authModal');
    if (!modal) return;

    modal.classList.remove('active');

    // Clear form inputs
    document.getElementById('loginForm')?.reset();
    document.getElementById('registerForm')?.reset();

    // Clear messages
    clearAuthMessages();
}

function switchAuthTab(tab) {
    // Update tab buttons
    const tabs = document.querySelectorAll('.auth-tab');
    tabs.forEach(t => {
        if (t.dataset.tab === tab) {
            t.classList.add('active');
        } else {
            t.classList.remove('active');
        }
    });

    // Update forms
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');

    if (tab === 'login') {
        loginForm?.classList.add('active');
        registerForm?.classList.remove('active');
    } else {
        registerForm?.classList.add('active');
        loginForm?.classList.remove('active');
    }

    // Clear messages when switching tabs
    clearAuthMessages();
}

function clearAuthMessages() {
    // Clear all error/success messages
    const messages = [
        'loginError',
        'registerError',
        'registerSuccess'
    ];

    messages.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.remove('active');
            el.textContent = '';
        }
    });
}

// Form Submission Handlers
async function handleLogin(event) {
    event.preventDefault();

    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const errorEl = document.getElementById('loginError');
    const submitBtn = document.getElementById('loginSubmitBtn');

    // Clear previous errors
    errorEl.classList.remove('active');
    errorEl.textContent = '';

    // Disable submit button
    submitBtn.disabled = true;
    submitBtn.textContent = 'Anmelden...';

    try {
        // Use AuthManager to login
        if (!window.authManager) {
            throw new Error('AuthManager not initialized');
        }

        const result = await window.authManager.login(email, password);

        if (!result.success) {
            throw new Error(result.error || 'Login fehlgeschlagen');
        }

        if (window.userManager && result.user) {
            window.userManager.setAuthenticatedUser(result.user);
        }

        // Close modal
        closeAuthModal();

        // Update UI
        if (window.updateAuthUI) {
            window.updateAuthUI();
        }

        // Show success notification (optional)
        showNotification('✅ Erfolgreich angemeldet!', 'success');

    } catch (error) {
        console.error('❌ Login failed:', error);

        // Show error message
        errorEl.textContent = error.message || 'Login fehlgeschlagen. Bitte überprüfe deine Eingaben.';
        errorEl.classList.add('active');

    } finally {
        // Re-enable submit button
        submitBtn.disabled = false;
        submitBtn.textContent = 'Anmelden';
    }
}

async function handleRegister(event) {
    event.preventDefault();

    const email = document.getElementById('registerEmail').value;
    const password = document.getElementById('registerPassword').value;
    const passwordConfirm = document.getElementById('registerPasswordConfirm').value;
    const errorEl = document.getElementById('registerError');
    const successEl = document.getElementById('registerSuccess');
    const submitBtn = document.getElementById('registerSubmitBtn');

    // Clear previous messages
    errorEl.classList.remove('active');
    successEl.classList.remove('active');
    errorEl.textContent = '';
    successEl.textContent = '';

    // Validate passwords match
    if (password !== passwordConfirm) {
        errorEl.textContent = 'Passwörter stimmen nicht überein!';
        errorEl.classList.add('active');
        return;
    }

    // Validate password strength (basic)
    if (password.length < 8) {
        errorEl.textContent = 'Passwort muss mindestens 8 Zeichen lang sein!';
        errorEl.classList.add('active');
        return;
    }

    if (!/[A-Z]/.test(password)) {
        errorEl.textContent = 'Passwort muss mindestens einen Großbuchstaben enthalten!';
        errorEl.classList.add('active');
        return;
    }

    if (!/\d/.test(password)) {
        errorEl.textContent = 'Passwort muss mindestens eine Zahl enthalten!';
        errorEl.classList.add('active');
        return;
    }

    // Disable submit button
    submitBtn.disabled = true;
    submitBtn.textContent = 'Registrierung läuft...';

    try {
        // Use AuthManager to register
        if (!window.authManager) {
            throw new Error('AuthManager not initialized');
        }

        const result = await window.authManager.register(email, password);

        if (!result.success) {
            throw new Error(result.error || 'Registrierung fehlgeschlagen');
        }

        if (window.userManager && result.user) {
            window.userManager.setAuthenticatedUser(result.user);
        }

        // Show success message
        successEl.textContent = '✅ Erfolgreich registriert! Du wirst angemeldet...';
        successEl.classList.add('active');

        // Wait 1.5 seconds, then close modal
        setTimeout(() => {
            closeAuthModal();

            // Update UI
            if (window.updateAuthUI) {
                window.updateAuthUI();
            }

            // Show success notification
            showNotification('✅ Willkommen bei Lexi!', 'success');
        }, 1500);

    } catch (error) {
        console.error('❌ Registration failed:', error);

        // Show error message
        let errorMessage = 'Registrierung fehlgeschlagen.';

        if (error.message.includes('409') || error.message.includes('already')) {
            errorMessage = 'Diese E-Mail ist bereits registriert. Bitte melde dich an.';
        } else if (error.message.includes('400')) {
            errorMessage = 'Ungültige E-Mail oder Passwort zu schwach.';
        } else {
            errorMessage = error.message || 'Registrierung fehlgeschlagen.';
        }

        errorEl.textContent = errorMessage;
        errorEl.classList.add('active');

        // Re-enable submit button
        submitBtn.disabled = false;
        submitBtn.textContent = 'Registrieren';
    }
}

// User Menu Dropdown Toggle
function toggleUserMenu() {
    const menuContent = document.getElementById('userMenuContent');
    if (!menuContent) return;

    menuContent.classList.toggle('active');
}

// Setup user menu button click handler
document.addEventListener('DOMContentLoaded', () => {
    const userMenuButton = document.getElementById('userMenuButton');
    if (userMenuButton) {
        userMenuButton.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleUserMenu();
        });
    }

    // Close user menu when clicking outside
    document.addEventListener('click', (e) => {
        const userMenuDropdown = document.getElementById('userMenuDropdown');
        const menuContent = document.getElementById('userMenuContent');

        if (userMenuDropdown && menuContent && !userMenuDropdown.contains(e.target)) {
            menuContent.classList.remove('active');
        }
    });

    // Close modal when clicking on overlay (outside modal content)
    const authModal = document.getElementById('authModal');
    if (authModal) {
        authModal.addEventListener('click', (e) => {
            if (e.target === authModal) {
                closeAuthModal();
            }
        });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // ESC to close modal
        if (e.key === 'Escape') {
            const modal = document.getElementById('authModal');
            if (modal && modal.classList.contains('active')) {
                closeAuthModal();
            }
        }
    });
});

// Notification Helper (optional)
// Limit concurrent notifications to prevent memory issues
const MAX_NOTIFICATIONS = 3;
let activeNotifications = [];

function showNotification(message, type = 'info') {
    // Remove oldest notification if limit reached
    if (activeNotifications.length >= MAX_NOTIFICATIONS) {
        const oldest = activeNotifications.shift();
        if (oldest && oldest.parentNode) {
            oldest.remove();
        }
    }

    // Simple notification - you can replace with a better UI component
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: ${20 + (activeNotifications.length * 70)}px;
        right: 20px;
        padding: 16px 24px;
        background: ${type === 'success' ? 'var(--success-color)' : 'var(--primary-color)'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 10001;
        font-size: 14px;
        font-weight: 500;
        animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);
    activeNotifications.push(notification);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            notification.remove();
            // Remove from tracking array
            const index = activeNotifications.indexOf(notification);
            if (index > -1) {
                activeNotifications.splice(index, 1);
            }
        }, 300);
    }, 3000);
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
