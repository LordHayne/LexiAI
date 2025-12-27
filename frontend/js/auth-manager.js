/**
 * AuthManager - Handles authentication with HttpOnly Cookie-based JWT tokens
 *
 * SECURITY MIGRATION (2025-01-23):
 * - Migrated from localStorage tokens to HttpOnly cookies
 * - Eliminates XSS attack vector for token theft
 * - Tokens are now only accessible by server (httponly=true)
 * - All requests use credentials: 'include' for automatic cookie sending
 *
 * Features:
 * - Register new users
 * - Login with email/password
 * - Logout and cookie cleanup (server-side)
 * - Automatic token refresh via /auth/refresh endpoint
 * - User data persistence in localStorage (non-sensitive)
 */

class AuthManager {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.userKey = 'lexi_user_data';
        this.refreshInterval = null;
        this.tokenRefreshInterval = 14 * 60 * 1000; // Refresh every 14 minutes (token expires in 15min)
        this.refreshPromise = null; // Promise lock for concurrent refresh prevention

        // Start automatic token refresh if logged in
        this.initTokenRefresh();
    }

    /**
     * DEPRECATED: Tokens are now in HttpOnly cookies, not accessible via JavaScript
     * This method is kept for backwards compatibility but always returns null
     */
    getToken() {
        console.warn('getToken() is deprecated. Tokens are now in HttpOnly cookies and not accessible via JavaScript.');
        return null;
    }

    /**
     * Get current user data (non-sensitive, stored in localStorage)
     */
    getUserData() {
        const data = localStorage.getItem(this.userKey);
        // Check if data exists and is not the string "undefined"
        if (!data || data === 'undefined' || data === 'null') {
            return null;
        }
        try {
            return JSON.parse(data);
        } catch (error) {
            console.error('Failed to parse user data:', error);
            return null;
        }
    }

    /**
     * Check if user is logged in
     * Since we can't access HttpOnly cookies, we check if user data exists
     */
    isLoggedIn() {
        return this.getUserData() !== null;
    }

    /**
     * DEPRECATED: JWT decoding is no longer needed (tokens are server-only)
     * Kept for backwards compatibility
     */
    decodeToken(token) {
        console.warn('decodeToken() is deprecated. Tokens are in HttpOnly cookies and cannot be decoded client-side.');
        return null;
    }

    /**
     * Register new user
     * Backend sets HttpOnly cookies automatically
     */
    async register(email, password, fullName = null) {
        try {
            const response = await fetch(`${this.baseUrl}/v1/auth/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include', // CRITICAL: Send/receive cookies
                body: JSON.stringify({
                    email,
                    password,
                    full_name: fullName
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Registrierung fehlgeschlagen');
            }

            // Store ONLY user data (tokens are in HttpOnly cookies now!)
            this.storeAuthData(data.user);
            this.initTokenRefresh();

            return {
                success: true,
                user: data.user
            };
        } catch (error) {
            console.error('Registration error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Login user
     * Backend sets HttpOnly cookies automatically
     */
    async login(email, password, rememberMe = false) {
        try {
            const response = await fetch(`${this.baseUrl}/v1/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include', // CRITICAL: Send/receive cookies
                body: JSON.stringify({
                    email,
                    password,
                    remember_me: Boolean(rememberMe)
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Login fehlgeschlagen');
            }

            // Store ONLY user data (tokens are in HttpOnly cookies now!)
            this.storeAuthData(data.user);
            this.initTokenRefresh();

            return {
                success: true,
                user: data.user
            };
        } catch (error) {
            console.error('Login error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Logout user
     * Backend clears HttpOnly cookies
     */
    async logout() {
        try {
            // Call backend logout endpoint (cookies are sent automatically)
            await fetch(`${this.baseUrl}/v1/auth/logout`, {
                method: 'POST',
                credentials: 'include' // CRITICAL: Send cookies for authentication
            });
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            // Clear user data from localStorage
            this.clearAuthData();
            this.stopTokenRefresh();

            // Redirect to login page
            window.location.href = '/frontend/pages/login.html';
        }
    }

    /**
     * Refresh authentication token (with race condition prevention)
     * Backend refreshes HttpOnly cookie automatically
     */
    async refreshToken() {
        // If a refresh is already in progress, return the existing promise
        if (this.refreshPromise) {
            return this.refreshPromise;
        }

        // Create new refresh promise
        this.refreshPromise = (async () => {
            try {
                // Refresh token is in HttpOnly cookie, sent automatically
                const response = await fetch(`${this.baseUrl}/v1/auth/refresh`, {
                    method: 'POST',
                    credentials: 'include' // CRITICAL: Send refresh_token cookie
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || 'Token refresh failed');
                }

                // Backend automatically sets new access_token and refresh_token cookies
                // No need to store anything client-side!
                console.log('Token refreshed successfully (via HttpOnly cookies)');

                return {
                    success: true
                };
            } catch (error) {
                console.error('Token refresh error:', error);

                // If refresh fails, logout user
                await this.logout();

                return {
                    success: false,
                    error: error.message
                };
            } finally {
                // Clear the promise lock after completion
                this.refreshPromise = null;
            }
        })();

        return this.refreshPromise;
    }

    /**
     * Migrate anonymous user to registered account
     */
    async migrateAnonymous(email, password, anonymousUserId) {
        try {
            const response = await fetch(`${this.baseUrl}/v1/auth/migrate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include', // CRITICAL: Send/receive cookies
                body: JSON.stringify({
                    email,
                    password,
                    anonymous_user_id: anonymousUserId
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Migration fehlgeschlagen');
            }

            // Store ONLY user data (tokens are in HttpOnly cookies now!)
            this.storeAuthData(data.user);
            this.initTokenRefresh();

            return {
                success: true,
                user: data.user
            };
        } catch (error) {
            console.error('Migration error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Store user data (SECURITY: NO tokens stored client-side!)
     */
    storeAuthData(user) {
        // ONLY store non-sensitive user data (no tokens!)
        localStorage.setItem(this.userKey, JSON.stringify(user));

        // Dispatch event for other components
        window.dispatchEvent(new CustomEvent('auth-changed', {
            detail: { user, isLoggedIn: true }
        }));
    }

    /**
     * Clear user data from localStorage
     * (Cookies are cleared server-side via /auth/logout)
     */
    clearAuthData() {
        localStorage.removeItem(this.userKey);

        // Dispatch event for other components
        window.dispatchEvent(new CustomEvent('auth-changed', {
            detail: { user: null, isLoggedIn: false }
        }));
    }

    /**
     * Initialize automatic token refresh
     * Refreshes every 14 minutes (token expires in 15min)
     */
    initTokenRefresh() {
        this.stopTokenRefresh();

        if (!this.isLoggedIn()) {
            return;
        }

        // Set up periodic refresh (every 14 minutes)
        this.refreshInterval = setInterval(() => {
            this.refreshToken();
        }, this.tokenRefreshInterval);

        console.log('Automatic token refresh initialized (every 14 minutes)');
    }

    /**
     * Stop automatic token refresh
     */
    stopTokenRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    /**
     * DEPRECATED: No longer needed, cookies are sent automatically
     * Kept for backwards compatibility
     */
    getAuthHeader() {
        return {};
    }

    /**
     * Make authenticated API request
     * Cookies are sent automatically via credentials: 'include'
     */
    async authenticatedFetch(url, options = {}) {
        const response = await fetch(url, {
            ...options,
            credentials: 'include', // CRITICAL: Send cookies automatically
            headers: {
                ...options.headers,
                // No Authorization header needed - cookies are sent automatically
            }
        });

        // If unauthorized, try refreshing token once
        if (response.status === 401) {
            const refreshResult = await this.refreshToken();
            if (refreshResult.success) {
                // Retry request (cookies are automatically updated)
                return fetch(url, {
                    ...options,
                    credentials: 'include'
                });
            }
        }

        return response;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AuthManager;
}
