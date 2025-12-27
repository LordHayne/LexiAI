/**
 * UserManager - Handles user initialization, authentication, and profile management
 *
 * Features:
 * - Auto-initialize anonymous users on first visit
 * - Store user_id in LocalStorage
 * - Inject X-User-ID header in all API requests
 * - Update and display user display_name
 * - User statistics retrieval
 * - Integration with AuthManager for JWT authentication
 */

class UserManager {
    constructor(authManager = null) {
        this.userId = null;
        this.displayName = null;
        this.userStats = null;
        this.isInitialized = false;
        this.authManager = authManager;

        // LocalStorage keys
        this.STORAGE_KEY_USER_ID = 'lexi_user_id';
        this.STORAGE_KEY_DISPLAY_NAME = 'lexi_display_name';

        // API endpoints
        this.API_BASE = '/v1/users';
    }

    /**
     * Set AuthManager instance
     */
    setAuthManager(authManager) {
        this.authManager = authManager;
    }

    /**
     * Initialize user - check LocalStorage or create new anonymous user
     */
    async init() {
        if (this.isInitialized) {
            return;
        }

        try {
            // 1. Check LocalStorage for existing user_id
            this.userId = localStorage.getItem(this.STORAGE_KEY_USER_ID);
            this.displayName = localStorage.getItem(this.STORAGE_KEY_DISPLAY_NAME);

            const authUser = this.authManager && this.authManager.isLoggedIn()
                ? this.authManager.getUserData()
                : null;

            if (authUser && authUser.user_id) {
                this.setAuthenticatedUser(authUser);
                try {
                    const profileUser = await this.fetchUserProfile();
                    this.syncAuthUserData(profileUser);
                } catch (error) {
                    console.warn('⚠️ Failed to sync authenticated user profile:', error);
                }
            } else if (this.userId) {
                // Verify user still exists on server
                try {
                    await this.fetchUserProfile();
                } catch (error) {
                    console.warn('⚠️ User not found on server, reinitializing...');
                    await this.createAnonymousUser();
                }
            } else {
                // 2. No user_id found, create new anonymous user
                await this.createAnonymousUser();
            }

            this.isInitialized = true;

        } catch (error) {
            console.error('❌ Failed to initialize UserManager:', error);

            // Retry once if initialization fails
            try {
                await this.createAnonymousUser();
                this.isInitialized = true;
            } catch (retryError) {
                console.error('❌ Retry failed:', retryError);
                throw new Error('User initialization failed after retry');
            }
        }
    }

    /**
     * Use authenticated user as primary identity for chat and profiles
     * @param {Object} user - Auth user object
     */
    setAuthenticatedUser(user) {
        if (!user || !user.user_id) {
            return;
        }

        this.userId = user.user_id;
        this.displayName = user.display_name || user.username || (user.email ? user.email.split('@')[0] : null);

        localStorage.setItem(this.STORAGE_KEY_USER_ID, this.userId);
        if (this.displayName) {
            localStorage.setItem(this.STORAGE_KEY_DISPLAY_NAME, this.displayName);
        }
    }

    /**
     * Create a new anonymous user via API
     */
    async createAnonymousUser() {
        try {
            const response = await fetch(`${this.API_BASE}/init`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`Failed to create user: ${response.status} - ${errorData.detail || 'Unknown error'}`);
            }

            const data = await response.json();

            // Check if user data is present (API returns {user: {...}, message: "..."})
            if (!data.user || !data.user.user_id) {
                throw new Error('Invalid response from user init API');
            }

            // Store user data
            this.userId = data.user.user_id;
            this.displayName = data.user.display_name || null;

            // Save to LocalStorage
            localStorage.setItem(this.STORAGE_KEY_USER_ID, this.userId);
            if (this.displayName) {
                localStorage.setItem(this.STORAGE_KEY_DISPLAY_NAME, this.displayName);
            }

        } catch (error) {
            console.error('❌ Failed to create anonymous user:', error);
            throw error;
        }
    }

    /**
     * Fetch current user profile from API
     */
    async fetchUserProfile() {
        try {
            const headers = this.injectUserIdHeader();

            const response = await fetch(`${this.API_BASE}/me`, {
                method: 'GET',
                headers
            });

            if (!response.ok) {
                if (response.status === 404) {
                    throw new Error('User not found');
                }
                throw new Error(`Failed to fetch profile: ${response.status}`);
            }

            const data = await response.json();

            // Check if user data is present (API returns {user: {...}, message: ...})
            if (data.user) {
                this.displayName = data.user.display_name || null;

                // Update LocalStorage cache
                if (this.displayName) {
                    localStorage.setItem(this.STORAGE_KEY_DISPLAY_NAME, this.displayName);
                }

                this.syncAuthUserData(data.user);
                return data.user;
            }

            throw new Error('Invalid response from profile API');

        } catch (error) {
            console.error('❌ Failed to fetch user profile:', error);
            throw error;
        }
    }

    /**
     * Update user display name
     * @param {string} newName - New display name
     */
    async updateDisplayName(newName) {
        if (!newName || newName.trim().length === 0) {
            throw new Error('Display name cannot be empty');
        }

        if (newName.length > 100) {
            throw new Error('Display name too long (max 100 characters)');
        }

        try {
            const headers = this.injectUserIdHeader({
                'Content-Type': 'application/json'
            });

            const response = await fetch(`${this.API_BASE}/me`, {
                method: 'PATCH',
                headers,
                body: JSON.stringify({
                    display_name: newName.trim()
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`Failed to update name: ${response.status} - ${errorData.detail || 'Unknown error'}`);
            }

            const data = await response.json();

            // API returns {user: {...}, message: "..."}
            if (data.user && data.user.user_id) {
                this.userId = data.user.user_id;
                localStorage.setItem(this.STORAGE_KEY_USER_ID, this.userId);
            }

            if (data.user && data.user.display_name !== undefined) {
                this.displayName = data.user.display_name;

                // Update LocalStorage cache
                localStorage.setItem(this.STORAGE_KEY_DISPLAY_NAME, this.displayName);

                this.syncAuthUserData(data.user);
                return data.user;
            }

            throw new Error('Invalid response from update API');

        } catch (error) {
            console.error('❌ Failed to update display name:', error);
            throw error;
        }
    }

    /**
     * Fetch user statistics
     */
    async fetchUserStats() {
        try {
            const headers = this.injectUserIdHeader();

            const response = await fetch(`${this.API_BASE}/stats`, {
                method: 'GET',
                headers
            });

            if (!response.ok) {
                throw new Error(`Failed to fetch stats: ${response.status}`);
            }

            const data = await response.json();

            // API returns stats directly (user_id, total_memories, categories, etc.)
            // Check if response has stats structure
            if (data && (data.user_id || data.total_memories !== undefined)) {
                this.userStats = data;
                return data;
            }

            throw new Error('Invalid response from stats API');

        } catch (error) {
            console.error('❌ Failed to fetch user stats:', error);
            throw error;
        }
    }

    /**
     * Inject X-User-ID header and JWT authorization into request headers
     * @param {Object} headers - Existing headers object (optional)
     * @returns {Object} Headers with X-User-ID and Authorization injected
     */
    injectUserIdHeader(headers = {}) {
        const result = { ...headers };

        // Add user ID header
        if (this.userId) {
            result['X-User-ID'] = this.userId;
        } else {
            console.warn('⚠️ No user_id available for header injection');
        }

        // Add JWT authorization if available
        if (this.authManager) {
            const authHeaders = this.authManager.getAuthHeader();
            Object.assign(result, authHeaders);
        }

        return result;
    }

    /**
     * Make authenticated fetch request (uses AuthManager if available)
     */
    async authenticatedFetch(url, options = {}) {
        if (this.authManager && this.authManager.isLoggedIn()) {
            // Use AuthManager's authenticated fetch (handles token refresh)
            return this.authManager.authenticatedFetch(url, {
                ...options,
                headers: {
                    ...options.headers,
                    ...this.injectUserIdHeader()
                }
            });
        } else {
            // Fallback to regular fetch with user ID
            return fetch(url, {
                ...options,
                headers: this.injectUserIdHeader(options.headers)
            });
        }
    }

    /**
     * Get current user_id
     * @returns {string|null}
     */
    getUserId() {
        return this.userId;
    }

    /**
     * Get current display_name
     * @returns {string|null}
     */
    getDisplayName() {
        return this.displayName || 'Anonymer Benutzer';
    }

    /**
     * Get user statistics
     * @returns {Object|null}
     */
    getUserStats() {
        return this.userStats;
    }

    /**
     * Check if user is initialized
     * @returns {boolean}
     */
    isReady() {
        return this.isInitialized && this.userId !== null;
    }

    /**
     * Clear user data (logout)
     */
    clearUserData() {
        this.userId = null;
        this.displayName = null;
        this.userStats = null;
        this.isInitialized = false;

        localStorage.removeItem(this.STORAGE_KEY_USER_ID);
        localStorage.removeItem(this.STORAGE_KEY_DISPLAY_NAME);
    }

    /**
     * Show user name update dialog
     */
    showNameUpdateDialog() {
        const currentName = this.getDisplayName();
        const newName = prompt('Benutzername eingeben:', currentName === 'Anonymer Benutzer' ? '' : currentName);

        if (newName === null) {
            // User cancelled
            return;
        }

        if (!newName || newName.trim().length === 0) {
            alert('❌ Benutzername darf nicht leer sein.');
            return;
        }

        // Update display name
        this.updateDisplayName(newName)
            .then(() => {
                alert('✅ Benutzername erfolgreich aktualisiert!');

                // Update UI elements
                this.updateUIElements();
            })
            .catch(error => {
                alert(`❌ Fehler beim Aktualisieren: ${error.message}`);
            });
    }

    /**
     * Update all UI elements showing user info
     */
    updateUIElements() {
        // Update all elements with class 'user-display-name'
        const nameElements = document.querySelectorAll('.user-display-name');
        nameElements.forEach(el => {
            el.textContent = this.getDisplayName();
        });

        // Update all elements with class 'user-id'
        const idElements = document.querySelectorAll('.user-id');
        idElements.forEach(el => {
            el.textContent = this.getUserId() || '-';
        });
    }

    /**
     * Sync display name and profile data into AuthManager storage if logged in
     * @param {Object} userData - Updated user data from /v1/users
     */
    syncAuthUserData(userData) {
        if (!this.authManager || !this.authManager.isLoggedIn()) {
            return;
        }

        const existing = this.authManager.getUserData();
        if (!existing) {
            return;
        }

        const merged = { ...existing };
        const updateFields = ['display_name', 'profile', 'preferences', 'user_id', 'email', 'username'];
        updateFields.forEach((field) => {
            if (userData && userData[field] !== undefined && userData[field] !== null) {
                merged[field] = userData[field];
            }
        });

        this.authManager.storeAuthData(merged);
    }
}

// Export for use in HTML files
window.UserManager = UserManager;
