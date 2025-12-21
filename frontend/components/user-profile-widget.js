/**
 * UserProfileWidget - Displays learned user profile
 * Shows: occupation, interests, skills (auto-learned from conversations)
 */

class UserProfileWidget {
    constructor(containerId, authManager) {
        this.container = document.getElementById(containerId);
        this.authManager = authManager;
        this.profile = null;
        this.isLoading = false;

        // Listen for auth changes
        window.addEventListener('auth-changed', (e) => {
            if (e.detail.isLoggedIn) {
                this.loadProfile();
            } else {
                this.clear();
            }
        });

        // Load profile if logged in
        if (this.authManager.isLoggedIn()) {
            this.loadProfile();
        }
    }

    /**
     * Load user profile from backend
     */
    async loadProfile() {
        if (this.isLoading) return;

        this.isLoading = true;
        this.showLoading();

        try {
            const response = await this.authManager.authenticatedFetch('/v1/users/me');

            if (!response.ok) {
                throw new Error('Failed to load profile');
            }

            const data = await response.json();
            // API returns {user: {...}, message: "..."}
            this.profile = data.user || data;
            this.render();
        } catch (error) {
            console.error('Error loading profile:', error);
            this.showError('Profil konnte nicht geladen werden');
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Show loading state
     */
    showLoading() {
        if (!this.container) return;

        // Safe: No user data in loading template
        this.container.innerHTML = `
            <div class="profile-widget loading">
                <div class="profile-header">
                    <div class="profile-avatar skeleton"></div>
                    <div class="profile-info">
                        <div class="skeleton-text"></div>
                        <div class="skeleton-text short"></div>
                    </div>
                </div>
                <div class="profile-content">
                    <div class="skeleton-block"></div>
                    <div class="skeleton-block"></div>
                    <div class="skeleton-block"></div>
                </div>
            </div>
        `;
    }

    /**
     * Show error message
     */
    showError(message) {
        if (!this.container) return;

        // XSS-safe: Error message could contain unsafe content
        window.sanitize.setInnerHTML(this.container, `
            <div class="profile-widget error">
                <div class="error-icon">⚠️</div>
                <p>${message}</p>
            </div>
        `);
    }

    /**
     * Clear widget
     */
    clear() {
        if (!this.container) return;
        this.container.innerHTML = '';
        this.profile = null;
    }

    /**
     * Render profile widget
     */
    render() {
        if (!this.container || !this.profile) return;

        const user = this.authManager.getUserData();
        const initials = this.getInitials(user?.email || user?.full_name);

        const occupation = this.profile.occupation || 'Nicht angegeben';
        const interests = this.profile.interests || [];
        const skills = this.profile.skills || [];
        const conversationCount = this.profile.conversation_count || 0;
        const lastActive = this.profile.last_active ? new Date(this.profile.last_active) : null;

        // XSS-safe: User data (email, name, occupation, etc.) must be sanitized
        window.sanitize.setInnerHTML(this.container, `
            <div class="profile-widget">
                <div class="profile-header">
                    <div class="profile-avatar" data-initials="${initials}">
                        ${initials}
                    </div>
                    <div class="profile-info">
                        <h3 class="profile-name">${user?.full_name || user?.email || 'Benutzer'}</h3>
                        <p class="profile-meta">
                            ${conversationCount} Gespräche
                            ${lastActive ? `• Zuletzt aktiv: ${this.formatDate(lastActive)}` : ''}
                        </p>
                    </div>
                    <button class="profile-edit-btn" onclick="window.fullProfileWidget.editProfile()" title="Profil bearbeiten">
                        ⚙️
                    </button>
                </div>

                <div class="profile-content">
                    ${this.renderOccupation(occupation)}
                    ${this.renderInterests(interests)}
                    ${this.renderSkills(skills)}
                </div>

                <div class="profile-footer">
                    <button class="btn-text" onclick="window.fullProfileWidget.viewFullProfile()">
                        Vollständiges Profil anzeigen →
                    </button>
                </div>
            </div>
        `);
    }

    /**
     * Render occupation section
     */
    renderOccupation(occupation) {
        if (!occupation || occupation === 'Nicht angegeben') {
            return `
                <div class="profile-section empty">
                    <h4>💼 Beruf</h4>
                    <p class="empty-state">Noch nicht bekannt. LexiAI lernt aus deinen Gesprächen.</p>
                </div>
            `;
        }

        return `
            <div class="profile-section">
                <h4>💼 Beruf</h4>
                <div class="profile-tag occupation">${this.escapeHtml(occupation)}</div>
            </div>
        `;
    }

    /**
     * Render interests section
     */
    renderInterests(interests) {
        if (!interests || interests.length === 0) {
            return `
                <div class="profile-section empty">
                    <h4>🎯 Interessen</h4>
                    <p class="empty-state">Noch keine Interessen erkannt.</p>
                </div>
            `;
        }

        const interestTags = interests.slice(0, 5).map(interest =>
            `<span class="profile-tag interest">${this.escapeHtml(interest)}</span>`
        ).join('');

        const moreCount = interests.length > 5 ? interests.length - 5 : 0;

        return `
            <div class="profile-section">
                <h4>🎯 Interessen</h4>
                <div class="profile-tags">
                    ${interestTags}
                    ${moreCount > 0 ? `<span class="profile-tag more">+${moreCount} weitere</span>` : ''}
                </div>
            </div>
        `;
    }

    /**
     * Render skills section
     */
    renderSkills(skills) {
        if (!skills || skills.length === 0) {
            return `
                <div class="profile-section empty">
                    <h4>⚡ Fähigkeiten</h4>
                    <p class="empty-state">Noch keine Fähigkeiten identifiziert.</p>
                </div>
            `;
        }

        const skillTags = skills.slice(0, 5).map(skill =>
            `<span class="profile-tag skill">${this.escapeHtml(skill)}</span>`
        ).join('');

        const moreCount = skills.length > 5 ? skills.length - 5 : 0;

        return `
            <div class="profile-section">
                <h4>⚡ Fähigkeiten</h4>
                <div class="profile-tags">
                    ${skillTags}
                    ${moreCount > 0 ? `<span class="profile-tag more">+${moreCount} weitere</span>` : ''}
                </div>
            </div>
        `;
    }

    /**
     * Get initials from name or email
     */
    getInitials(text) {
        if (!text) return '?';

        const parts = text.split(/[\s@.]+/);
        if (parts.length >= 2) {
            return (parts[0][0] + parts[1][0]).toUpperCase();
        }
        return text.substring(0, 2).toUpperCase();
    }

    /**
     * Format date
     */
    formatDate(date) {
        const now = new Date();
        const diff = now - date;
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));

        if (days === 0) return 'Heute';
        if (days === 1) return 'Gestern';
        if (days < 7) return `vor ${days} Tagen`;
        if (days < 30) return `vor ${Math.floor(days / 7)} Wochen`;
        if (days < 365) return `vor ${Math.floor(days / 30)} Monaten`;
        return `vor ${Math.floor(days / 365)} Jahren`;
    }

    /**
     * Escape HTML
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Edit profile (open config page)
     */
    editProfile() {
        window.location.href = '/frontend/pages/config_ui.html#profile';
    }

    /**
     * View full profile
     */
    viewFullProfile() {
        window.location.href = '/frontend/pages/config_ui.html#profile';
    }

    /**
     * Refresh profile data
     */
    async refresh() {
        await this.loadProfile();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UserProfileWidget;
}
