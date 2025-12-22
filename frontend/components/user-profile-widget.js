/**
 * UserProfileWidget - Displays learned user profile
 * Shows: occupation, interests, skills (auto-learned from conversations)
 */

class UserProfileWidget {
    constructor(containerId, authManager, userManager = null) {
        this.container = document.getElementById(containerId);
        this.authManager = authManager;
        this.userManager = userManager || window.userManager || null;
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
        this.loadProfile();
    }

    /**
     * Load user profile from backend
     */
    async loadProfile() {
        if (this.isLoading) return;

        this.isLoading = true;
        this.showLoading();

        try {
            const response = await this.fetchProfile();

            if (!response.ok) {
                throw new Error('Failed to load profile');
            }

            const data = await response.json();
            // API returns {user: {...}, message: "..."}
            this.profile = this.normalizeProfile(data);
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

        const user = this.profile.user || this.authManager?.getUserData() || {};
        const displayName = user.display_name || user.full_name || user.username || user.email || 'Benutzer';
        const initials = this.getInitials(displayName || user.email);

        const learnedProfile = this.profile.learnedProfile || {};
        const occupation = learnedProfile.user_profile_occupation || learnedProfile.occupation || 'Nicht angegeben';
        const interests = this.normalizeList(learnedProfile.user_profile_interests || learnedProfile.interests);
        const skills = this.normalizeList(learnedProfile.user_profile_skills || learnedProfile.skills);
        const conversationCount = Number.isFinite(learnedProfile.conversation_count) ? learnedProfile.conversation_count : null;
        const lastActive = user.last_seen ? new Date(user.last_seen) : null;
        const profileSummary = this.renderProfileSummary(learnedProfile);
        const preferencesSummary = this.renderPreferences(this.profile.preferences || {});

        // XSS-safe: User data (email, name, occupation, etc.) must be sanitized
        window.sanitize.setInnerHTML(this.container, `
            <div class="profile-widget">
                <div class="profile-header">
                    <div class="profile-avatar" data-initials="${initials}">
                        ${initials}
                    </div>
                    <div class="profile-info">
                        <h3 class="profile-name">${this.escapeHtml(displayName)}</h3>
                        <p class="profile-meta">
                            ${this.buildMetaLine(conversationCount, lastActive)}
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
                    ${profileSummary}
                    ${preferencesSummary}
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

    normalizeList(value) {
        if (!value) {
            return [];
        }
        if (Array.isArray(value)) {
            return value;
        }
        if (typeof value === 'string') {
            return [value];
        }
        return [String(value)];
    }

    renderProfileSummary(profile) {
        const summaryItems = this.getProfileSummaryItems(profile);
        if (summaryItems.length === 0) {
            return `
                <div class="profile-section empty">
                    <h4>🧾 Zusammenfassung</h4>
                    <p class="empty-state">Noch keine Zusammenfassung verfügbar.</p>
                </div>
            `;
        }

        const itemsHtml = summaryItems.map(({ label, value }) => `
            <div class="profile-summary-item">
                <span class="profile-summary-label">${this.escapeHtml(label)}:</span>
                <span class="profile-summary-value">${this.escapeHtml(value)}</span>
            </div>
        `).join('');

        return `
            <div class="profile-section">
                <h4>🧾 Zusammenfassung</h4>
                <div class="profile-summary-list">
                    ${itemsHtml}
                </div>
            </div>
        `;
    }

    renderPreferences(preferences) {
        const entries = this.getPreferenceItems(preferences);
        if (entries.length === 0) {
            return `
                <div class="profile-section empty">
                    <h4>🎛️ Präferenzen</h4>
                    <p class="empty-state">Noch keine Präferenzen gespeichert.</p>
                </div>
            `;
        }

        const itemsHtml = entries.map(({ label, value }) => `
            <div class="profile-summary-item">
                <span class="profile-summary-label">${this.escapeHtml(label)}:</span>
                <span class="profile-summary-value">${this.escapeHtml(value)}</span>
            </div>
        `).join('');

        return `
            <div class="profile-section">
                <h4>🎛️ Präferenzen</h4>
                <div class="profile-summary-list">
                    ${itemsHtml}
                </div>
            </div>
        `;
    }

    getProfileSummaryItems(profile) {
        if (!profile || typeof profile !== 'object') {
            return [];
        }

        const orderedKeys = [
            ["user_profile_occupation", "Beruf/Tätigkeit"],
            ["user_profile_interests", "Interessen"],
            ["user_profile_preferences", "Präferenzen"],
            ["user_profile_background", "Hintergrund"],
            ["user_profile_goals", "Ziele"],
            ["user_profile_location", "Wohnort"],
            ["user_profile_languages", "Sprachen"],
            ["user_profile_technical_level", "Technisches Niveau"],
            ["user_profile_communication_style", "Kommunikationsstil"],
            ["user_profile_topics", "Häufige Themen"]
        ];

        const items = [];
        orderedKeys.forEach(([key, label]) => {
            if (!Object.prototype.hasOwnProperty.call(profile, key)) {
                return;
            }
            const value = this.formatProfileValue(profile[key]);
            if (value) {
                items.push({ label, value });
            }
        });

        return items;
    }

    getPreferenceItems(preferences) {
        if (!preferences || typeof preferences !== 'object') {
            return [];
        }

        return Object.entries(preferences)
            .filter(([, value]) => value !== null && value !== undefined && value !== '')
            .map(([key, value]) => ({
                label: this.formatPreferenceKey(key),
                value: this.formatProfileValue(value)
            }))
            .filter((item) => item.value);
    }

    formatPreferenceKey(key) {
        if (!key) {
            return 'Einstellung';
        }

        const normalized = String(key).replace(/_/g, ' ').trim();
        return normalized.charAt(0).toUpperCase() + normalized.slice(1);
    }

    formatProfileValue(value) {
        if (value === null || value === undefined) {
            return '';
        }

        if (Array.isArray(value)) {
            return value.map((item) => String(item)).join(', ');
        }

        if (typeof value === 'object') {
            try {
                return JSON.stringify(value);
            } catch (error) {
                return String(value);
            }
        }

        return String(value);
    }

    buildMetaLine(conversationCount, lastActive) {
        const parts = [];
        if (Number.isFinite(conversationCount)) {
            parts.push(`${conversationCount} Gespräche`);
        }
        if (lastActive) {
            parts.push(`Zuletzt aktiv: ${this.formatDate(lastActive)}`);
        }
        return parts.length > 0 ? parts.join(' • ') : 'Keine Aktivität verfügbar';
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

    async fetchProfile() {
        if (this.userManager && this.userManager.isReady()) {
            return this.userManager.authenticatedFetch('/v1/users/me');
        }

        const authUser = this.authManager?.getUserData();
        const headers = {};
        if (authUser?.user_id) {
            headers['X-User-ID'] = authUser.user_id;
        }
        if (!headers['X-User-ID']) {
            const cachedUserId = localStorage.getItem('lexi_user_id');
            if (cachedUserId) {
                headers['X-User-ID'] = cachedUserId;
            }
        }

        if (this.authManager && this.authManager.isLoggedIn()) {
            return this.authManager.authenticatedFetch('/v1/users/me', { headers });
        }

        return fetch('/v1/users/me', { headers });
    }

    normalizeProfile(data) {
        const user = data?.user || data || {};
        return {
            user,
            learnedProfile: user.profile || data?.profile || {},
            preferences: user.preferences || {}
        };
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
