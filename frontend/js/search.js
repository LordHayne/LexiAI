/**
 * Lexi AI - Globale Suchfunktion
 * Mit Fuzzy Search, Cmd+K Shortcut, und Live-Search
 */

(function() {
    'use strict';

    /**
     * Suchindex mit allen durchsuchbaren Inhalten
     */
    const SEARCH_INDEX = [
        // Hauptseiten
        {
            id: 'dashboard',
            title: 'Dashboard',
            description: 'Übersicht und Schnellzugriff auf alle Features',
            keywords: ['home', 'start', 'übersicht', 'dashboard'],
            url: '/frontend/index.html',
            icon: '🏠',
            category: 'Navigation'
        },
        {
            id: 'chat',
            title: 'Chat',
            description: 'Intelligente Konversationen mit semantischem Gedächtnis',
            keywords: ['chat', 'gespräch', 'konversation', 'nachricht', 'talk'],
            url: '/frontend/chat_ui.html',
            icon: '💬',
            category: 'Hauptfunktionen'
        },

        // Intelligenz-Features
        {
            id: 'memory',
            title: 'Memory Management',
            description: 'ML-basierte Konsolidierung und Synthetisierung von Erinnerungen',
            keywords: ['memory', 'gedächtnis', 'erinnerung', 'speicher', 'konsolidierung'],
            url: '/frontend/pages/memory_management_ui.html',
            icon: '🧠',
            category: 'Intelligenz'
        },
        {
            id: 'patterns',
            title: 'Pattern Recognition',
            description: 'Automatische Erkennung von Verhaltensmustern',
            keywords: ['pattern', 'muster', 'erkennung', 'analyse', 'verhalten'],
            url: '/frontend/pages/patterns_ui.html',
            icon: '🔍',
            category: 'Intelligenz'
        },
        {
            id: 'gaps',
            title: 'Wissenslücken',
            description: 'Identifizierung von Bereichen mit fehlenden Informationen',
            keywords: ['wissenslücken', 'gaps', 'lücken', 'wissen', 'fehlt'],
            url: '/frontend/pages/knowledge_gaps_ui.html',
            icon: '💡',
            category: 'Intelligenz'
        },
        {
            id: 'goals',
            title: 'Ziele',
            description: 'Tracking und proaktive Erinnerung an deine Ziele',
            keywords: ['ziele', 'goals', 'tracking', 'erinnerung', 'aufgaben'],
            url: '/frontend/pages/goals_ui.html',
            icon: '🎯',
            category: 'Intelligenz'
        },

        // Verwaltung
        {
            id: 'settings',
            title: 'Einstellungen',
            description: 'Systemkonfiguration und Anpassungen',
            keywords: ['einstellungen', 'settings', 'config', 'konfiguration'],
            url: '/frontend/pages/config_ui.html',
            icon: '⚙️',
            category: 'Verwaltung'
        },
        {
            id: 'metrics',
            title: 'Metriken Dashboard',
            description: 'Echtzeit-Überwachung von Performance und Statistiken',
            keywords: ['metriken', 'metrics', 'dashboard', 'statistik', 'performance'],
            url: '/frontend/pages/metrics_dashboard.html',
            icon: '📊',
            category: 'Verwaltung'
        },

        // Actions
        {
            id: 'new-chat',
            title: 'Neue Konversation',
            description: 'Starte eine neue Chat-Sitzung',
            keywords: ['neu', 'new', 'chat', 'konversation', 'start'],
            url: '/frontend/chat_ui.html',
            icon: '🚀',
            category: 'Aktionen'
        },
        {
            id: 'heartbeat',
            title: 'Memory Konsolidierung',
            description: 'Starte manuelle Memory-Konsolidierung',
            keywords: ['konsolidierung', 'heartbeat', 'cleanup', 'bereinigung'],
            url: '/frontend/chat_ui.html#heartbeat',
            icon: '🫀',
            category: 'Aktionen'
        }
    ];

    /**
     * Fuzzy Search Algorithmus
     * Berechnet Ähnlichkeit zwischen Suchanfrage und Text
     */
    function fuzzyMatch(query, text) {
        query = query.toLowerCase();
        text = text.toLowerCase();

        // Exakte Übereinstimmung hat höchste Priorität
        if (text.includes(query)) {
            return 100;
        }

        // Levenshtein Distance für fuzzy matching
        let score = 0;
        let queryIndex = 0;

        for (let i = 0; i < text.length && queryIndex < query.length; i++) {
            if (text[i] === query[queryIndex]) {
                score += 1;
                queryIndex++;
            }
        }

        // Normalisieren auf 0-100
        return (score / query.length) * 80;
    }

    /**
     * Suche im Index durchführen
     */
    function search(query) {
        if (!query || query.trim().length < 2) {
            return [];
        }

        const results = [];

        SEARCH_INDEX.forEach(item => {
            let maxScore = 0;

            // Suche in Titel
            const titleScore = fuzzyMatch(query, item.title);
            maxScore = Math.max(maxScore, titleScore);

            // Suche in Beschreibung
            const descScore = fuzzyMatch(query, item.description) * 0.7;
            maxScore = Math.max(maxScore, descScore);

            // Suche in Keywords
            item.keywords.forEach(keyword => {
                const keywordScore = fuzzyMatch(query, keyword) * 0.9;
                maxScore = Math.max(maxScore, keywordScore);
            });

            // Nur Ergebnisse mit Mindest-Score
            if (maxScore > 30) {
                results.push({
                    ...item,
                    score: maxScore
                });
            }
        });

        // Sortiere nach Score (höchster zuerst)
        results.sort((a, b) => b.score - a.score);

        return results;
    }

    /**
     * Highlight matches in text (XSS-safe with regex escaping)
     */
    function highlightMatch(text, query) {
        if (!query) return text;

        // Escape regex special characters to prevent RegExp injection
        const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(${escapedQuery})`, 'gi');

        // First escape HTML entities in text
        const escapedText = text.replace(/&/g, '&amp;')
                                 .replace(/</g, '&lt;')
                                 .replace(/>/g, '&gt;')
                                 .replace(/"/g, '&quot;')
                                 .replace(/'/g, '&#039;');

        return escapedText.replace(regex, '<span class="search-result-match">$1</span>');
    }

    /**
     * Gruppiere Ergebnisse nach Kategorie
     */
    function groupByCategory(results) {
        const grouped = {};

        results.forEach(result => {
            if (!grouped[result.category]) {
                grouped[result.category] = [];
            }
            grouped[result.category].push(result);
        });

        return grouped;
    }

    /**
     * Rendere Suchergebnisse
     */
    function renderResults(results, query) {
        const resultsContainer = document.getElementById('searchResults');
        if (!resultsContainer) return;

        resultsContainer.innerHTML = '';

        if (results.length === 0) {
            // Safe: No user input in this HTML
            resultsContainer.innerHTML = `
                <div class="search-empty">
                    <div class="search-empty-icon">🔍</div>
                    <div class="search-empty-title">Keine Ergebnisse gefunden</div>
                    <div class="search-empty-text">Versuche andere Suchbegriffe</div>
                </div>
            `;
            return;
        }

        // Gruppiere nach Kategorie
        const grouped = groupByCategory(results);

        // Rendere jede Kategorie
        Object.keys(grouped).forEach(category => {
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'search-category';
            // category comes from SEARCH_INDEX (trusted), but sanitize for safety
            window.sanitize.setInnerHTML(categoryDiv, `<div class="search-category-title">${category}</div>`);
            resultsContainer.appendChild(categoryDiv);

            grouped[category].forEach((result, index) => {
                const resultItem = document.createElement('a');
                resultItem.className = 'search-result-item';
                resultItem.href = result.url;
                if (index === 0 && category === Object.keys(grouped)[0]) {
                    resultItem.classList.add('selected');
                }

                // XSS-safe: highlightMatch now escapes HTML, but sanitize for double protection
                window.sanitize.setInnerHTML(resultItem, `
                    <div class="search-result-icon">${result.icon}</div>
                    <div class="search-result-content">
                        <div class="search-result-title">${highlightMatch(result.title, query)}</div>
                        <div class="search-result-description">${highlightMatch(result.description, query)}</div>
                    </div>
                    <div class="search-result-badge">${Math.round(result.score)}%</div>
                `);

                resultItem.addEventListener('click', (e) => {
                    saveRecentSearch(result);
                    closeSearch();
                });

                resultsContainer.appendChild(resultItem);
            });
        });
    }

    /**
     * Recent Searches im LocalStorage
     */
    function saveRecentSearch(item) {
        let recent = JSON.parse(localStorage.getItem('lexi_recent_searches') || '[]');

        // Entferne Duplikate
        recent = recent.filter(r => r.id !== item.id);

        // Füge an den Anfang hinzu
        recent.unshift({
            id: item.id,
            title: item.title,
            icon: item.icon,
            url: item.url,
            timestamp: new Date().toISOString()
        });

        // Begrenze auf 5 Einträge
        recent = recent.slice(0, 5);

        localStorage.setItem('lexi_recent_searches', JSON.stringify(recent));
    }

    function loadRecentSearches() {
        return JSON.parse(localStorage.getItem('lexi_recent_searches') || '[]');
    }

    function renderRecentSearches() {
        const recent = loadRecentSearches();
        const resultsContainer = document.getElementById('searchResults');

        if (recent.length === 0) return;

        // XSS-safe: Sanitize data from localStorage (could be manipulated)
        window.sanitize.setInnerHTML(resultsContainer, `
            <div class="search-recent">
                <div class="search-recent-title">Zuletzt besucht</div>
                ${recent.map(item => `
                    <a href="${item.url}" class="search-recent-item">
                        <span class="search-recent-icon">${item.icon}</span>
                        <span>${item.title}</span>
                        <span class="search-recent-clear" data-id="${item.id}" title="Entfernen">×</span>
                    </a>
                `).join('')}
            </div>
        `);

        // Clear buttons
        document.querySelectorAll('.search-recent-clear').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const id = btn.getAttribute('data-id');
                let recent = loadRecentSearches();
                recent = recent.filter(r => r.id !== id);
                localStorage.setItem('lexi_recent_searches', JSON.stringify(recent));
                renderRecentSearches();
            });
        });
    }

    /**
     * Öffne Search Modal
     */
    function openSearch() {
        const overlay = document.getElementById('searchOverlay');
        const input = document.getElementById('searchInput');

        if (overlay && input) {
            overlay.classList.add('active');
            input.value = '';
            input.focus();
            renderRecentSearches();
            document.body.style.overflow = 'hidden';
        }
    }

    /**
     * Schließe Search Modal
     */
    function closeSearch() {
        const overlay = document.getElementById('searchOverlay');
        if (overlay) {
            overlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    }

    /**
     * Keyboard Navigation in Suchergebnissen
     */
    function setupKeyboardNavigation() {
        const input = document.getElementById('searchInput');
        if (!input) return;

        input.addEventListener('keydown', (e) => {
            const items = document.querySelectorAll('.search-result-item');
            const selected = document.querySelector('.search-result-item.selected');
            let currentIndex = Array.from(items).indexOf(selected);

            switch(e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    if (currentIndex < items.length - 1) {
                        selected?.classList.remove('selected');
                        items[currentIndex + 1]?.classList.add('selected');
                        items[currentIndex + 1]?.scrollIntoView({ block: 'nearest' });
                    }
                    break;

                case 'ArrowUp':
                    e.preventDefault();
                    if (currentIndex > 0) {
                        selected?.classList.remove('selected');
                        items[currentIndex - 1]?.classList.add('selected');
                        items[currentIndex - 1]?.scrollIntoView({ block: 'nearest' });
                    }
                    break;

                case 'Enter':
                    e.preventDefault();
                    if (selected) {
                        selected.click();
                    }
                    break;

                case 'Escape':
                    e.preventDefault();
                    closeSearch();
                    break;
            }
        });
    }

    /**
     * Live Search während Tippen
     */
    function setupLiveSearch() {
        const input = document.getElementById('searchInput');
        if (!input) return;

        let debounceTimer;

        input.addEventListener('input', (e) => {
            clearTimeout(debounceTimer);

            debounceTimer = setTimeout(() => {
                const query = e.target.value.trim();

                if (query.length < 2) {
                    renderRecentSearches();
                    return;
                }

                const results = search(query);
                renderResults(results, query);
            }, 200); // 200ms debounce
        });
    }

    /**
     * Initialisierung
     */
    function init() {
        // Füge Search Modal zum DOM hinzu
        const searchHTML = `
            <div class="search-overlay" id="searchOverlay">
                <div class="search-container">
                    <div class="search-header">
                        <div class="search-input-wrapper">
                            <span class="search-icon">🔍</span>
                            <input type="text"
                                   class="search-input"
                                   id="searchInput"
                                   placeholder="Suche nach Seiten, Features, Aktionen..."
                                   autocomplete="off"
                                   spellcheck="false">
                            <div class="search-shortcut-hint">
                                <span class="search-key">ESC</span>
                                <span>zum Schließen</span>
                            </div>
                        </div>
                    </div>

                    <div class="search-results" id="searchResults"></div>

                    <div class="search-footer">
                        <div class="search-footer-hints">
                            <div class="search-footer-hint">
                                <span class="search-key">↑↓</span>
                                <span>Navigieren</span>
                            </div>
                            <div class="search-footer-hint">
                                <span class="search-key">↵</span>
                                <span>Auswählen</span>
                            </div>
                        </div>
                        <div>
                            Powered by Lexi AI
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', searchHTML);

        // Setup Event Listeners
        setupLiveSearch();
        setupKeyboardNavigation();

        // Close on overlay click
        document.getElementById('searchOverlay')?.addEventListener('click', (e) => {
            if (e.target.id === 'searchOverlay') {
                closeSearch();
            }
        });

        // Global Keyboard Shortcut: Cmd+K / Ctrl+K
        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                openSearch();
            }
        });

        // Log Keyboard Shortcut (nur in Development)
        if (window.location.hostname === 'localhost') {
            console.log('🔍 Search: Cmd+K / Ctrl+K');
        }
    }

    /**
     * Starte wenn DOM geladen ist
     */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Cleanup function for SPA migration (future-proofing)
    function cleanup() {
        const overlay = document.getElementById('searchOverlay');
        if (overlay && overlay.parentNode) {
            overlay.remove();
        }
    }

    // Expose für andere Scripts
    window.LexiSearch = {
        open: openSearch,
        close: closeSearch,
        cleanup: cleanup
    };
})();
