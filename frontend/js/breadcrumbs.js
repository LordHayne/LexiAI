/**
 * Lexi AI - Breadcrumbs Navigation
 * Automatische Generierung basierend auf URL
 */

(function() {
    'use strict';

    /**
     * URL-zu-Name Mapping
     */
    const PAGE_NAMES = {
        '/': { name: 'Dashboard', icon: '🏠' },
        '/frontend/index.html': { name: 'Dashboard', icon: '🏠' },
        '/frontend/chat_ui.html': { name: 'Chat', icon: '💬' },
        '/frontend/pages/memory_management_ui.html': { name: 'Memory', icon: '🧠' },
        '/frontend/pages/patterns_ui.html': { name: 'Patterns', icon: '🔍' },
        '/frontend/pages/knowledge_gaps_ui.html': { name: 'Wissenslücken', icon: '💡' },
        '/frontend/pages/goals_ui.html': { name: 'Ziele', icon: '🎯' },
        '/frontend/pages/config_ui.html': { name: 'Einstellungen', icon: '⚙️' },
        '/frontend/pages/metrics_dashboard.html': { name: 'Metriken', icon: '📊' }
    };

    /**
     * Kategorie-Zuordnung für hierarchische Breadcrumbs
     */
    const PAGE_CATEGORIES = {
        '/frontend/pages/memory_management_ui.html': 'Intelligenz',
        '/frontend/pages/patterns_ui.html': 'Intelligenz',
        '/frontend/pages/knowledge_gaps_ui.html': 'Intelligenz',
        '/frontend/pages/goals_ui.html': 'Intelligenz'
    };

    /**
     * Generiere Breadcrumb-Pfad aus aktueller URL
     */
    function generateBreadcrumbs() {
        const currentPath = window.location.pathname;
        const breadcrumbs = [];

        // Immer Home zuerst (außer wir sind auf Home)
        if (currentPath !== '/' && currentPath !== '/frontend/index.html') {
            breadcrumbs.push({
                name: 'Home',
                icon: '🏠',
                url: '/',
                active: false
            });
        }

        // Füge Kategorie hinzu, falls vorhanden
        const category = PAGE_CATEGORIES[currentPath];
        if (category) {
            breadcrumbs.push({
                name: category,
                icon: '🧠',
                url: null, // Keine URL - nur Info
                active: false
            });
        }

        // Aktuelle Seite
        const currentPage = PAGE_NAMES[currentPath];
        if (currentPage) {
            breadcrumbs.push({
                name: currentPage.name,
                icon: currentPage.icon,
                url: currentPath,
                active: true
            });
        }

        return breadcrumbs;
    }

    /**
     * Rendere Breadcrumbs ins DOM
     */
    function renderBreadcrumbs() {
        const container = document.getElementById('breadcrumbsContainer');
        if (!container) return;

        const breadcrumbs = generateBreadcrumbs();

        // Wenn nur eine Seite (Home oder aktuelle), zeige nichts
        if (breadcrumbs.length <= 1) {
            container.style.display = 'none';
            return;
        }

        container.style.display = 'block';

        const list = container.querySelector('.breadcrumb-list');
        if (!list) return;

        list.innerHTML = '';

        breadcrumbs.forEach((crumb, index) => {
            const item = document.createElement('li');
            item.className = 'breadcrumb-item';
            if (crumb.active) {
                item.classList.add('active');
            }

            // Link oder Span
            if (crumb.url && !crumb.active) {
                const link = document.createElement('a');
                link.href = crumb.url;
                link.className = 'breadcrumb-link';
                // Data from PAGE_NAMES (trusted), but sanitize defensively
                window.sanitize.setInnerHTML(link, `
                    <span class="breadcrumb-icon">${crumb.icon}</span>
                    <span>${crumb.name}</span>
                `);
                item.appendChild(link);
            } else {
                const span = document.createElement('span');
                span.className = 'breadcrumb-link';
                // Data from PAGE_NAMES (trusted), but sanitize defensively
                window.sanitize.setInnerHTML(span, `
                    <span class="breadcrumb-icon">${crumb.icon}</span>
                    <span>${crumb.name}</span>
                `);
                item.appendChild(span);
            }

            list.appendChild(item);

            // Separator (außer beim letzten Item)
            if (index < breadcrumbs.length - 1) {
                const separator = document.createElement('li');
                separator.className = 'breadcrumb-separator';
                separator.setAttribute('aria-hidden', 'true');
                separator.textContent = '›';
                list.appendChild(separator);
            }
        });
    }

    /**
     * Initialisierung
     */
    function init() {
        renderBreadcrumbs();
    }

    /**
     * Starte wenn DOM geladen ist
     */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
