/**
 * Lexi AI - Verbesserte Navigation mit Dropdown-Menüs
 * Auto-Aktivierung, Dropdown-Funktionalität, Mobile-Menü
 */

(function() {
    'use strict';

    /**
     * Route Mapping für Clean URLs zu tatsächlichen Dateipfaden
     */
    const ROUTE_MAP = {
        '/': '/frontend/index.html',
        '/chat': '/frontend/chat_ui.html',
        '/memory': '/frontend/pages/memory_management_ui.html',
        '/patterns': '/frontend/pages/patterns_ui.html',
        '/gaps': '/frontend/pages/knowledge_gaps_ui.html',
        '/goals': '/frontend/pages/goals_ui.html',
        '/settings': '/frontend/pages/config_ui.html',
        '/dashboard': '/frontend/pages/metrics_dashboard.html'
    };

    /**
     * Aktiviert den passenden Navigation-Link basierend auf der aktuellen URL
     */
    function activateCurrentNavLink() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.nav-link, .nav-dropdown-item');

        navLinks.forEach(link => {
            link.classList.remove('active');

            // Hole data-route Attribut oder href
            const linkRoute = link.getAttribute('data-route') || link.getAttribute('href');

            if (!linkRoute) return;

            // Prüfe ob der Link zur aktuellen Seite passt
            if (currentPath === linkRoute ||
                currentPath === linkRoute.replace('/frontend', '') ||
                (currentPath === '/' && linkRoute.includes('index.html'))) {
                link.classList.add('active');

                // Wenn es ein Dropdown-Item ist, markiere auch das Dropdown
                const dropdown = link.closest('.nav-dropdown');
                if (dropdown) {
                    dropdown.querySelector('.nav-dropdown-toggle')?.classList.add('active');
                }
            }
        });
    }

    /**
     * Dropdown-Menü Funktionalität
     */
    function setupDropdowns() {
        const dropdowns = document.querySelectorAll('.nav-dropdown');

        dropdowns.forEach(dropdown => {
            const toggle = dropdown.querySelector('.nav-dropdown-toggle');
            const menu = dropdown.querySelector('.nav-dropdown-menu');

            if (!toggle || !menu) return;

            // Toggle beim Klick
            toggle.addEventListener('click', (e) => {
                e.stopPropagation();

                // Schließe andere Dropdowns
                dropdowns.forEach(otherDropdown => {
                    if (otherDropdown !== dropdown) {
                        otherDropdown.classList.remove('open');
                        otherDropdown.querySelector('.nav-dropdown-toggle')?.setAttribute('aria-expanded', 'false');
                    }
                });

                // Toggle current
                const isOpen = dropdown.classList.toggle('open');
                toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
            });

            // Schließe Dropdown beim Klick auf ein Item
            const items = menu.querySelectorAll('.nav-dropdown-item');
            items.forEach(item => {
                item.addEventListener('click', () => {
                    dropdown.classList.remove('open');
                    toggle.setAttribute('aria-expanded', 'false');
                });
            });
        });

        // Schließe alle Dropdowns beim Klick außerhalb
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.nav-dropdown')) {
                dropdowns.forEach(dropdown => {
                    dropdown.classList.remove('open');
                    dropdown.querySelector('.nav-dropdown-toggle')?.setAttribute('aria-expanded', 'false');
                });
            }
        });
    }

    /**
     * Mobile Hamburger-Menü
     */
    function setupMobileMenu() {
        const hamburger = document.getElementById('hamburgerMenu');
        const navLinks = document.getElementById('navLinks');

        if (!hamburger || !navLinks) return;

        hamburger.addEventListener('click', () => {
            const isOpen = navLinks.classList.toggle('show');
            hamburger.classList.toggle('active');
            hamburger.setAttribute('aria-label', isOpen ? 'Menü schließen' : 'Menü öffnen');
        });

        // Schließe Menü beim Klick auf einen Link
        const links = navLinks.querySelectorAll('.nav-link, .nav-dropdown-item');
        links.forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('show');
                hamburger.classList.remove('active');
                hamburger.setAttribute('aria-label', 'Menü öffnen');
            });
        });

        // Schließe Menü beim Klick außerhalb
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.nav-bar')) {
                navLinks.classList.remove('show');
                hamburger.classList.remove('active');
                hamburger.setAttribute('aria-label', 'Menü öffnen');
            }
        });
    }

    /**
     * Keyboard Navigation
     */
    function setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            // Alt + Shortcuts
            if (e.altKey) {
                switch(e.key.toLowerCase()) {
                    case 'c':
                        e.preventDefault();
                        window.location.href = ROUTE_MAP['/chat'];
                        break;
                    case 'm':
                        e.preventDefault();
                        window.location.href = ROUTE_MAP['/memory'];
                        break;
                    case 's':
                        e.preventDefault();
                        window.location.href = ROUTE_MAP['/settings'];
                        break;
                    case 'd':
                        e.preventDefault();
                        window.location.href = ROUTE_MAP['/dashboard'];
                        break;
                }
            }

            // ESC schließt Dropdowns und Mobile-Menü
            if (e.key === 'Escape') {
                document.querySelectorAll('.nav-dropdown.open').forEach(dropdown => {
                    dropdown.classList.remove('open');
                    dropdown.querySelector('.nav-dropdown-toggle')?.setAttribute('aria-expanded', 'false');
                });

                const navLinks = document.getElementById('navLinks');
                const hamburger = document.getElementById('hamburgerMenu');
                if (navLinks?.classList.contains('show')) {
                    navLinks.classList.remove('show');
                    hamburger?.classList.remove('active');
                }
            }
        });
    }

    /**
     * Smooth Transitions für Links
     */
    function setupSmoothTransitions() {
        document.querySelectorAll('.nav-link, .nav-dropdown-item').forEach(link => {
            link.addEventListener('click', function(e) {
                // Füge kurze Transition für bessere UX hinzu
                this.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    this.style.transform = '';
                }, 150);
            });
        });
    }

    /**
     * Initialisierung
     */
    function init() {
        activateCurrentNavLink();
        setupDropdowns();
        setupMobileMenu();
        setupKeyboardNavigation();
        setupSmoothTransitions();

        // Log Keyboard Shortcuts (nur in Development)
        if (window.location.hostname === 'localhost') {
            console.log('🎹 Keyboard Shortcuts:');
            console.log('  Alt+C → Chat');
            console.log('  Alt+M → Memory');
            console.log('  Alt+S → Settings');
            console.log('  Alt+D → Dashboard');
            console.log('  ESC   → Close menus');
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
})();
