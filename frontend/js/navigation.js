/**
 * Lexi AI - Navigation Helper
 * Auto-activates navigation links based on current page
 */

(function() {
    'use strict';

    /**
     * Aktiviert den passenden Navigation-Link basierend auf der aktuellen URL
     */
    function activateCurrentNavLink() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.nav-link');

        navLinks.forEach(link => {
            // Entferne active Klasse von allen
            link.classList.remove('active');

            // Prüfe ob der Link zur aktuellen Seite passt
            const linkPath = new URL(link.href, window.location.origin).pathname;

            if (currentPath === linkPath) {
                link.classList.add('active');
            }
            // Spezialfall: Root / sollte Home aktivieren
            else if (currentPath === '/' && linkPath === '/frontend/chat_ui.html') {
                link.classList.add('active');
            }
        });
    }

    /**
     * Smooth Scroll für Links
     */
    function setupSmoothNavigation() {
        document.querySelectorAll('.nav-link').forEach(link => {
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
     * Initialisierung wenn DOM geladen ist
     */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    function init() {
        activateCurrentNavLink();
        setupSmoothNavigation();
    }
})();
