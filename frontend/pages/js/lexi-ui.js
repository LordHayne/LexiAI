/**
 * Lexi AI UI JavaScript
 * Main entry point for the Lexi AI UI
 */

import { initializeConfig } from './config.js';
import { initializeStatus } from './status.js';


document.addEventListener('DOMContentLoaded', function() {
    initializeConfig();
    initializeStatus();
});
