/**
 * Configuration module for Lexi AI
 */

// Populate the form with configuration data
function populateForm(config) {
    document.getElementById('llm-model').value = config.llm_model || '';
    document.getElementById('embedding-model').value = config.embedding_model || '';

    if (config.ollama_url) {
        document.getElementById('ollama-url').value = config.ollama_url;
    }

    if (config.embedding_url) {
        document.getElementById('embedding-url').value = config.embedding_url;
        if (config.embedding_url !== config.ollama_url) {
            document.getElementById('same-url').checked = false;
            document.getElementById('embedding-url').disabled = false;
        }
    }

    if (config.qdrant_host) {
        document.getElementById('qdrant-host').value = config.qdrant_host;
    }

    if (config.qdrant_port) {
        document.getElementById('qdrant-port').value = config.qdrant_port;
    }

    if (config.features) {
        document.querySelectorAll('input[name="features"]').forEach(checkbox => {
            if (config.features[checkbox.value] !== undefined) {
                checkbox.checked = config.features[checkbox.value];
            }
        });
    }

    if (config.system_prompt !== undefined) {
        document.getElementById('system-prompt').value = config.system_prompt;
    }

    if (config.audit_log_path !== undefined) {
        document.getElementById('audit-log-path').value = config.audit_log_path;
    }
}

// Save the configuration to the API
function saveConfig() {
    const formData = {
        api_key: document.getElementById('api-key').value,
        LEXI_API_KEY: document.getElementById('api-key').value,
        llm_model: document.getElementById('llm-model').value,
        embedding_model: document.getElementById('embedding-model').value,
        ollama_url: document.getElementById('ollama-url').value,
        qdrant_host: document.getElementById('qdrant-host').value,
        qdrant_port: parseInt(document.getElementById('qdrant-port').value),
        force_recreate_collection: document.getElementById('force-recreate').checked,
        system_prompt: document.getElementById('system-prompt').value,
        audit_log_path: document.getElementById('audit-log-path').value,
        features: {}
    };

    let hasError = false;
    ['api-key', 'llm-model', 'embedding-model', 'ollama-url', 'qdrant-host', 'qdrant-port'].forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (!field.value.trim()) {
            field.classList.add('input-error');
            hasError = true;
        } else {
            field.classList.remove('input-error');
        }
    });

    if (hasError) {
        const statusMessage = document.getElementById('status-message');
        statusMessage.textContent = 'Bitte füllen Sie alle erforderlichen Felder aus.';
        statusMessage.className = 'status error';
        statusMessage.style.display = 'block';
        return;
    }

    if (document.getElementById('same-url').checked) {
        formData.embedding_url = document.getElementById('ollama-url').value;
    } else {
        formData.embedding_url = document.getElementById('embedding-url').value;
    }

    if (!formData.ollama_url) {
        formData.ollama_url = "http://localhost:11434";
    }

    if (!formData.embedding_url) {
        formData.embedding_url = formData.ollama_url;
    }

    if (!formData.embedding_model) {
        formData.embedding_model = "nomic-embed-text";
    }

    if (!formData.llm_model) {
        formData.llm_model = "gemma3:4b-it-qat";
    }

    document.querySelectorAll('input[name="features"]').forEach(checkbox => {
        formData.features[checkbox.value] = checkbox.checked;
    });

    const statusMessage = document.getElementById('status-message');
    statusMessage.textContent = 'Konfiguration wird gespeichert...';
    statusMessage.className = 'status';
    statusMessage.style.display = 'block';

    fetch('/ui/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            statusMessage.textContent = 'Konfiguration erfolgreich gespeichert';
            statusMessage.className = 'status success';
            if (data.current_config) populateForm(data.current_config);
            import('./status.js').then(module => module.checkSystemStatus());
        } else {
            statusMessage.textContent = `Fehler beim Speichern: ${data.detail || 'Unbekannter Fehler'}`;
            statusMessage.className = 'status error';
        }
    })
    .catch(error => {
        console.error('Error saving config:', error);
        statusMessage.textContent = `Fehler beim Speichern: ${error.message}`;
        statusMessage.className = 'status error';
    });

    setTimeout(() => {
        statusMessage.style.display = 'none';
    }, 5000);
}

// Fetch the current configuration from the API
function fetchCurrentConfig() {
    import('./status.js').then(module => {
        module.updateStatusIndicator('llm-status', 'llm-status-text', false, 'Wird geladen...');
        module.updateStatusIndicator('embedding-status', 'embedding-status-text', false, 'Wird geladen...');
        module.updateStatusIndicator('database-status', 'database-status-text', false, 'Wird geladen...');
        module.updateStatusIndicator('dimension-status', 'dimension-status-text', false, 'Wird geladen...');
    });

    fetch('/v1/config')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.config) {
                populateForm(data.config);
                import('./status.js').then(module => module.checkSystemStatus());
            } else {
                console.error('Error fetching configuration:', data.detail || 'Unknown error');
            }
        })
        .catch(error => {
            console.error('Error fetching configuration:', error);
        });
}

export function initializeConfig() {
    const sameUrlCheckbox = document.getElementById('same-url');
    const ollamaUrlInput = document.getElementById('ollama-url');
    const embeddingUrlInput = document.getElementById('embedding-url');

    sameUrlCheckbox.addEventListener('change', function() {
        embeddingUrlInput.disabled = this.checked;
        if (this.checked) embeddingUrlInput.value = ollamaUrlInput.value;
    });

    ollamaUrlInput.addEventListener('input', function() {
        if (sameUrlCheckbox.checked) {
            embeddingUrlInput.value = this.value;
        }
    });

    if (sameUrlCheckbox.checked) {
        embeddingUrlInput.disabled = true;
    }

    fetchCurrentConfig();

    document.getElementById('config-form').addEventListener('submit', function(e) {
        e.preventDefault();
        saveConfig();
    });

    ['api-key', 'llm-model', 'embedding-model', 'ollama-url', 'qdrant-host', 'qdrant-port'].forEach(fieldId => {
        document.getElementById(fieldId).addEventListener('input', function() {
            this.classList.remove('input-error');
        });
    });
}
